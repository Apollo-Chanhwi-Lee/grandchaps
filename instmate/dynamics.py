"""Song flow: a loudness arc across sections, drum fills into choruses, a short break before the last chorus.

Applied once to a whole song's arranged events (render.leadsheet_to_song), after every section is arranged:

  * arc     every section's velocities are scaled by its place in the song: quiet intro and first verse, each
            chorus a little louder than the last, the final chorus the loudest, a softer bridge, and an outro
            that fades over its last two bars.  The melody follows half as much so it stays in front.
  * fill    the last half bar before a chorus (not the final one) swaps its drum pattern for a snare/tom fill.
  * break   the last bar before the final chorus drops drums and bass, leaving a one-beat snare pickup, so the
            final chorus lands after a moment of space (the reference songs dip like this before the last chorus).
  * crash   a crash cymbal on the first beat of every chorus that has drums.

Fills, pickups and crashes are only added where the section itself plays drums.  Pitches never change, so
melody, chords and the range/length checks are unaffected; drum events stay inside their bar.
"""
from __future__ import annotations

from .arrange import CRASH, SNARE, Event
from .theory import TPQ

S16 = TPQ // 4
HI_TOM, MID_TOM, LOW_TOM = 50, 47, 45
OUTRO_FADE_BARS = 2
MELODY_FOLLOW = 0.5          # the melody moves half as much as the accompaniment


def arc_scales(kinds: list[str]) -> list[float]:
    """Velocity scale per section from its kind and position (see module doc)."""
    last_chorus = max((i for i, k in enumerate(kinds) if k == "chorus"), default=None)
    seen: dict[str, int] = {}
    out = []
    for i, k in enumerate(kinds):
        n = seen.get(k, 0)
        seen[k] = n + 1
        if k == "chorus":
            s = 1.2 if i == last_chorus and n > 0 else 1.0 + 0.05 * n
        elif k == "verse":
            s = 0.78 if n == 0 else 0.88
        elif k == "pre":
            s = 0.9 if n == 0 else 0.95
        elif k in ("bridge", "interlude"):
            s = 0.75
        else:                                       # intro, outro
            s = 0.7
        out.append(s)
    return out


def _scaled(e: Event, s: float) -> Event:
    if e.track == "melody":
        s = 1.0 + (s - 1.0) * MELODY_FOLLOW
    return Event(e.track, e.pitch, e.start, e.dur, max(1, min(127, round(e.velocity * s))))


KICK = 36
HATS, RIDE = (42, 46), 51
E8 = TPQ // 2
BASS_LO = 28


TAMBOURINE, OPEN_HAT, BASS_HI = 54, 46, 60
CHOP_FREE = ("ballad", "classical", "newage")      # piano patterns that already flow (arpeggios): no verse sustain


def phrase_variation(events: list[Event], sections: list[dict], bar_len: int,
                     genre: str | None = None) -> tuple[list[Event], list[str]]:
    """Band-arranging habits applied inside every section (the rule patterns are otherwise the same every bar):
      * drums in 4-bar phrases A-B-A-C / A-B-A-D: bar 2 answers the groove (extra kick on the "and" of 3, open hat on
        the "and" of 4), bar 4 ends with a one-beat fill, and the next phrase's fill is a different one
      * bass: the 2nd and 4th bar of a phrase split a long held note into eighths (root, fifth, root, octave)
      * piano: verses hold one chord per bar (long sustain), choruses keep the genre's short chops
      * chorus: a tambourine doubles every snare hit
    The last bar of a section is left to the section fills of apply_flow."""
    from .arrange import PATTERN
    piano_pat = PATTERN.get(genre, {}).get("piano", genre) if genre else None
    drop: set[int] = set()
    add: list[Event] = []
    notes: list[str] = []
    by_bar: dict[int, list[int]] = {}
    for idx, e in enumerate(events):
        by_bar.setdefault(e.start // bar_len, []).append(idx)
    for s in sections:
        inst, kind = s["instruments"], s["kind"]
        for i in range(s["bars"]):
            b = s["start_bar"] + i
            a = b * bar_len
            last = i == s["bars"] - 1
            pos = i % 4
            idxs = by_bar.get(b, [])
            if "drums" in inst and not last and bar_len >= 4 * (TPQ // 4) * 4:
                if pos == 1:
                    add += [Event("drums", KICK, a + 5 * E8, E8, 78), Event("drums", OPEN_HAT, a + 7 * E8, E8, 58)]
                    drop |= {j for j in idxs if events[j].track == "drums" and events[j].pitch in HATS
                             and events[j].start == a + 7 * E8}
                elif pos == 3:
                    w0 = a + bar_len - TPQ
                    drop |= {j for j in idxs if events[j].track == "drums" and events[j].start >= w0}
                    add += _fill(w0, TPQ, 92, 0 if (i // 4) % 2 == 0 else 3)
            if "bass" in inst and pos in (1, 3) and not last and kind not in ("intro", "outro"):
                long = [j for j in idxs if events[j].track == "bass" and events[j].start == a and events[j].dur >= bar_len // 2]
                if len(long) == 1 and not any(events[j].track == "bass" and events[j].start >= a + bar_len // 2 for j in idxs):
                    root = events[long[0]]
                    drop.add(long[0])
                    half = bar_len // 2
                    add.append(Event("bass", root.pitch, a, half - 20, root.velocity))
                    for k, step in enumerate((0, 7, 0, 12)):
                        p = root.pitch + step if root.pitch + step <= BASS_HI else root.pitch
                        add.append(Event("bass", p, a + half + k * E8, E8 - 20, max(1, root.velocity - 6)))
            if kind == "verse" and "piano" in inst and piano_pat not in CHOP_FREE:
                pn = [j for j in idxs if events[j].track == "piano"]
                onsets = sorted({events[j].start for j in pn})
                if len(onsets) > 2:
                    first = [events[j] for j in pn if events[j].start == onsets[0]]
                    drop |= set(pn)
                    add += [Event("piano", e.pitch, e.start, a + bar_len - e.start - 20, e.velocity - 4) for e in first]
            if kind == "chorus" and "drums" in inst:
                add += [Event("drums", TAMBOURINE, events[j].start, E8, max(1, round(events[j].velocity * 0.55)))
                        for j in idxs if events[j].track == "drums" and events[j].pitch in (SNARE, 39)]
    if drop or add:
        notes.append("밴드 편곡: 드럼 4마디 프레이즈(A-B-A-필인), 베이스 2·4마디 꾸밈, 벌스 피아노 서스테인, 후렴 탬버린")
    return [e for j, e in enumerate(events) if j not in drop] + add, notes


def vary_repeats(events: list[Event], sections: list[dict], bar_len: int) -> tuple[list[Event], list[str]]:
    """Make repeats differ from their first occurrence (rule arrangements are otherwise identical every time):
    from the 2nd verse/chorus on the bass leads into every next bar with a chromatic approach note (an eighth below the
    next root), and from the 2nd chorus on the hi-hats move to the ride cymbal."""
    seen: dict[str, int] = {}
    bass_windows, ride_spans, notes = [], [], []
    for s in sections:
        seen[s["kind"]] = seen.get(s["kind"], 0) + 1
        if s["kind"] not in ("verse", "chorus") or seen[s["kind"]] < 2:
            continue
        a, b = s["start_bar"] * bar_len, (s["start_bar"] + s["bars"]) * bar_len
        if "bass" in s["instruments"]:
            bass_windows.append((a, b))
        if s["kind"] == "chorus" and "drums" in s["instruments"]:
            ride_spans.append((a, b))
        notes.append(f"{s.get('label', s['kind'])}: 반복 변주")
    if not bass_windows and not ride_spans:
        return list(events), []
    out = [Event(e.track, RIDE if e.pitch in HATS and any(a <= e.start < b for a, b in ride_spans) else e.pitch,
                 e.start, e.dur, e.velocity) if e.track == "drums" else e for e in events]
    roots = {e.start: e for e in out if e.track == "bass" and e.start % bar_len == 0}
    added = []
    for a, b in bass_windows:
        for bar_end in range(a + bar_len, b, bar_len):           # into every next bar inside the section
            nxt = roots.get(bar_end)
            if nxt is None or nxt.pitch - 1 < BASS_LO:
                continue
            w0 = bar_end - E8
            out = [Event(e.track, e.pitch, e.start, min(e.dur, w0 - e.start), e.velocity)
                   if e.track == "bass" and e.start < w0 < e.start + e.dur else e for e in out]
            out = [e for e in out if not (e.track == "bass" and w0 <= e.start < bar_end)]
            added.append(Event("bass", nxt.pitch - 1, w0, E8 - 20, max(1, nxt.velocity - 10)))
    return out + added, notes
# fill length per genre (in bars of the bar before the chorus): J-pop plays long technical fills, loop genres short ones
# ---------------------------------------------------------------- interlocking parts
# Each part gets its own rhythmic job instead of every instrument hitting every eighth together: one part drives
# (strums), the other chord part holds long chords, the bass follows the kick (and moves a little more in the 2nd and
# 4th bar of a phrase).
DRIVER = {"ballad": "piano", "classical": "piano", "newage": "piano", "piano_rock": "piano"}    # others: guitar
STRAIGHT_GUITAR = ("pop_punk", "hard_rock")     # driving eighths and riffs are the idiom there
STRAIGHT_BASS = ("pop_punk",)
STRUM = (0, 2, 3, 5, 6, 7)                      # eighths of "down . down up . up down up"
SPARSE = ((0, 3), (0, 3, 6))                    # the part that steps back: odd bars, even bars
BUSY = 6                                        # from this many eighth onsets a part counts as playing every eighth
BASS_MOVE = (6, 7)                              # extra bass eighths in the 2nd and 4th bar of a phrase
E8 = TPQ // 2


def interlock(events: list[Event], sections: list[dict], bar_len: int,
              genre: str | None = None) -> tuple[list[Event], list[str]]:
    """(events, notes).  Only 4/4, and only bars where parts really double each other (see the constants above).
    Onsets where the chord changes are always kept; a kept note is held until the part's next kept onset."""
    if bar_len != 4 * TPQ or not sections:
        return list(events), []
    driver = DRIVER.get(genre, "guitar")
    other = "piano" if driver == "guitar" else "guitar"
    first, last = sections[0]["start_bar"], sections[-1]["start_bar"] + sections[-1]["bars"]
    sec_start = {b: s["start_bar"] for s in sections for b in range(s["start_bar"], s["start_bar"] + s["bars"])}
    by_bar: dict[tuple[int, str], dict[int, list[Event]]] = {}
    for e in events:
        if e.track in ("piano", "guitar", "bass", "drums"):
            by_bar.setdefault((e.start // bar_len, e.track), {}).setdefault(e.start % bar_len, []).append(e)
    drop: set[int] = set()
    hold: dict[int, int] = {}
    did = {"strum": 0, "hold": 0, "bass": 0}

    def busy(onsets: dict[int, list[Event]]) -> bool:
        return len(onsets) >= BUSY and all(t % E8 == 0 for t in onsets)

    def changes(bar: int) -> set[int]:
        """Eighth positions where the chord changes inside the bar, read from a part that plays chords."""
        for track in ("piano", "guitar"):
            on = by_bar.get((bar, track), {})
            if on and sum(len(v) for v in on.values()) >= 2 * len(on):
                out, prev = set(), None
                for t in sorted(on):
                    pcs = frozenset(e.pitch % 12 for e in on[t])
                    if prev is not None and pcs != prev:
                        out.add(t // E8)
                    prev = pcs
                return out
        return {4}

    def thin(bar: int, track: str, keep: set[int], sustain: bool) -> bool:
        on = by_bar[(bar, track)]
        kept = sorted(t for t in on if t // E8 in keep)
        if len(kept) == len(on) or not kept:
            return False
        for t in on:
            if t not in kept:
                drop.update(id(e) for e in on[t])
        if sustain:
            for t, nxt in zip(kept, kept[1:] + [bar_len]):
                for e in on[t]:
                    hold[id(e)] = max(e.dur, nxt - t - S16 // 2)
        return True

    for bar in range(first, last):
        parts = {t: by_bar.get((bar, t), {}) for t in ("piano", "guitar", "bass", "drums")}
        if not any(busy(parts[t]) for t in ("piano", "guitar", "bass")):
            continue
        ch = changes(bar)
        pos = (bar - sec_start[bar]) % 4
        both = bool(parts[driver]) and bool(parts[other])
        if both and busy(parts[other]):
            did["hold"] += thin(bar, other, set(SPARSE[pos % 2]) | ch, sustain=True)
        chordal = parts["guitar"] and sum(len(v) for v in parts["guitar"].values()) >= 3 * len(parts["guitar"])
        if (driver == "guitar" and genre not in STRAIGHT_GUITAR and busy(parts["guitar"]) and chordal
                and (parts["piano"] or parts["bass"])):
            did["strum"] += thin(bar, "guitar", set(STRUM) | ch, sustain=False)
        if busy(parts["bass"]) and genre not in STRAIGHT_BASS and (parts["piano"] or parts["guitar"]):
            kicks = {t // E8 for t, es in parts["drums"].items() if any(e.pitch == KICK for e in es)} or {0, 4}
            keep = kicks | {0} | ch | (set(BASS_MOVE) if pos in (1, 3) else set())
            did["bass"] += thin(bar, "bass", keep, sustain=True)
    if not any(did.values()):
        return list(events), []
    out = []
    for e in events:
        if id(e) in drop:
            continue
        out.append(Event(e.track, e.pitch, e.start, hold[id(e)], e.velocity) if id(e) in hold else e)
    label = {"strum": "기타는 스트럼 리듬", "hold": f"{'피아노' if other == 'piano' else '기타'}는 코드를 길게",
             "bass": "베이스는 킥에 맞춰(2·4번째 마디에서 더 움직임)"}
    return out, ["파트 리듬 분리: " + ", ".join(f"{label[k]} {v}마디" for k, v in did.items() if v)]


FILL_BARS = {"jpop": 1.0, "kpop": 0.25, "hiphop": 0.25}          # others: half a bar


def _fill(start: int, length: int, top: int, variant: int = 0) -> list[Event]:
    """16th-note fill over [start, start+length) with rising velocity.  Variants (rotated through a song):
    0 snare then toms downward, 1 tom run with a kick under every beat, 2 snare roll, 3 "da-da-dum" accents."""
    n = max(1, length // S16)
    if variant % 4 == 0:
        order = [SNARE] * (n - n // 2) + [HI_TOM, HI_TOM, MID_TOM, MID_TOM, LOW_TOM, LOW_TOM][: n // 2]
    elif variant % 4 == 1:
        order = [(HI_TOM, MID_TOM, LOW_TOM)[min(2, 3 * i // n)] for i in range(n)]
    elif variant % 4 == 2:
        order = [SNARE] * n
    else:
        order = [SNARE if i % 3 != 2 else LOW_TOM for i in range(n)]
    order += [LOW_TOM] * (n - len(order))
    ev = [Event("drums", p, start + i * S16, S16, max(1, min(127, round(top * (0.6 + 0.4 * i / max(1, n - 1))))))
          for i, p in enumerate(order[:n])]
    if variant % 4 == 1:
        ev += [Event("drums", KICK, start + i * S16, S16, 80) for i in range(0, n, 4)]
    return ev


def apply_flow(events: list[Event], sections: list[dict], bar_len: int,
               genre: str | None = None) -> tuple[list[Event], list[str]]:
    """(new events, Korean notes for the trace).  `sections`: dicts with kind, start_bar, bars, instruments.
    `genre` sets the fill length (FILL_BARS); fills rotate through the `_fill` variants."""
    if not sections:
        return list(events), []
    kinds = [s["kind"] for s in sections]
    scales = arc_scales(kinds)
    bounds = [(s["start_bar"] * bar_len, (s["start_bar"] + s["bars"]) * bar_len) for s in sections]
    total_end = bounds[-1][1]
    chorus_ids = [i for i, k in enumerate(kinds) if k == "chorus"]
    final = chorus_ids[-1] if len(chorus_ids) >= 2 else None
    notes: list[str] = []

    def sec_of(tick: int) -> int:
        for i, (a, b) in enumerate(bounds):
            if a <= tick < b:
                return i
        raise ValueError(f"event at tick {tick} is outside every section (song ends at {total_end})")

    events, lock_notes = interlock(events, sections, bar_len, genre)
    notes += lock_notes
    events, phrase_notes = phrase_variation(events, sections, bar_len, genre)
    events, vary_notes = vary_repeats(events, sections, bar_len)
    notes += phrase_notes
    notes += vary_notes
    out: list[Event] = []
    removed: list[tuple[int, int, set[str]]] = []    # (from, to, tracks) windows cleared before adding fills
    added: list[Event] = []
    fill_len = max(S16, int(bar_len * FILL_BARS.get(genre, 0.5)) // S16 * S16)
    n_fill = 0
    for i in range(1, len(sections)):
        if kinds[i] != "chorus":
            continue
        prev = sections[i - 1]
        prev_drums = "drums" in prev["instruments"]
        p_end = bounds[i - 1][1]
        if i == final:
            if bounds[i - 1][1] - bounds[i - 1][0] >= bar_len:
                removed.append((p_end - bar_len, p_end, {"drums", "bass"}))
                if prev_drums:
                    added += _fill(p_end - TPQ, TPQ, 100)
                notes.append(f"{prev['label'] if 'label' in prev else prev['kind']} 마지막 마디: 드럼·베이스를 비워 마지막 후렴 전에 쉼")
        elif prev_drums:
            span = min(fill_len, bounds[i - 1][1] - bounds[i - 1][0])
            removed.append((p_end - span, p_end, {"drums"}))
            added += _fill(p_end - span, span, 105, n_fill)
            n_fill += 1
            notes.append(f"{prev['label'] if 'label' in prev else prev['kind']} 끝: 드럼 필인")
        if "drums" in sections[i]["instruments"]:
            added.append(Event("drums", CRASH, bounds[i][0], TPQ, 100))

    for e in events:
        if any(a <= e.start < b and e.track in tracks for a, b, tracks in removed):
            continue
        i = sec_of(e.start)
        s = scales[i]
        if kinds[i] == "outro":
            fade_from = bounds[i][1] - OUTRO_FADE_BARS * bar_len
            if e.start >= fade_from:
                frac = (e.start - fade_from) / (OUTRO_FADE_BARS * bar_len)
                s *= 1.0 - 0.4 * frac
        out.append(_scaled(e, s))
    crash_ticks = {e.start for e in out if e.track == "drums" and e.pitch == CRASH}
    for e in added:
        if e.pitch == CRASH and e.start in crash_ticks:
            continue
        out.append(_scaled(e, scales[sec_of(e.start)]))
    out.sort(key=lambda e: (e.start, e.track, e.pitch))
    lo, hi = min(scales), max(scales)
    notes.insert(0, f"곡 흐름: 구간별 세기 {lo:.2f}~{hi:.2f}배 (마지막 후렴이 가장 큼), 후렴 첫 박 크래시")
    return out, notes
