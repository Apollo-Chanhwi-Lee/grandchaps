"""Genre pattern templates: chords -> per-instrument note events."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .chords import SEVENTHS, ChordDef, SetChord, chord_pcs
from .genres import PROFILES as _PROFILES
from .theory import BAR, METER_PULSE, METER_TICKS, TPQ, Key, Note

INSTRUMENTS = ["piano", "bass", "drums", "guitar", "strings", "keys"]
# keys = "melody piano": the melody harmonized in block chords (right hand) over a root (left hand), the piano
# sketch a song is usually written on; optional, never in the band default
BAND = ["piano", "bass", "drums", "guitar"]          # the default line-up: a school band (vocal = the melody track)
INSTRUMENT_KO = {"piano": "피아노", "bass": "베이스", "drums": "드럼", "guitar": "기타", "strings": "스트링",
                 "keys": "멜로디 피아노", "melody": "보컬 멜로디"}

# bass upper bound 60: a 24-fret bass reaches G4 (67); dance octave pattern peaks at root(<=47)+12
RANGES = {"piano": (48, 79), "bass": (28, 60), "strings": (55, 84), "guitar": (40, 76), "keys": (43, 88),
          "lead": (52, 88), "melody": (21, 108)}
KEYS_TOP = 84             # the harmonized right hand moves down an octave when the melody note is above this
# tempo, General MIDI programs and pattern names come from the genre profiles (instmate/genres.py)
TEMPO = {g: p["tempo"] for g, p in _PROFILES.items()}
PROGRAM = {track: {g: p["programs"][track] for g, p in _PROFILES.items()}
           for track in ("melody", "piano", "bass", "strings", "guitar")}
PROGRAM["keys"] = {g: 0 for g in _PROFILES}                      # grand piano for every genre
# intro lead guitar (render.intro_lead): distortion for the loud band genres, overdrive for rock, clean otherwise
_LEAD_DIST = {"hard_rock": 30, "pop_punk": 30, "jpop": 30, "rock": 29, "piano_rock": 29}
PROGRAM["lead"] = {g: _LEAD_DIST.get(g, 27) for g in _PROFILES}
PATTERN = {g: p["patterns"] for g, p in _PROFILES.items()}     # instrument -> whose pattern it plays (overlaps allowed)
S16 = TPQ // 4
KPOP_STAB = (0, 3, 6, 10, 12)          # 16th positions: a 3-3-4-2 syncopated stab (docs/style_analysis.md: 60% off-beat)
KPOP_BASS = (0, 3, 6, 8, 11, 14)
RIFF_16THS = (0, 3, 6, 8, 10, 14)       # hard-rock riff accents (16th positions)
KICK, SNARE, SIDESTICK, CLAP, HAT, OPEN_HAT, CRASH, RIDE = 36, 38, 37, 39, 42, 46, 49, 51
E8 = TPQ // 2


@dataclass
class Event:
    track: str
    pitch: int
    start: int
    dur: int
    velocity: int

    def to_dict(self):
        return asdict(self)


def voicing(pcs: list[int], center: int, prev: list[int] | None, lo: int, hi: int) -> list[int]:
    """Close-position voicing inside [lo, hi] nearest to the previous voicing (or to `center`)."""
    cands = []
    for inv in range(len(pcs)):
        order = pcs[inv:] + pcs[:inv]
        for base in range(lo, hi + 1):
            if base % 12 != order[0]:
                continue
            v = [base]
            for pc in order[1:]:
                nxt = v[-1] + 1
                while nxt % 12 != pc:
                    nxt += 1
                v.append(nxt)
            if v[-1] <= hi:
                cands.append(v)
    if not cands:
        raise ValueError(f"no voicing for {pcs} in range {lo}-{hi}")
    if prev:
        return min(cands, key=lambda v: sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev))) + abs(len(v) - len(prev)))
    return min(cands, key=lambda v: abs(sum(v) / len(v) - center))


def bass_root(pc: int) -> int:
    """Root in C2-B2 (36-47); fifth (<=54) and upper octave (<=59) stay inside the bass range."""
    return 36 + (pc - 36) % 12


def _piano(genre, bar_start, v, energy):
    ev = []
    vel = 58 + 12 * energy
    if genre == "ballad":
        top = v[0] + 12 if v[0] + 12 <= RANGES["piano"][1] else v[-1]
        seq = [v[0], v[1], v[2], top, v[2], v[1], v[2], v[1]]
        ev += [Event("piano", p, bar_start + i * E8, E8, vel - (6 if i % 2 else 0)) for i, p in enumerate(seq)]
    elif genre == "classical":
        seq = [v[0], v[2], v[1], v[2]] * 2           # Alberti bass
        ev += [Event("piano", p, bar_start + i * E8, E8, vel - (8 if i % 2 else 0)) for i, p in enumerate(seq)]
    elif genre == "rock":
        for b in range(4):
            ev += [Event("piano", p, bar_start + b * TPQ, TPQ - 40, vel + (8 if b in (0, 2) else 0)) for p in v]
    elif genre == "kpop":
        for i, s in enumerate(KPOP_STAB):
            ev += [Event("piano", p, bar_start + s * S16, 2 * S16 - 30, vel + (6 if i == 0 else 0)) for p in v]
    elif genre == "jpop":                         # driving eighths
        for i in range(8):
            ev += [Event("piano", p, bar_start + i * E8, E8 - 30, vel + (6 if i % 2 == 0 else -4)) for p in v]
    elif genre == "hiphop":                       # sparse keys: a held chord on 1 and a short stab on the "and" of 2
        ev += [Event("piano", p, bar_start, 3 * E8 - 20, vel - 6) for p in v]
        ev += [Event("piano", p, bar_start + 3 * E8, E8 - 30, vel - 10) for p in v]
    elif genre == "newage":                       # flowing arpeggio over two octaves
        hi = RANGES["piano"][1]
        up = [x + 12 if x + 12 <= hi else x for x in v]
        seq = [v[0], v[2], up[1], v[2], up[0], v[2], up[1], v[2]]
        ev += [Event("piano", p, bar_start + i * E8, 2 * E8 if i < 7 else E8, vel - 8 - (4 if i % 2 else 0))
               for i, p in enumerate(seq)]          # notes ring into the next one; the last ends on the bar line
    else:  # dance: offbeat stabs
        for b in range(4):
            ev += [Event("piano", p, bar_start + b * TPQ + E8, E8 - 40, vel + 6) for p in v]
    return ev


def _bass(genre, bar_start, root, fifth, energy):
    vel = 72 + 10 * energy
    if genre == "ballad":
        return [Event("bass", root, bar_start, 2 * TPQ, vel), Event("bass", fifth, bar_start + 2 * TPQ, 2 * TPQ, vel - 6)]
    if genre == "classical":
        return [Event("bass", root, bar_start, TPQ, vel), Event("bass", fifth, bar_start + 2 * TPQ, TPQ, vel - 6)]
    if genre in ("rock", "jpop"):
        return [Event("bass", root, bar_start + i * E8, E8 - 20, vel + (6 if i % 2 == 0 else 0)) for i in range(8)]
    if genre == "kpop":
        return [Event("bass", root + (12 if i in (2, 4) else 0), bar_start + s * S16, 2 * S16 - 20, vel + (6 if i == 0 else 0))
                for i, s in enumerate(KPOP_BASS)]
    if genre == "newage":
        return [Event("bass", root, bar_start, BAR - 40, vel - 20)]
    if genre == "hiphop":                         # long sub notes on 1 and the "and" of 2 (808 style)
        return [Event("bass", root, bar_start, 5 * E8 - 20, vel + 8), Event("bass", root, bar_start + 5 * E8, 3 * E8 - 20, vel)]
    return [Event("bass", root + (12 if i % 2 else 0), bar_start + i * E8, E8 - 30, vel + (8 if i % 2 else 0))
            for i in range(8)]


def _drums(genre, bar_start, bar_idx, energy):
    d = []
    add = lambda p, t, vel: d.append(Event("drums", p, bar_start + t, E8, min(127, vel + 10 * energy)))
    if genre == "ballad":
        add(KICK, 0, 70); add(KICK, 5 * E8, 55); add(SIDESTICK, 2 * TPQ, 60)
        for b in range(4):
            add(HAT, b * TPQ, 45)
    elif genre == "classical":
        add(KICK, 0, 50)
        for b in range(4):
            add(RIDE, b * TPQ, 35)
    elif genre in ("rock", "jpop"):
        if bar_idx == 0:
            add(CRASH, 0, 90)
        add(KICK, 0, 95); add(KICK, 2 * TPQ, 90); add(KICK, 5 * E8, 80)
        add(SNARE, TPQ, 95); add(SNARE, 3 * TPQ, 95)
        for i in range(8):
            add(HAT, i * E8, 70 if i % 2 == 0 else 55)
    elif genre == "kpop":                          # syncopated kick, clap on 2 and 4, 16th hats
        for s in (0, 6, 8, 11):
            add(KICK, s * S16, 100 if s in (0, 8) else 85)
        add(CLAP, TPQ, 92); add(CLAP, 3 * TPQ, 92)
        for i in range(16):                        # 16th hats last a 16th so the last one ends on the bar line
            d.append(Event("drums", HAT, bar_start + i * S16, S16,
                           min(127, (62 if i % 4 == 0 else 50 if i % 2 == 0 else 40) + 10 * energy)))
    elif genre == "newage":                        # no kit: one soft ride swell per bar if drums are kept
        add(RIDE, 0, 25)
    elif genre == "punk":                          # pop-punk two-beat: kick and snare alternate on eighths, ride eighths
        if bar_idx == 0:
            add(CRASH, 0, 95)
        for i in range(8):
            add(KICK if i % 2 == 0 else SNARE, i * E8, 96 if i % 2 == 0 else 92)
            add(RIDE, i * E8, 62 if i % 2 == 0 else 50)
    elif genre == "hiphop":                        # boom bap: kick 1 and the "and" of 2, snare 2 and 4, eighth hats
        add(KICK, 0, 100); add(KICK, 5 * E8, 90); add(KICK, 7 * E8, 60)
        add(SNARE, TPQ, 98); add(SNARE, 3 * TPQ, 98)
        for i in range(8):
            add(HAT, i * E8, 58 if i % 2 == 0 else 42)
    else:
        for b in range(4):
            add(KICK, b * TPQ, 100)
            add(OPEN_HAT, b * TPQ + E8, 65)
        add(CLAP, TPQ, 90); add(CLAP, 3 * TPQ, 90)
    return d


def guitar_root(pc: int) -> int:
    """Power-chord root in E2-D#3 (40-51); root + fifth + octave stays inside the guitar range."""
    return 40 + (pc - 40) % 12


def _guitar(genre, bar_start, v, root, energy, bar_len=BAR):
    """Band guitar: power-chord eighths (rock, J-pop), 16th off-beat chops (dance, K-pop), picked arpeggio otherwise."""
    vel = 66 + 10 * energy
    n8 = bar_len // E8
    if genre in ("rock", "jpop"):
        power = [root, root + 7, root + 12]
        return [Event("guitar", p, bar_start + i * E8, E8 - 40, vel + (8 if i % 2 == 0 else 0))
                for i in range(n8) for p in power]
    if genre in ("dance", "kpop"):
        return [Event("guitar", p, bar_start + s * S16, S16 - 20, vel - 4)
                for s in range(2, bar_len // S16, 4) for p in v]
    if genre == "riff":                           # hard-rock riff: syncopated palm-muted power chords on 16ths
        power = [root, root + 7, root + 12]
        hits = [s for s in RIFF_16THS if s * S16 < bar_len]
        return [Event("guitar", p, bar_start + s * S16, 2 * S16 - 30, vel + (10 if k == 0 else 0))
                for k, s in enumerate(hits) for p in power]
    seq = [v[0], v[1], v[2], v[1]]                      # picked arpeggio, eighths
    return [Event("guitar", seq[i % 4], bar_start + i * E8, E8, vel - 10 - (4 if i % 2 else 0)) for i in range(n8)]


def harmonize_note(pitch: int, pcs: list[int], below: int = 2) -> list[int]:
    """Melody note on top with up to `below` chord tones under it inside one octave (close voicing, nearest first).
    A non-chord melody note gets only the nearest chord tone below (a dyad), so a passing tone never stacks a cluster.
    The whole stack moves down an octave while the top note is above KEYS_TOP."""
    top = pitch
    while top > KEYS_TOP:
        top -= 12
    in_chord = top % 12 in pcs
    cands = sorted({top - d for d in range(1, 12) if (top - d) % 12 in pcs and (top - d) % 12 != top % 12}, reverse=True)
    return [top] + cands[: below if in_chord else 1]


def _keys(segment_notes, pcs, root_pc, seg_start, seg_len, energy):
    """Harmonized melody for one chord span: block chords on the melody's rhythm and a left-hand root on the span start
    (C3-B3, held through the span)."""
    vel = 62 + 8 * energy
    ev = []
    for n in segment_notes:
        for k, p in enumerate(harmonize_note(n.pitch, pcs)):
            if k and p < RANGES["keys"][0]:          # a very low melody note: keep it, drop the harmony under it
                continue
            ev.append(Event("keys", p, n.start, n.dur, vel + (8 if k == 0 else 0)))
    lh = 48 + (root_pc - 48) % 12
    ev = [e for e in ev if not (e.start == seg_start and e.pitch == lh)]      # no doubled key under the left hand
    ev.append(Event("keys", lh, seg_start, seg_len - 20, vel - 6))
    return ev


def _strings(genre, bar_start, v, energy):
    vel = 52 + 10 * energy
    if genre in ("dance", "kpop"):
        return [Event("strings", p, bar_start + h * 2 * TPQ, 2 * TPQ - 30, vel) for h in range(2) for p in v]
    return [Event("strings", p, bar_start, BAR - 20, vel) for p in v]


# ---- other meters (3/4, 2/4, 6/8): the same styles built from the bar length; 4/4 keeps the patterns above
def _pulses(bar_len: int, meter: str) -> list[int]:
    return list(range(0, bar_len, METER_PULSE[meter]))


def _piano_m(genre, bar_start, v, energy, bar_len, meter):
    vel = 58 + 12 * energy
    n8 = bar_len // E8
    if genre in ("ballad", "classical", "newage"):
        top = v[0] + 12 if v[0] + 12 <= RANGES["piano"][1] else v[-1]
        cycle = [v[0], v[1], v[2], top, v[2], v[1]] if genre in ("ballad", "newage") else [v[0], v[2], v[1], v[2]]
        return [Event("piano", cycle[i % len(cycle)], bar_start + i * E8, E8, vel - (6 if i % 2 else 0))
                for i in range(n8)]
    pulse = METER_PULSE[meter]
    ps = _pulses(bar_len, meter)
    if genre in ("rock", "jpop"):
        return [Event("piano", p, bar_start + t, pulse - 40, vel + (8 if k == 0 else 0)) for k, t in enumerate(ps) for p in v]
    return [Event("piano", p, bar_start + t + E8, E8 - 40, vel + 6) for t in ps for p in v]


def _bass_m(genre, bar_start, root, fifth, energy, bar_len):
    vel = 72 + 10 * energy
    half = bar_len // 2
    if genre in ("ballad", "newage"):
        return [Event("bass", root, bar_start, half, vel), Event("bass", fifth, bar_start + half, bar_len - half, vel - 6)]
    if genre == "classical":
        return [Event("bass", root, bar_start, TPQ, vel), Event("bass", fifth, bar_start + half, TPQ, vel - 6)]
    n8 = bar_len // E8
    if genre in ("rock", "jpop"):
        return [Event("bass", root, bar_start + i * E8, E8 - 20, vel + (6 if i % 2 == 0 else 0)) for i in range(n8)]
    return [Event("bass", root + (12 if i % 2 else 0), bar_start + i * E8, E8 - 30, vel + (8 if i % 2 else 0))
            for i in range(n8)]


def _drums_m(genre, bar_start, bar_idx, energy, bar_len, meter):
    d = []
    add = lambda p, t, vel: d.append(Event("drums", p, bar_start + t, E8, min(127, vel + 10 * energy)))
    ps = _pulses(bar_len, meter)
    if genre == "ballad":
        add(KICK, 0, 70)
        for t in ps[1:]:
            add(SIDESTICK, t, 60)
        for t in ps:
            add(HAT, t, 45)
    elif genre == "newage":
        add(RIDE, 0, 25)
    elif genre == "classical":
        add(KICK, 0, 50)
        for t in ps:
            add(RIDE, t, 35)
    elif genre in ("rock", "jpop"):
        if bar_idx == 0:
            add(CRASH, 0, 90)
        add(KICK, 0, 95)
        for t in ps[1:]:
            add(SNARE, t, 95)
        for i in range(bar_len // E8):
            add(HAT, i * E8, 70 if i % 2 == 0 else 55)
    else:
        for t in ps:
            add(KICK, t, 100)
            add(OPEN_HAT, t + E8, 65)
        for t in ps[1:]:
            add(CLAP, t, 90)
    return d


def _strings_m(genre, bar_start, v, energy, bar_len):
    vel = 52 + 10 * energy
    if genre in ("dance", "kpop"):
        half = bar_len // 2
        return [Event("strings", p, bar_start + h * half, half - 30, vel) for h in range(2) for p in v]
    return [Event("strings", p, bar_start, bar_len - 20, vel) for p in v]


def _voice_pcs(key: Key, c: ChordDef) -> list[int]:
    """Three pitch classes to voice: root/3rd/5th, or root/3rd/7th for a dominant seventh."""
    if isinstance(c, SetChord):
        return list(c.voice)
    return chord_pcs(key, c)[:3] if c.quality not in SEVENTHS else [chord_pcs(key, c)[k] for k in (0, 1, 3)]


def arrange(key: Key, genre: str, melody: list[Note], chords: list[ChordDef],
            instruments: list[str], energy: int = 0) -> list[Event]:
    """One chord per bar (the original model): a thin wrapper over `arrange_segments`."""
    return arrange_segments(key, genre, melody, [[(0, c)] for c in chords], instruments, energy)


def arrange_segments(key: Key, genre: str, melody: list[Note], bar_chords: list[list[tuple[int, ChordDef]]],
                     instruments: list[str], energy: int = 0, meter: str = "4/4") -> list[Event]:
    """Like `arrange`, but a bar may hold several chords: `bar_chords[i]` = [(tick in bar, chord), ...] with
    ascending ticks, the first at 0. Each chord plays the genre pattern of its bar restricted to its span:
    events starting inside [start, next start) are kept and clipped to the span end. Drums do not depend
    on the chord and play once per bar. With one chord per bar in 4/4 the output equals the original `arrange`.
    `meter`: 4/4 (original patterns), 3/4, 2/4 or 6/8 (patterns built from the bar length).
    """
    bar_len = METER_TICKS[meter]
    std = meter == "4/4"
    pat = PATTERN[genre]            # e.g. hip-hop plays K-pop guitar chops; the original genres play their own
    pp, pb, pd, pg, ps = pat["piano"], pat["bass"], pat["drums"], pat["guitar"], pat["strings"]
    unknown = set(instruments) - set(INSTRUMENTS)
    if unknown:
        raise ValueError(f"unknown instruments {sorted(unknown)}")
    events = [Event("melody", n.pitch, n.start, n.dur, 90) for n in melody if n.pitch is not None]
    prev_piano = prev_str = prev_gtr = None
    for i, segs in enumerate(bar_chords):
        start = i * bar_len
        if not segs or segs[0][0] != 0 or any(a[0] >= b[0] for a, b in zip(segs, segs[1:])) or segs[-1][0] >= bar_len:
            raise ValueError(f"bar {i + 1}: chord ticks must ascend from 0 inside the bar, got {[t for t, _ in segs]}")
        for j, (t0, c) in enumerate(segs):
            t1 = segs[j + 1][0] if j + 1 < len(segs) else bar_len

            def take(evs):                          # events starting inside this chord's span, clipped to its end
                if len(segs) == 1:
                    return evs
                return [Event(e.track, e.pitch, e.start, min(e.start + e.dur, start + t1) - e.start, e.velocity)
                        for e in evs if t0 <= e.start - start < t1]

            pcs = _voice_pcs(key, c)
            harmony: list[Event] = []
            if "piano" in instruments:
                lo, hi = RANGES["piano"]
                center = 55 if pp == "classical" else 62
                prev_piano = voicing(pcs, center, prev_piano, lo, hi)
                harmony += (_piano(pp, start, prev_piano, energy) if std
                            else _piano_m(pp, start, prev_piano, energy, bar_len, meter))
            if "bass" in instruments:
                bass_pc = c.bass if isinstance(c, SetChord) else chord_pcs(key, c)[0]
                root = bass_root(bass_pc)
                fifth = root if isinstance(c, SetChord) and c.bass != c.pcs[0] else root + 7   # slash bass: repeat it
                harmony += (_bass(pb, start, root, fifth, energy) if std
                            else _bass_m(pb, start, root, fifth, energy, bar_len))
            if len(segs) > 1 and "bass" in instruments and pb in ("ballad", "classical", "newage"):
                # the slow bass pattern restarts with each chord: root at the change, fifth halfway if there is room
                harmony = [e for e in harmony if e.track != "bass"]
                seg_len, vel = t1 - t0, 72 + 10 * energy
                held = seg_len // 2 // E8 * E8 if seg_len >= 2 * TPQ else seg_len
                slow = pb in ("ballad", "newage")
                harmony.append(Event("bass", root, start + t0, held if slow else min(TPQ, held), vel))
                if seg_len - held >= TPQ:
                    harmony.append(Event("bass", fifth, start + t0 + held, seg_len - held if slow else TPQ,
                                         vel - 6))
            if "guitar" in instruments:
                lo, hi = RANGES["guitar"]
                prev_gtr = voicing(pcs, 57, prev_gtr, 48, hi)
                gpc = c.bass if isinstance(c, SetChord) else chord_pcs(key, c)[0]
                harmony += _guitar(pg, start, prev_gtr, guitar_root(gpc), energy, bar_len)
            events += take(harmony)
            if "keys" in instruments:
                span0, span1 = start + t0, start + t1
                seg = [n for n in melody if n.pitch is not None and span0 <= n.start < span1]
                seg = [Note(n.pitch, n.start, min(n.dur, span1 - n.start)) for n in seg]
                full = chord_pcs(key, c) if not isinstance(c, SetChord) else list(c.pcs)
                kpc = c.bass if isinstance(c, SetChord) else chord_pcs(key, c)[0]
                events += _keys(seg, full, kpc, span0, span1 - span0, energy)
            if "drums" in instruments and j == 0:
                events += (_drums(pd, start, i, energy) if std
                           else _drums_m(pd, start, i, energy, bar_len, meter))
            if "strings" in instruments:
                lo, hi = RANGES["strings"]
                prev_str = voicing(pcs, 67, prev_str, lo, hi)
                if len(segs) == 1:
                    events += (_strings(ps, start, prev_str, energy) if std
                               else _strings_m(ps, start, prev_str, energy, bar_len))
                else:                               # sustained parts re-enter with each chord
                    pulsed = ps in ("dance", "kpop")
                    step = bar_len // 2 if pulsed else t1 - t0
                    for k in range(t0, t1, step):
                        events += [Event("strings", p, start + k, min(step, t1 - k) - (30 if pulsed else 20),
                                         52 + 10 * energy) for p in prev_str]
    return events
