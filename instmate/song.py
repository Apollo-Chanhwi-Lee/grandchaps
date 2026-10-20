"""작곡메이트: grow a seed melody (4-8 bars) into a 2-4 minute instrumental pop song.

Pop songs repeat a lot: only a few parts need new melody, the rest is repetition with a different arrangement.
  1. structure   - a genre template (ballad, band rock, dance pop) sized to the tempo so the song lasts 2-4 min
  2. melody      - chorus: the seed as written; verse: seed lowered and rhythmically simplified (rule);
                   pre-chorus and bridge: new material. Here a rule-based generator (sequenced / augmented seed
                   fragments) fills them; a learned infilling model can replace `PART_GENERATORS` entries later.
  3. harmony     - chorus: the adopted version's chords (user edits kept); other parts: the chord tool on that
                   part's melody, with the checked regeneration loop; the bridge is biased to darker chords
  4. arrangement - per-part instruments and energy; the final chorus goes up a semitone
  5. self-check  - melodic leaps between parts, clashing bars, identical consecutive blocks -> fixed/regenerated
No vocals or lyrics: the result is an instrumental demo (melody on an instrument).
"""
from __future__ import annotations

import base64

from . import agent, chords as ch
from .genres import PROFILES as _PROFILES
from .arrange import Event, arrange
from .check import CLASH_MAX, clash_rates, length_violations, range_violations
from .export import playback, to_midi, to_musicxml
from .instruments import place_extras
from .mix import apply_levels, resolve as resolve_mix
from .theory import BAR, TPQ, Key, Note, parse_number_input, split_by_bar

TARGET_MIN_S, TARGET_MAX_S = 120, 240          # 2-4 minutes
KEY_CHANGE = 1                                 # semitones for the last chorus
MAX_LEAP = 12                                  # semitones between the last note of a part and the first of the next
KO = {"intro": "인트로", "verse": "벌스", "pre": "프리코러스", "chorus": "후렴", "bridge": "브리지",
      "interlude": "간주", "outro": "아웃트로"}

# (kind, bars, instruments, energy, extra). "all" = every instrument the user kept on. Per genre: instmate/genres.py.
TEMPLATES = {g: [tuple(r) for r in p["form"]] for g, p in _PROFILES.items()}
LENGTH_S = {g: tuple(p["length_s"]) for g, p in _PROFILES.items()}      # preferred length inside 120-240 s
LOOP_BARS = {g: p["loop_bars"] for g, p in _PROFILES.items()}


def duration_note(duration: float, tempo: int) -> str | None:
    """None when the song is 2-4 minutes, else a Korean explanation that the UI shows (and status is not "ok")."""
    if TARGET_MIN_S <= duration <= TARGET_MAX_S:
        return None
    side = "2분에 못 미쳐요" if duration < TARGET_MIN_S else "4분을 넘어요"
    return f"길이 {duration}초: 템포 {tempo}에서는 2~4분에 맞출 수 없어 {side}. 템포를 바꾸면 맞출 수 있어요"


# ---------------------------------------------------------------- 1. structure
def _droppable(secs: list[dict]) -> list[int]:
    """Indices that may be removed to shorten a song, most expendable first: later verses, later pre-choruses, middle
    choruses, the bridge, outro, intro, then the first pre-chorus and verse. Never the first chorus or the key-change chorus."""
    idx = lambda kind: [i for i, s in enumerate(secs) if s["kind"] == kind]
    verses, pres, choruses = idx("verse"), idx("pre"), idx("chorus")
    middle = [i for i in choruses[1:] if not secs[i].get("shift")]
    order = verses[:0:-1] + pres[:0:-1] + middle[::-1] + idx("bridge") + idx("outro") + idx("intro") + pres[:1] + verses[:1]
    return order


FORMS = ("standard", "chorus_first")
# chorus_first ("C-A-B"): the hook comes right after the intro to catch the ear at once, then the usual verse ->
# pre-chorus -> chorus cycle (K-Pop practice; Hwang 2023, KHU thesis, pp. 8-10, CC BY-NC-ND 2.0 KR, see docs/theory_kb.md)


def form_rows(genre: str, form: str = "standard") -> list[tuple]:
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    rows = list(TEMPLATES[genre])
    if form == "chorus_first":
        first = next((r for r in rows if r[0] == "chorus"), None)
        if first is not None:
            at = 1 if rows and rows[0][0] == "intro" else 0
            rows = rows[:at] + [(first[0], first[1], first[2], first[3], {})] + rows[at:]
    return rows


def plan_structure(genre: str, bpm: int, unit: int, form: str = "standard",
                   seed_role: str = "chorus") -> tuple[list[dict], list[str]]:
    """Template sections, lengths rounded to the seed length `unit`, shrunk/extended into 2-4 minutes.

    Too long (slow tempo, long seed): parts are halved (never below the seed length / 4 bars), then repeated parts are
    dropped (`_droppable`). The seed chorus is kept whole, the tempo is never changed. Too short: the last chorus repeats.
    For every seed length 1-16 bars, every genre and every tempo 40-200 this lands in 120-240 s (tests/test_song.py)."""
    notes = []
    secs = [{"kind": k, "bars": b, "inst": i, "energy": e, **x} for k, b, i, e, x in form_rows(genre, form)]
    for s in secs:
        if s["kind"] not in ("verse", "chorus"):
            continue
        # the section that carries the input repeats it whole.  The other one keeps the template's length when the
        # input length does not fit it (a 5-bar input used to make 15-bar verses): its melody is a rule variation
        # cut to the section, so a plain 8 or 16 bars reads better than a multiple of 5.
        fits = s["bars"] % unit == 0 or unit % s["bars"] == 0
        if s["kind"] == seed_role or fits:
            s["bars"] = max(unit, round(s["bars"] / unit) * unit)
        else:
            s["free"] = True                 # not tied to the input's length: may be halved down to 4 bars
    if seed_role == "intro":                 # the input IS the intro: one intro, exactly as long as the input
        if not secs or secs[0]["kind"] != "intro":
            secs.insert(0, {"kind": "intro", "bars": unit, "inst": "all", "energy": 0})
        secs[0]["bars"] = unit
        secs[0]["seed"] = True
    seconds = lambda ss: sum(s["bars"] for s in ss) * 240.0 / bpm
    lo_s, hi_s = max(TARGET_MIN_S, LENGTH_S[genre][0]), min(TARGET_MAX_S, LENGTH_S[genre][1])   # the genre's own range
    # too long (slow tempo): halve parts one at a time, never below one seed length / 4 bars
    order = ["outro", "intro", "verse", "chorus", "pre", "bridge"]
    for kind in order * 2:
        if seconds(secs) <= hi_s:
            break
        for s in secs:
            floor = unit if kind in ("verse", "chorus") and not s.get("free") else 4
            if s["kind"] == kind and not s.get("seed") and s["bars"] // 2 >= floor and seconds(secs) > hi_s:
                s["bars"] //= 2
                notes.append(f"{KO[kind]} {s['bars'] * 2}→{s['bars']}마디")
    # still too long (very slow tempo with a long seed): drop repeated parts. The first chorus (the seed itself) and the
    # key-change chorus are never dropped and the chorus is never shortened below the seed; the tempo is never changed.
    while seconds(secs) > hi_s:
        drop = _droppable(secs)
        pick = next((i for i in drop if not secs[i].get("seed")
                     and seconds(secs) - secs[i]["bars"] * 240.0 / bpm >= lo_s), None)
        if pick is None:
            break
        notes.append(f"{KO[secs[pick]['kind']]} {secs[pick]['bars']}마디 생략")
        del secs[pick]
    # too short (fast tempo / short seed): repeat the last chorus
    while seconds(secs) < lo_s and sum(s["kind"] == "chorus" for s in secs) < 6:
        last = max(i for i, s in enumerate(secs) if s["kind"] == "chorus")
        secs.insert(last + 1, dict(secs[last]))
        notes.append("후렴 반복 추가")
    n = {}
    for s in secs:
        n[s["kind"]] = n.get(s["kind"], 0) + 1
        s["label"] = KO[s["kind"]] + (f" {n[s['kind']]}" if s["kind"] in ("verse", "pre", "chorus") else "")
        if s.get("shift"):
            s["label"] += " (전조)"
    return secs, notes


# ---------------------------------------------------------------- 2. melody per part
def _tile(motif: list[Note], motif_bars: int, bars: int) -> list[Note]:
    out = []
    for r in range(-(-bars // motif_bars)):      # a last partial repeat is started; the caller cuts it at the section's end
        out += [Note(n.pitch, n.start + r * motif_bars * BAR, n.dur) for n in motif]
    return out


def _pad(notes: list[Note], bars: int) -> list[Note]:
    notes = [n for n in notes if n.start < bars * BAR]
    notes = [Note(n.pitch, n.start, min(n.dur, bars * BAR - n.start)) for n in notes]
    end = max((n.start + n.dur for n in notes), default=0)
    return notes + ([Note(None, end, bars * BAR - end)] if end < bars * BAR else [])


def _octave_fit(notes: list[Note], lo: int, hi: int, prefer_down: bool) -> list[Note]:
    """Move the whole phrase by octaves into [lo, hi] (downwards first when prefer_down)."""
    ps = [n.pitch for n in notes if n.pitch is not None]
    if not ps:
        return notes
    for shift in ([-12, 0, -24, 12] if prefer_down else [0, -12, 12]):
        if lo <= min(ps) + shift and max(ps) + shift <= hi:
            return [Note(None if n.pitch is None else n.pitch + shift, n.start, n.dur) for n in notes]
    return notes


def simplify_rhythm(notes: list[Note]) -> list[Note]:
    """Keep only notes starting on a beat; each lasts until the next kept onset (sparser verse rhythm).

    A rest of the given melody is never sung through.  Notes less than half a beat apart form one sounding stretch (a
    shorter gap is articulation, not a rest); a kept note ends where its stretch ends when a silence of half a beat or
    more follows, even if that leaves it shorter than a beat: the user's rest comes before "at least a beat".  Written
    rests count as silence, also after the last note.  A melody that never rests comes out exactly as it always did."""
    breath = TPQ // 2
    sounding = sorted((n for n in notes if n.pitch is not None), key=lambda n: n.start)
    material_end = max((n.start + n.dur for n in notes), default=0)             # written rests included
    run_end: list[int] = []                      # per sounding note: where its unbroken stretch stops ...
    silence: list[int] = []                      # ... and how long the silence after that stretch is
    start_of_run = 0
    for i in range(len(sounding) + 1):
        end = max((m.start + m.dur for m in sounding[start_of_run:i]), default=0)
        if i == len(sounding) or (i > start_of_run and sounding[i].start - end >= breath):
            after = sounding[i].start if i < len(sounding) else material_end
            run_end += [end] * (i - start_of_run)
            silence += [after - end] * (i - start_of_run)
            start_of_run = i
    kept = [i for i, n in enumerate(sounding) if n.start % TPQ == 0]
    out = []
    for k, i in enumerate(kept):
        n = sounding[i]
        rests_after = silence[i] >= breath
        if k + 1 < len(kept):
            nxt = sounding[kept[k + 1]].start
            end = min(nxt, run_end[i]) if rests_after else nxt
        else:                                    # the last kept note: its own length, at least a beat ...
            end = max(n.start + TPQ, n.start + n.dur)
            if rests_after:                      # ... unless the melody itself ends with a rest
                end = min(end, run_end[i])
        out.append(Note(n.pitch, n.start, end - n.start))
    return out


def verse_melody(motif, motif_bars, bars, variant):
    """Seed lowered and simplified; the second verse starts the seed from its middle for some variety."""
    seed = motif
    if variant % 2 == 0 and motif_bars >= 2:          # second verse: seed halves swapped (B A instead of A B)
        half, length = (motif_bars // 2) * BAR, motif_bars * BAR
        back = [Note(n.pitch, n.start - half, n.dur) for n in motif if n.start >= half]
        front = [Note(n.pitch, n.start + length - half, min(n.dur, half - n.start)) for n in motif if n.start < half]
        seed = back + front
    base = _tile(seed, motif_bars, max(bars, motif_bars))
    return _pad(_octave_fit(simplify_rhythm(base), 48, 72, prefer_down=True), bars)


def _scale_step(key: Key, pitch: int, steps: int) -> int:
    scale = sorted({(key.tonic + s) % 12 for s in ([0, 2, 4, 5, 7, 9, 11] if key.mode == "major" else [0, 2, 3, 5, 7, 8, 10])})
    octave, pc = divmod(pitch, 12)
    idx = min(range(len(scale)), key=lambda i: abs(scale[i] - pc))
    i = idx + steps
    return (octave + i // len(scale)) * 12 + scale[i % len(scale)]


def pre_melody(motif, motif_bars, bars, key):
    """Rising sequence: the seed's first 2 bars repeated a scale step higher each time (builds toward the chorus)."""
    # a note that starts inside the cell but sounds past its end (e.g. a long note across bar 2) is cut at the cell end,
    # otherwise the next repetition (2 bars later) would overlap it
    cell = [Note(n.pitch, n.start, min(n.dur, 2 * BAR - n.start)) for n in motif if n.start < 2 * BAR]
    out = []
    for r in range(bars // 2):
        out += [Note(None if n.pitch is None else _scale_step(key, n.pitch, r), n.start + r * 2 * BAR, n.dur)
                for n in cell]
    return _pad(_octave_fit(out, 55, 84, prefer_down=False), bars)


def bridge_melody(motif, motif_bars, bars, key):
    """Augmentation: the seed's last 2 bars at half speed, lowered a third; calmer contrast before the last chorus."""
    cell_start = max(0, motif_bars - 2) * BAR
    cell = [n for n in motif if n.start >= cell_start]
    out = []
    for r in range(bars // 4):
        out += [Note(None if n.pitch is None else _scale_step(key, n.pitch, -2), (n.start - cell_start) * 2 + r * 4 * BAR,
                     n.dur * 2) for n in cell]
    return _pad(_octave_fit(out, 52, 79, prefer_down=True), bars)


def chorus_melody(motif, motif_bars, bars, key):
    """A chorus for a song whose input is the verse: the input's own rhythm a third higher in the scale, so the chorus
    lifts but is still made of the user's melody (a rule, no trained model)."""
    base = [Note(None if n.pitch is None else _scale_step(key, n.pitch, 2), n.start, n.dur)
            for n in _tile(motif, motif_bars, max(bars, motif_bars))]
    return _pad(_octave_fit(base, 55, 86, prefer_down=False), bars)


PART_GENERATORS = {"pre": ("규칙: 상승 반복", pre_melody), "bridge": ("규칙: 확대·하행", bridge_melody)}
SEED_ROLES = ("chorus", "verse", "intro")  # where the user's input melody goes
FILLS = ("own", "rest")                    # the other melodic sections: rule variations of the input, or no melody
MELODIC = ("verse", "pre", "chorus", "bridge")


# ---------------------------------------------------------------- 3-5. harmony, arrangement, checks
def _chunks(melody: list[Note], bars: int, size: int = 8):
    """Split a part's melody into chunks of at most `size` bars (the chord tools were trained on 8-bar segments)."""
    for b0 in range(0, bars, size):
        nb = min(size, bars - b0)
        sub = [Note(n.pitch, n.start - b0 * BAR, n.dur) for n in melody if b0 * BAR <= n.start < (b0 + nb) * BAR]
        yield b0, nb, _pad(sub, nb)


DARK_BIAS = 0.6


def build_song(state: dict, result: dict) -> dict:
    genre, instruments = state["genre"], list(state["instruments"])
    mix = resolve_mix(genre, result["tempo"], result["levels"])
    key = Key(int(state["key_choice"]["tonic"]), state["key_choice"]["mode"])
    motif = parse_number_input(state["tokens"], state["input_tonic"])
    motif_bars = len(split_by_bar(motif))
    motif = _pad(motif, motif_bars)
    romans = list(state["chords"])
    tonic = "I" if key.mode == "major" else "i"
    trace = []
    secs, notes = plan_structure(genre, mix["tempo"], motif_bars, state.get("form", "standard"),
                                 state.get("seed_role") or "chorus")
    trace.append({"step": "곡 구조 설계", "tool": "structure", "detail": f"{genre} 템플릿, 템포 {mix['tempo']}: "
                  + " → ".join(f"{s['label']} {s['bars']}" for s in secs) + (f" (조정: {', '.join(notes)})" if notes else "")})
    if state.get("key_change") is False:      # the user switched the last-chorus key change off
        for s in secs:
            if s.get("shift"):
                s["shift"], s["label"] = 0, s["label"].replace(" (전조)", "")
    tool_name, tool = agent._pick_tool(trace)
    dark_bias = {c.roman: DARK_BIAS for c in ch.vocab(key) if c.dark}
    style_bias = state.get("bias") or {}          # optional {roman: bonus} from a confirmed style; added to DARK_BIAS on the bridge

    seed_kind, fill = state.get("seed_role") or "chorus", state.get("fill") or "own"
    if seed_kind not in SEED_ROLES or fill not in FILLS:
        raise ValueError(f"seed_role must be one of {SEED_ROLES} and fill one of {FILLS}")
    events, melody_all, symbols, clash_all, sections = [], [], [], [], []
    bar0, prev_last = 0, None
    counts = {}
    for s in secs:
        counts[s["kind"]] = counts.get(s["kind"], 0) + 1
        k = Key((key.tonic + s.get("shift", 0)) % 12, key.mode)
        kind, bars = s["kind"], s["bars"]
        source, origin = "없음", "none"
        # --- melody
        if kind == seed_kind and (kind != "intro" or s.get("seed")):
            mel = _pad(_tile(motif, motif_bars, bars), bars)
            mel = [Note(None if n.pitch is None else n.pitch + s.get("shift", 0), n.start, n.dur) for n in mel]
            source, origin = "씨앗 멜로디", "input"
        elif kind in MELODIC and fill == "rest":             # the user asked for no melody outside the input
            mel, source, origin = [Note(None, 0, bars * BAR)], "쉼 (멜로디 없음)", "rest"
        elif kind == "chorus":                               # the input is the verse: the chorus is a rule variation
            mel = chorus_melody(motif, motif_bars, bars, k)
            mel = [Note(None if n.pitch is None else n.pitch + s.get("shift", 0), n.start, n.dur) for n in mel]
            source, origin = "규칙: 씨앗을 3도 위로", "rule"
        elif kind == "verse":
            mel, source, origin = verse_melody(motif, motif_bars, bars, counts["verse"]), "규칙: 씨앗 낮춤·리듬 단순화", "rule"
        elif kind in PART_GENERATORS:
            source, gen = PART_GENERATORS[kind]
            mel, origin = gen(motif, motif_bars, bars, k), "rule"
        else:
            mel = [Note(None, 0, bars * BAR)]
        # --- self-check: leap into this part -> move the part by an octave
        first = next((n.pitch for n in mel if n.pitch is not None), None)
        if prev_last is not None and first is not None and abs(first - prev_last) > MAX_LEAP and kind != seed_kind:
            step = -12 if first > prev_last else 12
            if all(21 <= n.pitch + step <= 108 for n in mel if n.pitch is not None):
                mel = [Note(None if n.pitch is None else n.pitch + step, n.start, n.dur) for n in mel]
                trace.append({"step": "연결 검사", "tool": "check",
                              "detail": f"{s['label']} 첫 음이 앞 파트와 {abs(first - prev_last)}반음 차이 → 옥타브 이동"})
        # --- harmony
        if kind == seed_kind or origin == "rest":            # the input's own chords; a melody-less section loops them
            rs = [romans[i % len(romans)] for i in range(bars)]
        elif kind == "intro":
            rs = [romans[(len(romans) - bars + i) % len(romans)] for i in range(bars)]
        elif kind == "outro":
            rs = [romans[(len(romans) - bars + 1 + i) % len(romans)] for i in range(bars - 1)] + [tonic]
        else:
            rs = []
            for b0, nb, sub in _chunks(mel, bars):
                sub_trace = []
                prog = agent._generate_with_checks(k, genre, sub, tool, sub_trace,
                                                   bias=agent.merge_bias(style_bias, dark_bias if kind == "bridge" else None))
                rs += [c.roman for c in prog]
                regen = [t for t in sub_trace if t["step"].startswith("재생성")]
                if regen:
                    trace.append({"step": "충돌 검사", "tool": "check",
                                  "detail": f"{s['label']} {b0 + 1}~{b0 + nb}마디 재생성 {len(regen)}회"})
            loop = LOOP_BARS.get(genre)
            if loop and kind in ("verse", "pre", "bridge") and len(rs) > loop:   # loop-based genres (K-pop, hip-hop)
                rs = [rs[i % loop] for i in range(len(rs))]
                trace.append({"step": "코드 루프", "tool": "genre",
                              "detail": f"{s['label']}: 앞 {loop}마디 코드 진행을 반복해요 ({' - '.join(rs[:loop])})"})
        prog = [ch.chord_by_roman(k, r) for r in rs]
        inst = instruments if s["inst"] == "all" else ([i for i in s["inst"] if i in instruments] or instruments)
        inst = place_extras(genre, kind, inst, instruments)
        sec_events = arrange(k, genre, mel, prog, inst, s["energy"])
        start = bar0 * BAR
        events += [Event(e.track, e.pitch, e.start + start, e.dur, e.velocity) for e in sec_events]
        melody_all += [Note(n.pitch, n.start + start, n.dur) for n in mel]
        symbols += [ch.chord_symbol(k, c) for c in prog]
        rates = clash_rates(k, mel, prog)
        clash_all += rates
        sections.append({"kind": kind, "label": s["label"], "start_bar": bar0, "bars": bars,
                         "start_sec": round(bar0 * 240.0 / mix["tempo"], 1),
                         "end_sec": round((bar0 + bars) * 240.0 / mix["tempo"], 1),
                         "melody_source": source, "origin": origin, "shift": s.get("shift", 0), "energy": s["energy"],
                         "instruments": inst, "romans": rs, "mean_clash": round(sum(rates) / len(rates), 3)})
        last = [n.pitch for n in mel if n.pitch is not None]
        prev_last = last[-1] if last else prev_last
        bar0 += bars

    # --- self-check: identical consecutive blocks (same melody, chords, instruments, energy) -> vary energy
    for a, b in zip(sections, sections[1:]):
        if (a["kind"] == b["kind"] and a["romans"] == b["romans"] and a["instruments"] == b["instruments"]
                and a["energy"] == b["energy"] and a["shift"] == b["shift"]):
            trace.append({"step": "반복 검사", "tool": "check",
                          "detail": f"{a['label']}와 {b['label']}가 완전히 같음 (편곡 변화 권장)"})

    # the same song flow as a song made from a lead sheet (render.leadsheet_to_song): loudness arc over the sections,
    # phrase variation, fills into the choruses, a break before the last one.  Pitches, starts and lengths of the
    # melody do not change, so the seed check below still compares the user's notes.
    from .dynamics import apply_flow
    events, flow_notes = apply_flow(events, sections, BAR, genre)
    trace.append({"step": "곡 흐름", "tool": "dynamics", "detail": "; ".join(flow_notes)})

    total_bars, duration = bar0, round(bar0 * 240.0 / mix["tempo"], 1)
    rv, lv = range_violations(events), length_violations(events, total_bars)
    problems = [f"음역·길이 위반 {len(rv) + len(lv)}건"] if rv or lv else []
    # chorus must reproduce the seed (notes and adopted chords)
    seed = sorted((n.pitch, n.start, n.dur) for n in motif if n.pitch is not None)
    for sec in (x for x in sections if x["kind"] == seed_kind):
        st = sec["start_bar"] * BAR
        got = sorted((e.pitch - sec["shift"], e.start - st, e.dur) for e in events
                     if e.track == "melody" and st <= e.start < st + motif_bars * BAR)
        if got != seed or sec["romans"][:motif_bars] != romans:
            problems.append(f"{sec['label']}가 씨앗과 다름")
    if problems:
        raise RuntimeError(f"song check failed: {problems}")
    boundary = [clash_all[s["start_bar"]] for s in sections[1:]]
    duration_ok = TARGET_MIN_S <= duration <= TARGET_MAX_S
    new_parts = [r for sec in sections if sec["kind"] not in (seed_kind, "intro", "outro")
                 for r in clash_all[sec["start_bar"]:sec["start_bar"] + sec["bars"]]]
    seed_over = [i + 1 for i, r in enumerate(clash_rates(key, motif, [ch.chord_by_roman(key, r) for r in romans]))
                 if r > CLASH_MAX]
    metrics = {"chorus_seed_match": 1.0, "boundary_mean_clash": round(sum(boundary) / len(boundary), 3),
               "mean_clash": round(sum(clash_all) / len(clash_all), 3),
               "bars_over_threshold": sum(r > CLASH_MAX for r in clash_all),
               "new_part_bars_over_threshold": sum(r > CLASH_MAX for r in new_parts),
               "seed_bars_over_threshold": seed_over}
    if seed_over:
        trace.append({"step": "후렴 충돌 안내", "tool": "check",
                      "detail": f"원본 버전의 마디 {seed_over}가 충돌 기준을 넘어 모든 후렴에 그대로 들어갔어요. "
                                "원본에서 그 마디를 수정하면 곡 전체의 후렴에 반영돼요"})
    trace.append({"step": "곡 검사", "tool": "check",
                  "detail": f"{len(sections)}파트 {total_bars}마디 {duration}초"
                            + ("" if duration_ok else f" (템포 {mix['tempo']}에서는 2~4분에 맞출 수 없음)")
                            + f". 후렴=씨앗 일치, 파트 경계 평균 충돌률 {metrics['boundary_mean_clash']}, 음역·길이 정상"})
    rendered = apply_levels(events, mix["levels"])
    return {
        # "ok" only for a 2-4 minute song; otherwise "duration_out_of_range" with the reason in duration_note (never a silent success)
        "status": "ok" if duration_ok else "duration_out_of_range", "title": "작곡메이트", "genre": genre,
        "sections": sections, "bars": total_bars, "duration_sec": duration, "duration_ok": duration_ok,
        "duration_note": duration_note(duration, mix["tempo"]), "tempo": mix["tempo"], "key": key.to_dict(),
        "chord_tool": tool_name, "symbols": symbols, "metrics": metrics,
        # one chord per bar in the per-bar shape every finished song has ([[tick in bar, symbol], ...]): the band
        # workspace and the band pack read this (a rule-made song could not be saved or packed without it)
        "bar_chords": [[[0, s]] for s in symbols],
        "check": {"clash_rates": [round(r, 3) for r in clash_all], "threshold": CLASH_MAX,
                  "range_violations": len(rv), "length_violations": len(lv)},
        "trace": trace,
        "playback": playback(rendered, genre, mix["tempo"]),
        "midi_b64": base64.b64encode(to_midi(rendered, genre, mix["tempo"])).decode(),
        "musicxml": to_musicxml(key, genre, melody_all, [ch.chord_by_roman(key, tonic)] * total_bars, mix["tempo"],
                                symbols=symbols),
    }
