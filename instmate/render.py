"""M3: render any lead sheet (instmate/leadsheet.py text format) end to end with the rule-based arrangement.

    lead sheet text -> parse -> (transpose) -> per-section arrangement -> MIDI / MusicXML / playback / checks

`leadsheet_to_song` returns the same result shape as `song.build_song`, so the UI song card, playback,
MIDI download and sheet code paths work unchanged. `song_to_leadsheet` goes the other way: any rule-based song
can be written in the common text format (round trip: song -> lead sheet -> song keeps melody and chords).

Chords
  * A symbol becomes a pitch-class set (`chords.SetChord`): all 16 qualities of the format
    ('', m, 7, maj7, m7, dim, dim7, aug, sus2, sus4, 6, m6, 9, add9, m7b5, 7sus4) and a slash bass (`C/E`).
    The bass plays the slash note; a slash note outside the chord counts as a chord tone in the clash check.
  * A bar may hold several chord marks. Marks are snapped to the nearest beat (later marks win a beat) and
    the arrangement follows the change: each chord plays its part of the bar (`arrange.arrange_segments`).
    So a change at beat 3 (half bar) is honoured; a change on beat 2 or 4 also works, but a change between
    beats (e.g. on an eighth note) is moved to the nearest beat. A bar without a mark keeps the previous
    chord; bars before the first mark of the sheet take the first chord of the sheet; a sheet with no chord
    at all gets chords from the rule-based chord tool.
  * The clash check judges each melody note against the chord in effect at its start.

Limitations
  * Meters 4/4, 3/4, 2/4 and 6/8 render. 4/4 uses the original genre patterns; the other meters use the same
    styles built from the bar length (`arrange._piano_m` and friends). Chord changes snap to the pulse
    (beat; the dotted quarter in 6/8). A non-4/4 sheet without any chord mark gets a fixed I-V-vi-IV
    (minor: i-VI-III-VII) cycle, since the chord tool only knows 4/4 bars.
  * Tempo above 200 BPM (valid in a sheet, up to 220) is clamped to 200, the limit of `mix.resolve`.
  * `target_key` must have the same mode as the sheet key (C -> G, Am -> Em); melody and chords are
    shifted by the smallest interval (-6..+5 semitones).
"""
from __future__ import annotations

import base64
import io

import mido

from . import agent, chords as ch
from .arrange import BAND, INSTRUMENTS, Event, arrange_segments
from .check import CLASH_MAX, clash_rates, length_violations, range_violations
from .dynamics import apply_flow
from .chords import GENRE_KO, SetChord
from .export import playback, to_midi, to_musicxml
from .instruments import place_extras
from .leadsheet import (KEYS, PITCH_MAX, PITCH_MIN, ChordMark, LeadNote, LeadSheet, LeadsheetError, _CHORD_RE,
                        from_notes, parse_leadsheet, serialize_leadsheet, to_notes, validate_leadsheet)
from .mix import TEMPO_MAX, apply_levels, resolve as resolve_mix
from .song import KEY_CHANGE, MAX_LEAP, TARGET_MAX_S, TARGET_MIN_S, TEMPLATES, _chunks, duration_note
from .theory import METER_PULSE, METER_TICKS, NOTE_NAMES, TPQ, Key, Note

# 팝/민요 are source tags of the v2 training data (docs/leadsheet_format.md), mapped to the nearest arrangement template
GENRE_EN = {"발라드": "ballad", "록": "rock", "댄스": "dance", "클래식": "classical", "팝": "ballad", "민요": "classical", "기타": "ballad",
            "케이팝": "kpop", "제이팝": "jpop", "뉴에이지": "newage", "힙합": "hiphop", "피아노 록": "piano_rock",
            "팝펑크": "pop_punk", "하드록": "hard_rock", "감성록": "emo_rock", "모던록": "modern_rock"}
KIND = {"인트로": "intro", "벌스": "verse", "프리": "pre", "후렴": "chorus", "브릿지": "bridge",
        "간주": "interlude", "아웃트로": "outro"}
NAME = {v: k for k, v in KIND.items()}
KIND["구간"] = "verse"                 # unlabeled clip: arranged like a verse (added after NAME so "verse" still maps back to 벌스)
# kinds whose energy is forced to 0 ("light"); chorus keeps the template's energy 1
LIGHT = {"intro", "bridge", "interlude", "outro"}

_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_MIXED_NAMES = NOTE_NAMES                                    # C C# D Eb E F F# G Ab A Bb B (the app's spelling)

# chord quality -> intervals above the root (root first)
QUALITY_IVS = {
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6), "dim7": (0, 3, 6, 9), "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "9": (0, 4, 7, 10, 2), "add9": (0, 4, 7, 2),
    "m7b5": (0, 3, 6, 10), "7sus4": (0, 5, 7, 10),
}
_TRIAD_VOICE = {"", "m", "dim", "aug", "sus2", "sus4", "add9"}     # voiced as first three tones; others R, 3rd, 7th/6th
_TENSIONS = {"": {2, 9, 11}, "6": {2, 9, 11}, "add9": {2, 9, 11}, "maj7": {2, 9, 11}, "m": {2, 5, 10},
             "m7": {2, 5, 10}, "m6": {2, 5, 10}, "7": {2, 9}, "9": {2, 9}, "sus2": {2, 9}, "sus4": {2, 9},
             "7sus4": {2, 9}, "dim": set(), "dim7": set(), "m7b5": set(), "aug": set()}
_MAPPED_QUALITY = {"m": "min", "m7": "min", "m6": "min", "dim": "dim", "dim7": "dim", "m7b5": "dim",
                   "7": "dom7", "9": "dom7", "7sus4": "dom7"}


# ---------------------------------------------------------------- keys and chord symbols
def key_of(name: str) -> Key:
    """'G' -> Key(7, major), 'F#m' -> Key(6, minor). Only names of `leadsheet.KEYS` are accepted."""
    if name not in KEYS:
        raise ValueError(f"알 수 없는 조: '{name}' (예: C, G, F#, Bb, Am, F#m, Ebm)")
    minor = name.endswith("m")
    root = name[:-1] if minor else name
    return Key(_root_pc(root), "minor" if minor else "major")


def _root_pc(text: str) -> int:
    return (_LETTER[text[0]] + {"#": 1, "b": -1}.get(text[1:], 0)) % 12


def parse_chord(symbol: str, key: Key) -> SetChord:
    """Lead-sheet chord symbol -> chord with explicit pitch classes (see the module docstring)."""
    m = _CHORD_RE.fullmatch(symbol)
    if not m or m.group(2) not in QUALITY_IVS:
        raise ValueError(f"알 수 없는 코드 이름: '{symbol}'")
    root_s, quality, bass_s = m.groups()
    root = _root_pc(root_s)
    ivs = QUALITY_IVS[quality]
    pcs = [(root + i) % 12 for i in ivs]
    bass = _root_pc(bass_s) if bass_s else root
    if bass not in pcs:
        pcs.append(bass)
    voice_ivs = ivs[:3] if quality in _TRIAD_VOICE else (ivs[0], ivs[1], ivs[3])
    minor = quality in ("m", "m7", "m6", "dim", "dim7", "m7b5")
    return SetChord(roman=symbol, offset=(root - key.tonic) % 12, quality=_MAPPED_QUALITY.get(quality, "maj"),
                    dark=minor, bright=quality in ("", "6", "add9", "maj7"),
                    pcs=tuple(pcs), voice=tuple((root + i) % 12 for i in voice_ivs), bass=bass,
                    tension_ivs=frozenset(_TENSIONS[quality] - set(ivs)), symbol=symbol)


def transpose_symbol(symbol: str, semitones: int, flats: bool | None = True) -> str:
    """Shift a chord symbol's root (and slash bass). Black keys are spelled with flats (True), sharps (False)
    or the app's mixed spelling (None, for C major / A minor)."""
    if semitones % 12 == 0:
        return symbol
    m = _CHORD_RE.fullmatch(symbol)
    if not m:
        raise ValueError(f"알 수 없는 코드 이름: '{symbol}'")
    root_s, quality, bass_s = m.groups()
    names = _MIXED_NAMES if flats is None else _FLAT_NAMES if flats else _SHARP_NAMES
    out = names[(_root_pc(root_s) + semitones) % 12] + quality
    return out + ("/" + names[(_root_pc(bass_s) + semitones) % 12] if bass_s else "")


def _shift_to(source: Key, target: Key) -> int:
    """Smallest shift (-6..+5) that moves `source` onto `target`."""
    if source.mode != target.mode:
        raise ValueError(f"target_key의 조성(장조/단조)이 악보의 조와 달라요: {source.name} → {target.name}")
    return (target.tonic - source.tonic + 6) % 12 - 6


# ---------------------------------------------------------------- intro lead melody
LEAD_RANGE = (52, 88)


def intro_lead(sec_info: list[dict], bar_len: int, meter: str) -> tuple[list[Event], str | None]:
    """A band intro states the hook: when an intro plays guitar and has (almost) no melody, a lead guitar plays the
    first chorus's opening bars over the intro chords. Bars where that line clashes with the intro chords
    (clash rate above CLASH_MAX) stay instrumental. Returns ("lead" events, Korean trace note)."""
    chorus = next((s for s in sec_info if s["kind"] == "chorus" and any(n.pitch is not None for n in s["mel"])), None)
    if chorus is None:
        return [], None
    out, used = [], 0
    for s in sec_info:
        if s["kind"] != "intro" or "guitar" not in s["inst"]:
            continue
        if sum(n.pitch is not None for n in s["mel"]) >= s["bars"]:
            continue                                   # the intro already has its own melody
        span = s["bars"] * bar_len
        line = [n for n in chorus["mel"] if n.pitch is not None and n.start < span]
        if not line:
            continue
        shift = 0
        while max(n.pitch for n in line) + shift > LEAD_RANGE[1]:
            shift -= 12
        while min(n.pitch for n in line) + shift < LEAD_RANGE[0]:
            shift += 12
        line = [Note(n.pitch + shift, n.start, min(n.dur, span - n.start)) for n in line]
        end = max(n.start + n.dur for n in line)
        if end < span:                                 # the chorus is silent in its last bar(s): the bars still exist
            line.append(Note(None, end, span - end))
        rates = clash_rates(s["key"], line, s["segs"], meter)
        ok_bars = {i for i, r in enumerate(rates) if r <= CLASH_MAX}
        keep = [n for n in line if n.pitch is not None and n.start // bar_len in ok_bars]
        out += [Event("lead", n.pitch, s["lo"] + n.start, n.dur, 84) for n in keep]
        used += len(ok_bars)
    if not out:
        return [], None
    return out, f"인트로에서 리드 기타가 후렴 멜로디를 먼저 들려줘요 ({used}마디, 코드와 부딪히는 마디는 쉼)"


# ---------------------------------------------------------------- section style (from song.TEMPLATES)
# The line-up before the band default (piano, bass, drums, strings; no guitar).  Only used to reproduce files made
# with it, e.g. the published listening-test clips (scripts/make_listening_samples.py, docs/listen/round1).
LEGACY_ALL = ["piano", "bass", "drums", "strings"]
_LS = ["piano", "strings"]
LEGACY_TEMPLATES = {
    "ballad": [("intro", 8, _LS, 0), ("verse", 16, ["piano", "bass", "strings"], 0), ("pre", 8, ["piano", "bass", "strings"], 0),
               ("chorus", 16, "all", 1), ("verse", 16, "all", 0), ("pre", 8, "all", 0), ("chorus", 16, "all", 1),
               ("bridge", 8, _LS, 0), ("chorus", 16, "all", 1), ("outro", 8, _LS, 0)],
    "rock": [("intro", 8, "all", 1), ("verse", 16, ["bass", "drums", "piano"], 0), ("pre", 8, "all", 0), ("chorus", 16, "all", 1),
             ("verse", 16, "all", 0), ("pre", 8, "all", 0), ("chorus", 16, "all", 1), ("bridge", 8, "all", 0),
             ("chorus", 16, "all", 1), ("outro", 8, "all", 1)],
    "dance": [("intro", 8, ["drums", "bass"], 0), ("verse", 16, ["drums", "bass", "piano"], 0), ("pre", 8, "all", 0),
              ("chorus", 16, "all", 1), ("verse", 8, "all", 0), ("pre", 8, "all", 0), ("chorus", 16, "all", 1),
              ("bridge", 8, _LS, 0), ("chorus", 16, "all", 1), ("outro", 8, ["drums", "bass"], 0)],
}
LEGACY_TEMPLATES["classical"] = LEGACY_TEMPLATES["ballad"]


def section_style(genre: str, kind: str, legacy_lineup: bool = False) -> tuple[list[str], int]:
    """(instruments, energy) of a section kind: the genre's first template entry of that kind (verse: fewer
    instruments; the last pre-chorus entry: more; chorus: all, energy 1); interlude follows the intro.
    "all" = the band line-up (BAND), or LEGACY_ALL with `legacy_lineup`."""
    if legacy_lineup and genre not in LEGACY_TEMPLATES:
        raise ValueError(f"no legacy line-up for genre {genre!r}")
    rows = LEGACY_TEMPLATES[genre] if legacy_lineup else TEMPLATES[genre]
    want = "intro" if kind == "interlude" else kind
    hits = [r for r in rows if r[0] == want] or [r for r in rows if r[0] == "verse"]   # e.g. a 프리 in hip-hop
    row = hits[-1] if kind == "pre" else hits[0]
    inst = list(LEGACY_ALL if legacy_lineup else BAND) if row[2] == "all" else list(row[2])
    return inst, 0 if kind in LIGHT else row[3]


# ---------------------------------------------------------------- lead sheet -> song
def _snap(tick: int, meter: str) -> int:
    """Nearest pulse start; never the bar end."""
    pulse = METER_PULSE[meter]
    if tick <= 0:
        return 0
    return min((tick + pulse // 2) // pulse * pulse, METER_TICKS[meter] - pulse)


def _bar_marks(sheet: LeadSheet) -> list[list[tuple[int, str]]]:
    """Per bar [(tick, symbol), ...] with the chord carried over from previous bars; [] at bars with no chord
    anywhere before them are filled from the first chord of the sheet (or left as [] if the sheet has none)."""
    raw = [sorted(b.chords, key=lambda c: c.start) for s in sheet.sections for b in s.bars]
    first = next((m.symbol for marks in raw for m in marks), None)
    if first is None:
        return [[] for _ in raw]
    cur, out = first, []
    for marks in raw:
        at: dict[int, str] = {}
        for m in marks:
            at[_snap(m.start, sheet.meter)] = m.symbol           # later marks win a beat
        at.setdefault(0, cur)
        segs: list[tuple[int, str]] = []
        for t in sorted(at):
            if not segs or segs[-1][1] != at[t]:    # a repeated chord is not a change
                segs.append((t, at[t]))
        cur = segs[-1][1]
        out.append(segs)
    return out


def _bar_leaps(sections: list[dict], firsts: list, lasts: list) -> list[dict]:
    out = []
    for i in range(1, len(sections)):
        a, b = lasts[i - 1], firsts[i]
        if a is not None and b is not None and abs(a - b) > MAX_LEAP:
            out.append({"from": sections[i - 1]["label"], "to": sections[i]["label"], "semitones": abs(a - b)})
    return out


# Who plays the melody.  It changes the sound written into the MIDI and whether the app adds its own lead guitar
# (the intro line that quotes the chorus): with a lead guitar as the melody, a second lead line would double it.
# The melody's notes are the same in every role.
MELODY_ROLES = {"vocal": {"label": "보컬", "program": None, "sound": None},
                "lead_guitar": {"label": "리드 기타", "program": 29, "sound": "electric_guitar"},     # overdriven guitar
                "acoustic_guitar": {"label": "어쿠스틱 기타", "program": 25, "sound": "acoustic_guitar"},  # steel strings
                "keys": {"label": "건반", "program": 0, "sound": "piano"}}


def leadsheet_to_song(sheet: LeadSheet | str, target_key: str | None = None, genre_override: str | None = None,
                      levels: dict | None = None, key_change_last_chorus: bool = False, flow: bool = True,
                      legacy_lineup: bool = False, instruments: list[str] | None = None,
                      melody_role: str = "vocal", target_shift: int | None = None) -> dict:
    """Render a lead sheet with the rule-based arrangement. Same result shape as `song.build_song`.

    `sheet`: LeadSheet or text (LeadsheetError, a ValueError, if the text is invalid).
    `target_key`: e.g. "G" or "Em"; melody and chords move by the smallest shift, the key is set accordingly.
    `genre_override`: Korean (발라드/록/댄스/클래식) or arrange name (ballad/rock/dance/classical).
    `key_change_last_chorus`: with 2+ 후렴 sections, the last one goes up a semitone (default: as written).
    `instruments`: the user's line-up; each section keeps the template's instruments that are in it (or the whole
    line-up when none are), and "keys" (melody piano) joins every section when chosen. None = the template as is.
    `flow`: apply dynamics.apply_flow (loudness arc, fills, break before the last chorus). False reproduces the
    arrangement made before song flow existed (used for the published listening-test clips).
    """
    if melody_role not in MELODY_ROLES:
        raise ValueError(f"melody_role must be one of {', '.join(MELODY_ROLES)}, got {melody_role!r}")
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    meter = sheet.meter
    if meter not in METER_TICKS:
        raise ValueError(f"지원하지 않는 박자: {meter}")
    bl = METER_TICKS[meter]
    bar_sec = bl / TPQ * 60.0
    genre_name = genre_override or sheet.genre
    genre = GENRE_EN.get(genre_name, genre_name)
    if genre not in ch.GENRES:
        raise ValueError(f"알 수 없는 장르: '{genre_name}' (가능: {', '.join(GENRE_EN)})")
    trace: list[dict] = []
    src = key_of(sheet.key)
    shift = _shift_to(src, key_of(target_key)) if target_key else 0
    if target_shift is not None:                           # the caller knows the way back: not worked out anew
        if not target_key or (src.tonic + target_shift) % 12 != key_of(target_key).tonic:
            raise ValueError(f"target_shift {target_shift} does not lead from {sheet.key} to {target_key}")
        shift = target_shift
    key = Key((src.tonic + shift) % 12, src.mode)
    signature = KEYS[target_key or sheet.key]
    flats = None if signature == 0 else signature < 0
    bpm = min(sheet.tempo, TEMPO_MAX)
    mix = resolve_mix(genre, bpm, levels)
    n_bars = sum(len(s.bars) for s in sheet.sections)
    trace.append({"step": "리드시트 해석", "tool": "leadsheet",
                  "detail": f"{genre_name}, 조 {sheet.key}, {meter}, 템포 {sheet.tempo}"
                            + (f" (재생 한도 {TEMPO_MAX}로 조정)" if bpm != sheet.tempo else "")
                            + f": {' → '.join(s.name for s in sheet.sections)} ({n_bars}마디)"})
    if shift:
        trace.append({"step": "조옮김", "tool": "transpose",
                      "detail": f"{sheet.key} → {key.name} ({shift:+d}반음, 멜로디와 코드 모두)"})

    all_notes, _ = to_notes(sheet)
    marks_by_bar = _bar_marks(sheet)
    chorus_ids = [i for i, s in enumerate(sheet.sections) if s.name == "후렴"]
    last_chorus = chorus_ids[-1] if key_change_last_chorus and len(chorus_ids) >= 2 else None
    if last_chorus is not None:
        trace.append({"step": "전조", "tool": "transpose", "detail": "마지막 후렴을 반음 올렸어요"})
    counts_total = {n: sum(s.name == n for s in sheet.sections) for n in {s.name for s in sheet.sections}}
    no_chords = not any(marks_by_bar)
    tool_name, tool = agent._pick_tool(trace) if no_chords and meter == "4/4" else ("leadsheet", None)
    if no_chords:
        tool_name = tool_name if tool else "cycle"
        trace.append({"step": "코드 생성", "tool": tool_name, "detail": "악보에 코드가 없어 " + (
            ("학습된 코드 모델(BiLSTM)로 채웠어요" if tool_name == "dl" else "규칙 기반 코드 도구로 채웠어요") if tool
            else "고정 코드 순환(I-V-vi-IV / i-VI-III-VII)으로 채웠어요")})

    events, melody_all, symbols, bar_chords, clash_all, sections = [], [], [], [], [], []
    sec_info: list[dict] = []
    firsts, lasts = [], []
    seen: dict[str, int] = {}
    bar0 = 0
    for si, sec in enumerate(sheet.sections):
        kind = KIND[sec.name]
        bars = len(sec.bars)
        extra = KEY_CHANGE if si == last_chorus else 0
        sk = Key((key.tonic + extra) % 12, key.mode)
        span = shift + extra
        seen[sec.name] = seen.get(sec.name, 0) + 1
        label = sec.name + (f" {seen[sec.name]}" if counts_total[sec.name] > 1 else "") + (" (전조)" if extra else "")
        lo, hi = bar0 * bl, (bar0 + bars) * bl
        mel = []
        for n in all_notes:
            if lo <= n.start < hi:
                p = None if n.pitch is None else n.pitch + span
                if p is not None and not PITCH_MIN <= p <= PITCH_MAX:
                    raise ValueError(f"조옮김 후 음역(MIDI {PITCH_MIN}~{PITCH_MAX})을 벗어나요: {label} MIDI {p}")
                mel.append(Note(p, n.start - lo, n.dur))
        # --- chords per bar: [(tick, ChordDef)]
        if tool is not None:
            prog = []
            for _, _, sub in _chunks(mel, bars):
                prog += agent._generate_with_checks(sk, genre, sub, tool, [], bias=None)
            names = [[(0, ch.chord_symbol(sk, c))] for c in prog]
        elif no_chords:
            cycle = ["I", "V", "vi", "IV"] if sk.mode == "major" else ["i", "VI", "III", "VII"]
            names = [[(0, ch.chord_symbol(sk, ch.chord_by_roman(sk, cycle[(bar0 + b) % 4])))] for b in range(bars)]
        else:
            names = [[(t, transpose_symbol(s, span, flats)) for t, s in marks_by_bar[bar0 + b]] for b in range(bars)]
        segs = [[(t, parse_chord(s, sk)) for t, s in bar] for bar in names]
        inst, energy = section_style(genre, kind, legacy_lineup)
        if instruments is not None:
            chosen = [i for i in inst if i in instruments] or [i for i in instruments if i != "keys"]
            inst = chosen + (["keys"] if "keys" in instruments and "keys" not in chosen else [])
            if not legacy_lineup:                    # e.g. strings the user kept join where the genre's table says
                inst = place_extras(genre, kind, inst, instruments)
        events += [Event(e.track, e.pitch, e.start + lo, e.dur, e.velocity)
                   for e in arrange_segments(sk, genre, mel, segs, inst, energy, meter)]
        sec_info.append({"kind": kind, "lo": lo, "bars": bars, "segs": segs, "key": sk, "inst": inst, "mel": mel})
        melody_all += [Note(n.pitch, n.start + lo, n.dur) for n in mel]
        symbols += [bar[0][1] for bar in names]
        bar_chords += [[[t, s] for t, s in bar] for bar in names]
        rates = clash_rates(sk, mel, segs, meter)
        clash_all += rates
        pitched = [n.pitch for n in mel if n.pitch is not None]
        firsts.append(pitched[0] if pitched else None)
        lasts.append(pitched[-1] if pitched else None)
        sections.append({"kind": kind, "label": label, "name": sec.name, "start_bar": bar0, "bars": bars,
                         "start_sec": round(bar0 * bar_sec / mix["tempo"], 1),
                         "end_sec": round((bar0 + bars) * bar_sec / mix["tempo"], 1),
                         "melody_source": "리드시트", "shift": extra, "energy": energy, "instruments": inst,
                         "romans": [bar[0][1] for bar in names], "mean_clash": round(sum(rates) / len(rates), 3)})
        bar0 += bars

    total_bars = bar0
    duration = round(total_bars * bar_sec / mix["tempo"], 1)
    if flow and not legacy_lineup and melody_role == "lead_guitar":
        trace.append({"step": "인트로 리드", "tool": "arrange",
                      "detail": "멜로디를 리드 기타가 맡으므로 앱이 따로 넣는 리드 기타는 뺐어요"})
    elif flow and not legacy_lineup:
        lead, lead_note = intro_lead(sec_info, bl, meter)
        events += lead
        if lead_note:
            trace.append({"step": "인트로 리드", "tool": "arrange", "detail": lead_note})
    if flow:
        events, flow_notes = apply_flow(events, sections, bl, genre)
        trace.append({"step": "곡 흐름", "tool": "dynamics", "detail": "; ".join(flow_notes)})
    rv, lv = range_violations(events), length_violations(events, total_bars, bl)
    leaps = _bar_leaps(sections, firsts, lasts)
    over = [i + 1 for i, r in enumerate(clash_all) if r > CLASH_MAX]
    boundary = [clash_all[s["start_bar"]] for s in sections[1:]]
    metrics = {"chorus_seed_match": None, "boundary_mean_clash": round(sum(boundary) / len(boundary), 3) if boundary else 0.0,
               "mean_clash": round(sum(clash_all) / len(clash_all), 3), "bars_over_threshold": len(over),
               "new_part_bars_over_threshold": None, "seed_bars_over_threshold": []}
    if leaps:
        trace.append({"step": "연결 검사", "tool": "check", "detail": "파트 경계 도약이 커요: " + ", ".join(
            f"{x['from']}→{x['to']} {x['semitones']}반음" for x in leaps) + " (악보 그대로 유지)"})
    trace.append({"step": "곡 검사", "tool": "check",
                  "detail": f"{len(sections)}파트 {total_bars}마디 {duration}초. 충돌 기준({CLASH_MAX}) 초과 마디 {len(over)}개, "
                            f"음역 위반 {len(rv)}건, 길이 위반 {len(lv)}건"})
    rendered = apply_levels(events, mix["levels"])
    role = MELODY_ROLES[melody_role]
    role_programs = {"melody": role["program"]} if role["program"] is not None else None
    return {
        "melody_role": melody_role, "melody_role_label": role["label"],
        "status": "ok", "source": "leadsheet", "meter": meter, "title": "작곡메이트", "genre": genre, "sections": sections,
        "bars": total_bars, "duration_sec": duration, "duration_ok": TARGET_MIN_S <= duration <= TARGET_MAX_S,
        "duration_note": duration_note(duration, mix["tempo"]), "tempo": mix["tempo"], "key": key.to_dict(), "chord_tool": tool_name, "symbols": symbols,
        "bar_chords": bar_chords, "metrics": metrics,
        "check": {"clash_rates": [round(r, 3) for r in clash_all], "threshold": CLASH_MAX,
                  "range_violations": len(rv), "length_violations": len(lv), "problem_bars": [i - 1 for i in over],
                  "leaps": leaps, "max_leap": MAX_LEAP},
        "trace": trace,
        "playback": playback(rendered, genre, mix["tempo"]),
        "midi_b64": base64.b64encode(to_midi(rendered, genre, mix["tempo"], meter, lineup=not legacy_lineup,
                                              programs=role_programs)).decode(),
        "musicxml": to_musicxml(key, genre, melody_all, [ch.chord_by_roman(key, "I" if key.mode == "major" else "i")] * total_bars,
                                mix["tempo"], symbols=[b[0][1] if len(b) == 1 else [(t, s) for t, s in b]
                                                       for b in bar_chords], meter=meter),
    }


# ---------------------------------------------------------------- song -> lead sheet
def melody_notes(result: dict) -> list[Note]:
    """Melody notes (pitch, start tick, duration) of a song result, read exactly from its MIDI."""
    mid = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
    if mid.ticks_per_beat != TPQ:
        raise ValueError(f"unexpected MIDI resolution {mid.ticks_per_beat}")
    for track in mid.tracks:
        if any(m.type == "track_name" and m.name == "melody" for m in track):
            break
    else:
        return []
    now, open_, notes = 0, {}, []
    for m in track:
        now += m.time
        if m.type == "note_on" and m.velocity > 0:
            open_[m.note] = now
        elif m.type in ("note_off", "note_on") and m.note in open_:
            start = open_.pop(m.note)
            notes.append(Note(m.note, start, now - start))
    return sorted(notes, key=lambda n: n.start)


def _place_mark(bar, tick: int, symbol: str) -> None:
    """Chord mark at `tick`; a note sounding across it is split there (tied), as the format needs."""
    for i, n in enumerate(bar.notes):
        if n.start < tick < n.start + n.dur:
            head = LeadNote(n.pitch, n.start, tick - n.start, tie=n.pitch is not None)
            tail = LeadNote(n.pitch, tick, n.start + n.dur - tick, n.tie)
            bar.notes[i:i + 1] = [head, tail]
            break
    bar.chords.append(ChordMark(symbol, tick))


def song_to_leadsheet(result: dict) -> str:
    """Serialize a song result (`build_song` or `leadsheet_to_song`) to lead sheet text (`result["meter"]`, default 4/4)."""
    notes = melody_notes(result)
    meter = result.get("meter", "4/4")
    bl = METER_TICKS[meter]
    bar_chords = result.get("bar_chords") or [[[0, s]] for s in result["symbols"]]
    sections = []
    for sec in result["sections"]:
        lo, hi = sec["start_bar"] * bl, (sec["start_bar"] + sec["bars"]) * bl
        sn = [Note(n.pitch, n.start - lo, min(n.dur, hi - n.start)) for n in notes if lo <= n.start < hi]
        first_marks = [bar_chords[b][0][1] for b in range(sec["start_bar"], sec["start_bar"] + sec["bars"])]
        sections.append((NAME[sec["kind"]], sn, first_marks))
    sheet = from_notes(GENRE_KO[result["genre"]], result["key"]["name"], meter, int(result["tempo"]), sections)
    b = 0
    for sec in sheet.sections:
        for bar in sec.bars:
            for tick, symbol in bar_chords[b][1:]:
                _place_mark(bar, tick, symbol)
            b += 1
    text = serialize_leadsheet(sheet)
    check = validate_leadsheet(text)
    if not check["ok"]:
        raise LeadsheetError("생성한 리드시트가 검증을 통과하지 못했어요: " + "; ".join(check["errors"]))
    return text
