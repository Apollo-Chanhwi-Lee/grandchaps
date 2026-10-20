"""Uploaded score check: listen to the user's own score and compare its chords with the AI's recommendation.

    MIDI (.mid/.midi) or MusicXML (.musicxml/.xml/.mxl) -> melody + original chords per bar -> two arrangements:
        original  the uploaded notes as they are (playback + MIDI), chords estimated from the accompaniment
        ai        the same melody re-harmonized by the app's chord tool for the chosen genre, arranged with the band
    plus a bar-by-bar comparison (same chord / same root / different) and melody-chord clash rates for both.

The file is read once and deleted by the caller (app.py); nothing is stored, and uploads never enter training data.
Only 4/4 with one main tempo is supported (same limits as the Lakh converter); at most MAX_BARS bars are compared.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np

from . import agent, chords as ch, lakh
from .arrange import BAND, INSTRUMENTS, Event, arrange_segments
from .check import clash_rates
from .export import playback, to_midi
from .genres import GENRES
from .render import parse_chord
from .theory import BAR, Key, Note

ALLOWED_EXT = {".mid", ".midi", ".musicxml", ".xml", ".mxl"}
MAX_BYTES = 2 * 1024 * 1024
MAX_BARS = 32
SKIP_KO = {
    "meter_not_4_4": "4/4 박자 악보만 비교할 수 있어요",
    "tempo_changes": "템포가 여러 번 바뀌는 악보는 아직 비교하지 못해요",
    "tempo_out_of_range": "템포가 40~220 BPM 밖이에요",
    "no_tempo": "템포 정보를 읽지 못했어요",
    "key_no_pitch_content": "음이 너무 적어 조를 알 수 없어요",
}


class ScoreError(ValueError):
    """The upload cannot be compared (Korean message for the UI)."""


def _load(path: Path):
    import pretty_midi
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ScoreError(f"지원하지 않는 파일 형식이에요: {ext} (MIDI 또는 MusicXML)")
    try:
        if ext in (".mid", ".midi"):
            return pretty_midi.PrettyMIDI(str(path))
        from music21 import converter, harmony
        score = converter.parse(str(path))
        # chord symbols are text above the staff, not notes: music21 would otherwise play them as chords in the MIDI
        # and they would land in the melody (scoreimport.xml_chords reads them separately)
        for symbol in list(score.recurse().getElementsByClass(harmony.Harmony)):
            symbol.activeSite.remove(symbol)
        with tempfile.TemporaryDirectory(prefix="instmate_score_") as tmp:
            mid = Path(tmp) / "score.mid"
            score.write("midi", fp=str(mid))
            return pretty_midi.PrettyMIDI(str(mid))
    except ScoreError:
        raise
    except Exception as e:                      # parsers raise many types
        raise ScoreError(f"악보 파일을 읽지 못했어요: {type(e).__name__}") from None


def _split_melody(pm) -> tuple[dict, list[dict]]:
    """(melody arrays, accompaniment arrays).  Several parts: the most melody-like one (lakh scoring without its
    minimum length); one polyphonic part (a piano score): its top voice is the melody, the rest accompaniment."""
    parts = [(i, inst) for i, inst in enumerate(pm.instruments) if not inst.is_drum and inst.notes]
    if not parts:
        raise ScoreError("음표가 있는 파트가 없어요")
    res = pm.resolution
    arrays = {i: lakh.inst_arrays(pm, inst) for i, inst in parts}
    if len(parts) == 1:
        a = arrays[parts[0][0]]
        order = np.lexsort((-a["p"], a["s"]))
        s, e, p, v = (a[k][order] for k in ("s", "e", "p", "v"))
        top = np.ones(len(s), bool)
        top[1:] = s[1:] != s[:-1]                # first of each onset group is the highest (sorted by -pitch)
        mel = {k: arr[top] for k, arr in zip(("s", "e", "p", "v"), (s, e, p, v))}
        acc = {k: arr[~top] for k, arr in zip(("s", "e", "p", "v"), (s, e, p, v))}
        return mel, ([acc] if len(acc["s"]) else [])

    def score(i):
        a = arrays[i]
        ov = lakh.overlap_ratio(a["s"], a["e"], lakh.OVERLAP_SLOP_BEATS * res)
        name = (pm.instruments[i].name or "").lower()
        return ((1.0 - ov) * 2 + float(a["p"].mean()) / 60 + (2.0 if lakh.NAME_HINT.search(name) else 0)
                - (2.0 if lakh.NAME_BAD.search(name) else 0))
    best = max((i for i, _ in parts), key=score)
    a = arrays[best]                             # a chord written in the melody part: its top note is the melody
    order = np.lexsort((-a["p"], a["s"]))
    top = np.ones(len(order), bool)
    top[1:] = a["s"][order][1:] != a["s"][order][:-1]
    mel = {k: a[k][order][top] for k in ("s", "e", "p", "v")}
    rest = {k: a[k][order][~top] for k in ("s", "e", "p", "v")}
    others = [arrays[i] for i, _ in parts if i != best]
    return mel, others + ([rest] if len(rest["s"]) else [])


def analyze(path: Path, genre: str = "ballad", instruments: list[str] | None = None) -> dict:
    if genre not in GENRES:
        raise ScoreError(f"알 수 없는 장르: {genre}")
    instruments = list(instruments or BAND)
    bad = set(instruments) - set(INSTRUMENTS)
    if bad:
        raise ScoreError(f"알 수 없는 악기: {sorted(bad)}")
    pm = _load(path)
    try:
        tempo, _ = lakh.check_meter_tempo(pm)
    except lakh.Skip as e:
        raise ScoreError(SKIP_KO.get(str(e), f"비교할 수 없는 악보예요 ({e})")) from None
    mel_a, acc_parts = _split_melody(pm)
    notes16, _ = lakh.quantize_melody(mel_a, pm.resolution)
    if len(notes16) < 4:
        raise ScoreError("멜로디 음이 너무 적어요")
    first_bar = notes16[0][1] // lakh.BAR16
    b0 = first_bar * lakh.BAR16
    melody = [Note(p, (s - b0) * lakh.SIXTEENTH, (e - s) * lakh.SIXTEENTH) for p, s, e in notes16
              if (s - b0) // lakh.BAR16 < MAX_BARS]
    n_bars = max(1, -(-max(n.start + n.dur for n in melody) // BAR))
    # key from every pitched note
    hist = np.zeros(12)
    for a in [mel_a, *acc_parts]:
        np.add.at(hist, a["p"] % 12, np.minimum(a["e"] - a["s"], 4 * pm.resolution))
    try:
        tonic, mode, _, _ = lakh.estimate_key(hist)
    except lakh.Skip:
        raise ScoreError(SKIP_KO["key_no_pitch_content"]) from None
    key = Key(tonic, "major" if mode == "maj" else "minor")
    # original chords: one per bar from the accompaniment (first half bar has the most weight in pop writing)
    tick16 = pm.resolution / 4
    original: list[str | None] = []
    for b in range(n_bars):
        w0, w1 = (b0 + b * lakh.BAR16) * tick16, (b0 + (b + 1) * lakh.BAR16) * tick16
        chroma, count, bass = np.zeros(12), 0, None
        for acc in acc_parts:
            c, k, bs = lakh.half_bar_chroma(acc, w0, w1)
            chroma += c
            count += k
            bass = bs if bass is None else bass
        if count < lakh.MIN_ACC_NOTES or chroma.sum() <= 0:
            original.append(None)
            continue
        root, q, _ = lakh.match_chord(chroma, bass)
        original.append(lakh.ROOT_NAMES[root] + q)
    # AI chords: the app's own chord tool for the genre (same as the 반주 button)
    _, tool = agent._pick_tool([])
    ai_prog = agent._generate_with_checks(key, genre, melody, tool, [])[:n_bars]
    ai = [ch.chord_symbol(key, c) for c in ai_prog]
    has_original = any(original)
    orig_defs = [parse_chord(s, key) if s else ai_prog[i] for i, s in enumerate(original)]

    def arranged(prog_defs):
        evs = arrange_segments(key, genre, melody, [[(0, c)] for c in prog_defs], instruments)
        return {"midi_b64": base64.b64encode(to_midi(evs, genre, tempo)).decode(), "playback": playback(evs, genre, tempo)}

    compare = []
    for i in range(n_bars):
        o, a = original[i], ai[i]
        ro = None if o is None else parse_chord(o, key)
        same = o is not None and _norm(o) == _norm(a)
        same_root = ro is not None and (key.tonic + ro.offset) % 12 == (key.tonic + ai_prog[i].offset) % 12
        compare.append({"bar": i + 1, "original": o, "ai": a, "same": same, "same_root": bool(same_root)})
    known = [c for c in compare if c["original"]]
    clash_ai = clash_rates(key, melody, ai_prog)
    clash_orig = clash_rates(key, melody, orig_defs) if has_original else None
    raw_events = [Event("melody" if a is mel_a else "piano", int(p), int(s), int(e - s), int(v))
                  for a in [mel_a, *acc_parts] for s, e, p, v in zip(a["s"], a["e"], a["p"], a["v"])]
    scale = 480 / pm.resolution
    raw_events = [Event(e.track, e.pitch, round(e.start * scale), max(1, round(e.dur * scale)), e.velocity)
                  for e in raw_events]
    return {
        "key": key.to_dict(), "tempo": tempo, "bars": n_bars, "genre": genre,
        "first_bar": int(first_bar),             # bars of silence before the melody that were skipped (bar 1 here = that bar + 1)
        "melody_notes": len(melody), "has_original_chords": has_original,
        "original": {"symbols": original, "playback": playback(raw_events, genre, tempo),
                     "midi_b64": base64.b64encode(to_midi(raw_events, genre, tempo)).decode(),
                     "mean_clash": None if clash_orig is None else round(float(np.mean(clash_orig)), 3)},
        "ai": {"symbols": ai, **arranged(ai_prog), "mean_clash": round(float(np.mean(clash_ai)), 3)},
        "original_arranged": arranged(orig_defs) if has_original else None,
        "compare": compare,
        "match_rate": round(sum(c["same"] for c in known) / len(known), 3) if known else None,
        "root_match_rate": round(sum(c["same_root"] for c in known) / len(known), 3) if known else None,
        "note": "올린 악보는 분석 후 바로 삭제하고, 저장하거나 학습에 쓰지 않아요.",
    }


def _norm(sym: str) -> str:
    """Compare chord names ignoring enharmonic spelling and the slash bass."""
    names = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}
    s = sym.split("/")[0]
    root = s[:2] if len(s) > 1 and s[1] in "#b" else s[:1]
    return names.get(root, root) + s[len(root):]
