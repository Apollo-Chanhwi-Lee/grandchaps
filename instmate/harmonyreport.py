"""Harmony report: things worth a second look in a finished song.  A diagnosis, never a fix.

    report(result) -> {"notes": [{"bar", "beat", "kind", "level", "on_input", "text"}], "summary": {...}}

It reads the song as it sounds (the result's MIDI and its chords per bar) and lists:

    melody_chord   a melody note that sounds against a chord it does not belong to for a beat or more, measured by the
                   time the note and the chord really overlap (so a note held into the next chord is seen too), unless
                   it moves by step on both sides and is short (a passing or neighbour tone)
    ending         how a melodic section ends: on the tonic, on the dominant (open), or somewhere else
    bass_leap      the bass jumping more than an octave between two notes
    cover          a section where the accompaniment sits above the melody for a good part of the time

Nothing here changes the song.  A note in the user's own input is reported like any other but marked `on_input`: the
app must not rewrite it.  A lower count does not mean better music; these are the same checks a bandmate would do by
ear, written down.
"""
from __future__ import annotations

import base64
import io

from .chords import note_role
from .render import parse_chord
from .theory import TPQ, Key

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
LONG_TICKS = TPQ                       # a note against a chord for at least one beat is worth reporting
STEP = 2                               # semitones: a whole step or less counts as stepwise motion
BASS_LEAP = 12                         # semitones
COVER_SHARE = 0.25                     # of the melody's sounding time in a section
MAX_NOTES = 120
ACCOMPANIMENT = ("guitar", "piano", "keys", "strings")
MELODIC = ("verse", "pre", "chorus", "bridge")


def _name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _tracks(midi: bytes) -> tuple[dict[str, list[tuple[int, int, int]]], int]:
    """{track: [(start, end, pitch)]} in lead-sheet ticks (480 per quarter), and the bar length for 4/4."""
    import mido
    mid = mido.MidiFile(file=io.BytesIO(midi))
    scale = TPQ / mid.ticks_per_beat
    out: dict[str, list[tuple[int, int, int]]] = {}
    for tr in mid.tracks:
        name = next((m.name for m in tr if m.type == "track_name"), None)
        if not name:
            continue
        now, on, notes = 0, {}, []
        for m in tr:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                on.setdefault(m.note, []).append(now)
            elif m.type in ("note_off", "note_on") and on.get(m.note):
                start = on[m.note].pop(0)
                notes.append((round(start * scale), round(now * scale), m.note))
        out[name] = sorted(notes)
    return out, TPQ * 4


def chord_segments(bar_chords: list, bar_ticks: int) -> list[tuple[int, int, str]]:
    """[(start, end, symbol)] over the whole song from the per-bar chord list ([[tick in bar, symbol], ...] per bar)."""
    points = []
    for b, bar in enumerate(bar_chords):
        for tick, symbol in bar:
            points.append((b * bar_ticks + int(tick), symbol))
    points.sort()
    total = len(bar_chords) * bar_ticks
    return [(s, points[i + 1][0] if i + 1 < len(points) else total, sym) for i, (s, sym) in enumerate(points)]


def melody_flags(melody: list[tuple[int, int, int]], segments: list[tuple[int, int, str]], key: Key,
                 bar_ticks: int) -> list[dict]:
    """Melody notes that sound for a beat or more against a chord they are not part of and are not passing tones.
    `melody`: [(start, end, pitch)] in time order."""
    out = []
    for i, (start, end, pitch) in enumerate(melody):
        for s, e, symbol in segments:
            overlap = min(end, e) - max(start, s)
            if overlap < LONG_TICKS:
                continue
            try:
                role = note_role(key, parse_chord(symbol, key), pitch % 12)
            except ValueError:
                continue                                   # a chord symbol the app cannot read: nothing to say
            if role in ("chord", "tension"):
                continue
            prev = melody[i - 1][2] if i > 0 else None
            nxt = melody[i + 1][2] if i + 1 < len(melody) else None
            by_step = prev is not None and nxt is not None and 0 < abs(pitch - prev) <= STEP and 0 < abs(nxt - pitch) <= STEP
            if by_step and end - start <= LONG_TICKS and start >= s:
                continue                                   # short, entered and left by step: a passing or neighbour tone
            at = max(start, s)
            held = start < s                               # the note began under the chord before and rings into this one
            out.append({"bar": at // bar_ticks + 1, "beat": round((at % bar_ticks) / TPQ + 1, 2), "kind": "melody_chord",
                        "level": "주의",
                        "text": f"멜로디 {_name(pitch)}가 코드 {symbol} 위에서 {overlap / TPQ:g}박 동안 코드 밖 음이에요"
                                + (" (앞 코드에서 끌어온 음)" if held else "")
                                + ("" if by_step else " · 앞뒤가 순차 진행이 아니에요")})
    return out


def _ending(symbol: str, key: Key) -> tuple[str, str]:
    try:
        root = parse_chord(symbol, key).pcs[0]
    except (ValueError, IndexError, AttributeError):
        return "unknown", f"끝 코드 {symbol}"
    interval = (root - key.tonic) % 12
    if interval == 0:
        return "tonic", f"으뜸화음({symbol})으로 끝나요"
    if interval == 7:
        return "dominant", f"딸림화음({symbol})으로 끝나 다음 구간으로 열려 있어요"
    return "other", f"{symbol}로 끝나요 (으뜸·딸림이 아님)"


def report(result: dict) -> dict:
    """See the module docstring.  `result`: a finished song (midi_b64, bar_chords, sections, key)."""
    tracks, bar_ticks = _tracks(base64.b64decode(result["midi_b64"]))
    base = Key(int(result["key"]["tonic"]), result["key"]["mode"])
    segments = chord_segments(result.get("bar_chords") or [], bar_ticks)
    melody = tracks.get("melody", [])
    notes: list[dict] = []
    for sec in result.get("sections") or []:
        lo, hi = sec["start_bar"] * bar_ticks, (sec["start_bar"] + sec["bars"]) * bar_ticks
        key = Key((base.tonic + int(sec.get("shift") or 0)) % 12, base.mode)
        on_input = sec.get("origin") == "input"
        mel = [n for n in melody if lo <= n[0] < hi]
        segs = [(max(s, lo), min(e, hi), sym) for s, e, sym in segments if s < hi and e > lo]
        for flag in melody_flags(mel, segs, key, bar_ticks):
            notes.append({**flag, "section": sec["label"], "on_input": on_input})
        if sec["kind"] in MELODIC and segs:
            kind, text = _ending(segs[-1][2], key)
            if kind != "tonic":
                notes.append({"bar": sec["start_bar"] + sec["bars"], "beat": 1, "kind": "ending", "level": "참고",
                              "section": sec["label"], "on_input": False, "text": f"{sec['label']}: {text}"})
        if mel:
            sung = sum(e - s for s, e, _ in mel)
            covered = 0
            for s, e, p in mel:
                above = [(max(s, a), min(e, b)) for part in ACCOMPANIMENT for a, b, q in tracks.get(part, [])
                         if q > p and a < e and b > s]
                if above:                                  # union of the covered stretches of this melody note
                    above.sort()
                    cur_s, cur_e = above[0]
                    for a, b in above[1:]:
                        if a <= cur_e:
                            cur_e = max(cur_e, b)
                        else:
                            covered += cur_e - cur_s
                            cur_s, cur_e = a, b
                    covered += cur_e - cur_s
            if sung and covered / sung >= COVER_SHARE:
                notes.append({"bar": sec["start_bar"] + 1, "beat": 1, "kind": "cover", "level": "참고",
                              "section": sec["label"], "on_input": False,
                              "text": f"{sec['label']}: 멜로디가 울리는 시간의 {round(100 * covered / sung)}% 동안 반주가 "
                                      "멜로디보다 높은 음을 연주해요"})
    bass = tracks.get("bass", [])
    for (s1, _, p1), (s2, _, p2) in zip(bass, bass[1:]):
        if abs(p2 - p1) > BASS_LEAP:
            notes.append({"bar": s2 // bar_ticks + 1, "beat": round((s2 % bar_ticks) / TPQ + 1, 2), "kind": "bass_leap",
                          "level": "참고", "section": None, "on_input": False,
                          "text": f"베이스가 {_name(p1)}에서 {_name(p2)}로 {abs(p2 - p1)}반음 뛰어요"})
    notes.sort(key=lambda n: (n["bar"], n["beat"]))
    counts: dict[str, int] = {}
    for n in notes:
        counts[n["kind"]] = counts.get(n["kind"], 0) + 1
    return {"notes": notes[:MAX_NOTES], "summary": {"total": len(notes), **counts,
                                                    "on_input": sum(1 for n in notes if n["on_input"])},
            "about": "참고용 점검이에요. 코드 밖 음이나 열린 끝이 틀린 것은 아니고, 숫자가 적다고 더 좋은 곡도 아니에요. "
                     "입력하신 멜로디와 직접 준 코드는 앱이 바꾸지 않아요."}
