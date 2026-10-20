"""A breathing place at the end of a phrase of the user's own melody, without changing that melody.

    plan(midi, bars)                    which of the named bars qualify, and why the others do not
    breathe_wave(wave, rate, targets, sec)   the sung wave with the last half beat of each target note silenced
    rest_band(played, targets)          a render copy of the prepared band that holds back in those half beats

Used by audio.render(..., breaths=[bar, ...]) and only when a voice sings.  Opt-in per bar (bars count from 1): there
is no rule such as "every fourth bar".  A bar qualifies only in 4/4 and only when its melody ends on one note that
starts on beat 3 and lasts exactly two beats to the bar line; anything else is left alone and the reason is returned.

What changes, all inside the last half beat of a qualifying bar (the "window"):
  * the voice sings the two-beat note for a beat and a half (the written note in the MIDI file stays two beats);
  * the guitar does not start a new note there, a piano note sounding into it stops at its start, and of the drum hits
    in it only the first stays (an existing fill is thinned, nothing is added);
  * the bass and whatever already rings (reverb, cymbals, the guitar's earlier notes) carry on: this is a breath, not
    a full stop of the band.
The band copy is made AFTER the performance was prepared (swing, humanizing), so the random timing and loudness
offsets of the rest of the song are the ones the song always had.  Whether this sounds better is for a listener to
say; the half beat and the choice of what holds back are design choices, not tested rules.
"""
from __future__ import annotations

import io

import mido

MAX_BPM = 120           # above this a strummed chord can spread past half a sixteenth: which note belongs to the window
                        # can no longer be read safely from the prepared performance, so nothing is changed
ATTACK_PARTS = ("guitar",)          # parts that do not start a new note in the window
SUSTAIN_PARTS = ("piano", "keys")   # parts whose sounding notes stop where the window starts
DRUMS = "drums"


def _name(track) -> str | None:
    return next((m.name for m in track if m.type == "track_name"), None)


def _notes(track) -> list[dict]:
    """[{"on": index, "off": index, "start", "end", "note"}] of one track (indices into the track's messages)."""
    now, open_, out = 0, {}, []
    for i, m in enumerate(track):
        now += m.time
        if m.type == "note_on" and m.velocity > 0:
            open_.setdefault((m.channel, m.note), []).append((i, now))
        elif m.type in ("note_off", "note_on") and open_.get((m.channel, m.note)):
            j, start = open_[(m.channel, m.note)].pop(0)
            out.append({"on": j, "off": i, "start": start, "end": now, "note": m.note})
    return out


def in_window(tick: int, window, tpb: int) -> bool:
    """Whether a note of the PREPARED performance belongs to the window.  Preparing moves starts a little (humanizing,
    strumming), so the start is read at the nearest sixteenth: a downbeat played a few ticks early still belongs to
    the next bar, and a hit just before the window stays outside it."""
    grid = tpb // 4
    return window[0] <= round(tick / grid) * grid < window[1]


def plan(midi: bytes, bars: list[int]) -> dict:
    """{"targets": [{"bar", "start", "end", "pitch", "window": [from, to]}], "skipped": [{"bar", "reason"}]} (ticks)."""
    wanted = []
    for b in bars or []:
        if b not in wanted:
            wanted.append(b)
    skip = lambda why: {"targets": [], "skipped": [{"bar": b, "reason": why} for b in wanted]}
    try:
        mid = mido.MidiFile(file=io.BytesIO(midi))
    except Exception:
        return skip("MIDI를 읽지 못했어요")
    sigs = {(m.numerator, m.denominator) for t in mid.tracks for m in t if m.type == "time_signature"}
    if sigs - {(4, 4)}:
        return skip("4/4 박자가 아니에요")
    melody = next((t for t in mid.tracks if _name(t) == "melody"), None)
    if melody is None:
        return skip("멜로디 트랙이 없어요")
    tempos = [m.tempo for t in mid.tracks for m in t if m.type == "set_tempo"]
    if len(set(tempos)) > 1 or (tempos and mido.tempo2bpm(tempos[0]) > MAX_BPM):
        return skip(f"템포가 바뀌거나 {MAX_BPM} BPM보다 빨라서 숨 자리의 음을 안전하게 가려낼 수 없어요")
    tpb = mid.ticks_per_beat
    bar_len, half = 4 * tpb, tpb // 2
    notes = _notes(melody)
    out = {"targets": [], "skipped": []}
    for b in wanted:
        if type(b) is not int or b < 1:
            out["skipped"].append({"bar": b, "reason": "마디 번호는 1부터예요"})
            continue
        start, end = (b - 1) * bar_len + 2 * tpb, b * bar_len
        late = [n for n in notes if start <= n["start"] < end]
        if len(late) != 1 or late[0]["start"] != start or late[0]["end"] != end:
            out["skipped"].append({"bar": b, "reason": "이 마디의 멜로디가 3박에서 시작해 마디 끝까지 두 박 이어지는 한 음으로 끝나지 않아요"})
            continue
        out["targets"].append({"bar": b, "start": start, "end": end, "pitch": late[0]["note"], "window": [end - half, end]})
    return out


AUTO_GAP_S = 6.0        # a breath is added only this long after the singer's last chance to breathe (design choice)


def auto_bars(midi: bytes, min_gap_s: float = AUTO_GAP_S) -> list[int]:
    """Bars (from 1) for `plan`, chosen without being named: a bar that qualifies (see the module text), whose phrase
    end has no rest of its own (the next sung note starts within an eighth of the bar line, so nothing marks the
    boundary and the singer has no place to breathe), and that comes at least `min_gap_s` after the last breath: the
    start of the song, the last rest of an eighth or more in the melody, or the last bar chosen here.
    Read from the MIDI alone: key and chords are not known at render time, so "no rest" is the one boundary cue used."""
    try:
        mid = mido.MidiFile(file=io.BytesIO(midi))
    except Exception:
        return []
    melody = next((t for t in mid.tracks if _name(t) == "melody"), None)
    if melody is None:
        return []
    tpb = mid.ticks_per_beat
    notes = sorted(_notes(melody), key=lambda n: n["start"])
    if not notes:
        return []
    last_bar = max(n["end"] for n in notes) // (4 * tpb) + 1
    targets = plan(midi, list(range(1, last_bar + 1)))["targets"]
    if not targets:
        return []
    sec = seconds_per_tick(midi)
    rests = [0] + [b["start"] for a, b in zip(notes, notes[1:]) if b["start"] - a["end"] >= tpb // 2]
    chosen, breathed = [], 0
    for t in targets:
        after = next((n["start"] for n in notes if n["start"] >= t["end"]), None)
        if after is None or after - t["end"] >= tpb // 2:
            continue                                   # the melody ends here, or it already rests after this phrase
        last = max([breathed] + [r for r in rests if r <= t["window"][0]])
        if (t["window"][0] - last) * sec >= min_gap_s:
            chosen.append(t["bar"])
            breathed = t["end"]
    return chosen


def seconds_per_tick(midi: bytes) -> float:
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tempo = next((m.tempo for m in mid.tracks[0] if m.type == "set_tempo"), 500000)
    return tempo / 1e6 / mid.ticks_per_beat


def breathe_wave(wave, rate: int, targets: list[dict], sec_per_tick: float, fade_s: float = 0.02):
    """The sung wave with the last half beat of every target note silenced (a short fade, then nothing until the bar
    line).  The voice is synthesised from the notes as written and only these samples are changed, so every other
    moment of the vocal is sample-for-sample what it was.  (Shortening the note before synthesis would shift the
    synthesiser's random pitch movement for the rest of the song.)  Returns a new array; `wave` is not modified."""
    import numpy as np
    if not targets:
        return wave
    out = np.array(wave, dtype=np.float64, copy=True)
    fade = max(1, int(fade_s * rate))
    for t in targets:
        a, b = int(t["window"][0] * sec_per_tick * rate), min(len(out), int(t["window"][1] * sec_per_tick * rate))
        if a >= len(out) or b <= a:
            continue
        k = min(fade, b - a)
        out[a:a + k] *= np.linspace(1.0, 0.0, k)
        out[a + k:b] = 0.0
    return out


def rest_band(played: bytes, targets: list[dict],
              exact: set[str] | frozenset[str] = frozenset()) -> bytes:
    """A copy of the prepared MIDI in which the band holds back inside every target window (see the module text).
    Protected tracks are passed through with every message intact. The melody track and every note outside
    the windows are byte-for-byte the notes of `played`."""
    if not targets:
        return played
    windows = [tuple(t["window"]) for t in targets]
    mid = mido.MidiFile(file=io.BytesIO(played))
    inside = lambda tick: any(in_window(tick, w, mid.ticks_per_beat) for w in windows)
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for track in mid.tracks:
        name = _name(track)
        if name in exact or name not in ATTACK_PARTS + SUSTAIN_PARTS + (DRUMS,):
            out.tracks.append(track)
            continue
        ticks, now = [], 0
        for m in track:
            now += m.time
            ticks.append(now)
        drop: set[int] = set()
        notes = _notes(track)
        if name in ATTACK_PARTS:
            for n in notes:
                if inside(n["start"]):
                    drop |= {n["on"], n["off"]}
        elif name in SUSTAIN_PARTS:
            for n in notes:
                for a, b in windows:
                    if not inside(n["start"]) and n["start"] < a and n["end"] > a:
                        ticks[n["off"]] = a                      # stops where the breath begins
        else:
            for a, b in windows:
                hits = sorted((n for n in notes if in_window(n["start"], (a, b), mid.ticks_per_beat)), key=lambda n: (n["start"], n["on"]))
                for n in hits[1:]:                               # the first hit of an existing fill stays
                    drop |= {n["on"], n["off"]}
        rows = sorted(((ticks[i], i, m) for i, m in enumerate(track) if i not in drop), key=lambda r: (r[0], r[1]))
        new, now = mido.MidiTrack(), 0
        for tick, _, m in rows:
            new.append(m.copy(time=tick - now))
            now = tick
        out.tracks.append(new)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()
