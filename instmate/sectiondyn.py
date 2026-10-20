"""Sound option "section_dynamics" (off unless asked for): the song's sections differ in size, not only in notes.

Two steps, both on the render copy (the MIDI file and the score are not changed):
  * thin    in the verses one chord part steps out: the guitar, when a piano or keys part is there to carry the
            harmony.  Bass, drums and the melody stay.
  * level   after the master, every section that is not a chorus is turned down until it lies CONTRAST_DB under the
            mean level of the choruses (measured on this render; a section that is already that quiet is left alone),
            the pre-chorus half as far, and the last of several choruses stands LAST_CHORUS_DB above the others.
            Nothing is raised: the loudest section keeps its mastered level and the rest go down from there.  The
            song is then brought back up as a whole (audio.relevel: one gain towards -12 LUFS and a limiter, no
            compressor), because turning sections down had left it 1.6 LU under the master's target.  Gains change
            over RAMP_S around a section's join, so nothing steps.  (Before the master the compressor took half of
            the contrast back: 5 dB asked for measured 2.6 dB.)

The arrangement already scales note velocities by section (instmate/dynamics.py); measured on a finished render that
moved the level by well under 3 dB, which is why this works on the audio.  CONTRAST_DB, the part that steps out and
the last-chorus step are design hypotheses to be judged by ear.
"""
from __future__ import annotations

import io

import mido
import numpy as np

CONTRAST_DB = 4.5
PRE_SHARE = 0.5              # the pre-chorus goes this share of the way down
LAST_CHORUS_DB = 1.5
RAMP_S = 0.25
STEP_OUT = "guitar"          # the part that leaves the verses ...
CARRIERS = ("piano", "keys")     # ... when one of these plays


def _name(track) -> str | None:
    return next((m.name for m in track if m.type == "track_name"), None)


def thin(midi: bytes, sections: list[dict] | None,
         exact: set[str] | frozenset[str] = frozenset()) -> bytes:
    """Let unprotected verse guitar step out only where a carrier actually plays.

    A carrier must sound for at least half the bar (merged note spans, not note count).
    This conservative arranging choice keeps the guitar under isolated piano accents.
    Notes extending into unsupported bars or another section are retained. Quantizing
    to the nearest sixteenth assigns small humanizing offsets to their intended bars.
    All protected tracks and all non-note messages are preserved.
    """
    from .breath import _notes
    verses = [s for s in sections or [] if s["kind"] == "verse"]
    if not verses or STEP_OUT in exact:
        return midi
    mid = mido.MidiFile(file=io.BytesIO(midi))
    names = {_name(t) for t in mid.tracks}
    if STEP_OUT not in names or not names & set(CARRIERS):
        return midi
    signatures = {(m.numerator, m.denominator) for t in mid.tracks for m in t
                  if m.type == "time_signature"}
    if len(signatures) > 1 or mid.ticks_per_beat < 4:
        return midi                         # a variable meter needs its own bar map
    num, den = next(iter(signatures), (4, 4))
    bar = mid.ticks_per_beat * 4 * num // den
    grid = mid.ticks_per_beat // 4
    verse_bars = {b for s in verses for b in range(s["start_bar"], s["start_bar"] + s["bars"])}
    spans = {}
    for track in mid.tracks:
        if _name(track) not in CARRIERS:
            continue
        for n in _notes(track):
            a, z = round(n["start"] / grid) * grid, round(n["end"] / grid) * grid
            for b in range(max(0, a // bar), (z - 1) // bar + 1):
                if b in verse_bars and z > a:
                    spans.setdefault(b, []).append((max(a, b * bar), min(z, (b + 1) * bar)))
    supported = set()
    for b, windows in spans.items():
        covered, end = 0, b * bar
        for a, z in sorted(windows):
            covered += max(0, z - max(a, end))
            end = max(end, z)
        if covered >= bar / 2:
            supported.add(b)
    if not supported:
        return midi
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    changed = False
    for track in mid.tracks:
        if _name(track) != STEP_OUT:
            out.tracks.append(track)
            continue
        drop = set()
        for n in _notes(track):
            a, z = round(n["start"] / grid) * grid, round(n["end"] / grid) * grid
            if z > a and all(b in supported for b in range(a // bar, (z - 1) // bar + 1)):
                drop.update((n["on"], n["off"]))
        if not drop:
            out.tracks.append(track)
            continue
        changed = True
        new, tick, written = mido.MidiTrack(), 0, 0
        for i, m in enumerate(track):
            tick += m.time
            if i not in drop:
                new.append(m.copy(time=tick - written))
                written = tick
        out.tracks.append(new)
    if not changed:
        return midi
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def plan(kinds: list[str], levels: list[float]) -> list[float]:
    """Gain in dB per section from its kind and its measured level (dBFS)."""
    choruses = [i for i, k in enumerate(kinds) if k == "chorus"]
    if not choruses:
        return [0.0] * len(kinds)
    ref = sum(levels[i] for i in choruses) / len(choruses)
    out = []
    for i, (k, lv) in enumerate(zip(kinds, levels)):
        if k == "chorus":
            out.append(LAST_CHORUS_DB if len(choruses) > 1 and i == choruses[-1] else 0.0)
        else:
            want = ref - CONTRAST_DB * (PRE_SHARE if k == "pre" else 1.0)
            out.append(round(min(0.0, want - lv), 2))
    return out


def curve(gains_db: list[float], spans: list[tuple[float, float]], n: int, sr: int) -> np.ndarray:
    """Per-sample gain (linear): each section's gain inside its span (seconds), ramped in dB over RAMP_S at the joins;
    before the first span and after the last the nearest section's gain holds."""
    t = np.arange(n) / sr
    points_t, points_g = [], []
    for g, (a, b) in zip(gains_db, spans):
        half = min(RAMP_S / 2, (b - a) / 2)
        points_t += [a + half, b - half]
        points_g += [g, g]
    return 10 ** (np.interp(t, points_t, points_g) / 20)


def level(wav: bytes, sections: list[dict], bar_s: float) -> tuple[bytes, list[float]]:
    """(the WAV with the section gains applied, the gains in dB as applied: the largest is 0)."""
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    spans = [(s["start_bar"] * bar_s, (s["start_bar"] + s["bars"]) * bar_s) for s in sections]
    rms = lambda seg: 20 * float(np.log10(np.sqrt(np.mean(seg.astype("float64") ** 2)) + 1e-9)) if len(seg) else -120.0
    levels = [rms(y[int(a * sr): int(b * sr)]) for a, b in spans]
    gains = plan([s["kind"] for s in sections], levels)
    if not any(gains):
        return wav, gains
    gains = [round(g - max(gains), 2) for g in gains]
    out = y * curve(gains, spans, len(y), sr)[:, None]
    buf = io.BytesIO()
    sf.write(buf, out, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), gains
