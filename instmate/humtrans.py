"""Hum or sing a seed melody: a short recording -> number-notation tokens (the melody input of the app).

    audio -> f0 (pyworld harvest, 5 ms frames) -> notes (voiced runs split at steady pitch changes) -> eighth-note grid
    at the chosen tempo -> key (Krumhansl-Kessler, instmate.lakh) -> movable-do tokens relative to the major tonic

A hum is one voice at a time, so a monophonic pitch tracker is enough (no polyphonic transcription model needed).
The recording is analysed once and deleted by the caller; nothing is stored.
"""
from __future__ import annotations

import io

import numpy as np

from . import lakh
from .theory import TPQ

FS = 16000
FRAME_MS = 5.0
MIN_NOTE_S = 0.09            # shorter voiced blips are dropped
SPLIT_SEMITONES = 0.7        # a pitch move this big, held for SPLIT_HOLD_S, starts a new note
SPLIT_HOLD_S = 0.06
STABLE_SEMITONES = 0.3       # ...and only once the new pitch is steady (a glide between notes is not a note)
MAX_BARS = 8
EIGHTH = TPQ // 2
STEP_TOKEN = {0: (1, False), 1: (1, True), 2: (2, False), 3: (2, True), 4: (3, False), 5: (4, False), 6: (4, True),
              7: (5, False), 8: (5, True), 9: (6, False), 10: (6, True), 11: (7, False)}


class HumError(ValueError):
    """Nothing usable in the recording (Korean message)."""


def _f0(wav_bytes: bytes) -> np.ndarray:
    import pyworld as pw
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if sr != FS:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=FS)
    if len(y) < FS * 0.5:
        raise HumError("녹음이 너무 짧아요")
    y = y / (np.abs(y).max() + 1e-9)
    f0, _ = pw.harvest(y, FS, frame_period=FRAME_MS, f0_floor=70, f0_ceil=1000)
    hop = int(FS * FRAME_MS / 1000)
    rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(y) else 0.0 for i in range(len(f0))])
    return clean_f0(f0, rms)


GAP_FILL_S = 0.2             # tracker dropouts inside a sounding note are bridged up to this length
SOUNDING = 0.12              # frame RMS above this share of the loudest frame counts as sounding


def clean_f0(f0: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Fix the two usual tracker errors on a sung note: octave jumps (folded back to the local median) and short
    unvoiced dropouts while the voice is still sounding (bridged when both sides have the same pitch)."""
    f0 = f0.copy()
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440), np.nan)
    n = len(midi)
    for i in range(n):                              # octave errors against the local median
        if np.isnan(midi[i]):
            continue
        ctx = midi[max(0, i - 10):i + 11]
        ctx = ctx[~np.isnan(ctx)]
        if len(ctx) < 5:
            continue
        d = midi[i] - np.median(ctx)
        if abs(abs(d) - 12) < 1.0:
            midi[i] -= 12 * np.sign(d)
    sounding = rms > SOUNDING * (rms.max() + 1e-12)
    max_gap = int(GAP_FILL_S * 1000 / FRAME_MS)
    i = 0
    while i < n:
        if not np.isnan(midi[i]) or not sounding[i] or i == 0 or np.isnan(midi[i - 1]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(midi[j]) and sounding[j]:
            j += 1
        if j < n and j - i <= max_gap and not np.isnan(midi[j]) and abs(midi[j] - midi[i - 1]) < 1.0:
            midi[i:j] = np.linspace(midi[i - 1], midi[j], j - i + 2)[1:-1]
        i = j + 1
    return np.where(np.isnan(midi), 0.0, 440 * 2 ** ((midi - 69) / 12))


def segment(f0: np.ndarray) -> list[tuple[float, float, float]]:
    """[(start_s, end_s, midi float)] from a frame-wise f0 (Hz, 0 = unvoiced)."""
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440), np.nan)
    hold = max(1, int(SPLIT_HOLD_S * 1000 / FRAME_MS))
    notes, i, n = [], 0, len(midi)
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(midi[j]):
            j += 1
        run = midi[i:j]                             # one voiced run: split where the pitch moves and stays moved
        start = 0
        for k in range(1, len(run) - hold + 1):
            ref = np.median(run[start:k])
            win = run[k:k + hold]
            if (k - start >= hold and np.all(np.abs(win - ref) > SPLIT_SEMITONES)
                    and float(np.ptp(win)) < STABLE_SEMITONES):
                notes.append((i + start, i + k, float(np.median(run[start:k]))))
                start = k
        notes.append((i + start, j, float(np.median(run[start:]))))
        i = j
    sec = FRAME_MS / 1000
    out: list[list[float]] = []
    for a, b, p in notes:                           # a too-short piece (a glide remnant) joins the note before it
        if (b - a) * sec < MIN_NOTE_S and out and abs(out[-1][1] - a) <= 1:
            out[-1][1] = b
        else:
            out.append([a, b, p])
    return [(a * sec, b * sec, p) for a, b, p in out if (b - a) * sec >= MIN_NOTE_S]


MAJOR_SCALE, MINOR_SCALE = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10, 11)


def estimate_key(notes: list[tuple[int, int]]) -> tuple[int, str]:
    """(tonic pc, 'maj'|'min') for a short tune [(duration, midi)]: Krumhansl-Kessler correlation, minus the share of
    time outside the scale, plus a bonus when the last (and first) note is the tonic. Short tunes are ambiguous for
    the profile alone (a few notes fit the relative key and its neighbours equally)."""
    hist = np.zeros(12)
    for d, p in notes:
        hist[p % 12] += d
    total = hist.sum() or 1.0
    first, last = notes[0][1] % 12, notes[-1][1] % 12
    best = None
    for tonic in range(12):
        for mode, prof, scale in (("maj", lakh.KK_MAJOR, MAJOR_SCALE), ("min", lakh.KK_MINOR, MINOR_SCALE)):
            r = float(np.corrcoef(hist, np.roll(prof, tonic))[0, 1]) if np.ptp(hist) > 0 else 0.0
            out = sum(hist[(tonic + i) % 12] for i in range(12) if i not in scale) / total
            score = r - 1.5 * out + 0.35 * (last == tonic) + 0.15 * (first == tonic)
            if best is None or score > best[0]:
                best = (score, tonic, mode)
    return best[1], best[2]


def to_tokens(notes: list[tuple[float, float, float]], bpm: int) -> dict:
    """Quantize to eighths at `bpm`, estimate the key, and write movable-do tokens (rests fill the gaps)."""
    if not notes:
        raise HumError("음을 찾지 못했어요. 조용한 곳에서 또렷하게 '음~' 또는 '라라라'로 불러 주세요")
    eighth_s = 60.0 / bpm / 2
    t0 = notes[0][0]
    grid = []
    for s, e, p in notes:
        a = int(round((s - t0) / eighth_s))
        b = max(a + 1, int(round((e - t0) / eighth_s)))
        if grid and a < grid[-1][1]:                 # overlapping after rounding: the earlier note gives way
            a = grid[-1][1]
            b = max(b, a + 1)
        grid.append([a, b, int(round(p))])
    limit = MAX_BARS * 8
    heard_notes, heard_bars = len(grid), -(-grid[-1][1] // 8)
    grid = [g for g in grid if g[0] < limit]            # the seed takes the first MAX_BARS bars; the rest is reported, not kept
    grid[-1][1] = min(grid[-1][1], limit)
    tonic, mode = estimate_key([(b - a, p) for a, b, p in grid])
    major_tonic = tonic if mode == "maj" else (tonic + 3) % 12   # degrees are relative to the major scale
    tokens, now = [], 0
    for a, b, p in grid:
        if a > now:
            tokens.append({"degree": 0, "sharp": False, "octave": 4, "dur": (a - now) * EIGHTH})
        x = p - major_tonic
        degree, sharp = STEP_TOKEN[x % 12]
        tokens.append({"degree": degree, "sharp": sharp, "octave": x // 12 - 1, "dur": (b - a) * EIGHTH})
        now = b
    names = lakh.ROOT_NAMES
    return {"tokens": tokens, "input_tonic": major_tonic, "bpm": bpm, "notes": len(grid),
            "key": f"{names[tonic]} {'장조' if mode == 'maj' else '단조'}", "bars": -(-now // 8),
            "bars_heard": heard_bars, "notes_heard": heard_notes, "cut": heard_bars > MAX_BARS,
            "kept_seconds": round(min(heard_bars, MAX_BARS) * 4 * 60.0 / bpm, 1)}   # counted from the first note


def transcribe(wav_bytes: bytes, bpm: int = 90) -> dict:
    if not 40 <= bpm <= 200:
        raise HumError("템포는 40~200 BPM이어야 해요")
    return to_tokens(segment(_f0(wav_bytes)), bpm)
