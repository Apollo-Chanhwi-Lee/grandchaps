"""Sound option "mix_v3" (off unless asked for): a wider, firmer mix made part by part.

Against the plain mix: the guitar is doubled left and right (the right side a few milliseconds late), piano, keys
and strings are spread, bass and the low end stay in the middle (nothing but the mid signal under MONO_BELOW_HZ),
voice and guitar get presence around 3 kHz, kick and bass get weight around 55 Hz, the whole mix loses a little at
200 Hz, and the master ends near LUFS under the TRUE_PEAK ceiling.  The numbers come from the user's order of
2026-10-10 and are design choices to be judged by ear; the mix must not lose more than MONO_DROP_DB when summed to mono.
"""
from __future__ import annotations

import numpy as np

from .mixv2 import _bell

LUFS, TRUE_PEAK = -11.0, -1.0
MONO_BELOW_HZ = 150.0
MONO_DROP_DB = 3.0
DOUBLE_MS, DOUBLE_GAIN = 13.0, 0.9          # the guitar's second take: this much later and a little softer, on the right
PRESENCE_HZ, PRESENCE_DB = 3000.0, 2.5      # voice and guitar
WEIGHT_HZ, WEIGHT_DB = 55.0, 2.5            # kick (the drum part) and bass
BOX_HZ, BOX_DB = 200.0, -2.0                # the whole mix
SPREAD_DB = {"piano": 5.0, "keys": 5.0, "strings": 6.0, "harmony": 3.0, "lead": 2.0}     # side signal raised
CENTRE = ("bass",)                          # summed to the middle


def _filter(x: np.ndarray, sr: int, hz: float, db: float, q: float = 1.0) -> np.ndarray:
    from scipy.signal import lfilter
    b, a = _bell(sr, hz, db, q)
    return lfilter(b, a, x.astype(np.float64), axis=-1)


def presence(mono: np.ndarray, sr: int) -> np.ndarray:
    """The guide voice with its presence lift (the voice is mixed in after the band)."""
    return _filter(mono, sr, PRESENCE_HZ, PRESENCE_DB).astype(np.float32)


def shape(stem: np.ndarray, sr: int, part: str) -> np.ndarray:
    """One part's (2, n) audio with its tone and place."""
    s = stem.astype(np.float64)
    if part == "guitar":
        mono = _filter(s.mean(axis=0), sr, PRESENCE_HZ, PRESENCE_DB)
        late = int(DOUBLE_MS / 1000 * sr)
        right = np.concatenate([np.zeros(late), mono[: len(mono) - late]]) * DOUBLE_GAIN
        return np.stack([mono, right])
    if part in CENTRE:
        mono = _filter(s.mean(axis=0), sr, WEIGHT_HZ, WEIGHT_DB)
        return np.stack([mono, mono])
    if part == "drums":
        return _filter(s, sr, WEIGHT_HZ, WEIGHT_DB)
    if part in SPREAD_DB:
        mid, side = (s[0] + s[1]) / 2, (s[0] - s[1]) / 2 * 10 ** (SPREAD_DB[part] / 20)
        return np.stack([mid + side, mid - side])
    return s


def mono_low(mix: np.ndarray, sr: int) -> np.ndarray:
    """The mix with only the mid signal under MONO_BELOW_HZ."""
    from scipy.signal import butter, sosfiltfilt
    mid, side = (mix[0] + mix[1]) / 2, (mix[0] - mix[1]) / 2
    side = side - sosfiltfilt(butter(8, MONO_BELOW_HZ, "lowpass", fs=sr, output="sos"), side)
    return np.stack([mid + side, mid - side])


def mixdown(stems: dict[str, np.ndarray], sr: int) -> np.ndarray:
    n = max(s.shape[1] for s in stems.values())
    out = np.zeros((2, n))
    for part in sorted(stems):
        s = shape(stems[part], sr, part)
        out[:, : s.shape[1]] += s
    out = mono_low(_filter(out, sr, BOX_HZ, BOX_DB), sr)
    peak = float(np.abs(out).max())
    return (out * (0.99 / peak) if peak > 0.99 else out).astype(np.float32)


def mono_drop_db(stereo: np.ndarray) -> float:
    """How much quieter the (n, 2) audio is when both channels are summed to one (dB, positive = quieter)."""
    x = stereo.astype(np.float64)
    both = np.mean(x ** 2)
    mono = np.mean(((x[:, 0] + x[:, 1]) / 2) ** 2)
    return float(10 * np.log10((both + 1e-20) / (mono + 1e-20)))
