"""Sound option "master_v2" (off unless asked for): a mix made part by part, then a louder master.

The plain render sums every SoundFont part inside FluidSynth and corrects the tone of the whole mix once.  Here each
part is rendered alone, gets its own tone correction (less 300 Hz, more 3 kHz, more air above 9 kHz, by part) and its
own place between the speakers, and the parts are summed afterwards.  The master keeps its whole-mix clarity EQ (the
per-part correction comes on top of it; without it this mix measured duller than the plain one), compresses a little
more and ends near LUFS under a TRUE_PEAK ceiling with a limiter.

Every number below is a design hypothesis, not a measured optimum: it is judged by ear (work/listening), not by a
score.  scipy is used for the filters (it is installed with librosa)."""
from __future__ import annotations

import numpy as np

LUFS, TRUE_PEAK = -12.0, -1.0
LIMIT_DB = -2.0              # sample-peak limit of the limiter; the margin under TRUE_PEAK is for inter-sample peaks
COMPRESSOR = "acompressor=threshold=-18dB:ratio=3:attack=20:release=250"
# per part, in dB: "mud" a bell at 300 Hz, "presence" a bell at 3 kHz, "air" a shelf from 9 kHz
TONE = {
    "guitar": {"mud": -3.0, "presence": 2.0, "air": 1.0},
    "piano": {"mud": -3.0, "presence": 1.5, "air": 2.0},
    "keys": {"mud": -3.0, "presence": 1.5, "air": 2.0},
    "strings": {"mud": -2.0, "presence": 0.0, "air": 2.0},
    "lead": {"mud": -2.0, "presence": 2.0, "air": 1.0},
    "harmony": {"mud": -2.0, "presence": 1.0, "air": 1.0},
    "melody": {"mud": -1.0, "presence": 2.0, "air": 1.0},
    "drums": {"mud": -3.0, "presence": 1.5, "air": 3.0},
    "bass": {"mud": -2.0, "presence": 1.0, "air": 0.0},
}
MUD_HZ, PRESENCE_HZ, AIR_HZ = 300.0, 3000.0, 9000.0
# per part: how far the quieter side is turned down (dB; negative = the part sits left, positive = right).  Bass,
# drums and the melody stay in the middle.  "strings" are widened instead (their side signal is raised).
SIDE = {"guitar": -5.0, "piano": 4.0, "keys": -2.5, "lead": 3.0, "harmony": -2.0}
WIDEN = {"strings": 3.0}


def _bell(sr: int, hz: float, gain_db: float, q: float = 1.0):
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * hz / sr
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
    return b / den[0], den / den[0]


def _high_shelf(sr: int, hz: float, gain_db: float):
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * hz / sr
    cos, alpha = np.cos(w), np.sin(w) / 2 * np.sqrt(2)
    b = np.array([a * ((a + 1) + (a - 1) * cos + 2 * np.sqrt(a) * alpha), -2 * a * ((a - 1) + (a + 1) * cos),
                  a * ((a + 1) + (a - 1) * cos - 2 * np.sqrt(a) * alpha)])
    den = np.array([(a + 1) - (a - 1) * cos + 2 * np.sqrt(a) * alpha, 2 * ((a - 1) - (a + 1) * cos),
                    (a + 1) - (a - 1) * cos - 2 * np.sqrt(a) * alpha])
    return b / den[0], den / den[0]


def tone(stem: np.ndarray, sr: int, part: str) -> np.ndarray:
    """(2, n) audio of one part with that part's tone correction; a part that is not in TONE is returned as it is."""
    row = TONE.get(part)
    if row is None:
        return stem
    from scipy.signal import lfilter
    out = stem.astype(np.float64)
    for gain, make, hz in ((row["mud"], _bell, MUD_HZ), (row["presence"], _bell, PRESENCE_HZ), (row["air"], _high_shelf, AIR_HZ)):
        if gain:
            b, a = make(sr, hz, gain)
            out = lfilter(b, a, out, axis=1)
    return out.astype(np.float32)


def place(stem: np.ndarray, part: str) -> np.ndarray:
    """The part moved to its side (the other channel turned down) or widened; centre parts are returned as they are."""
    if part in SIDE:
        out = stem.copy()
        out[1 if SIDE[part] < 0 else 0] *= 10 ** (-abs(SIDE[part]) / 20)
        return out
    if part in WIDEN:
        mid, side = (stem[0] + stem[1]) / 2, (stem[0] - stem[1]) / 2 * 10 ** (WIDEN[part] / 20)
        return np.stack([mid + side, mid - side]).astype(np.float32)
    return stem


def mixdown(stems: dict[str, np.ndarray], sr: int) -> np.ndarray:
    """(2, n) sum of the parts, each with its tone and place; turned down as a whole only if the sum would clip."""
    n = max(s.shape[1] for s in stems.values())
    out = np.zeros((2, n), dtype=np.float32)
    for part in sorted(stems):
        s = place(tone(stems[part], sr, part), part)
        out[:, : s.shape[1]] += s
    peak = float(np.abs(out).max())
    return out * (0.99 / peak) if peak > 0.99 else out
