"""Does a YuE2 take sing what its score says?  The voice is separated (the app's Demucs, as in yueblend), its pitch
is tracked (pYIN) and compared, frame by frame, with the pitch class the score's Vocal voice holds at that moment.
Octaves are ignored (the model sings an octave below the written notes); a lead-in of up to 0.6 s is fitted.

It answers one question only: did the singer sing the written notes.  On the two whole songs of 2026-10-12 it gave
0.97 and 0.94, yet the user heard the second one's guitar line off the beat: that fault is in the instruments and this
check does not see it.  (An earlier measurement with the lead-in fitted on the chords gave 0.86 and 0.59 and looked
like agreement with the ear; with the lead-in fitted on the voice the difference is gone.)  Only frames where a pitch
was found count.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

from . import yueimport
from .theory import TPQ

SR, HOP = 22050, 512
GOOD = 0.7                      # a take below this does not sing its score (takes that do measured 0.90-0.97; chance 0.1-0.2)
OFFSETS = [round(-0.2 + 0.05 * i, 2) for i in range(17)]        # -0.2 .. 0.6 s


class CheckError(RuntimeError):
    """The check cannot be made (no separator, an unreadable score)."""


def _vocals(wav: bytes) -> bytes:
    from . import yueblend
    python = yueblend.separator()
    if python is None:
        raise CheckError("no separator")
    with tempfile.TemporaryDirectory(prefix="instmate_yuecheck_") as tmp:
        src, dst = Path(tmp) / "take.wav", Path(tmp) / "vocals.wav"
        src.write_bytes(wav)
        proc = subprocess.run([str(python), str(yueblend.SEPARATE), str(src), str(dst)], capture_output=True, text=True, timeout=900)
        if proc.returncode != 0 or not dst.is_file():
            raise CheckError(f"separation failed: {(proc.stderr or '').strip()[-200:]}")
        return dst.read_bytes()


def follow(wav: bytes, abc: str, separate="app", shift: int = 0) -> dict:
    """{match, chance, frames, offset_s}: the share of pitched frames inside sung notes whose pitch class is the
    written one; `chance` is the same against the score rolled by half its length.  match None: too little was
    pitched to say.  `separate`: "app" separates the voice first; None takes the audio as it is (a voice alone).  `shift`: semitones the
    finished take was moved by (a male song, a lower key): the voice is compared in the key it was sung in."""
    import librosa
    import numpy as np
    try:
        sheet, _, _ = yueimport.read(abc)
    except yueimport.AbcError as exc:
        raise CheckError(f"the score cannot be read: {exc}") from None
    num, den = (int(x) for x in sheet.meter.split("/"))
    unit_s = 60.0 / sheet.tempo * 4 / den / (16 // den)          # one sixteenth
    line = []
    for bar in (b for s in sheet.sections for b in s.bars):
        row = [-1] * (num * 16 // den)
        for n in bar.notes:
            if n.pitch is not None:
                for u in range(n.start // (TPQ // 4), min(len(row), (n.start + n.dur) // (TPQ // 4))):
                    row[u] = n.pitch % 12
        line += row
    line = np.array(line)
    y, _ = librosa.load(io.BytesIO(_vocals(wav) if separate == "app" else wav), sr=SR, mono=True)
    if len(y) < SR // 4:
        return {"match": None, "chance": None, "frames": 0, "offset_s": 0.0}
    f0, _, prob = librosa.pyin(y, fmin=65, fmax=1047, sr=SR, hop_length=HOP)
    ok = (prob > 0.5) & ~np.isnan(f0)
    sung = np.full(len(f0), -1)
    sung[ok] = (np.round(librosa.hz_to_midi(f0[ok])).astype(int) - shift) % 12
    t = np.arange(len(sung)) * HOP / SR

    def at(offset, written):
        idx = np.floor((t - offset) / unit_s).astype(int)
        inside = (idx >= 0) & (idx < len(written))
        want = np.full(len(sung), -1)
        want[inside] = written[idx[inside]]
        both = (sung >= 0) & (want >= 0)
        return (float((sung[both] == want[both]).mean()) if both.sum() >= 20 else None), int(both.sum())

    best = max(OFFSETS, key=lambda o: (at(o, line)[0] or 0.0, -abs(o)))
    match, frames = at(best, line)
    chance, _ = at(best, np.roll(line, len(line) // 2))
    return {"match": None if match is None else round(match, 3), "chance": None if chance is None else round(chance, 3),
            "frames": frames, "offset_s": best}
