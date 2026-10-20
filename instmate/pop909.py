"""POP909 -> 8-bar 4/4 segments (melody Notes, ground-truth chord per bar, key) + song-level split."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .theory import BAR, TPQ, Key, Note

SEG_BARS = 8
MIN_MELODY_BARS = 6          # skip segments (intros, interludes) with melody in fewer bars
GRID = TPQ // 4              # quantize to 16th notes
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
      "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}

MAJ_CLASS = {"maj", "maj/3", "maj/5", "maj7", "maj7/7", "maj7/3", "maj7/5", "maj6", "7", "7/b7", "7/5", "7/3"}
MIN_CLASS = {"min", "min/b3", "min/5", "min7", "min7/b7", "min7/5", "min6"}


@dataclass(frozen=True)
class GTChord:
    root: int | None      # None for "N" (no chord)
    cls: str              # "maj" | "min" | "other" | "N"
    label: str


def parse_chord(label: str) -> GTChord:
    if label == "N":
        return GTChord(None, "N", label)
    root, qual = label.split(":", 1)
    if root not in PC:
        raise ValueError(f"unknown root in chord {label!r}")
    cls = "maj" if qual in MAJ_CLASS else "min" if qual in MIN_CLASS else "other"
    return GTChord(PC[root], cls, label)


def parse_key(label: str) -> Key:
    root, mode = label.split(":")
    if mode not in ("maj", "min"):
        raise ValueError(f"unknown key mode {label!r}")
    return Key(PC[root], "major" if mode == "maj" else "minor")


@dataclass
class Segment:
    song: str
    start_bar: int
    key: Key
    melody: list[Note]            # ticks relative to segment start, exactly SEG_BARS bars long
    chords: list[GTChord]         # one per bar


def _read_table(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            s, e, lab = line.split()[:3]
            out.append((float(s), float(e), lab))
    return out


def song_meter_ok(root: Path) -> set[str]:
    """Song ids annotated as 2 beats/measure with 2 quavers/beat (duple, pairs into 4/4 via the 4-beat downbeats)."""
    import openpyxl
    rows = list(openpyxl.load_workbook(root / "index.xlsx", read_only=True).active.iter_rows(values_only=True))[1:]
    return {str(r[0]) for r in rows if r[0] and str(r[4]) == "2" and str(r[5]) == "2"}


def load_segments(song_dir: Path) -> list[Segment]:
    import pretty_midi
    sid = song_dir.name
    beats = np.loadtxt(song_dir / "beat_midi.txt")
    beat_t = beats[:, 0]
    down = np.where(beats[:, 2] == 1)[0]
    pm = pretty_midi.PrettyMIDI(str(song_dir / f"{sid}.mid"))
    mel = [i for i in pm.instruments if i.name == "MELODY"]
    if not mel:
        return []
    idx = np.arange(len(beat_t), dtype=float)
    to_beat = lambda sec: np.interp(sec, beat_t, idx)
    notes = sorted(mel[0].notes, key=lambda n: n.start)
    on = np.round(to_beat([n.start for n in notes]) * 4) / 4       # beats, 16th-quantized
    off = np.round(to_beat([n.end for n in notes]) * 4) / 4
    chords = [(to_beat(s), to_beat(e), lab) for s, e, lab in _read_table(song_dir / "chord_midi.txt")]
    keys = [(to_beat(s), to_beat(e), lab) for s, e, lab in _read_table(song_dir / "key_audio.txt")]

    segs = []
    i = 0
    while i + SEG_BARS < len(down):
        bar_starts = down[i:i + SEG_BARS + 1]
        if not np.all(np.diff(bar_starts) == 4):         # need 8 regular 4-beat bars
            i += 1
            continue
        b0, b1 = float(bar_starts[0]), float(bar_starts[-1])
        seg_keys = [k for k in keys if k[0] < b1 and k[1] > b0]
        if len(seg_keys) != 1:                            # key change inside (or no key): skip
            i += SEG_BARS
            continue
        seg = _build_segment(sid, i, parse_key(seg_keys[0][2]), notes, on, off, chords, b0)
        if seg is not None:
            segs.append(seg)
        i += SEG_BARS
    return segs


def _build_segment(sid, start_bar, key, notes, on, off, chords, b0) -> Segment | None:
    b1 = b0 + 4 * SEG_BARS
    mel: list[Note] = []
    for n, s, e in zip(notes, on, off):
        s, e = max(s, b0), min(e, b1)
        if e <= s:
            continue
        st, en = int(round((s - b0) * TPQ)), int(round((e - b0) * TPQ))
        if mel and mel[-1].start + mel[-1].dur > st:     # enforce monophony: cut the previous note
            mel[-1].dur = st - mel[-1].start
            if mel[-1].dur <= 0:
                mel.pop()
        mel.append(Note(n.pitch, st, en - st))
    mel = [m for m in mel if m.dur > 0]
    if len({m.start // BAR for m in mel}) < MIN_MELODY_BARS:
        return None
    end = SEG_BARS * BAR
    last = max(m.start + m.dur for m in mel)
    if last < end:
        mel.append(Note(None, last, end - last))
    gt = [majority_chord(chords, b0 + 4 * b, b0 + 4 * (b + 1)) for b in range(SEG_BARS)]
    return Segment(sid, start_bar, key, mel, gt)


def majority_chord(chords, s, e) -> GTChord:
    """Chord with the largest overlap in [s, e); 'N' if nothing overlaps."""
    best, best_ov = None, 0.0
    for cs, ce, lab in chords:
        ov = min(ce, e) - max(cs, s)
        if ov > best_ov:
            best, best_ov = lab, ov
    return parse_chord(best) if best else GTChord(None, "N", "N")


def make_split(song_ids: list[str], seed: int = 42, frac=(0.8, 0.1, 0.1)) -> dict[str, list[str]]:
    ids = sorted(song_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    split = {"train": sorted(ids[:a]), "val": sorted(ids[a:b]), "test": sorted(ids[b:])}
    assert not (set(split["train"]) & set(split["val"]) | set(split["train"]) & set(split["test"])
                | set(split["val"]) & set(split["test"])), "song overlap between splits"
    return split


def fingerprint(root: Path, song_ids: list[str]) -> str:
    """sha256 over the annotation and MIDI files actually used."""
    h = hashlib.sha256()
    for sid in sorted(song_ids):
        for f in ("beat_midi.txt", "chord_midi.txt", "key_audio.txt", f"{sid}.mid"):
            h.update((root / sid / f).read_bytes())
    return h.hexdigest()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
