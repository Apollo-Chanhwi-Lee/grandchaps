"""Chord metrics against POP909 ground truth (bar level)."""
from __future__ import annotations

from .chords import ChordDef
from .pop909 import GTChord
from .theory import Key

PRED_CLASS = {"maj": "maj", "dom7": "maj", "min": "min", "dim": "other"}


def pred_root_cls(key: Key, c: ChordDef) -> tuple[int, str]:
    return (key.tonic + c.offset) % 12, PRED_CLASS[c.quality]


def score_bars(preds: list[tuple[int, str]], gts: list[GTChord]) -> dict:
    """Counts for one segment.

    root:   bars whose GT is not 'N'; correct when roots match.
    majmin: bars whose GT class is maj/min (MIREX majmin style, sus/dim/aug excluded); correct when root and class match.
    """
    if len(preds) != len(gts):
        raise ValueError(f"{len(preds)} predictions for {len(gts)} bars")
    c = {"root_n": 0, "root_ok": 0, "majmin_n": 0, "majmin_ok": 0}
    for (root, cls), gt in zip(preds, gts):
        if gt.cls == "N":
            continue
        c["root_n"] += 1
        c["root_ok"] += int(root == gt.root)
        if gt.cls in ("maj", "min"):
            c["majmin_n"] += 1
            c["majmin_ok"] += int(root == gt.root and cls == gt.cls)
    return c


GT_TONES = {"maj": ({0, 4, 7}, {2, 9, 11}), "min": ({0, 3, 7}, {2, 5, 10})}


def gt_clash_rates(melody, gts: list[GTChord]) -> list[float]:
    """Our clash metric (check.clash_rates rules) applied to ground-truth maj/min triads; other/N bars skipped."""
    from .check import BEAT3_WEIGHT
    from .theory import BAR, TPQ, beat_weight, split_by_bar
    rates = []
    for notes, gt in zip(split_by_bar(melody), gts):
        if gt.cls not in GT_TONES:
            continue
        tones, tens = GT_TONES[gt.cls]
        total = bad = 0.0
        for n in notes:
            bw = beat_weight(n.start % BAR)
            if n.pitch is None or bw < 0.8:
                continue
            w = n.dur / TPQ * (1.0 if bw == 1.0 else BEAT3_WEIGHT)
            total += w
            bad += w * ((n.pitch - gt.root) % 12 not in tones | tens)
        if total:
            rates.append(bad / total)
    return rates


def merge(counts: list[dict]) -> dict:
    tot = {k: sum(c[k] for c in counts) for k in ("root_n", "root_ok", "majmin_n", "majmin_ok")}
    tot["root_acc"] = round(tot["root_ok"] / tot["root_n"], 4) if tot["root_n"] else None
    tot["majmin_acc"] = round(tot["majmin_ok"] / tot["majmin_n"], 4) if tot["majmin_n"] else None
    return tot
