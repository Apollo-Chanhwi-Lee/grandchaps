"""Learned chord models: bar features, HMM (counts), BiLSTM tagger, constrained Viterbi decoding.

Classes are key-relative: id = rel_root * 2 + (0 maj | 1 min). 'other'/'N' bars are ignored in training.
"""
from __future__ import annotations

import numpy as np

from .chords import ChordDef, NoCandidate, chord_from_offset
from .theory import BAR, TPQ, Key, Note, beat_weight, split_by_bar

N_CLASSES = 24
IGNORE = -100
N_FEAT = 12 * 3 + 1 + 8 + 1
SEG_BARS = 8


def class_to_chord(key: Key, cid: int) -> ChordDef:
    return chord_from_offset(key, cid // 2, "min" if cid % 2 else "maj")


def chord_to_class(c: ChordDef) -> int | None:
    q = {"maj": 0, "dom7": 0, "min": 1}.get(c.quality)
    return None if q is None else c.offset * 2 + q


def bar_features(key: Key, melody: list[Note]) -> np.ndarray:
    """(n_bars, N_FEAT): key-relative pitch-class weights, beat-1 and beat-3 pitch, mode, bar position, has-melody."""
    bars = split_by_bar(melody)
    X = np.zeros((len(bars), N_FEAT), dtype=np.float32)
    for b, notes in enumerate(bars):
        for n in notes:
            if n.pitch is None:
                continue
            rel = (n.pitch - key.tonic) % 12
            pos = n.start % BAR
            X[b, rel] += n.dur / TPQ * (1 + 0.5 * beat_weight(pos))
            if pos == 0:
                X[b, 12 + rel] = 1
            elif pos == 2 * TPQ:
                X[b, 24 + rel] = 1
        s = X[b, :12].sum()
        if s:
            X[b, :12] /= s
        X[b, 36] = 1.0 if key.mode == "minor" else 0.0
        X[b, 37 + min(b, 7)] = 1
        X[b, 45] = 1.0 if s else 0.0
    return X


def gt_labels(key: Key, gts) -> np.ndarray:
    y = np.full(len(gts), IGNORE, dtype=np.int64)
    for i, g in enumerate(gts):
        if g.cls in ("maj", "min"):
            y[i] = ((g.root - key.tonic) % 12) * 2 + (1 if g.cls == "min" else 0)
    return y


def dataset(segments) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([bar_features(s.key, s.melody) for s in segments])
    Y = np.stack([gt_labels(s.key, s.chords) for s in segments])
    return X, Y


# ---------------- decoding ----------------
def viterbi(emis: np.ndarray, trans: np.ndarray, start: np.ndarray, allowed: np.ndarray | None = None) -> list[int]:
    """Max-sum path. emis (T,C) and trans (C,C) are log scores; allowed (T,C) bool masks classes per step."""
    T, C = emis.shape
    neg = -1e9
    e = emis if allowed is None else np.where(allowed, emis, neg)
    score = start + e[0]
    back = np.zeros((T, C), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def constraint_mask(key: Key, n: int, banned=None, forced=None) -> np.ndarray:
    allowed = np.ones((n, N_CLASSES), dtype=bool)
    for i, romans in (banned or {}).items():
        for r in romans:
            try:
                cid = chord_to_class(_by_roman(key, r))
            except ValueError:
                continue
            if cid is not None:
                allowed[i, cid] = False
    for i, r in (forced or {}).items():
        allowed[i] = False
        cid = chord_to_class(_by_roman(key, r))
        if cid is not None:
            allowed[i, cid] = True
    if not allowed.any(1).all():
        raise NoCandidate("constraints leave a bar with no chord")
    return allowed


def _by_roman(key, r):
    from .chords import chord_by_roman
    return chord_by_roman(key, r)


def bias_vector(key: Key, bias: dict[str, float] | None) -> np.ndarray:
    v = np.zeros(N_CLASSES)
    for r, b in (bias or {}).items():
        try:
            cid = chord_to_class(_by_roman(key, r))
        except ValueError:
            continue
        if cid is not None:
            v[cid] += b
    return v


# ---------------- HMM ----------------
class ChordHMM:
    """Emission: multinomial over key-relative melody pitch classes; transitions: bar-to-bar chord counts."""

    def __init__(self, emit_scale: float = 1.0):
        self.emit_scale = emit_scale

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> "ChordHMM":
        pc = np.full((N_CLASSES, 12), alpha)
        tr = np.full((N_CLASSES, N_CLASSES), alpha)
        st = np.full(N_CLASSES, alpha)
        for xs, ys in zip(X, Y):
            for t, (x, y) in enumerate(zip(xs, ys)):
                if y == IGNORE:
                    continue
                pc[y] += x[:12] + x[12:24] + 0.5 * x[24:36]
                if t == 0:
                    st[y] += 1
                if t > 0 and ys[t - 1] != IGNORE:
                    tr[ys[t - 1], y] += 1
        self.log_pc = np.log(pc / pc.sum(1, keepdims=True))
        self.log_tr = np.log(tr / tr.sum(1, keepdims=True))
        self.log_st = np.log(st / st.sum())
        prior = np.bincount(Y[Y != IGNORE], minlength=N_CLASSES) + alpha
        self.log_prior = np.log(prior / prior.sum())
        return self

    def emissions(self, x: np.ndarray) -> np.ndarray:
        w = x[:, :12] + x[:, 12:24] + 0.5 * x[:, 24:36]              # (T,12)
        return self.emit_scale * (w @ self.log_pc.T)                 # (T,C)

    def decode(self, x, allowed=None, bias=None) -> list[int]:
        e = self.emissions(x) + (bias if bias is not None else 0)
        return viterbi(e, self.log_tr, self.log_st, allowed)

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("log_pc", "log_tr", "log_st", "log_prior", "emit_scale")}

    @classmethod
    def from_state(cls, d) -> "ChordHMM":
        h = cls(float(d["emit_scale"]))
        for k in ("log_pc", "log_tr", "log_st", "log_prior"):
            setattr(h, k, np.asarray(d[k]))
        return h


# ---------------- BiLSTM ----------------
def make_bilstm(hidden: int = 128, layers: int = 2, dropout: float = 0.3):
    import torch.nn as nn

    class BiLSTMTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, N_CLASSES))

        def forward(self, x):
            h, _ = self.lstm(self.inp(x))
            return self.out(h)

    return BiLSTMTagger()


def masked_accuracy(logits_or_pred: np.ndarray, Y: np.ndarray) -> float:
    pred = logits_or_pred.argmax(-1) if logits_or_pred.ndim == Y.ndim + 1 else logits_or_pred
    m = Y != IGNORE
    return float((pred[m] == Y[m]).mean()) if m.any() else float("nan")
