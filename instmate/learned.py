"""Trained chord models as agent tools (same signature as chords.progression)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .chords import ChordDef
from .ml import ChordHMM, bar_features, bias_vector, class_to_chord, constraint_mask, make_bilstm, viterbi
from .theory import Key, Note

MODELS = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def load_hmm(path: str = str(MODELS / "chord_hmm.npz")) -> ChordHMM:
    return ChordHMM.from_state(np.load(path))


@lru_cache(maxsize=1)
def load_bilstm(path: str = str(MODELS / "chord_bilstm.pt")):
    import torch
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    m = make_bilstm(cfg["hidden"], cfg["layers"], cfg["dropout"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, float(ck["beta"])


def hmm_tool(key: Key, genre: str, melody: list[Note], banned=None, forced=None, bias=None) -> list[ChordDef]:
    x = bar_features(key, melody)
    h = load_hmm()
    path = h.decode(x, constraint_mask(key, len(x), banned, forced), bias_vector(key, bias))
    return [class_to_chord(key, c) for c in path]


def bilstm_tool(key: Key, genre: str, melody: list[Note], banned=None, forced=None, bias=None) -> list[ChordDef]:
    import torch
    x = bar_features(key, melody)
    model, beta = load_bilstm()
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor(x[None])), -1)[0].numpy()
    h = load_hmm()
    path = viterbi(lp + bias_vector(key, bias), beta * h.log_tr, beta * h.log_st,
                   constraint_mask(key, len(x), banned, forced))
    return [class_to_chord(key, c) for c in path]


def available() -> dict:
    return {"hmm": (MODELS / "chord_hmm.npz").exists(), "bilstm": (MODELS / "chord_bilstm.pt").exists()}
