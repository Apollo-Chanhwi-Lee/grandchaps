"""Diagnostic only: render one song with the level values that another render of it computed.

Three steps of audio.render set a level from the WHOLE song: the scale of a separately rendered stem (its overall RMS),
the gain of the guide voice against the band, and the loudness measurement of the master.  When two renders of one
song are compared (for example with and without a small local edit), those values move a little and every sample of
the song moves with them.  With this lock the second render reuses the first one's values, so what still differs is
the edit itself.

    with levellock.record() as kept:   wav_a = audio.render(...)
    with levellock.replay(kept):       wav_b = audio.render(..., some local edit)

Off by default: outside these two context managers `value` only calls `compute`, and nothing in the app uses them.
"""
from __future__ import annotations

from contextlib import contextmanager

_state: dict | None = None


def value(name: str, compute):
    """`compute()` normally; while recording, also kept under `name` in call order; while replaying, the kept value
    of the same call (a missing one is an error: the two renders did not take the same path)."""
    if _state is None:
        return compute()
    if _state["mode"] == "record":
        v = compute()
        _state["values"].setdefault(name, []).append(v)
        return v
    i = _state["pos"].get(name, 0)
    kept = _state["values"].get(name, [])
    if i >= len(kept):
        raise RuntimeError(f"level lock: no recorded value #{i + 1} for {name!r} (the two renders took different paths)")
    _state["pos"][name] = i + 1
    return kept[i]


@contextmanager
def record():
    global _state
    if _state is not None:
        raise RuntimeError("level lock is already in use")
    _state = {"mode": "record", "values": {}}
    try:
        yield _state["values"]
    finally:
        _state = None


@contextmanager
def replay(values: dict):
    global _state
    if _state is not None:
        raise RuntimeError("level lock is already in use")
    _state = {"mode": "replay", "values": values, "pos": {}}
    try:
        yield
    finally:
        _state = None
