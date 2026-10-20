"""Mix settings: tempo (BPM) and per-instrument level. Applied only when rendering (playback, MIDI, sheet).

Arrangement events stay canonical in a version, so bar edits and lock checks never see mixed velocities.
"""
from __future__ import annotations

from dataclasses import replace

from .arrange import INSTRUMENTS, TEMPO, Event

TEMPO_MIN, TEMPO_MAX = 40, 200
LEVEL_MIN, LEVEL_MAX = 0.25, 2.0
LEVEL_TRACKS = INSTRUMENTS + ["melody"]


def resolve(genre: str, tempo: int | None = None, levels: dict | None = None) -> dict:
    """Validated, complete mix: {"tempo": bpm, "levels": {track: level}}. Raises ValueError on bad input."""
    bpm = TEMPO[genre] if tempo is None else tempo
    if isinstance(bpm, bool) or not isinstance(bpm, int) or not TEMPO_MIN <= bpm <= TEMPO_MAX:
        raise ValueError(f"tempo must be an integer {TEMPO_MIN}-{TEMPO_MAX}, got {tempo!r}")
    full = {t: 1.0 for t in LEVEL_TRACKS}
    for track, lv in (levels or {}).items():
        if track not in full:
            raise ValueError(f"unknown track for level: {track!r}")
        if isinstance(lv, bool) or not isinstance(lv, (int, float)) or not LEVEL_MIN <= lv <= LEVEL_MAX:
            raise ValueError(f"level for {track} must be {LEVEL_MIN}-{LEVEL_MAX}, got {lv!r}")
        full[track] = round(float(lv), 3)
    return {"tempo": bpm, "levels": full}


def apply_levels(events: list[Event], levels: dict[str, float]) -> list[Event]:
    """Copies with velocity scaled per track (clamped to 1-127). The input list is not modified."""
    out = []
    for e in events:
        lv = levels.get(e.track, 1.0)
        out.append(e if lv == 1.0 else replace(e, velocity=max(1, min(127, round(e.velocity * lv)))))
    return out
