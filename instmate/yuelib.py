"""The library of songs YuE2 made: one folder per song under DATA/yue_library (the audio, the score YuE2 wrote, and
what the page shows about it).  Nothing is ever changed after it is saved."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

ID = re.compile(r"[0-9]{13}[0-9a-f]{6}")
UNTITLED = "제목 없음"


def _root(data) -> Path:
    return Path(data) / "yue_library"


def _dir(data, song_id: str) -> Path:
    if not isinstance(song_id, str) or not ID.fullmatch(song_id) or not (_root(data) / song_id / "meta.json").is_file():
        raise KeyError(song_id)
    return _root(data) / song_id


def save(data, audio: bytes, fmt: str, abc: str, meta: dict) -> str:
    """Store one song; returns its id (time first, so ids sort by age)."""
    song_id = f"{int(time.time() * 1000):013d}{uuid.uuid4().hex[:6]}"
    folder = _root(data) / song_id
    folder.mkdir(parents=True)
    (folder / f"audio.{fmt}").write_bytes(audio)
    (folder / "score.abc").write_text(abc, encoding="utf-8")
    row = {**meta, "id": song_id, "format": fmt, "title": (meta.get("title") or "").strip() or UNTITLED,
           "created": time.strftime("%Y-%m-%d %H:%M")}
    (folder / "meta.json").write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return song_id


def entries(data) -> list[dict]:
    """Every song, newest first (without the score text)."""
    root = _root(data)
    if not root.is_dir():
        return []
    rows = []
    for folder in sorted((p for p in root.iterdir() if ID.fullmatch(p.name) and (p / "meta.json").is_file()), reverse=True):
        rows.append(json.loads((folder / "meta.json").read_text(encoding="utf-8")))
    return rows


def get(data, song_id: str) -> dict:
    folder = _dir(data, song_id)
    return {**json.loads((folder / "meta.json").read_text(encoding="utf-8")), "abc": (folder / "score.abc").read_text(encoding="utf-8")}


def audio(data, song_id: str) -> tuple[bytes, str]:
    folder = _dir(data, song_id)
    fmt = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["format"]
    return (folder / f"audio.{fmt}").read_bytes(), fmt
