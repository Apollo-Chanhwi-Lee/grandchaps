"""User preference memory (Get Me): chords the user picked by editing are favored next time."""
from __future__ import annotations

import sqlite3
from pathlib import Path

BIAS_PER_PICK = 0.05
BIAS_CAP = 0.3


class Prefs:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS picks (genre TEXT, mode TEXT, roman TEXT, n INTEGER,"
                      " PRIMARY KEY (genre, mode, roman))")

    def _conn(self):
        return sqlite3.connect(self.path)

    def record(self, genre: str, mode: str, roman: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO picks VALUES (?, ?, ?, 1) ON CONFLICT(genre, mode, roman)"
                      " DO UPDATE SET n = n + 1", (genre, mode, roman))

    def bias(self, genre: str, mode: str) -> dict[str, float]:
        with self._conn() as c:
            rows = c.execute("SELECT roman, n FROM picks WHERE genre = ? AND mode = ?", (genre, mode)).fetchall()
        return {r: min(BIAS_CAP, n * BIAS_PER_PICK) for r, n in rows}

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM picks")
