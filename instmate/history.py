"""Composition history and user ratings (SQLite, same DB file as the preference memory).

Every composed song is stored with its seed, settings, lead sheet (model format: C major / A minor, the text the
lead sheet model reads and writes) and compose statistics.  A "good / bad" click is stored as a rating row
(`rating` = +1 / -1, `section` = index into the song's sections or NULL for the whole song).  The rows are the raw
material for later preference training (DPO); nothing here changes the generator.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class UnknownSong(KeyError):
    pass


class History:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS songs (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,"
                      " parent_id INTEGER, seed TEXT NOT NULL, settings TEXT NOT NULL, leadsheet TEXT NOT NULL,"
                      " compose_stats TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      " song_id INTEGER NOT NULL REFERENCES songs(id), section INTEGER,"
                      " rating INTEGER NOT NULL CHECK (rating IN (-1, 1)), created REAL NOT NULL)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def add_song(self, seed, settings: dict, leadsheet: str, compose_stats: dict, parent_id: int | None = None) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO songs (created, parent_id, seed, settings, leadsheet, compose_stats)"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            (time.time(), parent_id, json.dumps(seed, ensure_ascii=False),
                             json.dumps(settings, ensure_ascii=False), leadsheet,
                             json.dumps(compose_stats, ensure_ascii=False)))
            return int(cur.lastrowid)

    @staticmethod
    def _song(row) -> dict:
        return {"id": row[0], "created": row[1], "parent_id": row[2], "seed": json.loads(row[3]),
                "settings": json.loads(row[4]), "leadsheet": row[5], "compose_stats": json.loads(row[6])}

    def get_song(self, song_id: int) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT id, created, parent_id, seed, settings, leadsheet, compose_stats FROM songs"
                            " WHERE id = ?", (song_id,)).fetchone()
        if row is None:
            raise UnknownSong(song_id)
        return self._song(row)

    def list_songs(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, created, parent_id, seed, settings, leadsheet, compose_stats FROM songs"
                             " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._song(r) for r in rows]

    def rate(self, song_id: int, rating: int, section: int | None = None) -> int:
        """Append one rating (+1 / -1). Raises UnknownSong for an unknown song and ValueError for a bad rating."""
        if isinstance(rating, bool) or rating not in (-1, 1):
            raise ValueError(f"rating must be +1 or -1, got {rating!r}")
        if section is not None and (isinstance(section, bool) or not isinstance(section, int) or section < 0):
            raise ValueError(f"section must be a non-negative integer or None, got {section!r}")
        self.get_song(song_id)
        with self._conn() as c:
            cur = c.execute("INSERT INTO ratings (song_id, section, rating, created) VALUES (?, ?, ?, ?)",
                            (song_id, section, rating, time.time()))
            return int(cur.lastrowid)

    def ratings(self, song_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, song_id, section, rating, created FROM ratings WHERE song_id = ?"
                             " ORDER BY id", (song_id,)).fetchall()
        return [{"id": r[0], "song_id": r[1], "section": r[2], "rating": r[3], "created": r[4]} for r in rows]

    def similar_check(self, song_id: int) -> dict | None:
        """The similar-song report stored with a composed song (inside its compose stats), or None."""
        return self.get_song(song_id)["compose_stats"].get("similar_check")
