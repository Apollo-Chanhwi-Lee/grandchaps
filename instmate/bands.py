"""Band workspaces: each band keeps its own versions of a song (who changed what, from which version) in SQLite.

    create_band / list_bands / get_band        name, members, number of guitarists, the singer's range, where the song
                                               came from and on what terms (free text the band writes itself)
    save_version / list_versions / get_version  a full snapshot of the song as the band has it now, with author, note,
                                               credits and the parent version; nothing is ever overwritten
    compare(a, b)                              what differs between two versions: tempo, key, form, chords per bar,
                                               and per part the bars whose notes changed

Bands are independent by construction: a version belongs to one band and its parent must be in the same band.
The store records what the band enters; it does not judge authorship, shares or rights.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sqlite3
import time
from pathlib import Path

import mido


class UnknownBand(KeyError):
    pass


class UnknownVersion(KeyError):
    pass


class BandError(ValueError):
    """Bad input (Korean message)."""


BAND_FIELDS = ("name", "members", "guitarists", "vocal_low", "vocal_high", "source_note")
# A band's own direction: what the next song of THIS band starts from. Only what the band chose; nothing is learned
# from other bands or from reference songs, and one band's profile never changes another band's.
PROFILE_KEYS = ("mood", "genre", "tempo", "lead", "melody_role", "instruments", "energy", "key_change", "sounds")
LEADS = ("female", "male")


def clean_profile(profile: dict) -> dict:
    """Validated copy of a band profile (BandError names the bad field). Missing keys are simply not set."""
    from . import audio
    from .arrange import BAND
    from .genres import GENRES
    if not isinstance(profile, dict) or set(profile) - set(PROFILE_KEYS):
        raise BandError(f"밴드 방향에 쓸 수 있는 항목: {', '.join(PROFILE_KEYS)}")
    out = {}
    if profile.get("mood"):
        out["mood"] = str(profile["mood"]).strip()[:300]
    if profile.get("genre") is not None:
        if profile["genre"] not in GENRES:
            raise BandError(f"모르는 장르예요: {profile['genre']!r}")
        out["genre"] = profile["genre"]
    if profile.get("tempo") is not None:
        if isinstance(profile["tempo"], bool) or not isinstance(profile["tempo"], int) or not 40 <= profile["tempo"] <= 200:
            raise BandError("템포는 40~200 사이의 정수로 적어 주세요")
        out["tempo"] = profile["tempo"]
    if profile.get("lead") is not None:
        if profile["lead"] not in LEADS:
            raise BandError(f"리드 보컬은 {', '.join(LEADS)} 중 하나예요")
        out["lead"] = profile["lead"]
    if profile.get("melody_role") is not None:
        from .render import MELODY_ROLES
        if profile["melody_role"] not in MELODY_ROLES:
            raise BandError(f"멜로디 역할은 {', '.join(MELODY_ROLES)} 중 하나예요")
        out["melody_role"] = profile["melody_role"]
    if profile.get("instruments") is not None:
        parts = list(profile["instruments"])
        allowed = set(BAND) | {"strings", "keys"}
        if not parts or set(parts) - allowed or len(set(parts)) != len(parts):
            raise BandError(f"편성은 {', '.join(sorted(allowed))} 중에서 겹치지 않게 골라 주세요")
        out["instruments"] = parts
    for flag in ("energy", "key_change"):
        if profile.get(flag) is not None:
            if not isinstance(profile[flag], bool):
                raise BandError(f"{flag}는 true / false 로 적어 주세요")
            out[flag] = profile[flag]
    if profile.get("sounds"):
        try:
            audio.check_sounds(dict(profile["sounds"]))
        except (KeyError, ValueError) as e:
            raise BandError(str(e)) from None
        out["sounds"] = dict(profile["sounds"])
    return out


class Bands:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS bands (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,"
                      " name TEXT NOT NULL, members TEXT, guitarists INTEGER, vocal_low INTEGER, vocal_high INTEGER,"
                      " source_note TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS versions (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      " band_id INTEGER NOT NULL REFERENCES bands(id), parent_id INTEGER, created REAL NOT NULL,"
                      " author TEXT, note TEXT, credits TEXT NOT NULL, sha256 TEXT NOT NULL, result TEXT NOT NULL)")
            if "profile" not in [r[1] for r in c.execute("PRAGMA table_info(bands)")]:      # added 2026-10-06
                c.execute("ALTER TABLE bands ADD COLUMN profile TEXT")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    # ------------------------------------------------------------ bands
    def create_band(self, name: str, members: str | None = None, guitarists: int | None = None,
                    vocal_low: int | None = None, vocal_high: int | None = None, source_note: str | None = None) -> int:
        name = (name or "").strip()
        if not name:
            raise BandError("밴드 이름을 적어 주세요")
        if guitarists is not None and not 0 <= guitarists <= 4:
            raise BandError("기타 인원은 0~4명으로 적어 주세요")
        if (vocal_low is None) != (vocal_high is None):
            raise BandError("보컬 음역은 가장 낮은 음과 가장 높은 음을 함께 적어 주세요")
        if vocal_low is not None and not 24 <= vocal_low < vocal_high <= 96:
            raise BandError("보컬 음역이 맞지 않아요 (낮은 음 < 높은 음, MIDI 24~96)")
        with self._conn() as c:
            cur = c.execute("INSERT INTO bands (created, name, members, guitarists, vocal_low, vocal_high, source_note)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (time.time(), name[:80], (members or "")[:300] or None, guitarists, vocal_low, vocal_high,
                             (source_note or "")[:1000] or None))
            return cur.lastrowid

    def _band_row(self, row) -> dict:
        return {"id": row[0], "created": row[1], **dict(zip(BAND_FIELDS, row[2:8])),
                "profile": json.loads(row[8]) if row[8] else {}}

    def set_profile(self, band_id: int, profile: dict) -> dict:
        """Replace this band's direction (clean_profile); other bands are untouched. Returns the band."""
        clean = clean_profile(profile)
        with self._conn() as c:
            cur = c.execute("UPDATE bands SET profile = ? WHERE id = ?", (json.dumps(clean, ensure_ascii=False), band_id))
            if cur.rowcount != 1:
                raise UnknownBand(band_id)
        return self.get_band(band_id)

    def list_bands(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, created, name, members, guitarists, vocal_low, vocal_high, source_note, profile FROM bands"
                             " ORDER BY id").fetchall()
            counts = dict(c.execute("SELECT band_id, COUNT(*) FROM versions GROUP BY band_id").fetchall())
        return [{**self._band_row(r), "versions": counts.get(r[0], 0)} for r in rows]

    def get_band(self, band_id: int) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT id, created, name, members, guitarists, vocal_low, vocal_high, source_note, profile FROM bands"
                            " WHERE id = ?", (band_id,)).fetchone()
        if row is None:
            raise UnknownBand(band_id)
        return self._band_row(row)

    # ------------------------------------------------------------ versions
    def save_version(self, band_id: int, result: dict, author: str | None = None, note: str | None = None,
                     parent_id: int | None = None, credits: dict | None = None) -> dict:
        self.get_band(band_id)
        missing = [k for k in ("sections", "bar_chords", "midi_b64") if not result.get(k)]
        if missing:
            raise BandError(f"저장할 곡에 {', '.join(missing)} 값이 없어요")
        if parent_id is not None and self.get_version(parent_id)["band_id"] != band_id:
            raise BandError("부모 버전이 다른 밴드의 것이에요. 밴드마다 버전은 따로 관리해요")
        text = json.dumps(result, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if credits is None and parent_id is not None:        # credits stay with the song unless they are edited
            credits = self.get_version(parent_id)["credits"]
        clean = {str(k)[:40]: str(v)[:300] for k, v in (credits or {}).items() if str(v).strip()}
        with self._conn() as c:
            cur = c.execute("INSERT INTO versions (band_id, parent_id, created, author, note, credits, sha256, result)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (band_id, parent_id, time.time(), (author or "").strip()[:80] or None,
                             (note or "").strip()[:1000] or None, json.dumps(clean, ensure_ascii=False), digest, text))
            return {"id": cur.lastrowid, "sha256": digest}

    @staticmethod
    def _version_row(row, with_result: bool) -> dict:
        out = {"id": row[0], "band_id": row[1], "parent_id": row[2], "created": row[3],
               "created_text": time.strftime("%Y-%m-%d %H:%M", time.localtime(row[3])), "author": row[4], "note": row[5],
               "credits": json.loads(row[6]), "sha256": row[7]}
        result = json.loads(row[8])
        out["summary"] = {"bars": result.get("bars"), "tempo": result.get("tempo"), "key": (result.get("key") or {}).get("name"),
                          "genre": result.get("genre"), "melody_source": result.get("melody_source"),
                          "member_parts": [m["part"] for m in result.get("member_parts") or []]}
        if with_result:
            out["result"] = result
        return out

    COLS = "id, band_id, parent_id, created, author, note, credits, sha256, result"

    def list_versions(self, band_id: int) -> list[dict]:
        self.get_band(band_id)
        with self._conn() as c:
            rows = c.execute(f"SELECT {self.COLS} FROM versions WHERE band_id = ? ORDER BY id", (band_id,)).fetchall()
        return [self._version_row(r, with_result=False) for r in rows]

    def get_version(self, version_id: int) -> dict:
        with self._conn() as c:
            row = c.execute(f"SELECT {self.COLS} FROM versions WHERE id = ?", (version_id,)).fetchone()
        if row is None:
            raise UnknownVersion(version_id)
        return self._version_row(row, with_result=True)

    # ------------------------------------------------------------ compare
    def compare(self, a_id: int, b_id: int) -> dict:
        a, b = self.get_version(a_id), self.get_version(b_id)
        if a["band_id"] != b["band_id"]:
            raise BandError("서로 다른 밴드의 버전은 비교하지 않아요")
        return diff(a["result"], b["result"])


def _bar_notes(result: dict) -> tuple[dict, dict, dict]:
    """Per track and bar: the notes (start in bar, pitch, length), their velocities, and the performance messages."""
    mid = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
    num, den = next(((m.numerator, m.denominator) for m in mid.tracks[0] if m.type == "time_signature"), (4, 4))
    bar_len = mid.ticks_per_beat * 4 * num // den
    notes: dict[str, dict[int, set]] = {}
    dyn: dict[str, dict[int, set]] = {}
    expr: dict[str, dict[int, set]] = {}
    for tr in mid.tracks:
        name = next((m.name for m in tr if m.type == "track_name"), None)
        if name is None:
            continue
        now, open_ = 0, {}
        for m in tr:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                open_.setdefault(m.note, []).append((now, m.velocity))
            elif m.type in ("note_off", "note_on") and open_.get(m.note):
                start, vel = open_[m.note].pop(0)
                notes.setdefault(name, {}).setdefault(start // bar_len, set()).add((start % bar_len, m.note, now - start))
                dyn.setdefault(name, {}).setdefault(start // bar_len, set()).add((start % bar_len, m.note, vel))
            elif m.type in ("control_change", "pitchwheel", "aftertouch", "polytouch"):
                value = tuple(sorted((k, v) for k, v in m.dict().items() if k not in ("time", "channel")))
                expr.setdefault(name, {}).setdefault(now // bar_len, set()).add((now % bar_len, value))
    freeze = lambda d: {t: {b: frozenset(v) for b, v in bars.items()} for t, bars in d.items()}
    return freeze(notes), freeze(dyn), freeze(expr)


def _spans(bars: list[int]) -> str:
    out, start, prev = [], bars[0], bars[0]
    for b in bars[1:] + [None]:
        if b != prev + 1:
            out.append(str(start) if start == prev else f"{start}~{prev}")
            start = b
        prev = b
    return ", ".join(out)


def diff(a: dict, b: dict) -> dict:
    """What differs from version a to version b (bar numbers count from 1 across the song).  Per part: `bars` where
    a note's pitch, start or length changed, `dynamics_bars` where only how hard notes are played changed, and
    `expression_bars` where pedal / bend / controller messages changed.  `same` means none of these differ."""
    facts = []
    for key, label in (("tempo", "템포"), ("bars", "마디 수"), ("genre", "장르"), ("meter", "박자")):
        if a.get(key) != b.get(key):
            facts.append(f"{label}: {a.get(key)} → {b.get(key)}")
    ka, kb = (a.get("key") or {}).get("name"), (b.get("key") or {}).get("name")
    if ka != kb:
        facts.append(f"조: {ka} → {kb}")
    fa, fb = [s["label"] for s in a["sections"]], [s["label"] for s in b["sections"]]
    if fa != fb:
        facts.append(f"구성: {' → '.join(fa)}  ⇒  {' → '.join(fb)}")
    ca, cb = a["bar_chords"], b["bar_chords"]
    chord_bars = [i + 1 for i in range(max(len(ca), len(cb)))
                  if (ca[i] if i < len(ca) else None) != (cb[i] if i < len(cb) else None)]
    na, da, ea = _bar_notes(a)
    nb, db, eb = _bar_notes(b)
    parts = {}
    changed_in = lambda x, y, part: sorted(i + 1 for i in set(x.get(part, {})) | set(y.get(part, {}))
                                           if x.get(part, {}).get(i) != y.get(part, {}).get(i))
    for part in sorted(set(na) | set(nb)):
        if part not in na or part not in nb:
            parts[part] = {"change": "added" if part in nb else "removed", "bars": []}
            continue
        entry = {}
        for key, bars in (("bars", changed_in(na, nb, part)),                # pitch, start or length
                          ("dynamics_bars", [b for b in changed_in(da, db, part) if b not in changed_in(na, nb, part)]),
                          ("expression_bars", changed_in(ea, eb, part))):   # pedal, bend...
            if bars:
                entry[key] = bars
                entry[key + "_text"] = _spans(bars)
        if entry:
            parts[part] = {"change": "edited" if "bars" in entry else "performance", "bars": [], **entry}
    members_a = {m["part"]: m.get("sha256") for m in a.get("member_parts") or []}
    members_b = {m["part"]: m.get("sha256") for m in b.get("member_parts") or []}
    return {"facts": facts, "chord_bars": chord_bars, "chord_bars_text": _spans(chord_bars) if chord_bars else "",
            "parts": parts, "member_parts": {"before": sorted(members_a), "after": sorted(members_b)},
            "same": not facts and not chord_bars and not parts and members_a == members_b}
