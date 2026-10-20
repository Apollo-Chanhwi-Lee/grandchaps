"""Similar-song check ("유사곡 검사"): which PUBLISHED melodies does a composed song share long runs with?

This reports facts (matched note counts, where, and in which source). It is NOT a legal judgment.

Reference index
---------------
Two exact packed-key indexes over every reference melody (8-note windows, transposition invariant, see
`similarity.window_keys` / `similarity.rhythm_keys`):
  * models/reference_index_pitch.npz   pitch intervals only (info only)
  * models/reference_index_rhythm.npz  pitch intervals + relative rhythm (drives the risk level)
and a sidecar models/reference_index_meta.json (per source id: dataset, title, artist, year, split).
Arrays are stored uncompressed in the .npz and memory-mapped on load (`_mmap_npz`), sorted by key, so a query is a
vectorised np.searchsorted per window plus a diagonal chaining of the hits: the cost does not depend on the number of
reference songs except through log(N) and the number of hits.
Source id = "<dataset>:<song>#<unit>"; the part before "#" is the song (several windows or clips of one song share it).

Risk levels (ALARMS, not legal judgments)
-----------------------------------------
Level = f(longest NON-TRIVIAL pitch+rhythm run against one reference song). A run that consists only of repeated
pitches or a strict scale run is trivial and never counts.
  RISK_ATTENTION = 10 notes -> "주의"
  RISK_HIGH      = 16 notes -> "높음"
Pitch-only runs are listed as information and never change the level.
The thresholds were fixed against the real-song baseline in runs/similarity_index.json (valid songs vs train songs):
pitch-only 8-note matches flag 100% of real songs (so they carry no signal), pitch+rhythm runs >= 10 flag about 7%,
and runs >= 16 flag 0% of the real songs measured there. So "주의" means "unusual even among real songs" and "높음"
means "never seen between two different real songs in that baseline". Short, common figures (a few bars of a
cadence, a scale run, a repeated note) are normally not protected by copyright; a long run is a reason to look,
not proof of copying, and its absence is not proof of originality: only the published melodies in the index
are compared.
"""
from __future__ import annotations

import json
import re
import struct
import threading
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .leadsheet import LeadSheet, parse_leadsheet
from .similarity import DEFAULT_N, notes_from_leadsheet, rhythm_keys, window_keys

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PITCH_FILE, RHYTHM_FILE, META_FILE = ("reference_index_pitch.npz", "reference_index_rhythm.npz",
                                      "reference_index_meta.json")

# ---- pre-declared alarm thresholds (see module docstring) ----
RISK_ATTENTION = 10
RISK_HIGH = 16
LEVEL_NONE, LEVEL_ATTENTION, LEVEL_HIGH = "없음", "주의", "높음"
TOP_K = 3
HIT_CAP = 20000          # per query window; windows with more hits are counted in `capped_windows`
FORMAT_VERSION = 1
POS_MAX = 65535          # positions are stored as uint16: longer sources are cut (counted by the builder)

DATASET_LABEL = {"pop909": "POP909", "hooktheory": "Hooktheory", "nottingham": "Nottingham", "bach": "Bach 코랄",
                 "pdmx": "PDMX", "lakh": "Lakh MIDI"}
NOTE_TEMPLATE = ("공개 데이터셋 약 {n}곡과 비교한 결과입니다. 한국 가요 대부분은 비교 대상에 없으니 Shazam이나 Google "
                 "'허밍으로 검색'으로도 확인해 보세요. 이 결과는 표절 여부의 법적 판단이 아닙니다.")


def level_of(run_len: int) -> str:
    """Level for the longest non-trivial pitch+rhythm run (0 = none)."""
    if run_len >= RISK_HIGH:
        return LEVEL_HIGH
    if run_len >= RISK_ATTENTION:
        return LEVEL_ATTENTION
    return LEVEL_NONE


def approx_count(n: int) -> str:
    """'13,500' style rounding for the fixed note (nearest 100 from 1000, nearest 10 from 100)."""
    step = 100 if n >= 1000 else 10 if n >= 100 else 1
    return f"{int(round(n / step) * step):,}"


def song_of(source_id: str) -> str:
    return source_id.split("#", 1)[0]


# ---------------------------------------------------------------- npz memory mapping
def _mmap_npz(path) -> dict[str, np.ndarray]:
    """Arrays of an UNCOMPRESSED .npz (np.savez): big ones memory-mapped, small ones loaded."""
    path = Path(path)
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as zf:
        infos = list(zf.infolist())
    with open(path, "rb") as f:
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{path.name}: member {info.filename} is compressed, cannot memory-map")
            f.seek(info.header_offset)
            hdr = f.read(30)
            n, m = struct.unpack("<HH", hdr[26:30])
            f.seek(info.header_offset + 30 + n + m)
            ver = np.lib.format.read_magic(f)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                     else np.lib.format.read_array_header_2_0(f))
            off = f.tell()
            if dtype.hasobject or fortran:
                raise ValueError(f"{path.name}: unsupported array {info.filename}")
            count = int(np.prod(shape)) if shape else 1
            name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            if count == 0:
                out[name] = np.zeros(shape, dtype=dtype)
            elif not shape or count * dtype.itemsize < (1 << 20):
                f.seek(off - 0)
                out[name] = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype).reshape(shape).copy()
            else:
                out[name] = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape)
    return out


def save_flat(path, keys, rk, src, pos, source_ids: Sequence[str], mode: str, summary: dict) -> None:
    """Write one sorted index. `rk` is None for the pitch-only file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    arrays = dict(version=np.array(FORMAT_VERSION, dtype=np.int32), mode=np.array(mode, dtype=np.str_),
                  summary=np.array(json.dumps(summary, sort_keys=True, ensure_ascii=False), dtype=np.str_),
                  source_ids=np.array(list(source_ids), dtype=np.str_), keys=keys, src=src, pos=pos)
    if rk is not None:
        arrays["rk"] = rk
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def build_flat(per_source: Sequence[tuple[np.ndarray, np.ndarray | None]], with_rhythm: bool):
    """per_source[i] = (pitch keys, rhythm keys or None) of source i -> sorted (keys, rk, src, pos)."""
    ks, rs, ss, ps = [], [], [], []
    for i, (k, r) in enumerate(per_source):
        if k.size > POS_MAX + 1:
            k, r = k[:POS_MAX + 1], (r[:POS_MAX + 1] if r is not None else None)
        ks.append(k)
        ss.append(np.full(k.size, i, dtype=np.uint32))
        ps.append(np.arange(k.size, dtype=np.uint16))
        if with_rhythm:
            rs.append(r)
    if not ks:
        z = np.zeros(0, dtype=np.uint64)
        return z, (z if with_rhythm else None), np.zeros(0, np.uint32), np.zeros(0, np.uint16)
    keys, src, pos = np.concatenate(ks), np.concatenate(ss), np.concatenate(ps)
    if with_rhythm:
        rk = np.concatenate(rs)
        order = np.lexsort((pos, src, rk, keys))
        return keys[order], rk[order], src[order], pos[order]
    order = np.lexsort((pos, src, keys))
    return keys[order], None, src[order], pos[order]


class _Flat:
    def __init__(self, path):
        z = _mmap_npz(path)
        if int(z["version"]) != FORMAT_VERSION:
            raise ValueError(f"unsupported reference index version {int(z['version'])}")
        self.mode = str(z["mode"])
        self.summary = json.loads(str(z["summary"]))
        self.source_ids = z["source_ids"]
        self.keys, self.src, self.pos = z["keys"], z["src"], z["pos"]
        self.rk = z.get("rk")

    def source_id(self, i: int) -> str:
        return str(self.source_ids[int(i)])

    def lookup(self, qkeys: np.ndarray, qrk: np.ndarray | None):
        """-> (query index, source, position) of every hit (capped per window), number of capped windows."""
        lo = np.searchsorted(self.keys, qkeys, side="left")
        hi = np.searchsorted(self.keys, qkeys, side="right")
        qs, ss, ps, capped = [], [], [], 0
        for qi in np.nonzero(hi > lo)[0]:
            a, b = int(lo[qi]), int(hi[qi])
            if self.rk is not None:
                blk = np.asarray(self.rk[a:b])
                a, b = a + int(np.searchsorted(blk, qrk[qi], "left")), a + int(np.searchsorted(blk, qrk[qi], "right"))
                if b <= a:
                    continue
            if b - a > HIT_CAP:
                capped += 1
                b = a + HIT_CAP
            s, p = np.asarray(self.src[a:b], dtype=np.int64), np.asarray(self.pos[a:b], dtype=np.int64)
            qs.append(np.full(s.size, qi, dtype=np.int64))
            ss.append(s)
            ps.append(p)
        if not qs:
            e = np.zeros(0, dtype=np.int64)
            return e, e, e, capped
        return np.concatenate(qs), np.concatenate(ss), np.concatenate(ps), capped


def _chain(q, s, p, trivial_w, zero_w, n=DEFAULT_N) -> list[dict]:
    """Hits -> maximal runs per (source, diagonal). Longest first."""
    if q.size == 0:
        return []
    d = p - q
    order = np.lexsort((q, d, s))
    q, s, d = q[order], s[order], d[order]
    brk = (s[1:] != s[:-1]) | (d[1:] != d[:-1]) | (q[1:] != q[:-1] + 1)
    starts = np.concatenate(([0], np.nonzero(brk)[0] + 1))
    ends = np.concatenate((starts[1:], [q.size])) - 1
    cs_t = np.concatenate(([0], np.cumsum(~trivial_w)))
    cs_z = np.concatenate(([0], np.cumsum(~zero_w)))
    runs = []
    for a, b in zip(starts, ends):
        q0, q1 = int(q[a]), int(q[b])
        runs.append({"query_start": q0, "length": q1 - q0 + n, "source": int(s[a]), "source_start": q0 + int(d[a]),
                     "trivial": bool(cs_t[q1 + 1] - cs_t[q0] == 0), "zero_run": bool(cs_z[q1 + 1] - cs_z[q0] == 0)})
    runs.sort(key=lambda r: (-r["length"], r["query_start"], r["source"]))
    return runs


def _window_flags(pitches: np.ndarray, n=DEFAULT_N):
    m = pitches.size - n + 1
    if m <= 0:
        z = np.zeros(0, dtype=bool)
        return z, z
    iv = np.stack([np.diff(pitches)[k:k + m] for k in range(n - 1)], axis=1)
    zero = (iv == 0).all(axis=1)
    up = np.isin(iv, (1, 2)).all(axis=1)
    down = np.isin(iv, (-1, -2)).all(axis=1)
    return zero | up | down, zero


class ReferenceIndex:
    """Pitch-only and pitch+rhythm indexes plus the lazily loaded sidecar meta."""

    def __init__(self, pitch_path, rhythm_path, meta_path=None):
        self.pitch = _Flat(pitch_path)
        self.rhythm = _Flat(rhythm_path)
        self.meta_path = Path(meta_path) if meta_path else None
        self._meta: dict | None = None
        self._lock = threading.Lock()

    @classmethod
    def load(cls, directory=MODELS) -> "ReferenceIndex":
        d = Path(directory)
        return cls(d / PITCH_FILE, d / RHYTHM_FILE, d / META_FILE)

    @property
    def summary(self) -> dict:
        return self.rhythm.summary

    @property
    def n_songs(self) -> int:
        return int(self.summary.get("n_songs", 0))

    @property
    def n_sources(self) -> int:
        return int(self.summary.get("n_sources", 0))

    @property
    def note(self) -> str:
        return NOTE_TEMPLATE.format(n=approx_count(self.n_songs))

    def meta_of(self, source_id: str) -> dict:
        with self._lock:
            if self._meta is None:
                self._meta = {}
                if self.meta_path and self.meta_path.exists():
                    self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")).get("sources", {})
        m = self._meta.get(source_id)
        if m is None:
            ds = song_of(source_id).split(":", 1)[0]
            return {"song": song_of(source_id), "dataset": ds, "title": song_of(source_id), "artist": None,
                    "year": None, "split": None}
        return m

    def runs(self, pitches: Sequence[int], onsets: Sequence[int]) -> dict:
        """All runs >= 8 notes against the index, both modes. Source ids are resolved (strings)."""
        p = np.asarray(list(pitches), dtype=np.int64)
        if len(onsets) != p.size:
            raise ValueError("one onset per pitch required")
        out = {"n_notes": int(p.size), "capped_windows": 0}
        triv, zero = _window_flags(p)
        pk = window_keys(p.tolist())
        rk = rhythm_keys(list(onsets)) if p.size >= DEFAULT_N else np.zeros(0, dtype=np.uint64)
        for name, flat, qrk in (("rhythm", self.rhythm, rk), ("pitch", self.pitch, None)):
            q, s, pos, capped = flat.lookup(pk, qrk)
            out["capped_windows"] += capped
            runs = _chain(q, s, pos, triv, zero)
            for r in runs:
                r["source_id"] = flat.source_id(r["source"])
            out[name] = runs
        return out

    def top(self, runs: list[dict], k: int = TOP_K) -> tuple[list[dict], int]:
        """Best non-trivial run per reference SONG, longest first -> (top k, number of trivial runs ignored)."""
        best: dict[str, dict] = {}
        ignored = 0
        for r in runs:                                  # already sorted longest first
            if r["trivial"]:
                ignored += 1
                continue
            best.setdefault(song_of(r["source_id"]), r)
        return list(best.values())[:k], ignored


# ---------------------------------------------------------------- default index
_CACHE: dict = {}


def default_reference_index() -> ReferenceIndex | None:
    """The index in models/ (cached), or None when it has not been built."""
    if "idx" not in _CACHE:
        try:
            _CACHE["idx"] = ReferenceIndex.load()
        except Exception:
            _CACHE["idx"] = None
    return _CACHE["idx"]


# ---------------------------------------------------------------- reports
def _bar_ticks(meter: str) -> int:
    num, den = (int(x) for x in meter.split("/"))
    return num * (1920 // den)


def _match(idx: ReferenceIndex, r: dict, mode: str, name: str, first_bar: int, onsets: list[int], bt: int) -> dict:
    m = idx.meta_of(r["source_id"])
    q0, q1 = r["query_start"], r["query_start"] + r["length"] - 1
    return {"mode": mode, "notes": int(r["length"]), "section": name,
            "our_bars": [first_bar + onsets[q0] // bt, first_bar + onsets[q1] // bt],
            "source_id": r["source_id"], "song": m.get("song") or song_of(r["source_id"]),
            "dataset": m.get("dataset"), "dataset_label": DATASET_LABEL.get(m.get("dataset"), m.get("dataset")),
            "title": m.get("title"), "artist": m.get("artist"), "year": m.get("year"),
            "source_start_note": int(r["source_start"])}


def check_notes(idx: ReferenceIndex, pitches: Sequence[int], onsets: Sequence[int]) -> dict:
    """Longest non-trivial rhythm run and its song, for a hard-fail test: {"level","notes","title","capped_windows"}."""
    res = idx.runs(pitches, onsets)
    top, _ = idx.top(res["rhythm"], 1)
    n = top[0]["length"] if top else 0
    return {"level": level_of(n), "notes": n,
            "title": idx.meta_of(top[0]["source_id"]).get("title") if top else None,
            "capped_windows": res["capped_windows"]}


def section_level(idx: ReferenceIndex, section_sheet: LeadSheet | str) -> dict:
    pitches, onsets = notes_from_leadsheet(section_sheet)
    return check_notes(idx, pitches, onsets)


def _summary_part(label: str, rep: dict) -> str:
    top = rep["top_rhythm"][0] if rep["top_rhythm"] else None
    if rep["level"] == LEVEL_NONE or top is None:
        return f"{label}: 비슷한 곡 없음"
    return f"{label}: '{top['title']}' 과 {top['notes']}음 연속 일치 ({rep['level']})"


def _group_report(idx: ReferenceIndex, parts: list[dict]) -> dict:
    """parts: [{"name","first_bar","pitches","onsets","sources","repeats"}] -> one report for the group."""
    rhythm, pitch, ign, capped = [], [], 0, 0
    for part in parts:
        if len(part["pitches"]) < DEFAULT_N:
            continue
        bt = part["bar_ticks"]
        res = idx.runs(part["pitches"], part["onsets"])
        capped += res["capped_windows"]
        for mode, store in (("rhythm", rhythm), ("pitch", pitch)):
            top, i = idx.top(res[mode], 20)
            ign += i if mode == "rhythm" else 0
            store.extend(_match(idx, r, mode, part["name"], part["first_bar"], part["onsets"], bt) for r in top)

    def best(ms):
        seen, out = set(), []
        for m in sorted(ms, key=lambda m: (-m["notes"], m["our_bars"][0], m["source_id"])):
            if m["song"] in seen:
                continue
            seen.add(m["song"])
            out.append(m)
        return out[:TOP_K]

    tr, tp = best(rhythm), best(pitch)
    for i, m in enumerate(tr):
        m["rank"] = i + 1
    for i, m in enumerate(tp):
        m["rank"] = i + 1
    level = level_of(tr[0]["notes"]) if tr else LEVEL_NONE
    return {"level": level, "max_rhythm_run": tr[0]["notes"] if tr else 0,
            "max_pitch_run": tp[0]["notes"] if tp else 0, "top_rhythm": tr, "top_pitch": tp,
            "parts": [{"name": p["name"], "first_bar": p["first_bar"], "bars": p["bars"],
                       "source": p["source"], "repeats": p["repeats"]} for p in parts],
            "trivial_runs_ignored": ign, "capped_windows": capped}


def unavailable_report() -> dict:
    return {"available": False, "version": 1,
            "summary": "유사곡 검사 기준 데이터가 없어 검사하지 못했어요.",
            "note": "유사곡 검사용 참고 곡 색인이 없어 비교하지 않았습니다. 이 결과는 표절 여부의 법적 판단이 아닙니다."}


def check_song(result: dict, idx: ReferenceIndex | None = None, section_sources: list[str] | None = None) -> dict:
    """Separate reports for the seed (후렴) and for the AI/rule-written parts of a composed song.

    `result` needs "leadsheet_c" (the lead sheet in model key; transposition does not matter for the keys) and may carry
    "section_sources" (seed / model / rule per section). Each report holds the top 3 reference songs by longest
    non-trivial run, pitch+rhythm (`top_rhythm`, drives `level`) and pitch-only (`top_pitch`, information only).
    """
    idx = idx if idx is not None else default_reference_index()
    if idx is None:
        return unavailable_report()
    sheet = parse_leadsheet(result["leadsheet_c"])
    sources = section_sources or result.get("section_sources") or []
    bt = _bar_ticks(sheet.meter)
    first_of: dict[str, int] = {}
    groups: dict[str, dict] = {"seed": {}, "ai": {}}
    bar = 0
    order: dict[str, list] = {"seed": [], "ai": []}
    for i, sec in enumerate(sheet.sections):
        grp = "seed" if sec.name == "후렴" else "ai"
        key = f"{grp}:{sec.name}"
        if key in first_of:
            groups[grp][key]["repeats"] += 1
        else:
            first_of[key] = bar
            one = LeadSheet(sheet.genre, sheet.key, sheet.meter, sheet.tempo, [sec])
            try:
                pitches, onsets = notes_from_leadsheet(one)
            except ValueError:
                pitches, onsets = [], []
            src = "seed" if grp == "seed" else (sources[i] if i < len(sources) else "rule")
            groups[grp][key] = {"name": sec.name, "first_bar": bar + 1, "bars": len(sec.bars), "pitches": pitches,
                                "onsets": onsets, "bar_ticks": bt, "source": src, "repeats": 1}
            order[grp].append(key)
        bar += len(sec.bars)
    seed = _group_report(idx, [groups["seed"][k] for k in order["seed"]])
    ai = _group_report(idx, [groups["ai"][k] for k in order["ai"]])
    summary = _summary_part("AI가 쓴 파트", ai) + " / " + _summary_part("씨앗 멜로디", seed)
    return {"available": True, "version": 1, "summary": summary, "ai": ai, "seed": seed,
            "n_songs": idx.n_songs, "n_sources": idx.n_sources, "datasets": idx.summary.get("datasets", {}),
            "thresholds": {"attention": RISK_ATTENTION, "high": RISK_HIGH, "window": DEFAULT_N},
            "note": idx.note}
