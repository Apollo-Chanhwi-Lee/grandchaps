"""Output-similarity check (dev_instructions section 5): does a generated melody copy a known one?

Spec: a melody part is regenerated when it shares 8 or more consecutive notes with the
reference song or the training data, ignoring transposition.

Design
------
* Melody = sequence of non-rest MIDI pitches. Tied notes are already merged into one note by
  `leadsheet.to_notes`; rests are dropped, so a rest between two notes does NOT break a run
  (rhythm and rests are deliberately ignored: the check is about the pitch contour).
* Key of a window of n notes = the tuple of its n-1 pitch intervals (8 notes -> 7 intervals),
  which is transposition invariant. Repeated identical pitches DO count (interval 0); they
  are not collapsed, so "C C C C C C C C" is a genuine 8-note window with 7 zero intervals.
* Each interval is stored as (interval + 127) in 0..254 and the window is packed into one
  uint64 in base 255. This is an exact encoding (no hash collisions) for n-1 <= 8, i.e.
  n <= 9. Pitches must therefore lie in 0..127.
* Longer runs: a run of L >= n notes appears as L-n+1 overlapping windows. Windows belong to
  the same run only if they come from the same source on the same diagonal
  (source position - query position is constant). `max_run` chains windows per
  (source, diagonal), so a 12-note copy gives 12, not 8, and two unrelated 8-note matches to
  different songs never add up.
* `max_run` returns 0 when there is no window match: the index only knows n-note windows, so
  a shorter partial match (say 7 notes) is not measured.

Trivial patterns
----------------
A window is flagged `trivial=True` if all its intervals are 0 (repeated single pitch) or if it
is a strict scale run (every interval in {1, 2} or every interval in {-1, -2}). Nothing is
silently dropped: matches are always reported with the flag and the caller decides.

`should_regenerate` exempts only the all-zero case. Why: a same-pitch run carries no melodic
information (no contour), appears in nearly every song, and regenerating it would not remove
any copying. A stepwise scale run is also common, but a specific 8-note mixture of whole and
half steps is a real diatonic fragment, and the exemption of it would hide true copies of
scale-like hooks; so it is flagged, reported, and counted, but not exempt by default. Whether
to exempt it later should be decided from the cross-song overlap baseline
(runs/similarity_index.json).

Rhythm-aware mode
-----------------
An index is built in mode "pitch" (default, the spec's criterion) or "rhythm" (pitch + rhythm).
The mode is stored in the file. In rhythm mode a window key is the 7 pitch intervals AND the
7 inter-onset intervals (IOI) of the 8 notes. IOI = onset(next) - onset(this), so a rest before
a note lengthens the previous IOI (rests are part of the rhythm; note durations themselves are
not used). Each IOI is quantized to 120-tick units (a 16th note, minimum 1), then the 7 values
of the window are divided by their gcd, so a uniform tempo/augmentation scaling (e.g. all values
doubled) gives the same key; values are capped at 63 after normalization (very long rests are
treated as equal). Both parts must match exactly. Rhythm mode needs onsets for every melody.
A run in rhythm mode is a chain of consecutive matching windows; each window is normalized on
its own, so a run is "same contour and same relative rhythm within every 8-note window".

Regenerate policy
-----------------
`should_regenerate(result)` flags a result if some run of length >= policy["threshold"] exists
that is not an all-zero-interval run. The policy ({"mode", "threshold", ...}) is stored in the
index file (chosen by scripts/build_similarity_index.py under a rule fixed in advance) and is
carried in `check()` results; `spec_metric` reports the spec's raw 8-note pitch-only number
separately, as an evaluation metric.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .leadsheet import LeadSheet, parse_leadsheet, to_notes

DEFAULT_N = 8
MAX_N = 9                      # exact base-255 packing needs n-1 <= 8 intervals in uint64
MODES = ("pitch", "rhythm")
_BASE = 255
_OFFSET = 127
_RBASE = 64                    # rhythm digits 1..63
_RCAP = 63
IOI_UNIT = 120                 # ticks (a 16th note at 480 tpq)
_FORMAT_VERSION = 2
MAX_LISTED_SOURCES = 20        # per matched window; `n_sources` always has the full count
DEFAULT_POLICY = {"mode": "pitch", "threshold": DEFAULT_N}


# ---------- melody / keys ----------
def _as_pitches(pitches: Sequence[int]) -> np.ndarray:
    arr = np.asarray(list(pitches), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("pitches must be a flat sequence of ints")
    if arr.size and (arr.min() < 0 or arr.max() > 127):
        raise ValueError("pitches must be MIDI numbers in 0..127 (drop rests before calling)")
    return arr


def _check_n(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or not 2 <= n <= MAX_N:
        raise ValueError(f"n must be an integer in 2..{MAX_N}, got {n!r}")
    return n


def window_keys(pitches: Sequence[int], n: int = DEFAULT_N) -> np.ndarray:
    """Transposition-invariant key of every n-note window (uint64 array, len = max(0, L-n+1))."""
    _check_n(n)
    p = _as_pitches(pitches)
    m = p.size - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    iv = (np.diff(p) + _OFFSET).astype(np.uint64)          # 0..254
    keys = np.zeros(m, dtype=np.uint64)
    for k in range(n - 1):
        keys = keys * np.uint64(_BASE) + iv[k:k + m]
    return keys


def rhythm_keys(onsets: Sequence[int], n: int = DEFAULT_N) -> np.ndarray:
    """Tempo-scale-invariant key of the n-1 inter-onset intervals of every n-note window."""
    _check_n(n)
    o = np.asarray(list(onsets), dtype=np.int64)
    m = o.size - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    ioi = np.diff(o)
    if ioi.min() <= 0:
        raise ValueError("onsets must be strictly increasing (monophonic melody)")
    u = np.maximum(1, np.rint(ioi / IOI_UNIT)).astype(np.int64)
    w = np.stack([u[k:k + m] for k in range(n - 1)], axis=1)
    w = np.minimum(w // np.gcd.reduce(w, axis=1)[:, None], _RCAP).astype(np.uint64)
    keys = np.zeros(m, dtype=np.uint64)
    for k in range(n - 1):
        keys = keys * np.uint64(_RBASE) + w[:, k]
    return keys


def _intervals(pitches: Sequence[int], start: int, n: int) -> tuple[int, ...]:
    p = list(pitches)
    return tuple(p[start + k + 1] - p[start + k] for k in range(n - 1))


def is_trivial_intervals(iv: Sequence[int]) -> bool:
    """All-zero (repeated pitch) or a strict scale run (all steps in {1,2} or all in {-1,-2})."""
    return (all(x == 0 for x in iv) or all(x in (1, 2) for x in iv)
            or all(x in (-1, -2) for x in iv))


def is_zero_run(iv: Sequence[int]) -> bool:
    return all(x == 0 for x in iv)


# ---------- lead sheet helpers ----------
def notes_from_leadsheet(text_or_sheet: str | LeadSheet,
                         section_name: str | int | None = None) -> tuple[list[int], list[int]]:
    """(pitches, onset ticks) of the non-rest notes (ties merged by `to_notes`).

    section_name: None = whole song; a str selects every section with that name (in order);
    an int selects the section at that index. Raises ValueError if nothing matches.
    Onsets are absolute ticks, so a rest shows up as a longer gap to the next onset.
    """
    sheet = parse_leadsheet(text_or_sheet) if isinstance(text_or_sheet, str) else text_or_sheet
    if section_name is not None:
        if isinstance(section_name, int) and not isinstance(section_name, bool):
            secs = [sheet.sections[section_name]] if -len(sheet.sections) <= section_name < len(sheet.sections) else []
        else:
            secs = [s for s in sheet.sections if s.name == section_name]
        if not secs:
            raise ValueError(f"section not found: {section_name!r}")
        sheet = replace(sheet, sections=secs)
    notes, _ = to_notes(sheet)
    kept = [n for n in notes if n.pitch is not None]
    return [n.pitch for n in kept], [n.start for n in kept]


def melody_from_leadsheet(text_or_sheet: str | LeadSheet,
                          section_name: str | int | None = None) -> list[int]:
    """Non-rest pitches of a lead sheet (see `notes_from_leadsheet`)."""
    return notes_from_leadsheet(text_or_sheet, section_name)[0]


# ---------- index ----------
class SimilarityIndex:
    """Sorted array of (window key, source, position) built from many melodies."""

    def __init__(self, n: int = DEFAULT_N, mode: str = "pitch"):
        self.n = _check_n(n)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.policy: dict | None = None               # set by the build script, stored in the file
        self.source_ids: list[str] = []
        self._keys = np.zeros(0, dtype=np.uint64)     # pitch key, sorted by (key, rhythm, source, pos)
        self._rk = np.zeros(0, dtype=np.uint64)       # rhythm key (all 0 in pitch mode)
        self._src = np.zeros(0, dtype=np.int32)
        self._pos = np.zeros(0, dtype=np.int32)

    # -- build --
    @classmethod
    def build(cls, sources: Iterable[tuple], n: int = DEFAULT_N, mode: str = "pitch") -> "SimilarityIndex":
        """sources: (id, pitches) or (id, pitches, onsets); onsets are required in rhythm mode."""
        idx = cls(n, mode)
        per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for item in sources:
            sid, pitches = str(item[0]), item[1]
            onsets = item[2] if len(item) > 2 else None
            if sid in per:
                raise ValueError(f"duplicate source id: {sid!r}")
            pk = window_keys(pitches, idx.n)
            if mode == "rhythm":
                if onsets is None or len(onsets) != len(pitches):
                    raise ValueError(f"rhythm mode needs one onset per pitch (source {sid!r})")
                rk = rhythm_keys(onsets, idx.n)
            else:
                rk = np.zeros(pk.size, dtype=np.uint64)
            per[sid] = (pk, rk)
        idx.source_ids = sorted(per)                  # insertion order does not matter
        ks, rs, ss, ps = [], [], [], []
        for i, sid in enumerate(idx.source_ids):
            k, r = per[sid]
            ks.append(k)
            rs.append(r)
            ss.append(np.full(k.size, i, dtype=np.int32))
            ps.append(np.arange(k.size, dtype=np.int32))
        if ks:
            keys, rk, src, pos = np.concatenate(ks), np.concatenate(rs), np.concatenate(ss), np.concatenate(ps)
            order = np.lexsort((pos, src, rk, keys))
            idx._keys, idx._rk, idx._src, idx._pos = keys[order], rk[order], src[order], pos[order]
        return idx

    @property
    def n_sources(self) -> int:
        return len(self.source_ids)

    @property
    def n_windows(self) -> int:
        return int(self._keys.size)

    # -- query --
    def _query_keys(self, pitches, onsets, n):
        pk = window_keys(pitches, n)
        if self.mode == "rhythm":
            if onsets is None or len(onsets) != len(pitches):
                raise ValueError("rhythm mode needs one onset per pitch")
            rk = rhythm_keys(onsets, n)
        else:
            rk = np.zeros(pk.size, dtype=np.uint64)
        return pk, rk

    def _hits(self, key, rkey, exclude: set[int]):
        lo = int(np.searchsorted(self._keys, key, side="left"))
        hi = int(np.searchsorted(self._keys, key, side="right"))
        rmode = self.mode == "rhythm"
        return [(int(self._src[j]), int(self._pos[j])) for j in range(lo, hi)
                if int(self._src[j]) not in exclude and (not rmode or self._rk[j] == rkey)]

    def _exclude_idx(self, exclude_ids: Iterable[str]) -> set[int]:
        pos = {s: i for i, s in enumerate(self.source_ids)}
        return {pos[s] for s in (str(e) for e in exclude_ids) if s in pos}

    def matches(self, pitches: Sequence[int], n: int | None = None,
                exclude_ids: Iterable[str] = (), onsets: Sequence[int] | None = None) -> list[dict]:
        """Every query window (n notes) that occurs in some source, ignoring transposition.

        One dict per matched window, in query order:
          query_start / query_end  note positions in the query (end exclusive)
          sources                  [{"id", "pos"}] sorted, at most MAX_LISTED_SOURCES
          n_sources                number of (source, position) occurrences in total
          trivial                  all-zero intervals or strict scale run (see module doc)
          zero_run                 all-zero intervals only (the default exempt case)
          intervals                the n-1 pitch intervals
        n defaults to the index window size; a different n is an error. In rhythm mode `onsets`
        (tick of each note) is required and the rhythm must match too.
        """
        n = self.n if n is None else n
        if n != self.n:
            raise ValueError(f"index was built with n={self.n}, cannot query n={n}")
        excl = self._exclude_idx(exclude_ids)
        pks, rks = self._query_keys(pitches, onsets, n)
        out = []
        for qi in range(pks.size):
            hits = self._hits(pks[qi], rks[qi], excl)
            if not hits:
                continue
            hits.sort()
            iv = _intervals(pitches, qi, n)
            out.append({
                "query_start": qi, "query_end": qi + n,
                "sources": [{"id": self.source_ids[s], "pos": p} for s, p in hits[:MAX_LISTED_SOURCES]],
                "n_sources": len(hits),
                "trivial": is_trivial_intervals(iv), "zero_run": is_zero_run(iv),
                "intervals": list(iv),
            })
        return out

    def runs(self, pitches: Sequence[int], exclude_ids: Iterable[str] = (),
             onsets: Sequence[int] | None = None) -> list[dict]:
        """Maximal consecutive-note runs, chained per (source, diagonal). Sorted longest first.

        Each: {"query_start","length","source_id","source_start","trivial","zero_run"}; a run is
        trivial / zero_run only if every one of its windows is.
        """
        n = self.n
        excl = self._exclude_idx(exclude_ids)
        pks, rks = self._query_keys(pitches, onsets, n)
        chains: dict[tuple[int, int], list[int]] = {}      # (source, diagonal) -> query starts
        for qi in range(pks.size):
            for s, p in self._hits(pks[qi], rks[qi], excl):
                chains.setdefault((s, p - qi), []).append(qi)
        p_list = list(pitches)
        runs = []
        for (s, diag), qs in chains.items():
            qs.sort()
            start = prev = qs[0]
            for q in qs[1:] + [None]:
                if q is not None and q == prev + 1:
                    prev = q
                    continue
                length = prev - start + n
                ivs = [_intervals(p_list, w, n) for w in range(start, prev + 1)]
                runs.append({"query_start": start, "length": length, "source_id": self.source_ids[s],
                             "source_start": start + diag,
                             "trivial": all(is_trivial_intervals(v) for v in ivs),
                             "zero_run": all(is_zero_run(v) for v in ivs)})
                if q is not None:
                    start = prev = q
        runs.sort(key=lambda r: (-r["length"], r["query_start"], r["source_id"]))
        return runs

    def max_run(self, pitches: Sequence[int], exclude_ids: Iterable[str] = (),
                onsets: Sequence[int] | None = None) -> int:
        """Longest matching consecutive-note run vs any single source (0 = no n-note match)."""
        r = self.runs(pitches, exclude_ids, onsets)
        return r[0]["length"] if r else 0

    def check(self, pitches: Sequence[int], exclude_ids: Iterable[str] = (),
              onsets: Sequence[int] | None = None) -> dict:
        """One result object for `should_regenerate`: windows, runs, mode and the index policy."""
        windows = self.matches(pitches, exclude_ids=exclude_ids, onsets=onsets)
        runs = self.runs(pitches, exclude_ids, onsets)
        return {"n": self.n, "mode": self.mode, "policy": self.policy,
                "max_run": runs[0]["length"] if runs else 0,
                "n_windows": len(windows), "matches": windows, "runs": runs}

    # -- persistence --
    def save(self, path: str | os.PathLike) -> None:
        """Deterministic .npz (uncompressed, fixed zip timestamps, no pickles)."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, version=np.array(_FORMAT_VERSION, dtype=np.int32),
                     n=np.array(self.n, dtype=np.int32),
                     mode=np.array(self.mode, dtype=np.str_),
                     policy=np.array(json.dumps(self.policy, sort_keys=True), dtype=np.str_),
                     source_ids=np.array(self.source_ids, dtype=np.str_),
                     keys=self._keys, rk=self._rk, src=self._src, pos=self._pos)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "SimilarityIndex":
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != _FORMAT_VERSION:
                raise ValueError(f"unsupported similarity index version {int(z['version'])}")
            idx = cls(int(z["n"]), str(z["mode"]))
            idx.policy = json.loads(str(z["policy"]))
            idx.source_ids = [str(s) for s in z["source_ids"]]
            idx._keys, idx._rk, idx._src, idx._pos = z["keys"], z["rk"], z["src"], z["pos"]
        return idx


# ---------- policy ----------
def should_regenerate(result: dict, policy: dict | None = None, exempt_zero_run: bool = True) -> bool:
    """True if some run of length >= policy["threshold"] is a copy worth regenerating.

    policy defaults to the one stored in the index (`result["policy"]`), else 8 notes. Runs made
    only of all-zero-interval windows (repeated single pitch) are exempt; scale runs are NOT.
    Pass exempt_zero_run=False to count everything.
    """
    pol = policy or result.get("policy") or DEFAULT_POLICY
    t = pol["threshold"]
    return any(r["length"] >= t and not (exempt_zero_run and r["zero_run"]) for r in result["runs"])


def spec_metric(pitches: Sequence[int], index: SimilarityIndex, exclude_ids: Iterable[str] = ()) -> dict:
    """The spec's raw criterion, independent of the chosen policy: pitch-only, 8 notes.

    Returns {"flagged": any 8-note match (trivial ones included), "flagged_nonzero": same
    without all-zero runs, "max_run": longest run}. `index` must be a pitch-mode n=8 index.
    """
    if index.mode != "pitch" or index.n != DEFAULT_N:
        raise ValueError("spec_metric needs a pitch-mode index with n=8")
    runs = index.runs(pitches, exclude_ids)
    return {"flagged": bool(runs), "flagged_nonzero": any(not r["zero_run"] for r in runs),
            "max_run": runs[0]["length"] if runs else 0}
