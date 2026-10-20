"""Lakh MIDI -> lead-sheet piece conversion (research/non-commercial use only; see docs/LICENSES.md).

Pipeline for one MIDI file (``convert_midi``): load -> meter/tempo checks -> melody-track detection -> 16th-grid
quantisation -> key estimation (Krumhansl-Kessler profiles) and transposition to C / Am -> chords per half bar from
the other (non-melody, non-drum) tracks by chroma template matching.  No chord comes from any trained model.

Every rejection raises ``Skip(reason)`` with a short counted reason code; nothing is guessed silently.
The output is a plain dict (JSON-serialisable) that scripts/build_lakh.py turns into build_dataset_v2.Piece objects.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

TPQ = 480                       # lead-sheet ticks per quarter note
SIXTEENTH = TPQ // 4
BAR16 = 16                      # 16ths per 4/4 bar
HALF_BAR16 = 8
ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# ---- thresholds (documented in docs/lakh_stage2.md) ----
MIN_MELODY_NOTES = 40
MAX_OVERLAP_RATIO = 0.10        # share of notes overlapping another note of the same track
OVERLAP_SLOP_BEATS = 1 / 8      # overlaps shorter than a 32nd are legato slop
RANGE_LO, RANGE_HI = 55, 84
MIN_IN_RANGE = 0.75
AMBIGUITY_MARGIN = 0.30         # top two melody scores closer than this -> skip
MAX_OFF_GRID = 0.15             # share of melody note starts further than GRID_TOL from the 16th grid
GRID_TOL = 0.30                 # in 16th units
MAIN_TEMPO_SHARE = 0.90
KEY_MIN_CORR = 0.60
KEY_MIN_MARGIN = 0.05
MIN_ACC_NOTES = 3               # accompaniment notes sounding in a half bar for chord evidence
CHORD_MIN_SCORE = 0.55
MIN_IN_KEY = 0.88               # share of melody notes inside the chosen scale (minor: natural + raised 7th)
STACCATO_SHARE = 0.40           # share of note pairs with a 1-2 sixteenth gap that marks a staccato transcription
MAJOR_PCS = (0, 2, 4, 5, 7, 9, 11)
MINOR_PCS = (0, 2, 3, 5, 7, 8, 10, 11)

NAME_HINT = re.compile(r"melod|vocal|voice|vox|lead|sing|solo")
NAME_BAD = re.compile(r"bass|drum|perc|pad|chord|rhythm|accomp|comp|string|organ")


class Skip(Exception):
    """A documented filter; the message is the counted reason code."""


# =====================================================================================================
# text normalisation / grouping keys
# =====================================================================================================
def norm_text(s: str | None) -> str:
    """Lower-case ASCII-ish key: drop (...) / [...] parts, 'feat.' tails, accents and punctuation."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", s)
    s = re.sub(r"\b(feat|featuring|ft)\b.*$", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def song_group(artist: str | None, title: str | None, fallback: str) -> str:
    """Group key of all versions of one song: normalised artist + title; ``fallback`` when either is missing."""
    a, t = norm_text(artist), norm_text(title)
    return f"song:{a}|{t}" if a and t else fallback


# =====================================================================================================
# tick helpers (vectorised equivalent of PrettyMIDI.time_to_tick)
# =====================================================================================================
def times_to_ticks(pm, times: np.ndarray) -> np.ndarray:
    """Exact tick of note times that originate from ticks (nearest-tick search on pretty_midi's tick->time table)."""
    table = np.asarray(pm._PrettyMIDI__tick_to_time, dtype=float)
    idx = np.searchsorted(table, times, side="left")
    idx = np.clip(idx, 1, len(table) - 1)
    left, right = table[idx - 1], table[idx]
    idx = np.where(np.abs(times - left) < np.abs(times - right), idx - 1, idx)
    return idx.astype(np.int64)


def inst_arrays(pm, inst) -> dict:
    """start/end tick (float beats * resolution preserved as ticks), pitch, velocity arrays sorted by start."""
    if not inst.notes:
        return {"s": np.zeros(0, np.int64), "e": np.zeros(0, np.int64), "p": np.zeros(0, np.int64),
                "v": np.zeros(0, np.int64)}
    st = np.array([n.start for n in inst.notes])
    en = np.array([n.end for n in inst.notes])
    p = np.array([n.pitch for n in inst.notes], np.int64)
    v = np.array([n.velocity for n in inst.notes], np.int64)
    s, e = times_to_ticks(pm, st), times_to_ticks(pm, en)
    o = np.lexsort((-p, s))
    return {"s": s[o], "e": e[o], "p": p[o], "v": v[o]}


# =====================================================================================================
# melody detection
# =====================================================================================================
def overlap_ratio(s: np.ndarray, e: np.ndarray, slop: float) -> float:
    """Share of notes (sorted by start) that start before an earlier note has ended by more than ``slop`` ticks."""
    if len(s) == 0:
        return 1.0
    run_end = -np.inf
    over = 0
    for si, ei in zip(s, e):
        if si < run_end - slop:
            over += 1
        run_end = max(run_end, ei)
    return over / len(s)


def melody_candidates(pm) -> list[dict]:
    """Score every non-drum instrument that looks like a melody line.  Higher score = more melody-like."""
    res = pm.resolution
    out = []
    for idx, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < MIN_MELODY_NOTES:
            continue
        a = inst_arrays(pm, inst)
        ov = overlap_ratio(a["s"], a["e"], OVERLAP_SLOP_BEATS * res)
        if ov >= MAX_OVERLAP_RATIO:
            continue
        p = a["p"]
        in_range = float(np.mean((p >= RANGE_LO) & (p <= RANGE_HI)))
        if in_range < MIN_IN_RANGE:
            continue
        name = (inst.name or "").lower()
        score = (1.0 - ov) + in_range
        score += 2.0 if NAME_HINT.search(name) else 0.0
        score -= 2.0 if NAME_BAD.search(name) else 0.0
        score += 0.5 * min(1.0, max(0.0, (float(p.mean()) - 55) / 30))
        score += 0.5 * min(1.0, len(p) / 300)
        prog = inst.program
        if 32 <= prog <= 39:            # GM bass
            score -= 1.0
        elif prog in (52, 53, 54, 56, 57, 64, 65, 66, 68, 71, 72, 73, 40, 80, 81):   # voice, trumpet, sax, oboe, clarinet, flute, violin, leads
            score += 0.3
        out.append({"index": idx, "score": round(float(score), 4), "overlap": round(ov, 4), "in_range": round(in_range, 4),
                    "notes": len(p), "name": inst.name or "", "program": int(prog)})
    out.sort(key=lambda c: (-c["score"], c["index"]))
    return out


def pick_melody(pm) -> dict:
    """Best melody candidate, or Skip('no_melody_candidate' | 'melody_ambiguous')."""
    cands = melody_candidates(pm)
    if not cands:
        raise Skip("no_melody_candidate")
    if len(cands) > 1 and cands[0]["score"] - cands[1]["score"] < AMBIGUITY_MARGIN:
        raise Skip("melody_ambiguous")
    return {**cands[0], "n_candidates": len(cands)}


# =====================================================================================================
# key
# =====================================================================================================
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(hist: np.ndarray) -> tuple[int, str, float, float]:
    """-> (tonic pc, 'maj'|'min', best correlation, margin to the best key that is not the relative of the best).
    The relative key has the same pitch set, so confusing it changes only the header label, not the notes."""
    if hist.sum() <= 0 or np.ptp(hist) == 0:
        raise Skip("key_no_pitch_content")
    scores = []
    for tonic in range(12):
        for mode, prof in (("maj", KK_MAJOR), ("min", KK_MINOR)):
            scores.append((float(np.corrcoef(hist, np.roll(prof, tonic))[0, 1]), tonic, mode))
    scores.sort(key=lambda x: -x[0])
    r1, tonic, mode = scores[0]
    rel = ((tonic + 9) % 12, "min") if mode == "maj" else ((tonic + 3) % 12, "maj")
    r2 = next(r for r, t, m in scores[1:] if (t, m) != rel)
    return tonic, mode, r1, r1 - r2


def transpose_shift(tonic: int, mode: str) -> int:
    """Smallest shift (-6..+5) mapping the key to C major / A minor."""
    target = 0 if mode == "maj" else 9
    s = (target - tonic) % 12
    return s - 12 if s >= 6 else s


# =====================================================================================================
# chords
# =====================================================================================================
TEMPLATES = {            # quality suffix -> pitch-class offsets from the root
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}
_COMPLEX_PENALTY = {"dim": 0.05, "sus4": 0.08, "sus2": 0.08}


def match_chord(chroma: np.ndarray, bass_pc: int | None) -> tuple[int, str, float]:
    """Best (root pc, quality, score) for a chroma vector (any scale).  score = in-template energy share minus penalties
    for template tones that are absent, for 4-note and sus/dim templates, plus a bonus when the bass note is the root."""
    tot = float(chroma.sum())
    if tot <= 0:
        raise Skip("chord_no_energy")
    c = chroma / tot
    best = (-9.0, 0, "")
    for root in range(12):
        for q, tones in TEMPLATES.items():
            pcs = [(root + t) % 12 for t in tones]
            sc = float(sum(c[i] for i in pcs))
            sc -= 0.06 * sum(1 for i in pcs if c[i] < 0.04)
            sc -= 0.04 * max(0, len(tones) - 3) + _COMPLEX_PENALTY.get(q, 0.0)
            if bass_pc is not None and bass_pc == root:
                sc += 0.10
            if sc > best[0] + 1e-12:
                best = (sc, root, q)
    return best[1], best[2], best[0]


def half_bar_chroma(acc: dict, w0: int, w1: int, vel_weight: bool = True) -> tuple[np.ndarray, int, int | None]:
    """Chroma (overlap-duration weighted), number of sounding notes, bass pc for accompaniment notes in tick window."""
    s, e, p, v = acc["s"], acc["e"], acc["p"], acc["v"]
    m = (s < w1) & (e > w0)
    chroma = np.zeros(12)
    if not m.any():
        return chroma, 0, None
    ov = np.minimum(e[m], w1) - np.maximum(s[m], w0)
    w = ov * ((0.5 + v[m] / 254.0) if vel_weight else 1.0)
    pm_ = p[m]
    np.add.at(chroma, pm_ % 12, w)
    return chroma, int(m.sum()), int(pm_.min() % 12)


# =====================================================================================================
# whole-file conversion
# =====================================================================================================
def check_meter_tempo(pm) -> tuple[int, bool]:
    """-> (main tempo bpm, time signature defaulted).  Skip unless 4/4 only and one main tempo covers >= 90% of the time."""
    tsc = pm.time_signature_changes
    if any((t.numerator, t.denominator) != (4, 4) for t in tsc):
        raise Skip("meter_not_4_4")
    end = pm.get_end_time()
    times, tempi = pm.get_tempo_changes()
    if end <= 0 or len(tempi) == 0:
        raise Skip("no_tempo")
    bounds = list(times) + [max(end, times[-1] + 1e-9)]
    share: dict[int, float] = {}
    for i, t in enumerate(tempi):
        dur = max(0.0, bounds[i + 1] - bounds[i])
        share[int(round(float(t)))] = share.get(int(round(float(t))), 0.0) + dur
    tempo, dur = max(share.items(), key=lambda kv: (kv[1], -kv[0]))
    if dur / sum(share.values()) < MAIN_TEMPO_SHARE:
        raise Skip("tempo_changes")
    if not 40 <= tempo <= 220:
        raise Skip("tempo_out_of_range")
    return tempo, not tsc


def quantize_melody(a: dict, res: int) -> tuple[list[tuple[int, int, int]], float]:
    """-> ([(pitch, start16, end16)] monophonic, off-grid share of note starts).  Chord notes keep the highest pitch,
    overlaps are trimmed at the next onset, every note lasts at least one 16th."""
    u = res / 4.0
    x = a["s"] / u
    off = float(np.mean(np.abs(x - np.round(x)) > GRID_TOL)) if len(x) else 1.0
    notes = []
    last_s = None
    for s_t, e_t, p in zip(a["s"], a["e"], a["p"]):          # sorted by (start, -pitch): first of equal starts is highest
        s = int(round(s_t / u))
        e = max(int(round(e_t / u)), s + 1)
        if s == last_s:
            continue
        last_s = s
        notes.append([int(p), s, e])
    for i in range(len(notes) - 1):
        notes[i][2] = min(notes[i][2], notes[i + 1][1])
    return fill_gaps(notes), off


def fill_gaps(notes: list[list[int]]) -> list[tuple[int, int, int]]:
    """Absorb gate gaps: a one-16th rest between two notes is always an artefact; if at least STACCATO_SHARE of the note
    pairs have a 1-2 sixteenth gap the piece is a staccato transcription and gaps up to two 16ths are filled too."""
    gaps = [notes[i + 1][1] - notes[i][2] for i in range(len(notes) - 1)]
    stac = bool(gaps) and sum(1 for g in gaps if 1 <= g <= 2) / len(gaps) >= STACCATO_SHARE
    limit = 2 if stac else 1
    for i, g in enumerate(gaps):
        if 1 <= g <= limit:
            notes[i][2] = notes[i + 1][1]
    return [tuple(n) for n in notes]


def convert_midi(pm, hint_name: str = "") -> dict:
    """PrettyMIDI -> piece dict (C major / A minor, 4/4, 16th grid, chords per half bar) or raises Skip(reason)."""
    tempo, ts_default = check_meter_tempo(pm)
    mel = pick_melody(pm)
    res = pm.resolution
    if res < 96:
        raise Skip("resolution_too_low")
    a = inst_arrays(pm, pm.instruments[mel["index"]])
    notes16, off = quantize_melody(a, res)
    if off > MAX_OFF_GRID:
        raise Skip("off_grid")
    if len(notes16) < MIN_MELODY_NOTES:
        raise Skip("too_few_notes")
    first_bar = notes16[0][1] // BAR16
    end16 = max(n[2] for n in notes16)
    nbars = -(-end16 // BAR16) - first_bar
    if nbars < 4:
        raise Skip("too_short")
    b0 = first_bar * BAR16                                    # grid origin of the piece

    # key from every non-drum note (duration weighted, capped at 4 beats)
    hist = np.zeros(12)
    acc_parts = []
    for idx, inst in enumerate(pm.instruments):
        if inst.is_drum or not inst.notes:
            continue
        arr = a if idx == mel["index"] else inst_arrays(pm, inst)
        dur = np.minimum(arr["e"] - arr["s"], 4 * res)
        np.add.at(hist, arr["p"] % 12, dur)
        if idx != mel["index"]:
            acc_parts.append(arr)
    tonic, mode, r1, margin = estimate_key(hist)
    if r1 < KEY_MIN_CORR:
        raise Skip("key_low_correlation")
    if margin < KEY_MIN_MARGIN:
        raise Skip("key_ambiguous")
    shift = transpose_shift(tonic, mode)
    scale = MAJOR_PCS if mode == "maj" else MINOR_PCS
    in_key = float(np.mean([((p - tonic) % 12) in scale for p, _, _ in notes16]))
    if in_key < MIN_IN_KEY:
        raise Skip("melody_out_of_key")
    if not acc_parts:
        raise Skip("no_accompaniment")
    acc = {k: np.concatenate([x[k] for x in acc_parts]) for k in ("s", "e", "p", "v")}

    # melody events on the piece grid, transposed
    ev = []
    cur = 0
    total = nbars * BAR16
    for p, s, e in notes16:
        s, e = s - b0, min(e - b0, total)
        if s >= total:
            break
        pitch = p + shift
        if not 21 <= pitch <= 108:
            raise Skip("pitch_range")
        if s > cur:
            ev.append((None, cur * SIXTEENTH, (s - cur) * SIXTEENTH))
        ev.append((pitch, s * SIXTEENTH, (e - s) * SIXTEENTH))
        cur = e
    if cur < total:
        ev.append((None, cur * SIXTEENTH, (total - cur) * SIXTEENTH))

    # chords per half bar
    half_ticks = res * 2
    chords: list[tuple[str, int]] = []
    evidence: list[int] = []
    names: list[str | None] = []
    for h in range(nbars * 2):
        w0 = int(round((b0 + h * HALF_BAR16) * res / 4.0))
        w1 = int(round((b0 + (h + 1) * HALF_BAR16) * res / 4.0))
        chroma, n_notes, bass = half_bar_chroma(acc, w0, w1)
        name = None
        if n_notes >= MIN_ACC_NOTES and (chroma > 0).sum() >= 2:
            root, q, score = match_chord(chroma, bass)
            if score >= CHORD_MIN_SCORE:
                name = ROOT_NAMES[(root + shift) % 12] + q
        names.append(name)
        evidence.append(1 if name else 0)
    if not any(names):
        raise Skip("no_chord_evidence")
    first = next(i for i, n in enumerate(names) if n)       # backfill leading gaps, carry the previous chord afterwards
    prev = names[first]
    filled = []
    for n in names:
        prev = n or prev
        filled.append(prev)
    last = None
    for h, sym in enumerate(filled):
        if sym != last:
            chords.append((sym, h * HALF_BAR16 * SIXTEENTH))
            last = sym
    return {
        "nbars": nbars, "tempo": tempo, "key": "C" if mode == "maj" else "Am",
        "key_src": f"{ROOT_NAMES[tonic]}:{mode}", "shift": shift, "notes": ev, "chords": chords, "ev": evidence,
        "info": {"mel_name": mel["name"], "mel_program": mel["program"], "mel_score": mel["score"],
                 "n_candidates": mel["n_candidates"], "off_grid": round(off, 4), "key_r": round(r1, 4),
                 "key_margin": round(margin, 4), "in_key": round(in_key, 4), "ts_default": ts_default, "res": res},
    }
