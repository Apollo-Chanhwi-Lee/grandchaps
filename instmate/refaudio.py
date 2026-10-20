"""Reference audio analysis (M6). Only a file the user uploaded is analysed; the app never downloads audio.

Spec (docs/dev_instructions.md section 5): keep only tempo, key, structure, per-section energy and rough melody
statistics. The melody itself is NOT copied: the pitch track is reduced to three numbers inside this module and is
never returned (asserted on every result). The uploaded file is deleted when the analysis ends.

Public API
----------
analyze_reference(path, delete=True) -> dict
    {"file": {"name", "size_mb", "duration_s"}, "tempo": {"bpm", "alternatives": [half, double], "note"},
     "key": {"tonic", "mode", "name", "label", "score", "runner_up", "ambiguous"},
     "sections": [{"start_s", "end_s", "energy"}], "boundaries_s": [...],   energy: 0-1, relative to the loudest section
     "melody_stats": {"label": "rough, from mixed audio", "pitch_range_semitones", "notes_per_second",
                      "leap_ratio", "voiced_ratio"},
     "deleted": bool, "notes": [...]}
    Tempo: librosa beat tracking often lands on half or double time, so the two alternatives are reported next to
    the raw bpm; reference_to_style folds an implausible value (>180 or <50) and the user confirms.
    Melody stats come from librosa.pyin on the harmonic part of the MIXED audio (first MELODY_WINDOW_S seconds
    only, for speed). They are rough: accompaniment and vocals are not separated.
    Deletion: with delete=True the file is removed in a finally block on EVERY path once it exists - success,
    decode error, forced error, and also refusal for size/duration/extension (the upload is the user's temporary
    copy, so a refused upload is not kept either). Deletion is verified (os.path.exists is False). On success the
    result carries deleted=True; if removal failed, deleted=False and notes carries a warning (caller must check).
    Errors raise RefAudioError (Korean message) with .deleted set.
    Limits (checked before decoding): extension in wav/mp3/m4a/flac/ogg, size <= 30 MB, duration <= 10 min.
reference_to_style(analysis) -> dict
    Suggestion only, same shape as style.style_from_request (attrs, confirm_question, method="reference", notes);
    the user must still confirm it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .theory import KK_MAJOR, KK_MINOR, NOTE_NAMES, _corr

ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_BYTES = 30 * 1024 * 1024
MAX_DURATION_S = 600.0
SR = 22050
HOP = 512
MELODY_WINDOW_S = 90.0
LEAP_SEMITONES = 4
MAX_ARRAY_LEN = 64                                        # no list of numbers longer than this may leave the module


class RefAudioError(ValueError):
    def __init__(self, message: str, deleted: bool = False):
        super().__init__(message)
        self.deleted = deleted


def _assert_no_pitch_arrays(obj, path: str = "result") -> None:
    """Spec: the melody itself is never returned. Fails on ndarrays or long numeric lists anywhere in the result."""
    if isinstance(obj, np.ndarray):
        raise AssertionError(f"{path} is an ndarray")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_pitch_arrays(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        if len(obj) > MAX_ARRAY_LEN and all(isinstance(x, (int, float)) for x in obj):
            raise AssertionError(f"{path} is a long numeric list ({len(obj)})")
        for i, v in enumerate(obj):
            _assert_no_pitch_arrays(v, f"{path}[{i}]")


def _check_file(p: Path) -> float:
    """Loud refusals before anything is decoded. Returns duration in seconds."""
    import librosa
    if p.suffix.lower() not in ALLOWED_EXT:
        raise RefAudioError(f"지원하지 않는 파일 형식이에요({p.suffix or '확장자 없음'}). wav, mp3, m4a, flac, ogg만 올릴 수 있어요.")
    size = p.stat().st_size
    if size > MAX_BYTES:
        raise RefAudioError(f"파일이 너무 커요({size / 1048576:.1f}MB). 30MB 이하만 분석할 수 있어요.")
    try:
        dur = float(librosa.get_duration(path=str(p)))
    except Exception:
        raise RefAudioError("음원을 읽을 수 없어요. 손상된 파일이거나 지원하지 않는 인코딩일 수 있어요.") from None
    if dur > MAX_DURATION_S:
        raise RefAudioError(f"음원이 너무 길어요({dur / 60:.1f}분). 10분 이하만 분석할 수 있어요.")
    if dur < 3.0:
        raise RefAudioError(f"음원이 너무 짧아요({dur:.1f}초). 3초 이상이어야 분석할 수 있어요.")
    return dur


def _tempo(y, sr) -> dict:
    import librosa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0:
        return {"bpm": None, "alternatives": [], "note": "박자를 찾지 못했어요"}
    return {"bpm": round(bpm, 1), "alternatives": [round(bpm / 2, 1), round(bpm * 2, 1)],
            "note": "박자 추적은 절반/두 배 템포로 잡히는 경우가 많아 alternatives도 함께 알려요"}


def _key(chroma) -> dict:
    """Mean CQT chroma vs Krumhansl-Kessler profiles (same tables as theory.py)."""
    mean = [float(v) for v in chroma.mean(axis=1)]
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            rotated = [prof[(pc - tonic) % 12] for pc in range(12)]
            scores.append((_corr(mean, rotated), tonic, mode))
    scores.sort(reverse=True)
    (s1, t1, m1), (s2, t2, m2) = scores[0], scores[1]
    name = NOTE_NAMES[t1] + ("m" if m1 == "minor" else "")
    return {"tonic": t1, "mode": m1, "name": name, "label": f"{NOTE_NAMES[t1]} {'단조' if m1 == 'minor' else '장조'}",
            "score": round(s1, 3), "runner_up": NOTE_NAMES[t2] + ("m" if m2 == "minor" else ""),
            "ambiguous": bool(s1 - s2 < 0.03)}


def _sections(y, sr, chroma, duration: float) -> tuple[list[dict], list[float]]:
    import librosa
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    n = min(chroma.shape[1], mfcc.shape[1], rms.shape[0])
    feats = np.vstack([librosa.util.normalize(chroma[:, :n], axis=1),
                       librosa.util.normalize(mfcc[:, :n], axis=1)])
    k = int(min(max(round(duration / 20), 2), 10, n))
    bounds = librosa.segment.agglomerative(feats, k) if k > 1 else np.array([0])
    edges = sorted({int(b) for b in bounds} | {0}) + [n]
    times = [float(t) for t in librosa.frames_to_time(edges, sr=sr, hop_length=HOP)]
    times[-1] = round(duration, 2)
    means = [float(rms[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    top = max(means) or 1.0
    secs = [{"start_s": round(times[i], 2), "end_s": round(times[i + 1], 2), "energy": round(means[i] / top, 3)}
            for i in range(len(means))]
    return secs, [round(t, 2) for t in times[1:-1]]


def _melody_stats(y, sr) -> dict:
    """Three numbers from a pyin track. The pitch track itself stays local to this function."""
    import librosa
    y = y[: int(MELODY_WINDOW_S * sr)]
    harm = librosa.effects.harmonic(y)
    f0, voiced, _ = librosa.pyin(harm, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr,
                                 hop_length=HOP)
    midi = np.where(voiced & np.isfinite(f0), np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), np.nan)
    frame_s = HOP / sr
    n_voiced = int(np.isfinite(midi).sum())
    stats = {"label": "rough, from mixed audio", "pitch_range_semitones": None, "notes_per_second": 0.0,
             "leap_ratio": None, "voiced_ratio": round(n_voiced / max(len(midi), 1), 3)}
    if n_voiced < 8:
        return stats
    v = midi[np.isfinite(midi)]
    stats["pitch_range_semitones"] = int(round(np.percentile(v, 95) - np.percentile(v, 5)))
    # notes = runs of the same rounded pitch lasting >= 80 ms
    notes, run_p, run_n = [], None, 0
    for p in list(midi) + [np.nan]:
        if run_p is not None and np.isfinite(p) and p == run_p:
            run_n += 1
            continue
        if run_p is not None and run_n * frame_s >= 0.08:
            notes.append(run_p)
        run_p, run_n = (p, 1) if np.isfinite(p) else (None, 0)
    stats["notes_per_second"] = round(len(notes) / (len(midi) * frame_s), 2)
    steps = [abs(b - a) for a, b in zip(notes[:-1], notes[1:]) if b != a]
    stats["leap_ratio"] = round(sum(s > LEAP_SEMITONES for s in steps) / len(steps), 3) if steps else None
    return stats


def _analyze(p: Path) -> dict:
    import librosa
    duration = _check_file(p)
    try:
        y, sr = librosa.load(str(p), sr=SR, mono=True)
    except Exception:
        raise RefAudioError("음원을 읽을 수 없어요. 손상된 파일이거나 지원하지 않는 인코딩일 수 있어요.") from None
    if y.size == 0 or not np.isfinite(y).all() or float(np.abs(y).max()) < 1e-4:
        raise RefAudioError("음원이 비어 있거나 무음이에요.")
    duration = float(len(y) / sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    sections, bounds = _sections(y, sr, chroma, duration)
    return {"file": {"name": p.name, "size_mb": round(p.stat().st_size / 1048576, 2), "duration_s": round(duration, 1)},
            "tempo": _tempo(y, sr), "key": _key(chroma), "sections": sections, "boundaries_s": bounds,
            "melody_stats": _melody_stats(y, sr), "notes": []}


def analyze_reference(path, delete: bool = True) -> dict:
    """Analyse a user-uploaded file. See the module docstring for the result shape and the deletion contract."""
    p = Path(path)
    if not p.is_file():
        raise RefAudioError("파일을 찾을 수 없어요.")
    result, error = None, None
    try:
        result = _analyze(p)
        _assert_no_pitch_arrays(result)
    except RefAudioError as e:
        error = e
    except Exception as e:                                    # any other failure: still delete, report in Korean
        error = RefAudioError(f"음원 분석 중 오류가 났어요({type(e).__name__}).")
        error.__cause__ = e
    finally:
        deleted, warn = False, None
        if delete:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as e:
                warn = f"원본 삭제 실패({type(e).__name__}): 직접 지워 주세요"
            deleted = not os.path.exists(p)
            if not deleted and warn is None:
                warn = "원본 삭제를 확인하지 못했어요"
    if error is not None:
        error.deleted = deleted
        raise error
    result["deleted"] = deleted
    if warn:
        result["notes"].append(warn)
    return result


# ---------------------------------------------------------------- analysis -> style suggestion
def reference_to_style(analysis: dict) -> dict:
    """Map an analysis to a style suggestion (still to be confirmed by the user). Same shape as style_from_request."""
    from . import style
    notes: list[str] = []
    bpm = (analysis.get("tempo") or {}).get("bpm")
    if bpm:
        while bpm > 180:
            bpm /= 2
            notes.append("템포가 너무 빨라 절반으로 봤어요(박자 추적의 두 배 오류 가능)")
        while bpm < 50:
            bpm *= 2
            notes.append("템포가 너무 느려 두 배로 봤어요")
        lo, hi = round(bpm * 0.92), round(bpm * 1.08)
    else:
        lo = hi = None
        notes.append("템포를 찾지 못해 장르 기본 템포를 써요")
    en = [s["energy"] for s in analysis.get("sections", [])]
    mean_e = float(np.mean(en)) if en else 0.5
    spread = float(np.std(en)) if en else 0.0
    mid = bpm or 100
    if mid < 90:
        genre = "ballad"
    elif mid < 115:
        genre = "ballad" if mean_e < 0.6 else "rock"
    else:
        genre = "dance" if spread < 0.15 else "rock"
    mode = (analysis.get("key") or {}).get("mode")
    if genre in ("rock", "dance") and mean_e >= 0.6:
        mood = "energetic"
    elif mode == "minor":
        mood = "sad" if genre == "ballad" else "dark"
    else:
        mood = "warm" if genre == "ballad" else "bright"
    raw = {"genre": genre, "tempo_min": lo, "tempo_max": hi, "mood": mood,
           "instruments": list(style.DEFAULT_INSTRUMENTS[genre]), "structure_hint": []}
    src = f"레퍼런스 음원 분석(템포 {bpm and round(bpm)}, {(analysis.get('key') or {}).get('name', '?')})"
    attrs, vnotes = style.validate_attrs(raw, src)
    return {"attrs": attrs.to_dict(), "confirm_question": style.confirm_question(attrs), "method": "reference",
            "notes": notes + vnotes + ["제안일 뿐이에요. 사용자가 확인해야 적용돼요."]}
