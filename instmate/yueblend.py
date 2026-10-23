"""AI vocal on the app's own band: the YuE2 voice alone, laid over the song's accompaniment (option, button only).

The guide vocal is a rule-made voice and breaks or sounds mechanical (App/202610102026_문제3.md).  YuE2 sings in one
flow but brings its own backing and does not sing at the written octave.  This module follows the one procedure the
user heard and chose on 2026-10-10 (pair 9):

  1. the song's highlight (lead-in + first chorus) is prepared like the demo (yue.thin + yue.fit_whole) and written
     LOWER semitones down, with a male voice named in the request; the take is not pitch-shifted afterwards;
  2. YuE2 sings it (about 18 GB, about as long as the music);
  3. the voice is separated from the take (Demucs, its own program under models/demucs, see `separator`);
  4. the voice is moved in time onto the score: one offset for the whole take from the cross-correlation of pitch
     classes (the octave is ignored), and a section's own offset where it differs by more than SECTION_TOLERANCE_S;
  5. the app's band is rendered once from the same lowered sheet, the voice is mixed in like the guide vocal, and the
     result is mastered.

So the preview is in a key LOWER semitones below the song's, it covers the highlight only, and its melody is the
thinned one.  The alignment was heard to work on one song; it is not a guarantee for every song.  Nothing here
changes the song, its MIDI or its score.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import yue
from .leadsheet import KEYS, LeadSheet, parse_leadsheet, serialize_leadsheet
from .theory import TPQ

ROOT = Path(__file__).resolve().parent.parent
LOWER = 5                          # semitones the request and the band are lowered (the take the user chose)
VOICE = "expressive male voice"
SECTION_TOLERANCE_S = 0.05
ENV_PYTHON = "INSTMATE_DEMUCS_PYTHON"
SEPARATE = ROOT / "scripts" / "demucs_separate.py"
FRAME_S = 256 / 22050


class BlendUnavailable(RuntimeError):
    """The overlay cannot run here (Korean message)."""


def separator() -> Path | None:
    """The Python that has Demucs: INSTMATE_DEMUCS_PYTHON, else models/demucs/venv (not shipped; `python -m venv
    models/demucs/venv && models/demucs/venv/bin/pip install demucs soundfile librosa torchaudio`)."""
    given = os.environ.get(ENV_PYTHON)
    p = Path(given) if given else ROOT / "models" / "demucs" / "venv" / "bin" / "python"
    return p if p.is_file() else None


def unavailable_reason() -> str | None:
    reason = yue.unavailable_reason()
    if reason:
        return reason
    if separator() is None:
        return "보컬 분리 도구(Demucs)가 없어요. models/demucs/venv에 설치하거나 INSTMATE_DEMUCS_PYTHON으로 알려 주세요."
    return None


def lower_key(key: str, semitones: int = LOWER) -> str:
    """The name of the key `semitones` below `key` (the spelling with the fewest accidentals)."""
    from .render import key_of
    k = key_of(key)
    want = ((k.tonic - semitones) % 12, k.mode)
    names = [n for n in KEYS if (key_of(n).tonic, key_of(n).mode) == want]
    return min(names, key=lambda n: (abs(KEYS[n]), n))


def asked_sheet(sheet: LeadSheet | str, full: bool = False) -> LeadSheet:
    """The music as it is sent to the model and used for the band: thinned, one register, LOWER semitones down.
    `full`: the whole song when it is not longer than yue.MAX_S; otherwise (and by default) the highlight."""
    from .composer import transpose_sheet
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    part = sheet if full and yue.seconds(sheet) <= yue.MAX_S else yue.highlight(sheet)
    prepared, _ = yue.fit_whole(yue.thin(part), yue.STYLE["female_high"]["center"])
    out = transpose_sheet(prepared, [-LOWER] * len(prepared.sections), lower_key(prepared.key))
    out.tempo, out.meter, out.genre, out.country = part.tempo, part.meter, part.genre, part.country
    return out


def score_classes(sheet: LeadSheet, dt: float = FRAME_S) -> np.ndarray:
    """(12, frames) of the asked melody: 1 where a pitch class sounds (soft edges of half a semitone do not apply to
    a written note, so plain ones)."""
    sec = 60.0 / sheet.tempo / TPQ
    bar = 4 * TPQ
    total = sum(len(s.bars) for s in sheet.sections) * bar * sec
    out = np.zeros((12, int(np.ceil(total / dt))))
    at = 0
    for s in sheet.sections:
        for b in s.bars:
            for n in b.notes:
                if n.pitch is not None:
                    i, j = int((at + n.start) * sec / dt), int((at + n.start + n.dur) * sec / dt)
                    out[n.pitch % 12, i:j] = 1.0
            at += bar
    return out


def sung_classes(f0: np.ndarray) -> np.ndarray:
    """(12, frames) of a pitch track in Hz (NaN = no voice): how near each frame is to each pitch class."""
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = 69 + 12 * np.log2(f0 / 440.0)
    pcs = np.arange(12)[:, None]
    near = np.exp(-0.5 * ((((midi[None, :] - pcs + 6) % 12) - 6) / 0.5) ** 2)
    near[:, ~np.isfinite(midi)] = 0.0
    return near


def offset(observed: np.ndarray, ref: np.ndarray, expected: float = 0.0, limit: float = 8.0, dt: float = FRAME_S) -> float:
    """Seconds by which `ref` (a stretch of the score) lies later in `observed` (the sung track): the lag of the
    largest pitch-class cross-correlation within `limit` of `expected`."""
    from scipy.signal import correlate, correlation_lags
    scores = sum(correlate(y, x, method="fft") for y, x in zip(observed, ref))
    lags = correlation_lags(observed.shape[1], ref.shape[1]) * dt
    ok = (lags >= expected - limit) & (lags <= expected + limit)
    if not ok.any() or not np.any(ref) or float(np.max(scores[ok])) <= 0:
        return expected                    # no melody in this stretch, or the take ends before it: keep the expected place
    return float(lags[np.flatnonzero(ok)[np.argmax(scores[ok])]])


def plan(observed: np.ndarray, sheet: LeadSheet, dt: float = FRAME_S) -> list[dict]:
    """Per section {"section", "start", "end", "offset"}: where in the sung track the section's music is (offset in
    seconds added to the score's time).  The whole take's offset is used unless a section's own differs by more
    than SECTION_TOLERANCE_S."""
    ref = score_classes(sheet, dt)
    whole = offset(observed, ref, 0.0, 8.0, dt)
    bar_s = 240.0 / sheet.tempo
    out, at = [], 0.0
    for s in sheet.sections:
        a, b = at, at + len(s.bars) * bar_s
        local = offset(observed, ref[:, round(a / dt): round(b / dt)], a + whole, 0.75, dt) - a
        out.append({"section": s.name, "start": a, "end": b, "offset": local if abs(local - whole) > SECTION_TOLERANCE_S else whole})
        at = b
    return out


def align(voice: np.ndarray, sr: int, plans: list[dict], fade_s: float = 0.02) -> np.ndarray:
    """The mono voice moved onto the score's time section by section (constant shifts joined with short fades; no
    time-stretching, no pitch change)."""
    n = int((plans[-1]["end"] + 1.0) * sr)
    out, weights, fade = np.zeros(n), np.zeros(n), int(fade_s * sr)
    for p in plans:
        a, b = max(0, round(p["start"] * sr) - fade), min(n, round(p["end"] * sr) + fade)
        dst = np.arange(a, b)
        src = dst + round(p["offset"] * sr)
        v = np.zeros(len(dst))
        ok = (src >= 0) & (src < len(voice))
        v[ok] = voice[src[ok]]
        w = np.ones(len(dst))
        ramp = min(2 * fade, len(dst) // 2)
        if a > 0:
            w[:ramp] = np.linspace(0, 1, ramp)
        if b < n:
            w[-ramp:] = np.linspace(1, 0, ramp)
        out[dst] += v * w
        weights[dst] += w
    return out / np.maximum(weights, 1e-9)


def _take(asked: LeadSheet, genre: str | None, work: Path, lyrics: str | None = None) -> Path:
    """YuE2 sings the asked sheet into work/take.wav."""
    st = yue.STYLE["female_high"]
    band = yue.GENRE_STYLE.get(genre or "", "band pop, piano, guitar, bass, drums")
    words = yue.tagged(lyrics) if lyrics and lyrics.strip() else yue.lyrics(asked, st["syllable"])
    req = {"style": f"{yue.language_of(lyrics) if lyrics and lyrics.strip() else 'English'}, {band}, {VOICE}, {asked.tempo} BPM", "lyrics": words,
           "abc": yue.to_abc(asked, st["octave"]), "cot": "full", "seed": st["seed"], "cfg_scale": st["cfg"],
           "seconds": yue.seconds(asked)}
    (work / "lyrics.txt").write_text(req["lyrics"], encoding="utf-8")
    (work / "score.abc").write_text(req["abc"], encoding="utf-8")
    out = work / "take.wav"
    proc = subprocess.run(yue.command(req, work, out), cwd=yue.model_dir(), capture_output=True, text=True,
                          timeout=max(120.0, yue.TIMEOUT_PER_S * req["seconds"]), stdin=subprocess.DEVNULL)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"YuE2 생성 실패: {(proc.stderr or proc.stdout or '').strip()[-300:]}")
    return out


def _separate(take: Path, work: Path) -> Path:
    out = work / "vocals.wav"
    proc = subprocess.run([str(separator()), str(SEPARATE), str(take), str(out)], capture_output=True, text=True,
                          timeout=900, stdin=subprocess.DEVNULL)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"보컬 분리 실패: {(proc.stderr or proc.stdout or '').strip()[-300:]}")
    return out


def _pitch(voice: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    y = librosa.resample(voice.astype(np.float32), orig_sr=sr, target_sr=22050) if sr != 22050 else voice.astype(np.float32)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=22050,
                                 frame_length=2048, hop_length=256)
    f0[~voiced] = np.nan
    return f0


def overlay(leadsheet: str, genre: str | None = None, instruments: list[str] | None = None, mode: str | None = None,
            progress=None, take=_take, separate=_separate, full: bool = False, lyrics: str | None = None) -> tuple[bytes, dict]:
    """(mastered WAV of the highlight with the AI voice on the app's band, info).  `take` / `separate` are replaced
    in tests; everything else is the real path."""
    import soundfile as sf
    import base64
    from . import audio, render, vocal
    reason = unavailable_reason() if take is _take or separate is _separate else None
    if reason:
        raise BlendUnavailable(reason)
    say = progress or (lambda _: None)
    asked = asked_sheet(leadsheet, full)
    whole = len(asked.sections) == len(parse_leadsheet(leadsheet).sections)
    if full and not whole:
        say(f"곡이 {yue.MAX_S:.0f}초보다 길어 하이라이트(벌스~첫 후렴)만 만들어요")
    text = serialize_leadsheet(asked)
    song = render.leadsheet_to_song(text, genre_override=genre, instruments=instruments)
    midi = base64.b64decode(song["midi_b64"])
    with tempfile.TemporaryDirectory(prefix="instmate_blend_") as tmp:
        work = Path(tmp)
        say(f"1/4 YuE2가 {yue.seconds(asked):.0f}초 분량을 부르는 중 (대략 그만큼 걸려요)")
        take_path = take(asked, genre, work, lyrics) if lyrics and lyrics.strip() else take(asked, genre, work)
        say("2/4 목소리만 분리하는 중")
        raw, sr = sf.read(separate(take_path, work), always_2d=True)
    voice = raw.mean(axis=1)
    say("3/4 반주와 박자를 맞추는 중")
    plans = plan(sung_classes(_pitch(voice, sr)), asked)
    aligned = align(voice, sr, plans)
    say("4/4 앱의 반주에 얹는 중")
    notes: list[str] = []
    band = audio._mix_by_part(midi, sing_melody=True, genre=genre, user_parts=set(), sections=song["sections"],
                              notes=notes, overrides={}, mode=mode)
    mixed = audio.master_v2(vocal.mix_vocal(band, aligned, sr), "natural")
    info = {"key": asked.key, "lowered": LOWER, "seconds": round(yue.seconds(asked), 1), "full": whole,
            "sung_seconds": round(len(voice) / sr, 1),
            "sections": [s.name for s in asked.sections],
            "offsets_ms": [round(1000 * p["offset"]) for p in plans], "notes": notes[:6]}
    return mixed, info
