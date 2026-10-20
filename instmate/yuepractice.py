"""Practise my part against a YuE2 song (user's design, 2026-10-13): the song YuE2 wrote is played by the app's own band
(guitar, bass, keys, drums) from the score YuE2 wrote, my part left out, and the singing is YuE2's own voice, split
from the take once (Demucs, as in yueblend) and kept beside the song.  When my part is the vocal, there is no voice.
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

from . import yuecheck

PARTS = ("vocal", "guitar", "bass", "keys", "drums")
TRACK_OF = {"guitar": "guitar", "bass": "bass", "keys": "piano", "drums": "drums"}   # my part -> the app's MIDI track


class PracticeError(ValueError):
    """The request cannot be met (the message says why, in Korean)."""


def available() -> str | None:
    from . import yueblend
    return None if yueblend.separator() else "보컬 분리 도구(Demucs)가 없어 YuE2 목소리를 꺼낼 수 없어요"


def _voice(take: bytes) -> bytes:
    """The YuE2 voice alone (WAV bytes)."""
    import tempfile
    from . import yueblend
    with tempfile.TemporaryDirectory(prefix="instmate_practice_") as tmp:
        src = Path(tmp) / "take.wav"
        src.write_bytes(take)
        return yueblend._separate(src, Path(tmp)).read_bytes()


FINE_HOP = 64                    # onset strength hop at 22050 Hz: about 2.9 ms
SEARCH_S = 0.030                 # each beat is looked for within +-30 ms of the tracker's beat
ATTACK_HP_HZ = 1000              # attacks are found above 1 kHz, clear of the held notes below
ENVELOPE_S = 0.001               # the energy envelope's smoothing (centred, so it does not shift the attack)
RISE_SHARE = 0.5                 # the earliest rise that reaches half the window's steepest one is the attack


def _drums(y, sr):
    """The signal with the drums brought forward: kick (40-120 Hz), snare body (150-250 Hz) and attacks above 2 kHz."""
    import scipy.signal as ss
    band = lambda lo, hi: ss.butter(2, [lo, hi], "bandpass", fs=sr, output="sos")
    return (ss.sosfilt(band(40, 120), y) + ss.sosfilt(band(150, 250), y)
            + ss.sosfilt(ss.butter(2, 2000, "highpass", fs=sr, output="sos"), y))


def _onsets(y, sr):
    """(onset strength, its times) on the drum-weighted signal, one value every FINE_HOP samples at 22050 Hz.  A
    512-sample FFT keeps the strength close to the attack (the default 2048 put the take's beats about 33 ms late)."""
    import librosa
    import numpy as np
    d = librosa.resample(np.asarray(_drums(y, sr), dtype=float), orig_sr=sr, target_sr=22050)
    o = librosa.onset.onset_strength(y=d, sr=22050, hop_length=FINE_HOP, n_fft=512)
    return o, librosa.times_like(o, sr=22050, hop_length=FINE_HOP)


def coarse_beats(y, sr, bpm: float, onsets=None) -> list[float]:
    """The tracker's beats (librosa), near the score's tempo.  A tracker that settles on half or double the score's
    tempo (a backbeat can look like one slow beat) is run again held to the score's tempo."""
    import librosa
    import numpy as np
    o, times = onsets if onsets is not None else _onsets(y, sr)
    tempo, frames = librosa.beat.beat_track(onset_envelope=o, sr=22050, hop_length=FINE_HOP, start_bpm=bpm, tightness=400,
                                            trim=False)
    if not 0.75 < float(np.atleast_1d(tempo)[0]) / bpm < 1.33:
        _, frames = librosa.beat.beat_track(onset_envelope=o, sr=22050, hop_length=FINE_HOP, bpm=bpm, tightness=400,
                                            trim=False)
    return [float(times[f]) for f in frames]


class Grid(list):
    """Times (s) of the points of a take's pulse: `parts` points to a beat (1: the beats; 2: beats and off-beats).  A
    plain list of beats is a grid of one."""
    parts = 1


SUB_WINDOW = 0.30                # an off-beat is looked for within 30 % of its own span (at most 80 ms) of where it would be straight
SUB_CONTRAST = 15.0              # and counts when its attack rises 15 times the window's usual rise (clean hi-hats gave 45 and more,
                                 # a window with nothing in it 9 at most: measured on synthetic takes, 2026-10-13)


def _attacks(y, sr, onsets=None):
    """(onset strength, its times, the rise of the energy above 1 kHz per sample) of a take."""
    import numpy as np
    import scipy.signal as ss
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    o, times = onsets if onsets is not None else _onsets(y, sr)
    x = ss.sosfiltfilt(ss.butter(4, ATTACK_HP_HZ, "highpass", fs=sr, output="sos"), y)
    n = max(1, int(sr * ENVELOPE_S))
    k = np.hanning(n + 2)[1:-1]
    return o, times, np.diff(np.convolve(x * x, k / k.sum(), mode="same"))


def _locate(o, times, rise, sr, centre, half):
    """(time, steepness) of the attack nearest the strongest onset within `half` s of `centre`, or None when nothing
    rises there: first the strongest onset at 3 ms steps, then, in the 12 ms before and 4 ms after that, the sample where
    the energy rises fastest (the earliest rise reaching half the steepest one, so a later spike inside a hit does not win)."""
    import numpy as np
    near = (times >= centre - half) & (times <= centre + half)
    step = float(times[near][np.argmax(o[near])]) if near.any() else centre
    lo = max(int(max(step - 0.012, centre - half) * sr), 0)
    hi = min(int(min(step + 0.004, centre + half) * sr), len(rise))
    w = rise[lo:hi]
    if len(w) == 0 or w.max() <= 0:
        return None
    j = int(np.argmax(w > RISE_SHARE * w.max()))
    while j + 1 < len(w) and w[j + 1] > w[j]:                    # up to the top of that rise
        j += 1
    return (lo + j) / sr, float(w.max())


def refine_beats(y, sr, beats, onsets=None) -> list[float]:
    """Each beat moved onto its attack, within +-30 ms of where it was given (`_locate`).  A beat whose attack is under
    a tenth of the usual one (before the song starts) is dropped."""
    import numpy as np
    o, times, rise = _attacks(y, sr, onsets)
    found = []                                                # (time, how steep its attack is)
    for b in beats:
        hit = _locate(o, times, rise, sr, b, SEARCH_S)
        if hit is None:
            continue
        if not found or hit[0] - found[-1][0] > SEARCH_S:       # two beats never land on one hit
            found.append(hit)
    if not found:
        return []
    usual = float(np.median([steep for _, steep in found]))
    return [t for t, steep in found if steep >= 0.1 * usual]   # a tenth of a usual hit is no hit (before the song)


def refine_subbeats(y, sr, beats, parts=2, onsets=None) -> Grid:
    """The beats with `parts` - 1 points between each two: where the take has an attack near the straight position
    (a hi-hat, a ghost note) the point sits on it, so a swung or loose feel is kept; with no attack there it stays
    straight.  Points stay in order and inside their beat."""
    import numpy as np
    o, times, rise = _attacks(y, sr, onsets)
    grid = Grid()
    grid.parts = parts
    for k, b in enumerate(beats):
        grid.append(float(b))
        if k + 1 == len(beats):
            break
        iv = beats[k + 1] - b
        for j in range(1, parts):
            centre = b + iv * j / parts
            half = min(iv / parts * SUB_WINDOW, 0.08)
            hit = _locate(o, times, rise, sr, centre, half)
            window = np.abs(rise[max(int((centre - half) * sr), 0):int((centre + half) * sr)])
            standing = hit is not None and len(window) > 0 and hit[1] >= SUB_CONTRAST * (float(np.median(window)) + 1e-12)
            t = hit[0] if standing else centre
            floor = grid[-1] + iv * 0.02
            grid.append(float(min(max(t, floor), b + iv - iv * 0.02)))
    return grid


def take_subbeats(take: bytes, beats, parts=2) -> Grid:
    """`refine_subbeats` on the take's own audio (WAV or anything librosa reads)."""
    import librosa
    y, sr = librosa.load(io.BytesIO(take), sr=None, mono=True)
    return refine_subbeats(y, sr, list(beats), parts)


def sync_error(band: bytes, grid) -> dict:
    """How far the band's attacks are from the points of the grid it was built on, in ms (positive: the band is late).
    Each point is looked for within +-30 ms in the band's audio; a point with no attack there is `missed` (a rest, or a
    band that is not on the grid).  beat_* are the beats, sub_* the points between them."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(band), always_2d=True)
    o, times, rise = _attacks(y.mean(axis=1), sr, _onsets(y.mean(axis=1), sr))
    parts = getattr(grid, "parts", 1)
    beat_err, sub_err, missed = [], [], 0
    for i, t in enumerate(grid):
        hit = _locate(o, times, rise, sr, float(t), SEARCH_S)
        if hit is None:
            missed += 1
            continue
        (beat_err if i % parts == 0 else sub_err).append((hit[0] - float(t)) * 1000)

    def stats(errs, name):
        return {f"{name}_median_ms": round(float(np.median(errs)), 3), f"{name}_max_ms": round(float(np.max(np.abs(errs))), 3)} if errs else {}
    return {"beats": len(beat_err), "subbeats": len(sub_err), "missed": missed, **stats(beat_err, "beat"), **stats(sub_err, "sub")}


def take_beats(take: bytes, bpm: float) -> list[float]:
    """Beat times (s) of the YuE2 take itself (its drums carry the beat): librosa's tracker for where the beats
    roughly are, then each one refined onto its attack to the sample (`refine_beats`).  On synthetic takes with known
    beats (tests/test_yuepractice_beats.py) every beat lands within 1 ms; a real take's drums are not that clean."""
    import librosa
    y, sr = librosa.load(io.BytesIO(take), sr=None, mono=True)
    onsets = _onsets(y, sr)
    return refine_beats(y, sr, coarse_beats(y, sr, bpm, onsets), onsets)


def score_beats(beats: list[float], lead_in: float) -> list[float]:
    """The take's beats from the score's beat 0: the grid is extended back by whole beats and beat 0 is the grid
    point nearest the voice's measured lead-in."""
    step = (beats[-1] - beats[0]) / (len(beats) - 1)
    grid, t = list(beats), beats[0]
    while t - step > -step / 2:
        t -= step
        grid.insert(0, round(t, 3))
    k = min(range(len(grid)), key=lambda i: abs(grid[i] - lead_in))
    return grid[k:]


def beat_timing(grid) -> dict | None:
    """Measured quarter-note intervals; subdivisions never masquerade as beats."""
    import math
    beats = list(grid)[::max(1, getattr(grid, "parts", 1))]
    intervals = [b - a for a, b in zip(beats, beats[1:]) if math.isfinite(b - a) and b > a]
    if not intervals:
        return None
    return {"min_s": round(min(intervals), 3), "max_s": round(max(intervals), 3), "intervals": len(intervals)}


def follow_beats(midi: bytes, beats: list[float]) -> bytes:
    """The MIDI with one tempo per beat so that beat k sounds at beats[k] - beats[0] (after the last measured beat
    the last interval goes on).  Notes and ticks stay as they are; only the tempo map changes."""
    import mido
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tpb = mid.ticks_per_beat
    parts = getattr(beats, "parts", 1)                    # a Grid has points between the beats: the tempo changes there too
    if parts < 1 or tpb % parts:
        beats, parts = list(beats)[::max(1, parts)], 1
    step = tpb // parts
    keep, now = [], 0
    for m in mid.tracks[0]:                               # the old tempo map goes, other meta stays where it was
        now += m.time
        if m.type != "set_tempo":
            keep.append((now, m))
    end = max(sum(m.time for m in tr) for tr in mid.tracks)
    for k in range(end // step + 1):
        iv = beats[k + 1] - beats[k] if k + 1 < len(beats) else beats[-1] - beats[-2]
        keep.append((k * step, mido.MetaMessage("set_tempo", tempo=max(1, round(iv * parts * 1e6)), time=0)))   # per quarter note
    keep.sort(key=lambda p: (p[0], p[1].type != "set_tempo", p[1].type == "end_of_track"))
    track, last = mido.MidiTrack(), 0
    for at, m in keep:
        track.append(m.copy(time=at - last))
        last = at
    mid.tracks[0] = track
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _band(sheet_text: str, without: str | None, beats: list[float] | None = None) -> bytes:
    """The app's band for the lead sheet, its melody track always out (the voice is YuE2's), and `without` too.
    `beats`: the take's beats from the score's zero: the band plays on them, nothing humanized, and starts at beats[0]."""
    import mido
    from . import audio, bandpack
    from .render import leadsheet_to_song
    song = leadsheet_to_song(sheet_text)
    midi = bandpack.pick_tracks(base64.b64decode(song["midi_b64"]), without="melody")
    if without:
        midi = bandpack.pick_tracks(midi, without=without)
    if beats:
        midi = follow_beats(midi, beats)
    names = {next((m.name for m in tr if m.type == "track_name"), "") for tr in mido.MidiFile(file=io.BytesIO(midi)).tracks}
    wav, _ = audio.render(midi, "wav", genre=song["genre"], options=set(audio.DEFAULT_ON) - {"section_parts"},
                          exact_parts=frozenset(names - {""}) if beats else frozenset())
    if beats and beats[0] > 0:                            # the score's zero sits at beats[0] in the take: silence before it
        import numpy as np
        import soundfile as sf
        y, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        y = np.concatenate([np.zeros((round(beats[0] * sr), y.shape[1]), dtype="float32"), y])
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="FLOAT")
        wav = buf.getvalue()
    return wav


def _mix(band: bytes, voice: bytes | None, lead_in: float = 0.0) -> bytes:
    """Band plus voice; `lead_in`: seconds the voice starts after the score's zero (cut so it lands on the band's beat)."""
    import numpy as np
    import soundfile as sf
    b, sr = sf.read(io.BytesIO(band), dtype="float32", always_2d=True)
    if voice is not None:
        v, vsr = sf.read(io.BytesIO(voice), dtype="float32", always_2d=True)
        v = v[int(round(max(0.0, lead_in) * vsr)):]
        if vsr != sr:
            import librosa
            v = librosa.resample(v.T, orig_sr=vsr, target_sr=sr).T
        if v.shape[1] != b.shape[1]:
            v = np.repeat(v[:, :1], b.shape[1], axis=1)
        n = max(len(b), len(v))
        b = np.pad(b, ((0, n - len(b)), (0, 0))) + np.pad(v, ((0, n - len(v)), (0, 0)))
    peak = float(np.abs(b).max()) if len(b) else 0.0
    if peak > 0.99:                                       # one gain for the whole track: no clipping, nothing else changes
        b = b * (0.99 / peak)
    buf = io.BytesIO()
    sf.write(buf, b, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def minus(data, song_id: str, part: str, voice=None, band=None) -> tuple[bytes, dict]:
    """(WAV bytes, info) of the library song without `part`.  KeyError: no such song."""
    from . import yueimport, yuelib
    from .leadsheet import serialize_leadsheet
    if part not in PARTS:
        raise PracticeError(f"내 파트는 {', '.join(PARTS)} 중 하나여야 해요 (받은 값 {part!r})")
    voice, band = voice or _voice, band or _band
    row = yuelib.get(data, song_id)
    sheet = serialize_leadsheet(yueimport.read(row["abc"], row.get("sheet_genre") or "팝")[0])
    sung, lead_in = None, 0.0
    if part != "vocal":
        kept = Path(data) / "yue_library" / song_id / "voice.wav"
        if not kept.is_file():
            kept.write_bytes(voice(yuelib.audio(data, song_id)[0]))
        sung = kept.read_bytes()
        # YuE2 sings about a beat after the score's zero (0.35-0.5 s on 2026-10-13); the band starts on it
        lead_in = float(yuecheck.follow(sung, row["abc"], separate=None).get("offset_s") or 0.0)
    tempo = int(re.search(r"Q:1/4=(\d+)", row["abc"]).group(1))
    take = yuelib.audio(data, song_id)[0]
    found = take_beats(take, tempo) if band is _band else []
    if len(found) >= 2:                                   # the band plays on the take's own beats; the voice stays put
        grid = score_beats(found, lead_in)
        points, finer = len(grid), False
        try:                                              # and on its off-beats; a failed analysis leaves the plain beats
            grid, finer = take_subbeats(take, grid), True
        except Exception:
            pass
        out = _mix(band(sheet, TRACK_OF.get(part), grid), sung, 0.0)
        return out, {"part": part, "voice": "yue2" if sung is not None else None, "voice_moved_s": 0.0, "beats": points,
                     "subbeats": finer, "band_starts_s": grid[0], "beat_timing": beat_timing(grid)}
    out = _mix(band(sheet, TRACK_OF.get(part)), sung, lead_in)
    return out, {"part": part, "voice": "yue2" if sung is not None else None, "voice_moved_s": -round(lead_in, 2)}
