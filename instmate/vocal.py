"""Human-like humming guide vocal: a recorded hum (the user's own voice) re-sung on the melody with a WORLD vocoder.

    hum recording (a few seconds of "음~" / "아~") -> WORLD analysis (pyworld, MIT): spectral envelope + aperiodicity
    melody notes -> f0 contour (note pitches, short glides between legato notes, delayed vibrato, slight drift)
    -> WORLD synthesis with the voice's own timbre frames (looped back and forth) and a breath-like amplitude envelope

The voice keeps its formants when the pitch moves (WORLD separates pitch from timbre), so it sounds like the same
person humming the tune.  Everything runs locally; the recording is the user's own and stays in DATA/voice/.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np

FS = 24000                      # synthesis rate (enough for a voice; resampled to 44.1 kHz when mixed)
FRAME_MS = 5.0
GLIDE_S = 0.06                  # legato glide length into the next note
VIB_HZ, VIB_CENTS, VIB_DELAY_S = 5.6, 28.0, 0.28
ATTACK_S, RELEASE_S = 0.045, 0.09
LEGATO_GAP_S = 0.08             # notes closer than this glide into each other instead of breathing


# Option "vocal_polish" (off unless asked for; every number is a design hypothesis judged by ear):
PHRASE_GAP_S = 0.25             # a rest this long ends a phrase when the caller does not give the song's eighth note
XFADE_S = 0.03                  # inside a phrase one note passes into the next over this long (pitch and level)
SLIDE_S = 0.06                  # {"glide": True} (the open vowel): the pitch slides into the next note over this long,
SLIDE_PER_SEMITONE_S = 0.01     # ... longer for a wider step,
SLIDE_MAX_S = 0.14              # ... never longer than this
SLIDE_SHARE = 0.35              # ... nor than this share of the note it arrives on
REPEAT_DIP = 0.85               # a repeated pitch is said again: the level dips to this for a moment, never to silence
REPEAT_SAY = 0.5                # ... and its consonant is this share of a phrase's first one (a full one cut 10 dB and more)
LONG_NOTE_S = 0.8               # a note at least this long gets quieter towards its end and a late vibrato
LONG_FADE_FROM, LONG_FADE_TO = 0.55, 0.5      # the fall starts at this share of the note and ends at this level
VIB_LATE_FROM = 0.5             # on a long note the vibrato starts at this share of the note
LOW_CUT_HZ = 100.0              # the voice is cut below this (12 dB per octave)


ONSET_FRAMES = 26               # 130 ms: the consonant and the opening of the mouth at the start of the recording


class VoiceError(ValueError):
    """The recording cannot be used as a voice (Korean message)."""


def load_voice(wav_bytes: bytes) -> dict:
    """Analyse a hum recording: keep the steady voiced middle (sp/ap frames) and its median pitch."""
    import pyworld as pw
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if sr != FS:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=FS)
    if len(y) < FS * 1.0:
        raise VoiceError("녹음이 너무 짧아요 (1초 이상, 3~5초 권장)")
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    f0, t = pw.harvest(y, FS, frame_period=FRAME_MS, f0_floor=70, f0_ceil=900)
    sp = pw.cheaptrick(y, f0, t, FS)
    ap = pw.d4c(y, f0, t, FS)
    voiced = np.where(f0 > 0)[0]
    if len(voiced) < 100:
        raise VoiceError("목소리(음높이)를 찾지 못했어요. 조용한 곳에서 '음~' 하고 3~5초 길게 녹음해 주세요")
    a, b = voiced[0], voiced[-1]
    lo, hi = a + (b - a) // 5, b - (b - a) // 5          # the steady middle 60%
    keep = [i for i in range(lo, hi) if f0[i] > 0]
    if len(keep) < 60:
        raise VoiceError("길게 이어지는 소리가 부족해요. 끊지 말고 '음~' 하고 3초 이상 유지해 주세요")
    onset = list(range(a, min(a + ONSET_FRAMES, lo)))       # how this voice starts a note (a consonant, if sung)
    return {"sp": sp[keep], "ap": ap[keep], "f0": float(np.median(f0[keep])), "frames": len(keep),
            "onset_sp": sp[onset], "onset_ap": ap[onset]}


def phrasing(notes: list[tuple[float, float, int]], gap_s: float | None = None, breaks=()) -> dict:
    """How the polished voice groups `notes` into phrases:
    {"notes": the notes with every note inside a phrase held until the next one starts (no silence inside a phrase),
     "first": per note, it starts a phrase (it gets the syllable and an attack),
     "last": per note, it ends a phrase (only such a note is released, and only such a long note falls),
     "again": per note, it repeats the pitch before it inside a phrase (said again, with a short dip)}.
    A phrase ends before a rest of `gap_s` or more (the song's eighth note; PHRASE_GAP_S when not given) and at every
    time in `breaks` (seconds: section ends, breathing places)."""
    gap = PHRASE_GAP_S if gap_s is None else gap_s
    n = len(notes)
    first = [i == 0 or notes[i][0] - notes[i - 1][1] >= gap - 1e-6
             or any(notes[i - 1][0] < b <= notes[i][0] + 1e-6 for b in breaks) for i in range(n)]
    last = [i == n - 1 or first[i + 1] for i in range(n)]
    again = [not first[i] and notes[i][2] == notes[i - 1][2] for i in range(n)]
    held = [(s, e if last[i] else max(e, notes[i + 1][0]), p) for i, (s, e, p) in enumerate(notes)]
    return {"notes": held, "first": first, "last": last, "again": again}


def _plan(notes, polish) -> dict | None:
    """The phrasing for `polish`: False -> None, True -> the default phrasing, a phrasing -> itself,
    {"gap_s", "breaks"} -> the phrasing made with them."""
    if not polish:
        return None
    if isinstance(polish, dict):
        if "first" in polish:
            return polish
        plan = phrasing(notes, polish.get("gap_s"), polish.get("breaks") or ())
        if polish.get("glide"):
            plan["glide"] = True                # carried with the phrasing: `sing` plans once and passes the plan on
        return plan
    return phrasing(notes)


def low_cut(y: np.ndarray, sr: int = FS) -> np.ndarray:
    """The voice without what lies under LOW_CUT_HZ (second-order high-pass; scipy comes with librosa)."""
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, LOW_CUT_HZ, "highpass", fs=sr, output="sos"), y)


def f0_contour(notes: list[tuple[float, float, int]], total_s: float, seed: int = 0, polish=False) -> np.ndarray:
    """Frame-wise f0 (Hz, 0 = silent) for notes [(start_s, end_s, midi)].
    `polish` (see `_plan`): notes inside a phrase are held to the next note and pass into it over XFADE_S; the vibrato
    only on long notes and only in their later part (VIB_LATE_FROM)."""
    rng = np.random.default_rng(seed)
    n = int(total_s * 1000 / FRAME_MS) + 1
    f0 = np.zeros(n)
    t = np.arange(n) * FRAME_MS / 1000
    prev = None
    plan = _plan(notes, polish)
    for j, (s, e, p) in enumerate(plan["notes"] if plan else notes):
        i0, i1 = int(s * 1000 / FRAME_MS), min(n, int(e * 1000 / FRAME_MS))
        if i1 <= i0:
            continue
        hz = 440.0 * 2 ** ((p - 69) / 12)
        cents = np.zeros(i1 - i0)
        tt = t[i0:i1] - s
        ramp = np.clip((tt - VIB_DELAY_S) / 0.25, 0, 1)
        if polish:
            late = VIB_LATE_FROM * (e - s)
            ramp = np.clip((tt - late) / 0.3, 0, 1) if e - s >= LONG_NOTE_S else np.zeros(i1 - i0)
        cents += VIB_CENTS * ramp * np.sin(2 * math.pi * VIB_HZ * tt + rng.uniform(0, 2 * math.pi))
        cents += rng.normal(0, 4)                           # each note slightly off the grid, like a person
        if plan:
            if prev is not None and not plan["first"][j] and plan.get("glide"):
                step = 1200 * math.log2(prev[2] / hz)       # an eased slide whose length follows the interval
                span = min(SLIDE_MAX_S, SLIDE_S + SLIDE_PER_SEMITONE_S * abs(step) / 100, SLIDE_SHARE * (e - s))
                span = max(span, XFADE_S)
                g = tt < span
                cents[g] += step * 0.5 * (1 + np.cos(math.pi * tt[g] / span))
            elif prev is not None and not plan["first"][j]:
                g = tt < XFADE_S
                cents[g] += 1200 * math.log2(prev[2] / hz) * (1 - tt[g] / XFADE_S)
        elif prev is not None and s - prev[1] < LEGATO_GAP_S:
            g = tt < GLIDE_S
            cents[g] += 1200 * math.log2(prev[2] / hz) * (1 - tt[g] / GLIDE_S)
        f0[i0:i1] = hz * 2 ** (cents / 1200)
        prev = (s, e, hz)
    return f0


def amp_envelope(notes: list[tuple[float, float, int]], n: int, polish=False) -> np.ndarray:
    """`polish` (see `_plan`): the voice sounds through every join inside a phrase (a repeated pitch dips to
    REPEAT_DIP for a moment); only a phrase's last note is released, and only when it is long does it fall to
    LONG_FADE_TO of its level."""
    env = np.zeros(n)
    plan = _plan(notes, polish)
    for j, (s, e, _) in enumerate(plan["notes"] if plan else notes):
        i0, i1 = int(s * 1000 / FRAME_MS), min(n, int(e * 1000 / FRAME_MS))
        if i1 <= i0:
            continue
        k = np.arange(i1 - i0) * FRAME_MS / 1000
        dur = (i1 - i0) * FRAME_MS / 1000
        a = np.clip(k / ATTACK_S, 0, 1)
        r = np.clip((dur - k) / RELEASE_S, 0, 1)
        if plan:
            dip = lambda x: REPEAT_DIP + (1 - REPEAT_DIP) * np.clip(x / XFADE_S, 0, 1)
            if not plan["first"][j]:
                a = dip(k) if plan["again"][j] else np.ones(i1 - i0)
            if not plan["last"][j]:
                r = dip(dur - k) if plan["again"][j + 1] else np.ones(i1 - i0)
            fall = np.clip((k / dur - LONG_FADE_FROM) / (1 - LONG_FADE_FROM), 0, 1) if plan["last"][j] and dur >= LONG_NOTE_S else 0.0
            env[i0:i1] = np.maximum(env[i0:i1], np.minimum(a, r) * 0.96 * (1 - (1 - LONG_FADE_TO) * fall))
            continue
        env[i0:i1] = np.maximum(env[i0:i1], np.minimum(a, r) * (0.92 + 0.08 * np.cos(k * 1.7)))
    return env


# ---------------------------------------------------------------- syllables ("na na na" / "la la la")
# A consonant is a short change of the mouth shape at the start of every note, applied to the voice's own spectral
# envelope: (closure s, transition s, [(from Hz, to Hz, dB)]).  "n": the mouth is closed and the sound leaves through
# the nose (only the lowest band stays).  "l": the tongue tip touches, the upper formants dip briefly.
SYLLABLES = {
    "na": (0.070, 0.045, [(0, 320, 0.0), (320, 650, -14.0), (650, 12000, -32.0)]),
    "la": (0.045, 0.040, [(0, 420, -3.0), (420, 1100, -9.0), (1100, 3200, -22.0), (3200, 12000, -14.0)]),
}
SYLLABLE_SHARE = 0.4            # a consonant never takes more than this share of its note
STYLES = {"male": {"syllable": "na", "label": "남성 '아~'", "top": 67},        # highest sung note: G4
          "female": {"syllable": "la", "label": "여성 '아~'", "top": 79}}      # G5


def consonant_curve(notes: list[tuple[float, float, int]], n: int, syllable: str, polish=False) -> np.ndarray:
    """Per frame 0..1: 1 while the consonant is held at a note start, falling to 0 as the mouth opens.
    `polish` (see `_plan`): only a phrase's first note and a repeated pitch get the consonant."""
    closure, trans, _ = SYLLABLES[syllable]
    c = np.zeros(n)
    plan = _plan(notes, polish)
    said = [1.0 if plan["first"][i] else REPEAT_SAY if plan["again"][i] else 0.0 for i in range(len(notes))] if plan else [1.0] * len(notes)
    for (s, e, _), say in zip(plan["notes"] if plan else notes, said):
        if not say:
            continue
        k = min(1.0, SYLLABLE_SHARE * (e - s) / (closure + trans))
        i0 = int(s * 1000 / FRAME_MS)
        a, b = int(closure * k * 1000 / FRAME_MS), int(trans * k * 1000 / FRAME_MS)
        for j in range(a + b):
            if 0 <= i0 + j < n:
                c[i0 + j] = max(c[i0 + j], say * (1.0 if j < a else 1.0 - (j - a + 1) / (b + 1)))
    return c


def consonant_gain(syllable: str, bins: int) -> np.ndarray:
    """Power gain per spectral bin while the consonant is fully held."""
    f = np.linspace(0, FS / 2, bins)
    db = np.zeros(bins)
    for lo, hi, g in SYLLABLES[syllable][2]:
        db[(f >= lo) & (f < hi)] = g
    k = max(1, bins // 64)                                   # soften the band edges
    return 10 ** (np.convolve(np.pad(db, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid") / 10)


def fit_register(notes: list[tuple[float, float, int]], style: str) -> list[tuple[float, float, int]]:
    """Move the melody by whole octaves to the highest place where its top note is still within the singer's reach
    (a man usually ends up an octave below the written tune)."""
    if not notes:
        return notes
    top, hi = STYLES[style]["top"], max(p for _, _, p in notes)
    shift = -12 * max(0, -(-(hi - top) // 12)) if hi > top else 12 * ((top - hi) // 12)
    return [(s, e, p + shift) for s, e, p in notes]


# Built-in voices: an /a/ vowel made from formants (no recording, nothing to license), for when the user has not
# recorded a hum.  (formant Hz, bandwidth Hz); female formants sit about 15% higher.
FORMANTS = {"male": [(730, 90), (1090, 100), (2440, 140), (3400, 180), (4500, 240)],
            "female": [(850, 100), (1220, 110), (2810, 160), (3900, 200), (4950, 260)]}
BUILTIN_FRAMES = 8


def builtin_voice(style: str) -> dict:
    """A voice dict like load_voice(): a steady synthetic /a/ for "male" or "female"."""
    import pyworld as pw
    bins = pw.get_cheaptrick_fft_size(FS) // 2 + 1
    f = np.linspace(0, FS / 2, bins)
    amp = 1.0 / np.sqrt(1 + (f / 180.0) ** 2)                 # glottal tilt (with lip radiation: about -6 dB/octave)
    for fk, bw in FORMANTS[style]:
        amp *= fk ** 2 / np.sqrt((fk ** 2 - f ** 2) ** 2 + (bw * f) ** 2)
    sp = (amp / amp.max()) ** 2 * 1e-2 + 1e-12
    ap = np.clip(0.03 + 0.6 * (f / (FS / 2)) ** 1.5, 0, 1)    # a little breath, more of it higher up
    return {"sp": np.tile(sp, (BUILTIN_FRAMES, 1)), "ap": np.tile(ap, (BUILTIN_FRAMES, 1)),
            "f0": 110.0 if style == "male" else 220.0, "frames": BUILTIN_FRAMES, "builtin": style}


ONSET_FADE = 5                  # frames over which the recorded onset blends into the held vowel


def _paste_onsets(sp: np.ndarray, ap: np.ndarray, notes: list[tuple[float, float, int]], voice: dict) -> None:
    """Start every note with the voice's recorded onset frames (squeezed for short notes), blended into the vowel."""
    osp, oap = voice["onset_sp"], voice["onset_ap"]
    for s, e, _ in notes:
        i0 = int(s * 1000 / FRAME_MS)
        k = min(len(osp), max(4, int(SYLLABLE_SHARE * (e - s) * 1000 / FRAME_MS)), len(sp) - i0)
        if k < 2:
            continue
        src = np.round(np.linspace(0, len(osp) - 1, k)).astype(int)
        w = np.ones(k)
        w[-min(ONSET_FADE, k):] = np.linspace(1, 0, min(ONSET_FADE, k) + 1)[:-1]
        sp[i0:i0 + k] = np.exp(w[:, None] * np.log(osp[src] + 1e-16) + (1 - w[:, None]) * np.log(sp[i0:i0 + k] + 1e-16))
        ap[i0:i0 + k] = w[:, None] * oap[src] + (1 - w[:, None]) * ap[i0:i0 + k]


def shift_formants(voice: dict, ratio: float) -> dict:
    """The same voice with every formant moved by `ratio` (0.85: a larger vocal tract, i.e. a man's timbre)."""
    out = dict(voice)
    for key in ("sp", "ap", "onset_sp", "onset_ap"):
        if key in voice and len(voice[key]):
            x = voice[key]
            src = np.clip(np.arange(x.shape[1]) / ratio, 0, x.shape[1] - 1)
            lo = np.floor(src).astype(int)
            hi = np.minimum(lo + 1, x.shape[1] - 1)
            out[key] = np.ascontiguousarray(x[:, lo] * (1 - (src - lo)) + x[:, hi] * (src - lo))
    return out


SYSTEM_VOICE = "Yuna"            # macOS Korean voice used as the default singer (generated on this Mac, never stored)
SYSTEM_TEXT = {"na": "나아아아아아아", "la": "라아아아아아아"}
MALE_FORMANTS = 0.84
_system_cache: dict[str, dict] = {}


def system_voice(style: str) -> dict | None:
    """The default singer of a style from the Mac's own speech voice saying a long "na" / "la" (so the consonant is
    real); the male voice is the same recording with lower formants.  None when `say` or the voice is missing."""
    import shutil
    import subprocess
    import tempfile
    if style in _system_cache:
        return _system_cache[style]
    say = shutil.which("say")
    if say is None:
        return None
    syllable = STYLES[style]["syllable"]
    try:
        with tempfile.TemporaryDirectory(prefix="instmate_voice_") as tmp:
            out = Path(tmp) / "v.aiff"
            subprocess.run([say, "-v", SYSTEM_VOICE, "-r", "70", "-o", str(out), SYSTEM_TEXT[syllable]], check=True,
                           capture_output=True, timeout=30)
            voice = load_voice(out.read_bytes())
    except (OSError, subprocess.SubprocessError, VoiceError, RuntimeError):
        return None
    voice["onset_syllable"] = syllable
    voice["system"] = SYSTEM_VOICE
    if style == "male":
        voice = shift_formants(voice, MALE_FORMANTS)
    _system_cache[style] = voice
    return voice


def sing(notes: list[tuple[float, float, int]], voice: dict, total_s: float, seed: int = 0,
         syllable: str | None = None, polish=False) -> np.ndarray:
    """Mono float32 at FS: the voice humming `notes` [(start_s, end_s, midi)].
    `syllable`: "na" or "la" puts that consonant at the start of every note ("na na na"); None = one long hum.
    `polish` (option "vocal_polish"): True, or {"gap_s": the song's eighth note in seconds, "breaks": [seconds]}.
    A phrase is sung in one breath on one syllable (see `phrasing`), its last note is released, a long last note
    falls, long notes get a late vibrato, and the voice is cut below LOW_CUT_HZ."""
    import pyworld as pw
    if syllable is not None and syllable not in SYLLABLES:
        raise ValueError(f"unknown syllable {syllable!r} (known: {', '.join(SYLLABLES)})")
    polish = _plan(notes, polish) or False
    f0 = f0_contour(notes, total_s, seed, polish)
    n = len(f0)
    m = voice["frames"]
    idx = np.arange(n) % (2 * m - 2) if m > 1 else np.zeros(n, int)
    idx = np.where(idx < m, idx, 2 * m - 2 - idx)            # ping-pong through the steady frames
    sp = voice["sp"][idx].copy()
    ap = voice["ap"][idx].copy()
    env = amp_envelope(notes, n, polish)
    if syllable is not None and voice.get("onset_syllable") == syllable and len(voice.get("onset_sp", ())) >= 8:
        said = [x for i, x in enumerate(polish["notes"]) if polish["first"][i] or polish["again"][i]] if polish else notes
        _paste_onsets(sp, ap, said, voice)                   # the voice's own recorded consonant
    elif syllable is not None:
        c = consonant_curve(notes, n, syllable, polish)
        sp *= consonant_gain(syllable, sp.shape[1])[None, :] ** c[:, None]
    sp *= (env[:, None] ** 2) + 1e-12                         # sp is a power spectrum
    ap[f0 == 0] = 1.0
    y = pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(ap), FS, FRAME_MS)
    if polish:
        y = low_cut(y)
    y[np.repeat(env, int(FS * FRAME_MS / 1000))[: len(y)] < 1e-4] = 0.0
    return (y / (np.abs(y).max() + 1e-9) * 0.8).astype(np.float32)


# ---------------------------------------------------------------- mixing into a rendered song
VOCAL_REL_DB = -1.0             # vocal RMS relative to the band (a lead vocal sits about level with the band)
REVERB_MIX = 0.18


def melody_notes(midi: bytes) -> tuple[list[tuple[float, float, int]], float]:
    """[(start_s, end_s, pitch)] of the 'melody' track and the song length in seconds (single tempo)."""
    import mido
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tempo = next((m.tempo for m in mid.tracks[0] if m.type == "set_tempo"), 500000)
    sec = tempo / 1e6 / mid.ticks_per_beat
    out, open_ = [], {}
    for tr in mid.tracks:
        if not any(m.type == "track_name" and m.name == "melody" for m in tr):
            continue
        now = 0
        for m in tr:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                open_[m.note] = now
            elif m.type in ("note_off", "note_on") and m.note in open_:
                s = open_.pop(m.note)
                out.append((s * sec, now * sec, m.note))
    out.sort()
    return out, mid.length


def strip_track(midi: bytes, name: str) -> bytes:
    import mido
    mid = mido.MidiFile(file=io.BytesIO(midi))
    keep = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for tr in mid.tracks:
        if not any(m.type == "track_name" and m.name == name for m in tr):
            keep.tracks.append(tr)
    buf = io.BytesIO()
    keep.save(file=buf)
    return buf.getvalue()


def _room(y: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """A small room: exponentially decaying noise impulse response (about 0.9 s), mixed in lightly."""
    rng = np.random.default_rng(seed)
    n = int(sr * 0.9)
    ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (sr * 0.18))
    ir[: int(sr * 0.012)] = 0                              # pre-delay
    ir /= np.sqrt((ir ** 2).sum()) + 1e-12
    wet = np.convolve(y, ir)[: len(y)]
    return (1 - REVERB_MIX) * y + REVERB_MIX * wet


def solo_vocal(vocal: np.ndarray, vocal_sr: int = FS, sample_rate: int = 44100) -> bytes:
    """Mono vocal alone -> 16-bit stereo WAV at the band's sample rate, peak at -3 dBFS (practice: the sung part only)."""
    import soundfile as sf
    if vocal_sr != sample_rate:
        import librosa
        vocal = librosa.resample(vocal.astype(np.float32), orig_sr=vocal_sr, target_sr=sample_rate)
    out = _room(vocal.astype(np.float64), sample_rate)
    peak = float(np.abs(out).max())
    if peak > 0:
        out = out * (10 ** (-3 / 20) / peak)
    buf = io.BytesIO()
    sf.write(buf, np.column_stack([out, out]), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def mix_vocal(band_wav: bytes, vocal: np.ndarray, vocal_sr: int = FS) -> bytes:
    """Band WAV + mono vocal -> 16-bit stereo WAV, the vocal set VOCAL_REL_DB against the band RMS."""
    import soundfile as sf
    band, sr = sf.read(io.BytesIO(band_wav), dtype="float32", always_2d=True)
    if vocal_sr != sr:
        import librosa
        vocal = librosa.resample(vocal.astype(np.float32), orig_sr=vocal_sr, target_sr=sr)
    vocal = _room(vocal.astype(np.float64), sr)
    n = max(len(band), len(vocal))
    band = np.pad(band, ((0, n - len(band)), (0, 0)))
    vocal = np.pad(vocal, (0, n - len(vocal)))
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    from . import levellock
    sung = vocal[np.abs(vocal) > 1e-4]
    if len(sung):
        vocal *= levellock.value("vocal_gain", lambda: rms(band[np.abs(vocal) > 1e-4]) * 10 ** (VOCAL_REL_DB / 20) / rms(sung))
    out = band + vocal[:, None]
    factor = levellock.value("vocal_mix_peak", lambda: 0.99 / float(np.abs(out).max()) if float(np.abs(out).max()) > 0.99 else 1.0)
    if factor != 1.0:
        out *= factor
    buf = io.BytesIO()
    sf.write(buf, out, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
