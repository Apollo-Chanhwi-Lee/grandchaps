"""M7: offline audio rendering. Arranged MIDI bytes -> WAV (FluidSynth + a General MIDI SoundFont) -> MP3 (ffmpeg).

    midi_b64 of any result (version, song, composed song) -> render_wav -> (to_mp3)

Runs on the CPU with local programs only; nothing is sent anywhere. Both programs are optional at install time:
  * FluidSynth: the `fluidsynth` command on PATH, or the binary given by INSTMATE_FLUIDSYNTH.
  * SoundFont:  the .sf2 file given by INSTMATE_SOUNDFONT, else the first of DEFAULT_SOUNDFONTS that exists
                (`models/soundfonts/` is git-ignored like every model file). See docs/LICENSES.md for the banks.
  * ffmpeg:     only for MP3 (`ffmpeg` on PATH, or INSTMATE_FFMPEG). Without it WAV still works.
A missing piece raises AudioUnavailable with a Korean message saying what to install; it never crashes the app.
The MIDI already carries the mix (tempo, track levels), so the audio matches the MIDI download.

Audio-only polish (`render`; the MIDI download, sheet and edits stay exactly as arranged):
  * humanize  small seeded timing/velocity variation per instrument and a downbeat accent (`humanize`), deterministic
              for the same MIDI; same-pitch notes never overlap after the shift.
  * placement per-instrument pan, reverb and chorus sends (MIDI CC 10/91/93) and FluidSynth's reverb/chorus on.
  * master    with ffmpeg: low cut, gentle compression, loudness normalised to MASTER_LUFS with a true-peak ceiling.
"""
from __future__ import annotations

import hashlib
import io
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENV_SOUNDFONT = "INSTMATE_SOUNDFONT"
ENV_FLUIDSYNTH = "INSTMATE_FLUIDSYNTH"
ENV_FFMPEG = "INSTMATE_FFMPEG"
DEFAULT_SOUNDFONTS = (ROOT / "models" / "soundfonts" / "GeneralUser-GS.sf2",
                      ROOT / "models" / "soundfonts" / "FluidR3_GM.sf2")
SAMPLE_RATE = 44100
GAIN = 0.5                 # FluidSynth's default 0.2 is quiet for a 5-track arrangement; 0.5 leaves headroom
TIMEOUT_S = 120            # a 4-minute song renders in seconds; this only stops a hung process
MAX_MIDI_BYTES = 2_000_000
MASTER_LUFS, MASTER_TP = -16.0, -1.5   # loudness target (LUFS) and true-peak ceiling (dBTP)
# per-genre tone before compression (genre profile "master"): sub = kick/808 weight (K-pop, hip-hop), bright = vocal
# and guitar presence (J-pop), mid = rock guitars, punch = dance kick, warm = ballad, natural = untouched
MASTER_EQ = {"natural": "", "warm": "bass=g=1:f=150,treble=g=-1:f=9000", "mid": "equalizer=f=1500:t=q:w=1:g=1.5",
             "punch": "bass=g=2:f=80", "sub": "bass=g=3.5:f=60,equalizer=f=350:t=q:w=1:g=-1.5",
             "bright": "treble=g=2:f=4000,equalizer=f=2500:t=q:w=1:g=1"}
# Applied to every master before the genre's own EQ: the SoundFont band piled up between 250 and 500 Hz (46% of a rock
# chorus's energy, 2-6 kHz 15 dB below it, measured 2026-10-06) and sounded boomy.  A wide cut there and a gentle lift
# of the top bring the balance closer to a band recording.  INSTMATE_MASTER_CLARITY=off restores the older sound.
CLARITY_EQ = "equalizer=f=320:t=q:w=0.9:g=-4,equalizer=f=180:t=q:w=1.2:g=-1.5,treble=g=3.5:f=3500"
REVERB = {"room-size": 0.65, "damp": 0.35, "width": 0.9, "level": 0.7}
# per track: pan (0 left, 64 centre, 127 right), reverb send, chorus send, timing sd (ticks), velocity sd
# reverb sends are 40% of the first values: the user compared 100 / 40 / 0 on the same song (2026-10-06) and chose 40
PLACE = {"melody": (64, 14, 12, 6, 5), "piano": (82, 18, 10, 9, 7), "bass": (64, 5, 0, 5, 4),
         "guitar": (40, 12, 12, 8, 6), "strings": (64, 32, 35, 14, 5), "drums": (64, 11, 0, 4, 8),
         "keys": (58, 18, 8, 6, 6), "lead": (74, 20, 20, 7, 6), "harmony": (52, 16, 14, 6, 5)}
TIMING_CLIP = 2.5            # timing offsets are cut at 2.5 sd
# channel volume (CC 7) per track: the melody is the (humming guide) vocal and sits on top like a lead vocal; dense
# parts step back (measured with work/audio_q/stem_level.py: the melody was 6-7 dB under the band in rock)
VOLUME = {"melody": 127, "lead": 112, "bass": 104, "drums": 104, "piano": 92, "keys": 96, "guitar": 86, "strings": 84,
          "harmony": 104}
MELODY_BOOST = 10            # velocity added to melody notes (soft-attack voice patches need it to speak)
# Selectable sounds per part (General MIDI programs of the SoundFont), for new age pieces, covers and band demos.
# key -> (Korean label, program).  Drums keep the standard kit.  Not given = the genre's default program.
SOUNDS = {
    "melody": {"voice": ("가이드 보컬", 53), "piano": ("피아노", 0), "violin": ("바이올린", 40), "cello": ("첼로", 42),
               "flute": ("플루트", 73), "acoustic_guitar": ("어쿠스틱 기타", 25), "electric_guitar": ("일렉 기타", 27),
               "sax": ("색소폰", 65), "synth": ("신스 리드", 80)},
    "piano": {"grand": ("그랜드 피아노", 0), "electric": ("일렉 피아노", 4), "organ": ("오르간", 16), "harp": ("하프", 46)},
    "keys": {"grand": ("그랜드 피아노", 0), "bright": ("브라이트 피아노", 1), "electric": ("일렉 피아노", 4),
             "music_box": ("뮤직박스", 10), "vibraphone": ("비브라폰", 11)},
    "guitar": {"acoustic": ("어쿠스틱", 25), "nylon": ("나일론", 24), "clean": ("클린 일렉", 27), "jazz": ("재즈 기타", 26),
               "overdrive": ("오버드라이브", 29), "distortion": ("디스토션", 30), "muted": ("뮤트", 28)},
    "bass": {"finger": ("핑거 베이스", 33), "pick": ("피크 베이스", 34), "acoustic": ("어쿠스틱 베이스", 32),
             "fretless": ("프렛리스 베이스", 35), "slap": ("슬랩 베이스", 36),
             "synth": ("신스 베이스", 38)},
    "strings": {"ensemble": ("현악 합주", 48), "slow": ("느린 현악", 49), "violin": ("바이올린", 40), "viola": ("비올라", 41),
                "cello": ("첼로", 42), "contrabass": ("콘트라베이스", 43), "harp": ("하프", 46), "pizzicato": ("피치카토", 45)},
    # drum kits: GS kit numbers on the drum channel (GeneralUser GS ships these kits)
    "drums": {"standard": ("스탠다드 드럼", 0), "room": ("룸 드럼", 8), "power": ("파워 드럼", 16),
              "electronic": ("일렉트로닉 드럼", 24), "tr808": ("808 드럼머신", 25), "jazz": ("재즈 드럼", 32),
              "brush": ("브러시 드럼", 40), "orchestra": ("오케스트라 타악기", 48)},
}


def check_sounds(sounds: dict | None) -> dict[str, int]:
    """{track: sound key} -> {track: program}.  Raises ValueError naming the bad track or key."""
    out = {}
    for track, key in (sounds or {}).items():
        if track not in SOUNDS:
            raise ValueError(f"소리를 바꿀 수 없는 파트: {track!r} (가능: {', '.join(SOUNDS)})")
        if key in SOUNDS[track]:
            out[track] = SOUNDS[track][key][1]
            continue
        inst = _chosen_instrument(track, key)        # an instrument of configs/instruments.yaml offered for this part
        if inst is None:
            raise ValueError(f"{track}에 없는 소리: {key!r} (가능: {', '.join(SOUNDS[track])})")
        out[track] = SOUNDS[track][inst.get("fallback", inst["sound"])][1]    # what a MIDI file can name
    return out


def _chosen_instrument(track: str, key: str, mode: str | None = None) -> dict | None:
    """The table instrument a user picked for a part (configs/instruments.yaml `choices`), or None.  In release mode a
    tier-B choice comes back as its tier-A replacement (instruments.resolve)."""
    from . import instruments
    cfg = instruments.load()
    if key not in cfg.get("choices", {}).get(track, []):
        return None
    return instruments.resolve(key, mode)


def set_sounds(midi: bytes, sounds: dict | None) -> bytes:
    """Replace (or add) each named track's program change.  Unknown tracks/keys raise ValueError (see check_sounds)."""
    return _set_programs(midi, check_sounds(sounds))


def _set_programs(midi: bytes, programs: dict[str, int]) -> bytes:
    """Replace (or add) the program change of each named track: {track: program}."""
    if not programs:
        return midi
    mid = mido.MidiFile(file=io.BytesIO(midi))
    for track in mid.tracks:
        name = next((m.name for m in track if m.type == "track_name"), None)
        if name not in programs:
            continue
        ch = next((m.channel for m in track if hasattr(m, "channel")), None)
        if ch is None:
            continue
        if any(m.type == "program_change" and m.time for m in track):   # removing it would shift later events
            raise ValueError(f"track {name}: program change after time 0 is not supported")
        track[:] = [m for m in track if m.type != "program_change"]
        pos = next(i for i, m in enumerate(track) if m.type == "track_name") + 1
        track.insert(pos, mido.Message("program_change", channel=ch, program=programs[name], time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


ACCENT = 4                   # velocity added to non-drum notes on a bar's first beat


class AudioUnavailable(Exception):
    """FluidSynth, the SoundFont or (for MP3) ffmpeg is missing. The message says what to install."""


class AudioRenderError(Exception):
    """A renderer ran but failed (or the input could not be used, see BadMidi)."""


class BadMidi(AudioRenderError):
    """The input is not a usable MIDI file (the caller's fault, HTTP 400)."""


def _program(env: str, name: str) -> str | None:
    """Absolute path of a program: the env var (a path or a command name) or `name` on PATH."""
    given = os.environ.get(env)
    if given:
        return shutil.which(given) or (given if Path(given).is_file() and os.access(given, os.X_OK) else None)
    return shutil.which(name)


def fluidsynth_path() -> str | None:
    return _program(ENV_FLUIDSYNTH, "fluidsynth")


def ffmpeg_path() -> str | None:
    return _program(ENV_FFMPEG, "ffmpeg")


def soundfont_path() -> Path | None:
    given = os.environ.get(ENV_SOUNDFONT)
    if given:                                  # an explicit setting is never silently replaced by a default
        p = Path(given).expanduser()
        return p if p.is_file() else None
    return next((p for p in DEFAULT_SOUNDFONTS if p.is_file()), None)


ENV_PIANO = "INSTMATE_PIANO_SOUNDFONT"
PIANO_SOUNDFONTS = sorted((ROOT / "models" / "soundfonts").glob("SalamanderGrandPiano*.sf2"))


def piano_soundfont_path() -> Path | None:
    """A sampled grand piano loaded on top of the GM bank (FluidSynth prefers the last loaded font for the same
    program, so GM program 0 plays it). INSTMATE_PIANO_SOUNDFONT names one ("off" disables); by default the Salamander
    Grand Piano in models/soundfonts is used only together with the default GM bank (not with INSTMATE_SOUNDFONT)."""
    given = os.environ.get(ENV_PIANO)
    if given:
        if given.lower() == "off":
            return None
        p = Path(given).expanduser()
        return p if p.is_file() else None
    if os.environ.get(ENV_SOUNDFONT):
        return None
    return PIANO_SOUNDFONTS[-1] if PIANO_SOUNDFONTS else None


def unavailable_reason() -> str | None:
    """None when WAV rendering can run, else why not (Korean, for the UI)."""
    if fluidsynth_path() is None:
        hint = f" ({ENV_FLUIDSYNTH}={os.environ[ENV_FLUIDSYNTH]} 실행 파일을 찾지 못함)" if os.environ.get(ENV_FLUIDSYNTH) else ""
        return f"FluidSynth가 설치되어 있지 않아요{hint}. 설치: macOS `brew install fluid-synth`, Ubuntu `apt install fluidsynth`"
    if soundfont_path() is None:
        if os.environ.get(ENV_SOUNDFONT):
            return f"사운드폰트 파일이 없어요: {ENV_SOUNDFONT}={os.environ[ENV_SOUNDFONT]}"
        return (f"사운드폰트(.sf2)가 없어요. GeneralUser GS 또는 FluidR3_GM을 받아 {ENV_SOUNDFONT}에 경로를 지정하거나 "
                f"models/soundfonts/에 두세요 (docs/LICENSES.md)")
    return None


def status() -> dict:
    reason = unavailable_reason()
    sf = soundfont_path()
    piano = piano_soundfont_path()
    return {"available": reason is None, "reason": reason, "soundfont": sf.name if sf else None,
            "piano": piano.name if piano else None,
            "mp3": reason is None and ffmpeg_path() is not None}


def _check_midi(midi: bytes) -> None:
    if not midi:
        raise BadMidi("MIDI 데이터가 비어 있어요")
    if len(midi) > MAX_MIDI_BYTES:
        raise BadMidi(f"MIDI가 너무 커요 ({len(midi)} bytes, 최대 {MAX_MIDI_BYTES})")
    try:
        mid = mido.MidiFile(file=io.BytesIO(midi))
    except Exception as e:                     # mido raises several types (OSError, EOFError, ValueError, KeyError)
        raise BadMidi(f"MIDI 파일을 읽지 못했어요: {type(e).__name__}: {e}") from None
    if not any(msg.type == "note_on" and msg.velocity > 0 for tr in mid.tracks for msg in tr):
        raise BadMidi("MIDI에 음표가 없어요")


def _run(cmd: list[str], what: str) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise AudioRenderError(f"{what}가 {TIMEOUT_S}초 안에 끝나지 않았어요") from None
    except OSError as e:                       # e.g. the binary vanished or is not executable
        raise AudioUnavailable(f"{what}를 실행하지 못했어요: {e}") from None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise AudioRenderError(f"{what} 실패 (종료 코드 {proc.returncode}): {tail}")


def render_wav(midi: bytes, sample_rate: int = SAMPLE_RATE, reverb: bool = True) -> bytes:
    """MIDI bytes -> 16-bit stereo WAV bytes. Raises AudioUnavailable or AudioRenderError.
    `reverb=False`: no reverb or chorus (a part that goes through an effect chain gets its room afterwards)."""
    reason = unavailable_reason()
    if reason:
        raise AudioUnavailable(reason)
    _check_midi(midi)
    fs, sf = fluidsynth_path(), soundfont_path()
    with tempfile.TemporaryDirectory(prefix="instmate_audio_") as tmp:
        src, out = Path(tmp) / "in.mid", Path(tmp) / "out.wav"
        src.write_bytes(midi)
        # -n no MIDI input driver, -i no shell, -q quiet; options must come before the SoundFont and the MIDI file
        opts = [x for k, v in REVERB.items() for x in ("-o", f"synth.reverb.{k}={v}")]
        piano = piano_soundfont_path()
        fonts = [str(sf)] + ([str(piano)] if piano else [])
        _run([fs, "-n", "-i", "-q", "-R", "1" if reverb else "0", "-C", "1" if reverb else "0", *opts, "-F", str(out), "-T", "wav", "-O", "s16",
              "-r", str(sample_rate), "-g", str(GAIN), *fonts, str(src)], "FluidSynth")
        data = out.read_bytes() if out.is_file() else b""
    if len(data) <= 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioRenderError("FluidSynth가 WAV 파일을 만들지 않았어요")
    return data


# how far each part sits behind the drums (ms): a band does not land on one sample; the bass and the pads lay back
FEEL_MS = {"bass": 7.0, "piano": 4.0, "keys": 4.0, "strings": 12.0}
STRUMMED = ("guitar",)       # chords of these parts are strummed: the strings sound one after another
STRUM_MS = (5.0, 15.0)       # delay from one string to the next
STRUM_MAX_MS = 60.0          # a whole stroke never takes longer (fast songs, six-string chords)
STRUM_MIN_NOTES = 3
UPSTROKE_DV = -6             # up-strokes (off the beat) are lighter


def _strum_offsets(notes: list, rng: random.Random, tpq: int, ms_ticks: float) -> dict[int, tuple[int, int, int]]:
    """{note index: (chord timing offset, extra delay of this string, velocity change)} for every chord (3+ notes
    starting together).  On the beat = down-stroke (low string first), off the beat = up-stroke (high string first,
    lighter); each string follows the one before by 5-15 ms and is hit a little differently."""
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(notes):
        groups.setdefault(n[0], []).append(i)
    out = {}
    for on in sorted(groups):
        idx = groups[on]
        if len(idx) < STRUM_MIN_NOTES:
            continue
        down = on % tpq == 0
        idx.sort(key=lambda i: notes[i][3], reverse=not down)
        gaps = [rng.uniform(*STRUM_MS) for _ in idx[1:]]
        total = sum(gaps)
        if total > STRUM_MAX_MS:
            gaps = [g * STRUM_MAX_MS / total for g in gaps]
        dt = round(max(-TIMING_CLIP, min(TIMING_CLIP, rng.gauss(0, 1))) * PLACE["guitar"][3] / 2)
        late = 0.0
        for k, i in enumerate(idx):
            if k:
                late += gaps[k - 1]
            dv = round(rng.gauss(0, PLACE["guitar"][4]) - 1.5 * k) + (0 if down else UPSTROKE_DV)
            out[i] = (dt, max(k, round(late * ms_ticks)), dv)      # at least one tick apart, in string order
    return out


def humanize(midi: bytes, exact: frozenset[str] | set[str] = frozenset()) -> bytes:
    """Arranged MIDI -> the same notes played a little less mechanically (see module doc). Deterministic.
    `exact`: tracks a person already played (a member's own part): their timing and velocities are left as they are;
    only the mix settings (volume, pan, reverb send) are added."""
    mid = mido.MidiFile(file=io.BytesIO(midi))
    rng = random.Random(hashlib.sha256(midi).hexdigest())
    tpq = mid.ticks_per_beat
    bar = 4 * tpq
    for msg in mid.tracks[0]:
        if msg.type == "time_signature":
            bar = msg.numerator * tpq * 4 // msg.denominator
            break
    tempo = next((m.tempo for m in mid.tracks[0] if m.type == "set_tempo"), 500000)
    ms_ticks = tpq * 1000.0 / tempo                 # ticks per millisecond
    out = mido.MidiFile(ticks_per_beat=tpq, type=mid.type)
    for track in mid.tracks:
        name = next((m.name for m in track if m.type == "track_name"), None)
        if name not in PLACE:
            out.tracks.append(track)
            continue
        pan, rev, cho, t_sd, v_sd = PLACE[name]
        now, other, notes, open_ = 0, [], [], {}
        channel = 9 if name == "drums" else 0
        for m in track:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                channel = m.channel
                open_.setdefault((m.channel, m.note), []).append(len(notes))
                notes.append([now, None, m.channel, m.note, m.velocity])
            elif m.type in ("note_off", "note_on"):
                stack = open_.get((m.channel, m.note))
                if not stack:
                    raise ValueError(f"track {name}: note_off without note_on (note {m.note} at {now})")
                notes[stack.pop(0)][1] = now
            elif m.type != "end_of_track":
                other.append((now, m))
        if any(n[1] is None for n in notes):
            raise ValueError(f"track {name}: a note never ends")
        strum = _strum_offsets(notes, rng, tpq, ms_ticks) if name in STRUMMED and name not in exact else {}
        feel = 0 if name in exact else round(FEEL_MS.get(name, 0.0) * ms_ticks)
        moved = []
        for i, (on, off, ch, pitch, vel) in enumerate(notes):
            if name in exact:
                moved.append([on, off, ch, pitch, vel])
                continue
            if i in strum:                          # one stroke: the chord moves together, the strings follow in order
                dt, late, dv = strum[i]
            else:
                dt = round(max(-TIMING_CLIP, min(TIMING_CLIP, rng.gauss(0, 1))) * t_sd)
                late, dv = 0, round(rng.gauss(0, v_sd))
            dv += (ACCENT if name != "drums" and on % bar == 0 else 0) + (MELODY_BOOST if name == "melody" else 0)
            new_on = max(0, on + dt) + late + feel
            moved.append([new_on, max(new_on + 1, off + dt + feel), ch, pitch, max(1, min(127, vel + dv))])
        by_key: dict[tuple, list] = {}
        for n in moved:
            by_key.setdefault((n[2], n[3]), []).append(n)
        for seq in by_key.values():                 # a repeated pitch must end before it starts again
            seq.sort(key=lambda n: n[0])
            for a, b in zip(seq, seq[1:]):
                if b[0] <= a[0]:
                    b[0] = a[0] + 1
                    b[1] = max(b[1], b[0] + 1)
                a[1] = min(a[1], b[0])
        msgs = [(t, 0, m) for t, m in other]
        msgs += [(0, 0, mido.Message("control_change", channel=channel, control=c, value=v, time=0))
                 for c, v in ((7, VOLUME.get(name, 100)), (10, pan), (91, rev), (93, cho))]
        for on, off, ch, pitch, vel in moved:
            msgs.append((on, 2, mido.Message("note_on", channel=ch, note=pitch, velocity=vel)))
            msgs.append((off, 1, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
        msgs.sort(key=lambda x: (x[0], x[1]))
        t = mido.MidiTrack()
        last = 0
        for tick, _, m in msgs:
            t.append(m.copy(time=tick - last))
            last = tick
        out.tracks.append(t)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def swing(midi: bytes, amount: float, exact: frozenset[str] | set[str] = frozenset()) -> bytes:
    """Swing the off-beat eighths: inside every beat time is warped so the "and" moves later by amount x half a beat
    (0.33 = triplet feel), on and off times alike; on-beat events stay put.  amount 0 returns the input.
    `exact`: tracks a person played or edited: they keep their own timing (a member who wants swing plays it)."""
    if not 0 <= amount <= 0.5:
        raise ValueError(f"swing must be 0-0.5, got {amount}")
    if amount == 0:
        return midi
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tpq = mid.ticks_per_beat
    half = tpq / 2
    shift = amount * half

    def warp(t: int) -> int:
        beat, x = divmod(t, tpq)
        y = x * (half + shift) / half if x <= half else half + shift + (x - half) * (tpq - half - shift) / (tpq - half)
        return beat * tpq + round(y)

    out = mido.MidiFile(ticks_per_beat=tpq, type=mid.type)
    for track in mid.tracks:
        if next((m.name for m in track if m.type == "track_name"), None) in exact:
            out.tracks.append(track)
            continue
        now, absmsgs = 0, []
        for m in track:
            now += m.time
            absmsgs.append((warp(now), m))
        t = mido.MidiTrack()
        last = 0
        for tick, m in absmsgs:                  # warp is monotonic, so the order is kept
            t.append(m.copy(time=tick - last))
            last = tick
        out.tracks.append(t)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def prepare(midi: bytes, swing_amount: float, exact: frozenset[str] = frozenset()) -> bytes:
    """What render() does to the notes before synthesis: swing, then humanize.  Tracks in `exact` (members' own
    parts) come out with the same note starts, lengths and velocities they went in with."""
    return humanize(swing(midi, swing_amount, exact), exact)


def select_tracks(midi: bytes, names: set[str], keep: bool) -> bytes:
    """MIDI with only (keep=True) or without (keep=False) the named tracks; the tempo track always stays."""
    mid = mido.MidiFile(file=io.BytesIO(midi))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for i, tr in enumerate(mid.tracks):
        name = next((m.name for m in tr if m.type == "track_name"), None)
        if i == 0 or ((name in names) == keep):
            out.tracks.append(tr)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def _track_names(midi: bytes) -> set[str]:
    mid = mido.MidiFile(file=io.BytesIO(midi))
    return {m.name for tr in mid.tracks for m in tr if m.type == "track_name"}


def _bar_seconds(midi: bytes) -> float:
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tempo = next((m.tempo for m in mid.tracks[0] if m.type == "set_tempo"), 500000)
    num, den = next(((m.numerator, m.denominator) for m in mid.tracks[0] if m.type == "time_signature"), (4, 4))
    return num * 4 / den * tempo / 1e6


def _program_of(part: str, key: str) -> int:
    from .instruments import SOUND_TABLE
    return SOUNDS[SOUND_TABLE.get(part, part)][key][1]


def _stem(wav: bytes):
    import soundfile as sf
    data, _ = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    return data.T


def _sfz_stem(solo: bytes, sfz: Path) -> np.ndarray:
    """(2, n) audio of a one-part MIDI file played by an SFZ sample instrument (sfizz_render)."""
    import shutil
    import soundfile as sf
    exe = shutil.which("sfizz_render") or str(Path.home() / ".local/bin/sfizz_render")
    with tempfile.TemporaryDirectory(prefix="instmate_sfz_") as tmp:
        mid, out = Path(tmp) / "part.mid", Path(tmp) / "part.wav"
        mid.write_bytes(solo)
        _run([exe, "--sfz", str(sfz), "--midi", str(mid), "--wav", str(out), "--samplerate", str(SAMPLE_RATE)], "sfizz_render")
        if not out.is_file():
            raise AudioRenderError("sfizz_render가 WAV를 만들지 않았어요")
        data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    if float(np.abs(data).max()) < 1e-6:
        raise AudioRenderError("샘플 악기가 소리를 내지 않았어요")
    return data.T


def _variant_stem(played: bytes, part: str, inst: dict, entry: dict | None, notes: list[str], mode: str | None = None):
    """(2, n) audio of one part played as `inst`: its plugin, its SFZ samples, or its SoundFont sound through its effect
    chain, or (plugin / samples missing, effects unavailable, any of them failing) the SoundFont fallback.  Reasons go
    to `notes`."""
    from . import instruments, vst
    solo = select_tracks(played, {part}, keep=True)
    sfz, sfz_reason = instruments.sfz_entry(inst)
    if sfz_reason:
        notes.append(f"{part}: {sfz_reason}, 사운드폰트로 대체")
    if sfz is not None:
        try:
            return _sfz_stem(solo, sfz)
        except Exception as e:
            notes.append(f"{part}: 샘플 악기 실패 ({str(e)[:80]}), 사운드폰트로 대체")
    if entry is not None:
        try:
            return vst.render_stem(played, part, entry)
        except Exception as e:                   # plugin missing or broken: keep the SoundFont part
            notes.append(f"{part}: plugin failed ({str(e)[:80]}), SoundFont used")
    if "fx" in inst:
        if instruments.fx_available():
            try:
                dry = _stem(render_wav(_set_programs(solo, {part: _program_of(part, inst["sound"])}), reverb=False))
                return instruments.apply_fx(dry, SAMPLE_RATE, inst["fx"], mode, notes)
            except (AudioUnavailable, AudioRenderError):
                raise
            except Exception as e:
                notes.append(f"{part}: effect chain failed ({str(e)[:80]}), SoundFont {inst['fallback']} used")
        else:
            notes.append(f"{part}: pedalboard not installed, SoundFont {inst['fallback']} used")
        return _stem(render_wav(_set_programs(solo, {part: _program_of(part, inst["fallback"])})))
    return _stem(render_wav(_set_programs(solo, {part: _program_of(part, inst["sound"])})))


def _render_parts(played: bytes, sing_melody: bool = False, genre: str | None = None, user_parts: set[str] = frozenset(),
                  sections: list[dict] | None = None, notes: list[str] | None = None,
                  overrides: dict[str, dict] | None = None, mode: str | None = None) -> bytes:
    """FluidSynth for every part, except the parts that need their own stem: a plugin (instruments table, or
    models/vst/tracks.json), an effect chain, or a guitar tone that changes with the section.  Each stem is
    level-matched to the SoundFont version of the part.  A part the user changed on screen (`user_parts`) is never
    replaced; a failing plugin or effect falls back to the SoundFont (reason appended to `notes`).
    `sections`: [{"kind", "start_bar", "bars"}] of the song, needed only for the per-section guitar tone."""
    from . import instruments, vst
    notes = notes if notes is not None else []
    skip = {"melody"} if sing_melody else set()
    tracks = _track_names(played)
    manual = vst.config()
    if manual and not instruments.allowed("B", mode):     # a hand-picked plugin has no recorded licence: tier B
        notes.extend(f"{p}: 배포 모드라 직접 고른 플러그인 대신 기본 편성 사용" for p in sorted(set(manual) & tracks))
        manual = {}
    lu = instruments.lineup(genre, mode) if genre else None
    if overrides:                                # gear the user picked: it plays in every section of that part
        lu = {"parts": {**(lu["parts"] if lu else {}), **overrides}, "sections": lu["sections"] if lu else {}}
    plans: dict[str, list[tuple[dict, dict | None, set[str]]]] = {}    # part -> [(instrument, plugin entry, kinds)]
    for part in sorted(tracks - skip - set(user_parts)):
        if part in manual:
            plans[part] = [({"id": "manual", "sound": None}, manual[part], set())]
            continue
        if lu is None or part not in lu["parts"]:
            continue
        base = lu["parts"][part]
        variants = [(base, set())]
        if part == "guitar" and sections and part not in (overrides or {}):
            for kind, sec in lu["sections"].items():
                alt = sec.get("guitar")
                if alt is None or alt["id"] == base["id"] or not any(s["kind"] == kind for s in sections):
                    continue
                hit = next((v for v in variants if v[0]["id"] == alt["id"]), None)
                if hit:
                    hit[1].add(kind)
                else:
                    variants.append((alt, {kind}))
        plan = []
        for inst, kinds in variants:
            entry, reason = instruments.plugin_entry(inst)
            if reason:
                notes.append(f"{part}: {reason}, 사운드폰트로 대체")
            if inst.get("replaced"):
                notes.append(f"{part}: 배포 모드라 {inst['replaced']} 대신 {inst['label']} 사용")
            plan.append((inst, entry, kinds))
        if len(plan) > 1 or plan[0][1] is not None or "fx" in base or "sfz" in base:
            plans[part] = plan
    if not plans:
        return render_wav(select_tracks(played, skip, keep=False) if skip else played)
    try:
        base_wav = render_wav(select_tracks(played, set(plans) | skip, keep=False))
    except BadMidi:                              # every sounding part has its own stem: start from silence
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros((SAMPLE_RATE // 10, 2), dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        base_wav = buf.getvalue()
    stems = []
    for part, plan in plans.items():
        from . import levellock
        target = levellock.value("stem_target", lambda: vst.rms(_stem(render_wav(select_tracks(played, {part}, keep=True)))))
        if plan[0][0]["id"] == "manual":
            try:
                stems.append((vst.render_stem(played, part, plan[0][1]), target))
            except Exception as e:
                notes.append(f"{part}: plugin failed ({str(e)[:80]}), SoundFont used")
                stems.append((_stem(render_wav(select_tracks(played, {part}, keep=True))), target))
            continue
        rendered = [_variant_stem(played, part, inst, entry, notes, mode) for inst, entry, _ in plan]
        if len(rendered) == 1:
            stems.append((rendered[0], target))
            continue
        n = max(r.shape[1] for r in rendered)
        bar_s = _bar_seconds(played)
        spans = [(s["start_bar"] * bar_s, (s["start_bar"] + s["bars"]) * bar_s, s["kind"]) for s in sections]
        gains = instruments.section_gains(spans, {inst["id"]: kinds for inst, _, kinds in plan[1:]},
                                          plan[0][0]["id"], n, SAMPLE_RATE)
        mixed = np.zeros((2, n), dtype=np.float32)
        for (inst, _, _), r in zip(plan, rendered):
            from . import levellock
            r = np.pad(r, ((0, 0), (0, n - r.shape[1]))) * levellock.value("variant_scale", lambda: target / vst.rms(r))
            mixed += (r * gains[inst["id"]]).astype(np.float32)
        stems.append((mixed, 0.0))                 # already at the part's level
    return vst.mix(base_wav, stems)


def _mix_by_part(band: bytes, sing_melody: bool, mixer=None, **kw) -> bytes:
    """Options "master_v2" / "mix_v3": every part rendered alone (the same way as in the plain mix: plugin, effect
    chain or SoundFont), given its own tone and place (`mixer`: instmate/mixv2.py or mixv3.py) and summed."""
    import soundfile as sf
    from . import mixv2
    mixv2 = mixer or mixv2
    parts = sorted(_track_names(band) - ({"melody"} if sing_melody else set()))
    if not parts:
        raise BadMidi("no part to render")
    stems = {p: _stem(_render_parts(select_tracks(band, {p}, keep=True), sing_melody=False, **kw)) for p in parts}
    buf = io.BytesIO()
    sf.write(buf, mixv2.mixdown(stems, SAMPLE_RATE).T, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def master_v2(wav: bytes, eq: str = "natural") -> bytes:
    """Option "master_v2": low cut, the same clarity and genre EQ as the plain master, a firmer
    compressor, one gain towards mixv2.LUFS and a limiter under mixv2.TRUE_PEAK.  The loudness lands a little under
    the target when the limiter has to work."""
    from . import mixv2
    ff = ffmpeg_path()
    if ff is None:
        raise AudioUnavailable("마스터링에는 ffmpeg가 필요해요 (macOS `brew install ffmpeg`)")
    if eq not in MASTER_EQ:
        raise ValueError(f"unknown master EQ {eq!r} (known: {', '.join(MASTER_EQ)})")
    clarity = "" if os.environ.get("INSTMATE_MASTER_CLARITY", "").lower() == "off" else CLARITY_EQ
    pre = ",".join(x for x in ("highpass=f=30", clarity, MASTER_EQ[eq], mixv2.COMPRESSOR) if x)
    with tempfile.TemporaryDirectory(prefix="instmate_audio_") as tmp:
        src, out = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        src.write_bytes(wav)
        measured = _measure_loudness(ff, src, f"{pre},loudnorm=I={mixv2.LUFS}:TP={mixv2.TRUE_PEAK}:LRA=20:print_format=json")
        gain = mixv2.LUFS - float(measured["input_i"])
        limit = 10 ** (mixv2.LIMIT_DB / 20)
        _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
              "-af", f"{pre},volume={gain:.2f}dB,alimiter=limit={limit:.4f}:level=false:attack=5:release=80",
              "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)], "ffmpeg")
        # the limiter works on samples; a peak between two samples can still pass the ceiling: measure, then turn down
        over = float(_measure_loudness(ff, out, "loudnorm=print_format=json")["input_tp"]) - mixv2.TRUE_PEAK
        if over > 0:
            fixed = Path(tmp) / "fixed.wav"
            _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(out),
                  "-af", f"volume={-(over + 0.1):.2f}dB", "-c:a", "pcm_s16le", str(fixed)], "ffmpeg")
            out = fixed
        data = out.read_bytes() if out.is_file() else b""
    if len(data) <= 44 or data[:4] != b"RIFF":
        raise AudioRenderError("ffmpeg가 마스터링한 WAV를 만들지 않았어요")
    return data


def relevel(wav: bytes, lufs: float | None = None) -> bytes:
    """Option "section_dynamics", last step: the finished song brought back up after its quiet sections were turned
    down.  One gain towards mixv2.LUFS and a limiter under mixv2.TRUE_PEAK; no EQ and no compressor, so the contrast
    between the sections is kept except where the limiter has to hold the loudest peaks."""
    from . import mixv2
    goal = mixv2.LUFS if lufs is None else lufs          # "mix_v3" gives its own loudness
    ff = ffmpeg_path()
    if ff is None:
        return wav
    with tempfile.TemporaryDirectory(prefix="instmate_audio_") as tmp:
        src, out = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        src.write_bytes(wav)
        gain = goal - float(_measure_loudness(ff, src, "loudnorm=print_format=json")["input_i"])
        limit = 10 ** (mixv2.LIMIT_DB / 20)
        _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
              "-af", f"volume={gain:.2f}dB,alimiter=limit={limit:.4f}:level=false:attack=5:release=80",
              "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)], "ffmpeg")
        over = float(_measure_loudness(ff, out, "loudnorm=print_format=json")["input_tp"]) - mixv2.TRUE_PEAK
        if over > 0:
            fixed = Path(tmp) / "fixed.wav"
            _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(out),
                  "-af", f"volume={-(over + 0.1):.2f}dB", "-c:a", "pcm_s16le", str(fixed)], "ffmpeg")
            out = fixed
        data = out.read_bytes() if out.is_file() else b""
    if len(data) <= 44 or data[:4] != b"RIFF":
        raise AudioRenderError("ffmpeg가 음량을 맞춘 WAV를 만들지 않았어요")
    return data


def master(wav: bytes, eq: str = "natural") -> bytes:
    """WAV -> mastered 16-bit WAV (ffmpeg): low cut, light compression, then two-pass loudness normalisation.

    Pass 1 measures the whole song; pass 2 applies one linear gain (loudnorm linear=true) so the song's own
    build-up (quiet intro, loud final chorus) is kept instead of being levelled out in real time.
    """
    ff = ffmpeg_path()
    if ff is None:
        raise AudioUnavailable("마스터링에는 ffmpeg가 필요해요 (macOS `brew install ffmpeg`)")
    if eq not in MASTER_EQ:
        raise ValueError(f"unknown master EQ {eq!r} (known: {', '.join(MASTER_EQ)})")
    clarity = "" if os.environ.get("INSTMATE_MASTER_CLARITY", "").lower() == "off" else CLARITY_EQ
    pre = ",".join(x for x in ("highpass=f=30", clarity, MASTER_EQ[eq],
                               "acompressor=threshold=-16dB:ratio=2:attack=25:release=300") if x)
    target = f"I={MASTER_LUFS}:TP={MASTER_TP}:LRA=20"
    with tempfile.TemporaryDirectory(prefix="instmate_audio_") as tmp:
        src, out = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        src.write_bytes(wav)
        from . import levellock
        measured = levellock.value("loudnorm", lambda: _measure_loudness(ff, src, f"{pre},loudnorm={target}:print_format=json"))
        m = ":".join(f"{k}={measured[v]}" for k, v in (("measured_I", "input_i"), ("measured_TP", "input_tp"),
                                                      ("measured_LRA", "input_lra"), ("measured_thresh", "input_thresh"),
                                                      ("offset", "target_offset")))
        _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
              "-af", f"{pre},loudnorm={target}:{m}:linear=true", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
              str(out)], "ffmpeg")
        data = out.read_bytes() if out.is_file() else b""
    if len(data) <= 44 or data[:4] != b"RIFF":
        raise AudioRenderError("ffmpeg가 마스터링한 WAV를 만들지 않았어요")
    return data


def match_loudness(wav: bytes) -> bytes:
    """WAV -> the same sound at the app's own loudness (MASTER_LUFS, true-peak ceiling), 16-bit WAV.  One linear gain
    for the whole file: no EQ, no compression, so the voice of an outside model (the sung demo) is not coloured; it
    only stops being quieter than the app's own renders."""
    ff = ffmpeg_path()
    if ff is None:
        raise AudioUnavailable("음량 맞춤에는 ffmpeg가 필요해요")
    target = f"I={MASTER_LUFS}:TP={MASTER_TP}:LRA=20"
    with tempfile.TemporaryDirectory(prefix="instmate_audio_") as tmp:
        src, out = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        src.write_bytes(wav)
        measured = _measure_loudness(ff, src, f"loudnorm={target}:print_format=json")
        m = ":".join(f"{k}={measured[v]}" for k, v in (("measured_I", "input_i"), ("measured_TP", "input_tp"),
                                                      ("measured_LRA", "input_lra"), ("measured_thresh", "input_thresh"),
                                                      ("offset", "target_offset")))
        _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
              "-af", f"loudnorm={target}:{m}:linear=true", "-c:a", "pcm_s16le", str(out)], "ffmpeg")
        data = out.read_bytes() if out.is_file() else b""
    if len(data) <= 44 or data[:4] != b"RIFF":
        raise AudioRenderError("ffmpeg가 음량을 맞춘 WAV를 만들지 않았어요")
    return data


def _measure_loudness(ff: str, src: Path, chain: str) -> dict:
    """loudnorm's first-pass measurement (the JSON block it prints on stderr)."""
    try:
        proc = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", str(src), "-af", chain, "-f", "null", "-"],
                              capture_output=True, text=True, timeout=TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise AudioRenderError(f"ffmpeg가 {TIMEOUT_S}초 안에 끝나지 않았어요") from None
    except OSError as e:
        raise AudioUnavailable(f"ffmpeg를 실행하지 못했어요: {e}") from None
    err = proc.stderr or ""
    start, end = err.rfind("{"), err.rfind("}")
    if proc.returncode != 0 or start < 0 or end < start:
        raise AudioRenderError(f"ffmpeg 음량 측정 실패 (종료 코드 {proc.returncode}): {err.strip()[-300:]}")
    import json
    data = json.loads(err[start:end + 1])
    need = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [k for k in need if k not in data]
    if missing:
        raise AudioRenderError(f"ffmpeg 음량 측정 결과에 {missing} 값이 없어요")
    return data


def to_mp3(wav: bytes) -> bytes:
    """WAV bytes -> MP3 bytes (ffmpeg, libmp3lame VBR ~190 kbps). Raises AudioUnavailable without ffmpeg."""
    ff = ffmpeg_path()
    if ff is None:
        raise AudioUnavailable("MP3로 바꾸려면 ffmpeg가 필요해요. WAV로 받으세요 (macOS `brew install ffmpeg`)")
    with tempfile.TemporaryDirectory(prefix="instmate_audio_") as tmp:
        src, out = Path(tmp) / "in.wav", Path(tmp) / "out.mp3"
        src.write_bytes(wav)
        _run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
              "-codec:a", "libmp3lame", "-q:a", "2", str(out)], "ffmpeg")
        data = out.read_bytes() if out.is_file() else b""
    if not data:
        raise AudioRenderError("ffmpeg가 MP3 파일을 만들지 않았어요")
    return data


MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg"}

# YuE2 weights are CC BY-NC 4.0 (docs/LICENSES.md): its takes carry the attribution and terms in the file and its name
YUE_FILENAME = "instmate_yue2_CC-BY-NC-4.0.{ext}"
YUE_NOTICE = ("Vocal take made with YuE2-3B (m-a-p, https://huggingface.co/m-a-p/YuE2-3B), weights licensed CC BY-NC 4.0 "
              "(https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only.")


def yue_attributed(data: bytes, fmt: str) -> bytes:
    """A WAV gets a LIST/INFO chunk (ICMT notice, ISFT tool) after its chunks; other formats and broken input pass through."""
    import struct
    if fmt != "wav" or len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE" or b"CC BY-NC 4.0" in data:
        return data

    def sub_chunk(tag: bytes, text: str) -> bytes:
        raw = text.encode("utf-8") + b"\0"
        return tag + struct.pack("<I", len(raw)) + raw + (b"\0" if len(raw) % 2 else b"")

    info = b"INFO" + sub_chunk(b"ICMT", YUE_NOTICE) + sub_chunk(b"ISFT", "Grandchaps InstMate")
    body = data[12:12 + struct.unpack("<I", data[4:8])[0] - 4]
    if len(body) % 2:
        body += b"\0"
    body += b"LIST" + struct.pack("<I", len(info)) + info
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


# Opt-in sound features: key -> (label on the page, one-line explanation).  All off unless asked for
# (`render(options=...)`, `AudioReq.options`); a feature becomes a default only after the user has heard it and said so.
OPTIONS = {
    "mix_v3": ("믹스 3안 (넓게)", "기타를 좌우로 겹치고 건반·패드를 넓히며, 저역과 보컬은 가운데에 두고 -11 LUFS로 마무리해요. 켜면 2안 대신 적용돼요."),
    "master_v2": ("믹스·마스터 2안", "악기마다 따로 음색을 다듬고 좌우로 벌린 뒤, 더 크게(-12 LUFS 근처) 마무리해요."),
    "vocal_polish": ("가이드 보컬 다듬기", "한 프레이즈를 한 숨에 이어 부르고(라~아~아), 프레이즈 끝에서만 소리를 줄이며, 긴 음은 늦게 떨고, 100Hz 아래를 걸러요."),
    "vocal_ah": ("가이드 보컬 '아' 발음", "음마다 '라'·'나' 자음을 붙이지 않고 '아~'로 이어 부르며, 앞뒤 음의 간격에 맞춰 음높이가 부드럽게 넘어가요."),
    "section_parts": ("구간별 편성", "벌스는 킥·하이햇·베이스와 건반(없으면 기타)만 연주하고, 프리코러스에서 스네어가 들어오며, 후렴은 전부 연주해요. 브릿지는 드럼을 절반으로 줄이고, 마지막 후렴은 선율을 한 옥타브 위에서 겹쳐요. 음량은 줄이지 않아요."),
    "section_dynamics": ("구간별 강약", "벌스에서는 기타가 빠지고 작게, 후렴은 3~6 dB 크게, 마지막 후렴은 한 단계 더 크게 들려줘요."),
    "auto_breath": ("자동 숨 자리", "쉼 없이 6초 넘게 이어진 프레이즈 끝에서 가이드 보컬이 반 박 먼저 끊고 숨을 쉬어요. 악보의 음은 그대로예요."),
    "tail_trim": ("끝 무음 자르기", "마지막 소리가 사라진 뒤 1.5초만 남기고 파일을 끝내요."),
}
# Options the page switches on by itself: the ones the user chose by ear in the blind listening of 2026-10-10
# (pair 2: master_v2; 2026-10-10 19:39: vocal_polish restored because syllable breaks were worse).
# The function/API default stays "no option": callers must opt in; page and chat share these UI defaults.
DEFAULT_ON = ("master_v2", "vocal_polish", "vocal_ah")     # vocal_ah: asked for by the user on 2026-10-11
TAIL_KEEP = 1.5              # seconds of the file kept after the last sound
TAIL_FLOOR = 1e-3            # a sample below this (about -60 dBFS) is silence


def trim_tail(wav: bytes) -> bytes:
    """The WAV cut TAIL_KEEP seconds after its last sound, with a short fade to nothing.  FluidSynth renders until every
    voice has fully died away, and some drum sounds (toms, open hi-hat) keep an inaudible voice alive for 15-30
    seconds: a song ending on a fill carried that much silence.  Nothing above TAIL_FLOOR is removed; a file that is
    silent throughout, already short, or not readable as audio is returned as it is (trimming never costs the render)."""
    import numpy as np
    import soundfile as sf
    try:
        y, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
    except Exception:
        return wav
    loud = np.nonzero(np.abs(y.astype("int32")).max(axis=1) > TAIL_FLOOR * 32768)[0]
    if not len(loud):
        return wav
    end = int(loud[-1]) + 1 + int(TAIL_KEEP * sr)
    if end >= len(y):
        return wav
    out = y[:end].copy()
    fade = min(end, int(0.05 * sr))
    out[-fade:] = (out[-fade:] * np.linspace(1.0, 0.0, fade)[:, None]).astype("int16")
    buf = io.BytesIO()
    sf.write(buf, out, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def render(midi: bytes, fmt: str = "auto", sounds: dict | None = None, genre: str | None = None,
           swing_amount: float | None = None, voice: dict | None = None, sections: list[dict] | None = None,
           notes: list[str] | None = None, vocal_style: str | None = None,
           exact_parts: set[str] | frozenset[str] = frozenset(), mode: str | None = None,
           vocal_fit: bool = True, breaths: list[int] | None = None,
           options: set[str] | frozenset[str] | list[str] = frozenset()) -> tuple[bytes, str]:
    """(audio bytes, format). fmt: "wav", "mp3", or "auto" (MP3 when ffmpeg exists, else WAV).
    `sounds`: {track: key of SOUNDS[track]} to play a part with another instrument (e.g. melody on violin).
    `genre`: its profile sets the swing and the master EQ; `swing_amount` overrides the profile's swing.
    `voice`: a vocal.load_voice() result: the melody is sung by that recorded hum (instmate.vocal) instead of played
    by the MIDI melody sound.
    `vocal_style`: "male" ("na na na", an octave lower when needed) or "female" ("la la la"): the melody is sung with
    that syllable, by `voice` when given, else by the built-in voice of that style (vocal.builtin_voice).
    `exact_parts`: parts a band member played themselves: no timing or velocity changes are added to them.
    `sections`: the result's sections ({"kind", "start_bar", "bars"}); with them the guitar tone follows the genre's
    per-section table (configs/instruments.yaml).  `notes`: a list that receives fallback reasons (plugin missing...).
    `mode`: "demo" (default: every installed instrument) or "release" (licence tier A only, instruments.resolve).
    `breaths`: bars (from 1) at whose end the guide voice takes a breath and the band holds back for half a beat
    (instmate/breath.py).  Opt-in, only when a voice sings, and only for bars that qualify; the MIDI melody is not changed.
    `options`: keys of OPTIONS to switch on (none by default; an unknown key is a ValueError).
    `vocal_fit`: True moves the sung melody by whole octaves into the singer's range (vocal.fit_register); False sings
    the melody at its written pitch and reports the notes above the singer's reach in `notes`."""
    from . import instruments as _inst
    options = frozenset(options or ())
    if options - set(OPTIONS):
        raise ValueError(f"unknown sound option: {', '.join(sorted(options - set(OPTIONS)))} (known: {', '.join(OPTIONS)})")
    _inst.allowed("A", mode)                                           # refuses an unknown mode
    from .genres import PROFILES
    if fmt not in ("auto", "wav", "mp3"):
        raise ValueError(f"format must be auto, wav or mp3, got {fmt!r}")
    if genre is not None and genre not in PROFILES:
        raise ValueError(f"unknown genre {genre!r}")
    prof = PROFILES.get(genre) if genre else None
    amount = swing_amount if swing_amount is not None else (prof["swing"] if prof else 0.0)
    check_sounds(sounds)
    _check_midi(midi)
    if vocal_style is not None:
        from . import vocal
        if vocal_style not in vocal.STYLES:
            raise ValueError(f"vocal_style must be one of {', '.join(vocal.STYLES)}, got {vocal_style!r}")
        if voice is None:                                  # no recording: the Mac's voice, else the formant voice
            if _inst.allowed("B", mode):
                voice = vocal.system_voice(vocal_style) or vocal.builtin_voice(vocal_style)
            else:                                          # the Mac's speech voice is Apple's: not in release mode
                voice = vocal.builtin_voice(vocal_style)
                if notes is not None:
                    notes.append("가이드 보컬: 배포 모드라 맥 시스템 음성 대신 내장 합성 음성 사용")
    programs = check_sounds(sounds)                                    # the user's choice wins over the genre table
    # a choice that names a table instrument (gear with an effect chain or a plugin) is rendered like the table's own
    overrides = {p: inst for p, k in (sounds or {}).items() if k not in SOUNDS[p] and (inst := _chosen_instrument(p, k, mode))}
    for part, inst in overrides.items():
        programs[part] = _program_of(part, inst["sound"])              # the effect chain starts from its clean sound
    if prof:
        from . import instruments
        for part, inst in instruments.lineup(genre, mode)["parts"].items():
            programs.setdefault(part, _program_of(part, inst["sound"]))
    played = prepare(_set_programs(midi, programs), amount, frozenset(exact_parts))
    # practice "this part only" on the sung melody: there is no band to render, the voice sings alone
    alone = voice is not None and _track_names(played) <= {"melody"}
    breath_targets: list[dict] = []
    band = played
    if "section_parts" in options and not alone:
        if sections:
            from . import sectionparts
            band = sectionparts.shape(band, sections, exact=exact_parts)
        elif notes is not None:
            notes.append("구간별 편성: 곡의 구간 정보가 없어 적용하지 않았어요")
    dyn_sections = sections if "section_dynamics" in options and not alone else None
    if "section_dynamics" in options and not sections and notes is not None:
        notes.append("구간별 강약: 곡의 구간 정보가 없어 적용하지 않았어요")
    if dyn_sections:
        from . import sectiondyn
        band = sectiondyn.thin(band, dyn_sections, exact=exact_parts)
        if "guitar" in exact_parts and notes is not None:
            notes.append("구간별 강약: 직접 연주한 기타 파트는 빼지 않고 유지했어요")
    if "auto_breath" in options and not breaths and voice is not None:
        from . import breath
        breaths = breath.auto_bars(midi)
        if notes is not None and breaths:
            notes.append("자동 숨 자리: " + ", ".join(str(b) for b in breaths) + "마디 끝")
    if breaths and voice is not None:
        from . import breath
        chosen = breath.plan(midi, list(breaths))
        breath_targets = chosen["targets"]
        if notes is not None:
            notes.extend(f"숨 자리: {s['bar']}마디는 그대로 둠 ({s['reason']})" for s in chosen["skipped"])
        band = breath.rest_band(band, breath_targets, exact=exact_parts)
        protected = set(exact_parts) & set(breath.ATTACK_PARTS + breath.SUSTAIN_PARTS + (breath.DRUMS,))
        if protected and breath_targets and notes is not None:
            notes.append("숨 자리: 직접 연주한 반주 파트는 유지했어요 (" + ", ".join(sorted(protected)) + ")")
    from functools import partial
    by_part = partial(_mix_by_part, mixer=__import__("instmate.mixv3", fromlist=["x"])) if "mix_v3" in options else _mix_by_part
    wav = None if alone else (by_part if {"master_v2", "mix_v3"} & options else _render_parts)(
        band, sing_melody=voice is not None, genre=genre if prof else None,
        user_parts=set(sounds or {}) - set(overrides), sections=sections, notes=notes, overrides=overrides, mode=mode)
    if voice is not None:
        from . import vocal
        sung, length = vocal.melody_notes(played)
        if vocal_style is not None and vocal_fit:
            fitted = vocal.fit_register(sung, vocal_style)
            if notes is not None and fitted and sung and fitted[0][2] != sung[0][2]:
                notes.append(f"가이드 보컬: 음역에 맞춰 멜로디를 {abs(fitted[0][2] - sung[0][2]) // 12}옥타브 "
                             f"{'내려' if fitted[0][2] < sung[0][2] else '올려'} 불렀어요 (악보의 음높이와 달라요)")
            sung = fitted
        elif vocal_style is not None:
            top = vocal.STYLES[vocal_style]["top"]
            high = sum(1 for _, _, p in sung if p > top)
            if high and notes is not None:
                notes.append(f"가이드 보컬: 멜로디의 {high}음이 이 목소리의 편한 음역보다 높아요. 원래 음높이 그대로 불렀어요 "
                             f"(옥타브를 옮겨 부르려면 \"보컬 음역에 맞추기\"를 켜세요)")
        syllable = vocal.STYLES[vocal_style]["syllable"] if vocal_style else None
        if "vocal_ah" in options:                            # the open vowel alone: no consonant at the note starts
            syllable = None
        polish = {}                                          # the plain call stays exactly what it was
        if "vocal_polish" in options:                        # phrases end at rests of an eighth, section ends, breaths
            from . import breath as _breath
            tick = _breath.seconds_per_tick(played)
            eighth = tick * mido.MidiFile(file=io.BytesIO(played)).ticks_per_beat / 2
            ends = [(s["start_bar"] + s["bars"]) * _bar_seconds(played) for s in sections or []]
            ends += [t["end"] * tick for t in breath_targets]
            polish = {"polish": {"gap_s": eighth, "breaks": ends}}
            if "vocal_ah" in options:                        # with no consonant the notes slide into each other
                polish["polish"]["glide"] = True
        voice_wave = vocal.sing(sung, voice, length + 1.0, syllable=syllable, **polish)
        if breath_targets:
            from . import breath
            voice_wave = breath.breathe_wave(voice_wave, vocal.FS, breath_targets, breath.seconds_per_tick(played))
        if "mix_v3" in options and not alone:
            from . import mixv3
            voice_wave = mixv3.presence(voice_wave, vocal.FS)
        wav = vocal.solo_vocal(voice_wave) if alone else vocal.mix_vocal(wav, voice_wave)
    if ffmpeg_path() is not None:
        wav = (master_v2 if {"master_v2", "mix_v3"} & options else master)(wav, prof["master"] if prof else "natural")
        if "mix_v3" in options:
            from . import mixv3
            wav = relevel(wav, mixv3.LUFS)
    if dyn_sections:
        from . import sectiondyn
        wav, dyn_gains = sectiondyn.level(wav, dyn_sections, _bar_seconds(played))
        if any(dyn_gains):
            wav = relevel(wav)
        if notes is not None:
            notes.append("구간별 강약: " + ", ".join(f"{s['kind']} {g:+.1f}dB" for s, g in zip(dyn_sections, dyn_gains)))
    if "tail_trim" in options:
        wav = trim_tail(wav)
    if fmt == "wav" or (fmt == "auto" and ffmpeg_path() is None):
        return wav, "wav"
    return to_mp3(wav), "mp3"
