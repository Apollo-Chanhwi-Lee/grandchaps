"""Play chosen parts with real instrument plugins (VST3/AU) through Spotify's Pedalboard (GPL-3.0).

    models/vst/tracks.json  {"piano": {"plugin": "~/Library/Audio/Plug-Ins/VST3/Surge XT.vst3", "state": "piano.state"}}
    a part listed there -> its MIDI notes rendered by that plugin (with the sound saved by scripts/vst_pick.py)
                        -> level-matched to what the SoundFont would have played for the same part
    every other part    -> FluidSynth as before; the stems are summed before mastering

Plugins must be installed on this Mac (user plug-in folders work without admin rights). Without tracks.json, or when a
plugin cannot be loaded, rendering falls back to FluidSynth for that part (the reason is returned, never a crash).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VST_DIR = ROOT / "models" / "vst"
CONFIG = VST_DIR / "tracks.json"
SR = 44100


def config() -> dict:
    """{track: {"plugin": path, "state": file name}} for the parts that should use a plugin (may be empty)."""
    env = os.environ.get("INSTMATE_VST")
    if env and env.lower() == "off":
        return {}
    try:
        return json.loads(CONFIG.read_text()) if CONFIG.is_file() else {}
    except (OSError, ValueError):
        return {}


def _plugin(entry: dict):
    import pedalboard
    plug = pedalboard.load_plugin(os.path.expanduser(entry["plugin"]))
    state = VST_DIR / entry["state"] if entry.get("state") else None
    if state is not None and state.is_file():
        plug.raw_state = state.read_bytes()
        if plug.is_instrument:               # a restored sound is applied on the next block: without this short run
            plug([], duration=0.1, sample_rate=SR, num_channels=2)     # Surge XT's first render came out silent
    return plug


def track_messages(midi: bytes, name: str) -> tuple[list, float]:
    """The named track's note/control messages with absolute times in seconds, and the song length."""
    import mido
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tempo = next((m.tempo for m in mid.tracks[0] if m.type == "set_tempo"), 500000)
    sec = tempo / 1e6 / mid.ticks_per_beat
    out = []
    for tr in mid.tracks:
        if not any(m.type == "track_name" and m.name == name for m in tr):
            continue
        now = 0
        for m in tr:
            now += m.time
            if m.type in ("note_on", "note_off", "control_change", "pitchwheel"):
                out.append(m.copy(time=now * sec, channel=0))
    return out, mid.length


X86_PYTHON = ROOT / ".venv-x86" / "bin" / "python"
WORKER = ROOT / "scripts" / "vst_worker.py"
WORKER_TIMEOUT_S = 300


def plugin_archs(path: str) -> set[str]:
    """CPU architectures of a plugin bundle's binary (e.g. {"x86_64"} for an Intel-only plugin)."""
    import subprocess
    macos = Path(os.path.expanduser(path)) / "Contents" / "MacOS"
    bins = [b for b in macos.iterdir() if b.is_file()] if macos.is_dir() else []
    archs: set[str] = set()
    for b in bins:
        out = subprocess.run(["lipo", "-archs", str(b)], capture_output=True, text=True)
        archs |= set(out.stdout.split())
    return archs


def needs_rosetta(path: str) -> bool:
    import platform
    archs = plugin_archs(path)
    return platform.machine() == "arm64" and bool(archs) and "arm64" not in archs


def _render_in_rosetta(msgs: list, duration: float, entry: dict) -> np.ndarray:
    import subprocess
    import tempfile
    if not X86_PYTHON.exists():
        raise RuntimeError("Intel-only plugin needs .venv-x86 (uv venv with an x86_64 Python + pedalboard)")
    state = VST_DIR / entry["state"] if entry.get("state") else None
    job = {"plugin": entry["plugin"], "state": str(state) if state else None, "duration": duration, "sample_rate": SR,
           "messages": [[m.type, getattr(m, "note", 0), getattr(m, "velocity", 0), getattr(m, "control", 0),
                         getattr(m, "value", 0), m.time] for m in msgs if m.type in ("note_on", "note_off", "control_change")]}
    with tempfile.TemporaryDirectory(prefix="instmate_vst_") as tmp:
        jp, op = Path(tmp) / "job.json", Path(tmp) / "out.npy"
        jp.write_text(json.dumps(job))
        proc = subprocess.run(["arch", "-x86_64", str(X86_PYTHON), str(WORKER), str(jp), str(op)],
                              capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)
        if proc.returncode != 0 or not op.exists():
            raise RuntimeError(f"plugin worker failed: {(proc.stderr or '').strip()[-300:]}")
        return np.load(op)


def render_stem(midi: bytes, name: str, entry: dict) -> np.ndarray:
    """(2, n) float32 audio of one part played by its plugin (Intel-only plugins run under Rosetta)."""
    msgs, length = track_messages(midi, name)
    if needs_rosetta(entry["plugin"]):
        return _render_in_rosetta(msgs, length + 2.0, entry)
    plug = _plugin(entry)
    audio = plug(msgs, duration=length + 2.0, sample_rate=SR, num_channels=2)
    return np.asarray(audio, dtype=np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)


def mix(base_wav: bytes, stems: list[tuple[np.ndarray, float]]) -> bytes:
    """Base WAV (FluidSynth, without the plugin parts) + plugin stems, each scaled to its target RMS."""
    import soundfile as sf
    base, sr = sf.read(io.BytesIO(base_wav), dtype="float32", always_2d=True)
    out = base.T.copy()
    for stem, target in stems:
        if stem.shape[1] > out.shape[1]:
            out = np.pad(out, ((0, 0), (0, stem.shape[1] - out.shape[1])))
        from . import levellock
        s = stem * levellock.value("stem_scale", lambda: target / rms(stem)) if target > 0 else stem
        out[:, : s.shape[1]] += s
    from . import levellock
    factor = levellock.value("stem_mix_peak", lambda: 0.99 / float(np.abs(out).max()) if float(np.abs(out).max()) > 0.99 else 1.0)
    if factor != 1.0:
        out *= factor
    buf = io.BytesIO()
    sf.write(buf, out.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
