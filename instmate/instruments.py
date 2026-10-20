"""Genre -> part -> instrument table (configs/instruments.yaml) and how each instrument is rendered.

    table        instruments (SoundFont sound, optional plugin, optional effect chain), effect chains, and per genre
                 the instrument of every part plus what changes per section kind (guitar tone, parts that join)
    priority     a part the user changed on screen > models/vst/tracks.json (scripts/vst_pick.py <part>) > this table
    fallback     a plugin that is not installed (or has no saved sound yet) plays the instrument's SoundFont sound;
                 an effect chain that cannot run (no pedalboard) plays the instrument's `fallback` sound

Nothing here imports audio at module load (audio and genres both read this module).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "instruments.yaml"
VST_DIR = ROOT / "models" / "vst"
PARTS = ("melody", "lead", "piano", "keys", "bass", "guitar", "strings", "drums")
SOUND_TABLE = {"lead": "guitar"}          # the intro lead guitar chooses from the guitar sounds
KINDS = ("intro", "verse", "pre", "chorus", "bridge", "interlude", "outro")
FX_KEYS = {"drive_db", "low_cut_hz", "high_cut_hz", "presence_hz", "presence_db"}
FX_OPTIONAL = {"mid_hz", "mid_db", "chorus", "plugin", "state", "tier", "license"}
TIERS = ("A", "B")
INSTRUMENT_KEYS = {"label", "sound", "vst", "state", "fx", "fallback", "sfz", "tier", "license", "credit", "release"}
TOP_KEYS = {"instruments", "fx", "genres", "choices", "modes", "sources"}
SECTION_KEYS = {"guitar", "add"}
ADDABLE = ("strings",)                    # parts a section may add (band parts follow the genre's form instead)


class InstrumentConfigError(ValueError):
    """configs/instruments.yaml is inconsistent (the message names the entry)."""


def validate(cfg: dict, sounds: dict) -> None:
    """Raise InstrumentConfigError unless every reference in the table resolves. `sounds` = audio.SOUNDS."""
    if not isinstance(cfg, dict) or not {"instruments", "fx", "genres"} <= set(cfg) or set(cfg) - TOP_KEYS:
        raise InstrumentConfigError(f"top level must be: instruments, fx, genres (optional: {sorted(TOP_KEYS)})")
    for mode, tiers in (cfg.get("modes") or {}).items():
        if mode not in ("demo", "release") or not tiers or set(tiers) - set(TIERS):
            raise InstrumentConfigError(f"modes: {mode!r} must be demo or release with tiers from {TIERS}")
    for src in cfg.get("sources") or []:
        if set(src) != {"name", "use", "tier", "license", "credit"} or src["tier"] not in TIERS:
            raise InstrumentConfigError(f"sources: each needs name, use, tier, license, credit ({src.get('name')!r})")
    for fid, chain in cfg["fx"].items():
        if not FX_KEYS <= set(chain) or set(chain) - FX_KEYS - FX_OPTIONAL or ("mid_hz" in chain) != ("mid_db" in chain):
            raise InstrumentConfigError(f"fx {fid}: keys must be {sorted(FX_KEYS)} (optional: {sorted(FX_OPTIONAL)})")
        if ("plugin" in chain) != ("state" in chain) or (set(chain) & {"tier", "license"} and "plugin" not in chain):
            raise InstrumentConfigError(f"fx {fid}: plugin and state go together; tier and license describe the plugin")
        if chain.get("tier", "A") not in TIERS:
            raise InstrumentConfigError(f"fx {fid}: tier must be one of {TIERS}")
    for part, ids in (cfg.get("choices") or {}).items():
        if part not in PARTS:
            raise InstrumentConfigError(f"choices: unknown part {part!r}")
        for iid in ids:
            _check_use(cfg, sounds, "choices", part, iid)
    for iid, inst in cfg["instruments"].items():
        extra = set(inst) - INSTRUMENT_KEYS
        if extra or "label" not in inst or "sound" not in inst:
            raise InstrumentConfigError(f"instrument {iid}: needs label and sound; unknown keys {sorted(extra)}")
        if inst.get("tier", "A") not in TIERS:
            raise InstrumentConfigError(f"instrument {iid}: tier must be one of {TIERS}")
        if sum(k in inst for k in ("vst", "fx", "sfz")) > 1:
            raise InstrumentConfigError(f"instrument {iid}: use one of vst, fx, sfz")
        if "release" in inst:
            alt = cfg["instruments"].get(inst["release"])
            if inst.get("tier", "A") == "A" or alt is None or alt.get("tier", "A") != "A" or alt["sound"] != inst["sound"]:
                raise InstrumentConfigError(f"instrument {iid}: release must name a tier-A instrument with the same "
                                            f"SoundFont sound, and only a tier-B instrument needs one")
        if "fx" in inst and inst["fx"] not in cfg["fx"]:
            raise InstrumentConfigError(f"instrument {iid}: unknown fx {inst['fx']!r}")
        if "vst" in inst and "fx" in inst:
            raise InstrumentConfigError(f"instrument {iid}: use vst or fx, not both")
        if ("fallback" in inst) != ("fx" in inst):
            raise InstrumentConfigError(f"instrument {iid}: fallback goes with fx (and fx needs a fallback)")
    for g, entry in cfg["genres"].items():
        if set(entry) - {"parts", "sections"} or set(entry.get("parts", {})) != set(PARTS):
            raise InstrumentConfigError(f"genre {g}: parts must list exactly {', '.join(PARTS)}")
        for part, iid in entry["parts"].items():
            _check_use(cfg, sounds, g, part, iid)
        for kind, sec in (entry.get("sections") or {}).items():
            if kind not in KINDS or set(sec) - SECTION_KEYS:
                raise InstrumentConfigError(f"genre {g}: bad section entry {kind!r}")
            if "guitar" in sec:
                _check_use(cfg, sounds, g, "guitar", sec["guitar"])
            bad = [p for p in sec.get("add", []) if p not in ADDABLE]
            if bad:
                raise InstrumentConfigError(f"genre {g} {kind}: cannot add {bad} (addable: {', '.join(ADDABLE)})")


def _check_use(cfg: dict, sounds: dict, genre: str, part: str, iid: str) -> None:
    inst = cfg["instruments"].get(iid)
    if inst is None:
        raise InstrumentConfigError(f"genre {genre} {part}: unknown instrument {iid!r}")
    table = sounds[SOUND_TABLE.get(part, part)]
    for key in (inst["sound"], inst.get("fallback")):
        if key is not None and key not in table:
            raise InstrumentConfigError(f"genre {genre} {part}: instrument {iid} uses sound {key!r}, not a {part} sound")


@lru_cache(maxsize=1)
def load() -> dict:
    import yaml
    from . import audio
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate(cfg, audio.SOUNDS)
    return cfg


@lru_cache(maxsize=1)
def _raw_kits() -> dict:
    """{genre: {part: SoundFont sound key}} without validation (genres.py reads it while audio may still be loading)."""
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {g: {p: cfg["instruments"][i]["sound"] for p, i in e["parts"].items() if p != "lead"}
            for g, e in cfg["genres"].items()}


def kit(genre: str) -> dict[str, str]:
    """The genre's SoundFont sound per part (keys of audio.SOUNDS): what plays with no plugin and no effect."""
    return dict(_raw_kits()[genre])


def default_mode() -> str:
    """"demo" (every installed instrument) unless INSTMATE_MODE=release (tier A only)."""
    return "release" if os.environ.get("INSTMATE_MODE", "").lower() == "release" else "demo"


def allowed(tier: str, mode: str | None) -> bool:
    mode = mode or default_mode()
    modes = load().get("modes") or {}
    if mode not in ("demo", "release"):
        raise ValueError(f"mode must be demo or release, got {mode!r}")
    return tier in modes.get(mode, TIERS if mode == "demo" else ("A",))


def resolve(iid: str, mode: str | None = None) -> dict:
    """The instrument `iid` as it may play in `mode`: {"id", ...instrument}.  An instrument whose tier the mode does
    not allow becomes its `release` instrument, or (none given) its own SoundFont sound with the plugin / samples
    removed; "replaced" then names the instrument that was asked for."""
    cfg = load()
    inst = {"id": iid, **cfg["instruments"][iid]}
    if allowed(inst.get("tier", "A"), mode):
        return inst
    if "release" in inst:
        return {"id": inst["release"], **cfg["instruments"][inst["release"]], "replaced": inst["label"]}
    plain = {k: v for k, v in inst.items() if k not in ("vst", "state", "sfz", "tier", "license", "credit")}
    return {**plain, "label": f"{inst['label']} → 기본 음원", "replaced": inst["label"]}


def lineup(genre: str, mode: str | None = None) -> dict:
    """{"parts": {part: {"id", "label", ...instrument}}, "sections": {kind: {"guitar": {...}, "add": [...]}}}."""
    cfg = load()
    entry = cfg["genres"][genre]
    inst = lambda iid: resolve(iid, mode)
    sections = {}
    for kind, sec in (entry.get("sections") or {}).items():
        sections[kind] = {"add": list(sec.get("add", []))}
        if "guitar" in sec:
            sections[kind]["guitar"] = inst(sec["guitar"])
    return {"parts": {p: inst(i) for p, i in entry["parts"].items()}, "sections": sections}


def section_adds(genre: str, part: str) -> set[str]:
    """Section kinds in which `part` joins according to the table (empty: the table says nothing about it)."""
    secs = load()["genres"].get(genre, {}).get("sections") or {}
    return {kind for kind, sec in secs.items() if part in sec.get("add", [])}


def place_extras(genre: str, kind: str, inst: list[str], chosen: list[str]) -> list[str]:
    """A section's instruments after the table's `add` rule: a part the user keeps in the line-up (`chosen`) that the
    table adds to some section kinds plays in exactly those kinds; other parts are left as the form set them."""
    out = list(inst)
    for part in ADDABLE:
        kinds = section_adds(genre, part)
        if not kinds or part not in chosen:
            continue
        if kind in kinds and part not in out:
            out.append(part)
        elif kind not in kinds and part in out:
            out.remove(part)
    return out


# ---------------------------------------------------------------- rendering choices
def plugin_entry(inst: dict) -> tuple[dict | None, str | None]:
    """(vst entry {"plugin", "state"}, None) when the instrument's plugin can be used here, else (None, reason)."""
    if "vst" not in inst:
        return None, None
    if os.environ.get("INSTMATE_VST", "").lower() == "off":
        return None, "플러그인 사용 꺼짐 (INSTMATE_VST=off)"
    path = Path(os.path.expanduser(inst["vst"]))
    if not path.exists():
        return None, f"플러그인 미설치: {path.name}"
    state = inst.get("state")
    if state and not (VST_DIR / state).is_file():
        return None, f"저장된 소리 없음 (scripts/vst_pick.py --instrument {inst.get('id', '?')})"
    return {"plugin": str(path), "state": state}, None


def sfz_entry(inst: dict) -> tuple[Path | None, str | None]:
    """(SFZ file, None) when the instrument's samples can be rendered here, else (None, reason)."""
    if "sfz" not in inst:
        return None, None
    import shutil
    if os.environ.get("INSTMATE_VST", "").lower() == "off":            # the switch covers plugins and sample instruments
        return None, "플러그인·샘플 악기 사용 꺼짐 (INSTMATE_VST=off)"
    path = Path(os.path.expanduser(inst["sfz"]))
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        return None, f"샘플 악기 미설치: {path.name}"
    if shutil.which("sfizz_render") is None and not (Path.home() / ".local/bin/sfizz_render").is_file():
        return None, "sfizz_render 없음"
    return path, None


def fx_plugin(fx_id: str, mode: str | None = None) -> tuple[dict | None, str | None]:
    """(amp plugin entry {"plugin", "state"}, None) when the chain's amp plugin can be used, else (None, reason);
    (None, None) when the chain has no amp plugin."""
    c = load()["fx"][fx_id]
    if "plugin" not in c:
        return None, None
    if not allowed(c.get("tier", "A"), mode):
        return None, "배포 모드: 앰프 플러그인 대신 내장 이펙트"
    if os.environ.get("INSTMATE_VST", "").lower() == "off":
        return None, "플러그인 사용 꺼짐 (INSTMATE_VST=off)"
    path = Path(os.path.expanduser(c["plugin"]))
    if not path.exists():
        return None, f"앰프 플러그인 미설치: {path.name}, 내장 이펙트 사용"
    if not (VST_DIR / c["state"]).is_file():
        return None, f"앰프 설정 없음 (scripts/vst_pick.py --fx {fx_id}), 내장 이펙트 사용"
    return {"plugin": str(path), "state": c["state"]}, None


def fx_available() -> bool:
    try:
        import pedalboard  # noqa: F401
        return True
    except Exception:
        return False


def apply_fx(audio: np.ndarray, sr: int, fx_id: str, mode: str | None = None, notes: list[str] | None = None) -> np.ndarray:
    """(2, n) float32 -> the same part through drive and a guitar-cabinet shape, plus a short room (pedalboard).
    When the chain names an amp plugin that is installed, has a saved setting and is allowed in `mode`, the plugin is
    the amp and cabinet instead; why it was not used (or that it failed) goes to `notes`."""
    import pedalboard as pb
    c = load()["fx"][fx_id]
    entry, reason = fx_plugin(fx_id, mode)
    if reason and notes is not None:
        notes.append(f"{fx_id}: {reason}")
    if entry is not None:
        try:
            amp = pb.load_plugin(entry["plugin"])
            amp.raw_state = (VST_DIR / entry["state"]).read_bytes()
            amp(np.zeros((2, sr // 10), dtype=np.float32), sr)          # a restored setting needs one block to settle
            peak = float(np.abs(audio).max())
            x = audio / peak * 0.5 if peak > 0 else audio
            wet = np.asarray(amp(x.astype(np.float32), sr), dtype=np.float32)
            if not np.isfinite(wet).all() or float(np.abs(wet).max()) < 1e-6:
                raise RuntimeError("the amp plugin returned silence")
            room = pb.Pedalboard([pb.Reverb(room_size=0.25, damping=0.5, wet_level=0.1, dry_level=0.9, width=0.8)])
            return np.asarray(room(wet, sr), dtype=np.float32)
        except Exception as e:                                         # broken plugin: the built-in chain still plays
            if notes is not None:
                notes.append(f"{fx_id}: 앰프 플러그인 실패 ({str(e)[:80]}), 내장 이펙트 사용")
    chain = [pb.HighpassFilter(cutoff_frequency_hz=c["low_cut_hz"])]
    if c.get("chorus"):
        chain.append(pb.Chorus(rate_hz=1.1, depth=0.3, centre_delay_ms=8.0, feedback=0.1, mix=0.45))
    if c["drive_db"] > 0:
        chain.append(pb.Distortion(drive_db=c["drive_db"]))
    if "mid_hz" in c:                                              # e.g. the scooped middle of a metal tone
        chain.append(pb.PeakFilter(cutoff_frequency_hz=c["mid_hz"], gain_db=c["mid_db"], q=0.8))
    chain += [
        pb.PeakFilter(cutoff_frequency_hz=c["presence_hz"], gain_db=c["presence_db"], q=1.0),
        pb.LowpassFilter(cutoff_frequency_hz=c["high_cut_hz"]),
        pb.LowpassFilter(cutoff_frequency_hz=c["high_cut_hz"]),     # twice: a speaker rolls off steeply
        pb.HighpassFilter(cutoff_frequency_hz=c["low_cut_hz"]),
        pb.Reverb(room_size=0.25, damping=0.5, wet_level=0.1, dry_level=0.9, width=0.8),
    ]
    board = pb.Pedalboard(chain)
    peak = float(np.abs(audio).max())
    x = audio / peak * 0.5 if peak > 0 else audio                  # the same drive whatever the part's level
    return np.asarray(board(x.astype(np.float32), sr), dtype=np.float32)


def section_gains(spans: list[tuple[float, float, str]], variants: dict[str, set[str]], default: str, n: int,
                  sr: int, fade_s: float = 0.03) -> dict[str, np.ndarray]:
    """Per variant a 0..1 gain curve of n samples: 1 inside the sections that use it, short fades at the borders.
    spans: [(start_s, end_s, kind)]; variants: {variant id: section kinds}; kinds nobody claims use `default`.
    The curves sum to 1 everywhere (the time after the last section keeps the last section's variant)."""
    ids = list(dict.fromkeys([default, *variants]))
    owner = np.zeros(n, dtype=np.int16)
    last = 0
    for start, end, kind in sorted(spans):
        who = next((ids.index(v) for v, kinds in variants.items() if kind in kinds), 0)
        a, b = max(0, int(round(start * sr))), min(n, int(round(end * sr)))
        owner[a:b] = who
        last = who
        tail = b
    if spans:
        owner[tail:] = last
    k = max(1, int(fade_s * sr))
    kernel = np.ones(k) / k
    out = {}
    for i, v in enumerate(ids):
        g = (owner == i).astype(np.float64)
        out[v] = np.convolve(np.pad(g, (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")
    # Normalize rounding error without changing the crossfade's proportions.
    total = sum(out.values())
    for v in out:
        out[v] = out[v] / total
    return out


def credits(mode: str | None = None) -> dict:
    """What the app's info box shows: the sound sources and every table instrument with its tier, licence and
    attribution line, and whether the given mode may use it."""
    cfg = load()
    rows = [{"name": s["name"], "use": s["use"], "tier": s["tier"], "license": s["license"], "credit": s["credit"],
             "allowed": allowed(s["tier"], mode)} for s in cfg.get("sources") or []]
    for iid, inst in cfg["instruments"].items():
        if not ({"vst", "sfz"} & set(inst)):
            continue                                                   # a SoundFont sound: covered by the sources
        tier = inst.get("tier", "A")
        rows.append({"name": inst["label"], "use": iid, "tier": tier, "license": inst.get("license", "미기록"),
                     "credit": inst.get("credit"), "allowed": allowed(tier, mode)})
    for fid, c in cfg["fx"].items():
        if "plugin" in c:
            tier = c.get("tier", "A")
            rows.append({"name": f"앰프 플러그인 ({fid})", "use": Path(c["plugin"]).name, "tier": tier,
                         "license": c.get("license", "미기록"), "credit": None, "allowed": allowed(tier, mode)})
    return {"mode": mode or default_mode(), "rows": rows}


def api_lineup(genre: str, mode: str | None = None) -> dict:
    """The genre's line-up for the UI: per part the instrument label and whether it plays as a plugin, with effects,
    or as the SoundFont fallback here; and what changes per section kind."""
    part_ko = {"melody": "멜로디", "lead": "리드 기타", "piano": "피아노", "keys": "멜로디 피아노", "bass": "베이스",
               "guitar": "기타", "strings": "스트링", "drums": "드럼"}
    kind_ko = {"intro": "인트로", "verse": "벌스", "pre": "프리코러스", "chorus": "후렴", "bridge": "브리지",
               "interlude": "간주", "outro": "아웃트로"}
    lu = lineup(genre, mode)

    def how(inst: dict) -> dict:
        entry, reason = plugin_entry(inst)
        sfz, sfz_reason = sfz_entry(inst)
        reason = reason or sfz_reason
        kind = "plugin" if entry else "sfz" if sfz else "fx" if "fx" in inst and fx_available() else "soundfont"
        note = f"사운드폰트로 대체 ({reason})" if reason else None
        if inst.get("replaced"):
            note = f"배포 모드: {inst['replaced']} 대신 사용" + (f", {note}" if note else "")
        return {"id": inst["id"], "label": inst["label"], "mode": kind, "tier": inst.get("tier", "A"), "note": note}

    parts = [{"part": p, "part_label": part_ko[p], **how(i)} for p, i in lu["parts"].items()]
    sections = []
    for kind, sec in lu["sections"].items():
        row = {"kind": kind, "kind_label": kind_ko[kind], "add": [part_ko[p] for p in sec["add"]]}
        if "guitar" in sec:
            row["guitar"] = how(sec["guitar"])
        sections.append(row)
    return {"parts": parts, "sections": sections}
