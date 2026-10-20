"""Export a lead sheet as a request for YuE2 (open-weight song model, m-a-p/YuE2-3B): the model sings the melody.

    lead sheet -> ABC score in the dialect of YuE2's examples (L:1/16, a "Vocal" voice with chord symbols, an empty
                  "Ins" voice, one "% section" comment per part)
               -> lyrics with one syllable per sung note ("na" for a male guide vocal, "la" for a female one)
               -> request JSON for `yue2 generate --request ...` (style text, lyrics, abc, cot="melody", seed)

YuE2 runs outside the app (a CUDA machine; weights are CC BY-NC 4.0), so this module only writes the request; nothing
here needs the model.  The song it returns is a demo of the tune with a sung voice, not the app's own arrangement.
"""
from __future__ import annotations

import json

from .leadsheet import LeadSheet, parse_leadsheet
from .theory import TPQ

S16 = TPQ // 4
ABC_NAMES = ("C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B")
SECTION_EN = {"인트로": "intro", "벌스": "verse", "프리": "pre-chorus", "후렴": "chorus", "브리지": "bridge",
              "간주": "interlude", "아웃트로": "outro"}
LYRIC_TAG = {"intro": "Intro", "verse": "Verse", "pre-chorus": "Pre-Chorus", "chorus": "Chorus", "bridge": "Bridge",
             "interlude": "Interlude", "outro": "Outro"}
# Voice presets, chosen by ear on 2026-10-05 (work/yue: candidates A-F).  With the default guidance (1.0) the style text
# barely steered the voice and "female" came out male; guidance 2.0 fixed that.  octave: the melody is written that many
# octaves higher for the model.
#
# 2026-10-06, by ear again: the presets sounded good on a sparse tune (two long notes per bar) and fell apart on the
# app's real songs (four or more notes per bar, a verse written an octave below the chorus).  So the demo now sings a
# THINNED melody (`simplify`) with every section moved by octaves into the voice's register (`fit_register`); the
# "octave" of a preset is no longer used.  The native male preset did not sound male: the male demo is the female E
# take lowered five semitones (`shift`).  The user compared three, four and five semitones of the same take and chose
# five; lowering the F take howled at the end of the pre-chorus (the F take itself broke down there), so E is the base.
# The first entry is the default on screen: E was the one take called perfect on an app song.
STYLE = {
    "female_high": {"label": "여성 E 'la la' (맑은 고음)", "syllable": "la", "voice": "expressive female voice",
                    "octave": 0, "cfg": 2.0, "seed": 42, "center": 72, "shift": 0},
    "male": {"label": "남성 (여성 E를 5키 낮춤)", "syllable": "la", "voice": "expressive female voice",
             "octave": 0, "cfg": 2.0, "seed": 42, "center": 72, "shift": -5, "input": "sections"},
    "female": {"label": "여성 F 'la la' (담백한 음색)", "syllable": "la", "voice": "solo female singer, young woman soprano",
               "octave": 0, "cfg": 2.0, "seed": 99, "center": 67, "shift": 0},
}
REGISTER_TOP = 86               # D6: no section is raised above this
MAX_DROP = 5                    # a section may not sit more than a fourth below the one before it
GENRE_STYLE = {"ballad": "piano ballad, acoustic piano, soft drums, bass", "rock": "band rock, electric guitars, bass, live drums",
               "emo_rock": "emotional band rock, clean electric guitar, piano, bass, drums",
               "kpop": "K-pop band sound, electric piano, clean guitar, punchy drums, synth bass",
               "jpop": "J-pop band rock, bright piano, driving electric guitars, bass, drums",
               "piano_rock": "piano rock, grand piano, overdriven guitar, bass, drums",
               "pop_punk": "pop punk, fast distorted guitars, bass, energetic drums",
               "hard_rock": "hard rock, heavy guitar riffs, organ, bass, power drums",
               "modern_rock": "modern rock, clean and crunch guitars, piano, bass, drums"}
METERS = ("4/4", "3/4", "6/8")


class YueExportError(ValueError):
    """The lead sheet cannot be written for YuE2 (the message says why)."""


def _pitch(midi: int) -> str:
    """ABC pitch with an explicit accidental (the score is written in K:C): C4 = "C", C5 = "c", C3 = "C,"."""
    name, octave = ABC_NAMES[midi % 12], midi // 12 - 1
    acc, letter = (name[0], name[1]) if len(name) == 2 else ("", name)
    if octave >= 5:
        return acc + letter.lower() + "'" * (octave - 5)
    return acc + letter + "," * (4 - octave)


def _units(ticks: int) -> int:
    return max(1, round(ticks / S16))


def to_abc(sheet: LeadSheet | str, octave: int = 0) -> str:
    """The melody and chords as a YuE2 score.  Notes are rounded to sixteenths; every bar is filled to its length.
    `octave`: write the melody that many octaves higher."""
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    if sheet.meter not in METERS:
        raise YueExportError(f"{sheet.meter} 박자는 YuE2 악보로 내보낼 수 없어요")
    num, den = (int(x) for x in sheet.meter.split("/"))
    bar_units = num * 16 // den
    lines = ["X:1", "T:", f"M:{sheet.meter}", "L:1/16", f"Q:1/4={sheet.tempo}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"', 'V: Ins clef=treble name="Ins Melody" snm="Inst."', "K:C"]
    if not any(n.pitch is not None for s in sheet.sections for b in s.bars for n in b.notes):
        raise YueExportError("리드시트에 멜로디 음이 하나도 없어요")
    for sec in sheet.sections:
        bars = []
        for bar in sec.bars:
            chords = {_units(c.start) if c.start else 0: c.symbol for c in bar.chords}
            out, now, sharp = [], 0, set()
            for n in sorted(bar.notes, key=lambda x: x.start):
                start = min(bar_units, round(n.start / S16))
                length = min(_units(n.dur), bar_units - start)
                if length <= 0:
                    continue
                if start > now:
                    out.append(_chord_at(chords, now) + f"z{start - now}")
                    now = start
                text = "z" if n.pitch is None else _pitch(n.pitch + 12 * octave)
                if n.pitch is not None:                      # a natural after a sharp on the same letter in this bar
                    letter = text.lstrip("^").rstrip(",'")
                    if text.startswith("^"):
                        sharp.add(letter)
                    elif letter in sharp:
                        text = "=" + text
                        sharp.discard(letter)
                out.append(_chord_at(chords, now) + f"{text}{length}" + ("-" if n.tie and n.pitch is not None else ""))
                now += length
            if now < bar_units:
                out.append(_chord_at(chords, now) + f"z{bar_units - now}")
            bars.append("".join(out))
        lines += [f"% {SECTION_EN.get(sec.name, 'verse')}", "V: Vocal", "|".join(bars) + "|", "V: Ins", f"Z{len(sec.bars)}|"]
    return "\n".join(lines) + "\n"


def _chord_at(chords: dict[int, str], unit: int) -> str:
    return f'"{chords.pop(unit)}"' if unit in chords else ""


def lyrics(sheet: LeadSheet | str, syllable: str) -> str:
    """One syllable per sung note (a tied continuation gets none), a line per bar, a tag per section; sections with
    no sung note are left out."""
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    blocks = []
    for sec in sheet.sections:
        rows, tied = [], False
        for bar in sec.bars:
            n = 0
            for note in sorted(bar.notes, key=lambda x: x.start):
                if note.pitch is None:
                    tied = False
                    continue
                n += 0 if tied else 1
                tied = note.tie
            if n:
                rows.append(" ".join([syllable] * n))
        if rows:
            blocks.append(f"[{LYRIC_TAG[SECTION_EN.get(sec.name, 'verse')]}]\n" + "\n".join(rows))
    return "\n\n".join(blocks) + "\n"


def seconds(sheet: LeadSheet) -> float:
    num, den = (int(x) for x in sheet.meter.split("/"))
    return sum(len(sec.bars) for sec in sheet.sections) * num * 4 / den * 60.0 / sheet.tempo


def highlight(sheet: LeadSheet | str, max_s: float = 80.0) -> LeadSheet:
    """A short demo cut: the first chorus with the verse / pre-chorus that leads into it (as much of that as fits in
    `max_s`); without a chorus, the opening sections that fit.  At least one section is always kept."""
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    names = [s.name for s in sheet.sections]
    cut = lambda a, b: LeadSheet(sheet.genre, sheet.key, sheet.meter, sheet.tempo, sheet.sections[a:b], sheet.country)
    if "후렴" in names:
        end = names.index("후렴") + 1
        start = end - 1
        while start > 0 and names[start - 1] in ("벌스", "프리") and seconds(cut(start - 1, end)) <= max_s:
            start -= 1
        return cut(start, end)
    end = 1
    while end < len(names) and seconds(cut(0, end + 1)) <= max_s:
        end += 1
    return cut(0, end)


def simplify(sheet: LeadSheet | str) -> LeadSheet:
    """The melody thinned for the sung demo: per bar the note sounding at its start and the note sounding at its
    middle, each held for half the bar (one note for the whole bar in an odd meter, or when both are the same).
    The chords stay.  This is NOT the user's melody any more: the demo says so."""
    import copy
    from .leadsheet import LeadNote
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    num, den = (int(x) for x in sheet.meter.split("/"))
    total = num * 4 * TPQ // den
    points = [0, total // 2] if num % 2 == 0 else [0]
    out = copy.deepcopy(sheet)
    for sec in out.sections:
        for bar in sec.bars:
            picks = [next((n.pitch for n in sorted(bar.notes, key=lambda x: x.start)
                           if n.pitch is not None and n.start <= at < n.start + n.dur), None) for at in points]
            if len(points) == 1 or (picks[0] is not None and picks[0] == picks[1]):
                bar.notes = [LeadNote(picks[0], 0, total)] if picks[0] is not None else []
            else:
                bar.notes = [LeadNote(p, at, total // 2) for p, at in zip(picks, points) if p is not None]
    return out


def thin(sheet: LeadSheet | str) -> LeadSheet:
    """Experiment (2026-10-10): a lighter thinning than `simplify`.  A note of an eighth or less is joined to the
    note before it in its bar (that note is held through it); a note of a quarter or more keeps its place, length
    and pitch.  A short note with nothing before it in the bar stays and takes the short notes after it."""
    import copy
    from .leadsheet import LeadNote
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    out = copy.deepcopy(sheet)
    for sec in out.sections:
        for bar in sec.bars:
            kept: list[LeadNote] = []
            for n in sorted(bar.notes, key=lambda x: x.start):
                if n.pitch is None:
                    kept.append(LeadNote(None, n.start, n.dur))          # a rest stays a rest
                    continue
                if kept and kept[-1].pitch is not None and n.dur <= TPQ // 2 and kept[-1].start + kept[-1].dur >= n.start:
                    kept[-1].dur = n.start + n.dur - kept[-1].start
                else:
                    kept.append(LeadNote(n.pitch, n.start, n.dur))
            bar.notes = kept
    return out


def fit_whole(sheet: LeadSheet, center: int) -> tuple[LeadSheet, list[dict]]:
    """Experiment (2026-10-10): the whole song moved by ONE number of octaves so that the middle of all its melody
    is nearest `center`; only a section that would pass REGISTER_TOP is lowered an octave more."""
    import copy
    out, log = copy.deepcopy(sheet), []
    every = sorted(n.pitch for s in out.sections for b in s.bars for n in b.notes if n.pitch is not None)
    base = 12 * round((center - every[len(every) // 2]) / 12) if every else 0
    for sec in out.sections:
        ps = sorted(n.pitch for b in sec.bars for n in b.notes if n.pitch is not None)
        shift = base
        while ps and ps[-1] + shift > REGISTER_TOP:
            shift -= 12
        for b in sec.bars:
            for n in b.notes:
                if n.pitch is not None:
                    n.pitch += shift
        log.append({"section": sec.name, "shift": shift, **({"low": ps[0] + shift, "high": ps[-1] + shift} if ps else {})})
    return out, log


def fit_chorus(sheet: LeadSheet, center: int) -> tuple[LeadSheet, list[dict]]:
    """Experiment (2026-10-10, "C3"): the chorus sets the register.  Every other section is moved by whole octaves to
    where the middle of its melody is nearest the middle of the chorus; then the WHOLE song is moved by one number of
    octaves so that the chorus's middle is nearest `center`, and lowered an octave as a whole while any note passes
    REGISTER_TOP (no exception per section).  Without a chorus the middle of the whole melody is the reference.
    The log holds each section's shift and its middle, and "jump": the step from the section before (semitones)."""
    import copy
    out, log = copy.deepcopy(sheet), []
    mid = lambda ps: sorted(ps)[len(ps) // 2]
    of = lambda sec: [n.pitch for b in sec.bars for n in b.notes if n.pitch is not None]
    chorus = [p for s in out.sections if s.name.split()[0] == "후렴" for p in of(s)]
    every = [p for s in out.sections for p in of(s)]
    if not every:
        return out, [{"section": s.name, "shift": 0} for s in out.sections]
    ref = mid(chorus or every)
    shifts = [0 if not of(s) or s.name.split()[0] == "후렴" else 12 * round((ref - mid(of(s))) / 12) for s in out.sections]
    whole = 12 * round((center - ref) / 12)
    while max(p + sh for s, sh in zip(out.sections, shifts) for p in of(s)) + whole > REGISTER_TOP:
        whole -= 12
    before = None
    for sec, sh in zip(out.sections, shifts):
        ps = of(sec)
        for b in sec.bars:
            for n in b.notes:
                if n.pitch is not None:
                    n.pitch += sh + whole
        row = {"section": sec.name, "shift": sh + whole}
        if ps:
            m = mid(ps) + sh + whole
            row.update(low=min(ps) + sh + whole, high=max(ps) + sh + whole, middle=m, jump=None if before is None else m - before)
            before = m
        log.append(row)
    return out, log


def fit_register(sheet: LeadSheet, center: int) -> tuple[LeadSheet, list[dict]]:
    """Every section moved by whole octaves so that the middle of its melody is nearest `center`, never above
    REGISTER_TOP, and never more than MAX_DROP semitones below the section before it (a pre-chorus must not suddenly
    sink under the verse).  Returns the sheet and what was done per section."""
    import copy
    out, log, before = copy.deepcopy(sheet), [], None
    for sec in out.sections:
        ps = sorted(n.pitch for b in sec.bars for n in b.notes if n.pitch is not None)
        if not ps:
            log.append({"section": sec.name, "shift": 0})
            continue
        mid = ps[len(ps) // 2]
        shift = 12 * round((center - mid) / 12)
        while before is not None and mid + shift < before - MAX_DROP and ps[-1] + shift + 12 <= REGISTER_TOP:
            shift += 12
        while ps[-1] + shift > REGISTER_TOP:
            shift -= 12
        for b in sec.bars:
            for n in b.notes:
                if n.pitch is not None:
                    n.pitch += shift
        before = mid + shift
        log.append({"section": sec.name, "shift": shift, "low": ps[0] + shift, "high": ps[-1] + shift})
    return out, log


ENV_INPUT = "INSTMATE_YUE_INPUT"
INPUTS = ("whole", "sections")


def input_mode() -> str:
    """How the score is prepared for the model.  "whole" (default since 2026-10-10, chosen by the user in a blind
    comparison of the same song and seed): short notes joined to the note before them (`thin`) and the whole song
    moved by one number of octaves (`fit_whole`).  "sections": the earlier way, two notes per bar (`simplify`) and
    every section moved on its own (`fit_register`).  The male demo keeps "sections" (STYLE["male"]["input"]): of
    the takes compared on 2026-10-10 the earlier male take lay nearest the reference statistics the user named as
    the yardstick.  INSTMATE_YUE_INPUT=whole | sections sets one way for every style."""
    import os
    mode = os.environ.get(ENV_INPUT, "").strip().lower() or "whole"
    if mode not in INPUTS:
        raise YueExportError(f"{ENV_INPUT} 값은 {', '.join(INPUTS)} 중 하나여야 해요 (받은 값 {mode!r})")
    return mode


def prepared(sheet: LeadSheet, style: str, simple: bool = True, mode: str | None = None) -> tuple[LeadSheet, list[dict] | None]:
    """The sheet as it is sent for a style: thinned and moved into that voice's register (and what was moved), or
    untouched with simple=False.  `mode`: see `input_mode` (None = the configured one)."""
    if not simple:
        return sheet, None
    import os
    if mode is None:                                   # the style's own way unless INSTMATE_YUE_INPUT says otherwise
        mode = input_mode() if os.environ.get(ENV_INPUT, "").strip() else STYLE[style].get("input", input_mode())
    if mode == "sections":
        return fit_register(simplify(sheet), STYLE[style]["center"])
    return fit_whole(thin(sheet), STYLE[style]["center"])


def request(sheet: LeadSheet | str, style: str = "female", genre: str | None = None, seed: int | None = None,
            simple: bool = True) -> dict:
    """The request for `yue2 generate`: the model follows the given melody and chords (cot="full" with our score).
    `simple` (default): the melody is thinned and moved into the voice's register first; "register" in the result
    says how each section was moved.  simple=False sends the melody as written (the earlier behaviour)."""
    if style not in STYLE:
        raise YueExportError(f"목소리(style)는 {', '.join(STYLE)} 중 하나여야 해요 (받은 값 {style!r})")
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    st = STYLE[style]
    sheet, _ = prepared(sheet, style, simple)
    band = GENRE_STYLE.get(genre or "", "band pop, piano, guitar, bass, drums")
    return {"id": f"instmate_{style}", "style": f"English, {band}, {st['voice']}, {sheet.tempo} BPM",
            "lyrics": lyrics(sheet, st["syllable"]), "abc": to_abc(sheet, st["octave"]), "cot": "full",
            "seed": st["seed"] if seed is None else seed, "cfg_scale": st["cfg"]}


def request_json(sheet: LeadSheet | str, style: str = "female", genre: str | None = None, seed: int | None = None) -> str:
    return json.dumps(request(sheet, style, genre, seed), ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------- running the model on this Mac (MLX port)
ENV_DIR = "INSTMATE_YUE_DIR"
FRAME_S = 0.04                  # one semantic token = 40 ms of audio
TAIL_S = 4.0                    # room for the last note to ring out
TIMEOUT_PER_S = 6.0             # generation ran at about 1-1.4x real time on an M5; this is a generous limit
MAX_S = 300.0


class YueUnavailable(Exception):
    """The model is not installed here (Korean message)."""


def model_dir():
    import os
    from pathlib import Path
    root = Path(os.environ.get(ENV_DIR) or Path(__file__).resolve().parent.parent / "models" / "yue2_mlx" / "repo")
    return root


def unavailable_reason() -> str | None:
    d = model_dir()
    if not (d / "generate.py").is_file():
        return f"YuE2(MLX)가 설치되어 있지 않아요 ({d})"
    if not (d / "8bit" / "model.safetensors").is_file():
        return "YuE2 8비트 모델 파일이 없어요 (8bit/model.safetensors)"
    try:
        import mlx.core  # noqa: F401
        import tiktoken  # noqa: F401
    except Exception:
        return "YuE2 실행에는 mlx와 tiktoken이 필요해요 (Apple Silicon 맥)"
    return None


def command(req: dict, work, out) -> list[str]:
    """The generate.py call for a request whose style / lyrics / abc were written into `work`."""
    import sys
    frames = int((req["seconds"] + TAIL_S) / FRAME_S)
    return [sys.executable, "generate.py", "--model", "8bit", "--style", req["style"], "--lyrics-file", str(work / "lyrics.txt"),
            "--cot", req["cot"], "--abc-file", str(work / "score.abc"), "--seed", str(req["seed"]),
            "--cfg-scale", str(req["cfg_scale"]), "--max-semantic-tokens", str(frames), "--out", str(out)]


def pitch_shift(wav: bytes, semitones: int) -> bytes:
    """The whole take (voice and backing) moved by `semitones`, same length (pedalboard / Rubber Band)."""
    import io
    import soundfile as sf
    try:
        from pedalboard import Pedalboard, PitchShift
    except Exception:
        raise YueUnavailable("남성 데모는 음높이를 옮기는 데 pedalboard가 필요해요") from None
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    moved = Pedalboard([PitchShift(semitones=semitones)])(audio.T, sr)
    buf = io.BytesIO()
    sf.write(buf, moved.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def generate(sheet: LeadSheet | str, style: str = "female", genre: str | None = None, full: bool = False,
             progress=None) -> tuple[bytes, dict]:
    """(WAV bytes, info).  Runs the MLX port in its own process (about real time; around 18 GB of memory at the peak,
    so nothing else heavy should run).  `full=False` sings the highlight cut, `full=True` the whole song."""
    import subprocess
    import tempfile
    import time
    from pathlib import Path
    reason = unavailable_reason()
    if reason:
        raise YueUnavailable(reason)
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    part = sheet if full else highlight(sheet)
    length = seconds(part)
    if length > MAX_S:
        raise YueExportError(f"곡이 너무 길어요 ({length:.0f}초, 한도 {MAX_S:.0f}초). 하이라이트로 만들어 보세요")
    _, register = prepared(part, style)
    shift = STYLE[style].get("shift", 0)
    req = {**request(part, style, genre), "seconds": length}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="instmate_yue_") as tmp:
        work = Path(tmp)
        (work / "lyrics.txt").write_text(req["lyrics"], encoding="utf-8")
        (work / "score.abc").write_text(req["abc"], encoding="utf-8")
        out = work / "song.wav"
        if progress:
            progress(f"YuE2가 {length:.0f}초 분량을 부르는 중 (대략 그만큼 걸려요)")
        try:
            proc = subprocess.run(command(req, work, out), cwd=model_dir(), capture_output=True, text=True,
                                  timeout=max(120.0, TIMEOUT_PER_S * length), stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise RuntimeError("YuE2 생성이 제한 시간 안에 끝나지 않았어요") from None
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"YuE2 생성 실패: {(proc.stderr or proc.stdout or '').strip()[-300:]}")
        data = out.read_bytes()
    if shift:
        data = pitch_shift(data, shift)
    return data, {"melody": "simplified", "register": register, "pitch_shift": shift,
                  "style": style, "label": STYLE[style]["label"], "score_seconds": round(length, 1), "full": full,
                  "sections": [s.name for s in part.sections], "elapsed_sec": round(time.time() - t0, 1)}
