"""A whole song from YuE2 (open-weight song model) that starts with the user's bars.

    the user's bars with their chords -> the head of a score in YuE2's own dialect, left open (`seed_prefix`)
    YuE2 continues that score, then sings and plays all of it (scripts/yue_continue.py, the MLX port as a subprocess)
    the score it wrote -> the app's lead sheet (`yueimport`), so chords, parts and edits work on the song it made

The user's bars are the start of the score as given; `make` checks that they come back note for note.  Without words
every line is sung on "아" (the user's choice, 2026-10-12).  The runner caps MLX's buffer cache and decodes in small
tiles: the same audio at about 6.5 GB instead of 18 GB (measured that day).
"""
from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

from . import yue, yueimport
from .leadsheet import KEYS, LeadSheet, _signature, _spell, parse_leadsheet, serialize_leadsheet
from .theory import TPQ

S16 = TPQ // 4
MAX_BARS = 200                  # the given bars (the user's limit, 2026-10-12; a 3-4 minute song is 90-120 bars)
WINDOW_TOKENS, TOKENS_PER_BAR = 24576, 21      # YuE2's window; a bar of score measured at 19-22 tokens
CHUNK = 4                       # YuE2 writes four bars to a chunk
MAX_SECONDS = 90.0              # a continued song, unless asked otherwise
WHOLE_SECONDS = 900.0           # a whole given song: as long as it is; the runner cuts what the model's window cannot hold
CACHE_GB, TILE = 0.2, 32        # 6.5 GB at the peak for a 36 s take; the default (no cap, one tile) took 18.3 GB
# 2026-10-12: asked for "male lead vocal", the take's voice sat at the same pitch as the female one (median D4 both), so
# a male song is the female take lowered five semitones, as the sung demo already does (yue.pitch_shift, yue.STYLE).
MALE_SHIFT = -5
# Presets chosen by ear on 2026-10-12 / 13 (work/yue_score_1012/voices): the style words set the colour of the voice;
# `shift` lowers the finished take by semitones.  The four the user kept on 2026-10-13 (soprano howled; the lowered male was dropped).
VOICES = {
    "female": {"label": "여성 (맑은 음색)", "style": "clear female lead vocal", "shift": 0},
    "mezzo": {"label": "여성 메조소프라노 (따뜻하게)", "style": "warm female mezzo-soprano lead vocal", "shift": 0},
    "female_low": {"label": "저음 여성 (허스키 알토)", "style": "low husky female alto lead vocal, warm soulful tone", "shift": 0},
    "male_low": {"label": "중저음 남성 (바리톤)", "style": "deep male baritone lead vocal, warm low voice", "shift": 0},
}
VOICE = {k: v["style"] for k, v in VOICES.items()}
KEY_SHIFTS = (0, -1, -2, -3, -4)    # a lower key on request: the whole finished take, by semitones
AH_LINE = "아 아 아 아 아 아"
RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "yue_continue.py"
ACC = {1: "^", 2: "^^", -1: "_", -2: "__", 0: "="}


class YueSongError(ValueError):
    """The given bars cannot be handed to YuE2 (the message says why)."""


class YueCancelled(RuntimeError):
    """The user cancelled the take; the runner was stopped."""


_cancel = threading.Event()     # one take runs at a time (the server refuses a second), so one flag is enough


def request_cancel() -> None:
    _cancel.set()


def cancelled() -> bool:
    return _cancel.is_set()


def _note(midi: int, sig: dict[str, int], flats: bool, local: dict) -> str:
    """ABC pitch under a real key signature; an accidental is written only where the bar's state differs."""
    letter, acc, octave = _spell(midi, sig, flats)
    mark = "" if local.get((letter, octave), sig[letter]) == acc else ACC[acc]
    local[(letter, octave)] = acc
    return mark + (letter.lower() + "'" * (octave - 5) if octave >= 5 else letter + "," * (4 - octave))


def _bar(bar, chords: dict, bar_units: int, sig, flats, sung: bool) -> str:
    """One bar as ABC.  `chords`: unit -> symbol, written where they fall (none for an Ins bar); `sung` False writes a
    whole-bar rest under the chords (the singer waits while an instrument plays)."""
    chords = dict(chords)
    if not sung:
        cuts = sorted(chords) + [bar_units]
        return "".join(f'"{chords[u]}"z{cuts[i + 1] - u}' for i, u in enumerate(cuts[:-1])) or f"z{bar_units}"
    out, now, local = [], 0, {}
    for n in sorted(bar.notes, key=lambda x: x.start):
        start = min(bar_units, round(n.start / S16))
        length = min(max(1, round(n.dur / S16)), bar_units - start)
        if length <= 0:
            continue
        if start > now:
            out.append(yue._chord_at(chords, now) + f"z{start - now}")
            now = start
        text = "z" if n.pitch is None else _note(n.pitch, sig, flats, local)
        out.append(yue._chord_at(chords, now) + f"{text}{length}" + ("-" if n.tie and n.pitch is not None else ""))
        now += length
    if now < bar_units:
        out.append(yue._chord_at(chords, now) + f"z{bar_units - now}")
    return "".join(out)


def seed_prefix(sheet: LeadSheet | str, ins_sections=None, open_end: bool = True) -> str:
    """The head of a YuE2 score: header, then every given section under its own name, four bars to a chunk as YuE2
    writes them.  A section listed in `ins_sections` is played by an instrument: its notes go to the Ins voice and the
    singer rests under the chords.  `open_end`: the score ends on an open "V: Vocal" line for the model to go on
    from; False closes it (a whole song, sung and played as given)."""
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    ins_sections = sorted(set(ins_sections or []))
    total = sum(len(s.bars) for s in sheet.sections)
    if not sheet.sections or not 1 <= total <= MAX_BARS:
        raise YueSongError(f"넘겨줄 마디는 1~{MAX_BARS}마디여야 해요 (지금 {total}마디)")
    if any(not 0 <= i < len(sheet.sections) for i in ins_sections):
        raise YueSongError(f"악기가 연주할 구간 {ins_sections}이(가) 맞지 않아요: 넘겨준 부분은 {len(sheet.sections)}개 구간이에요")
    if sheet.meter not in yue.METERS or sheet.key not in KEYS:
        raise YueSongError(f"{sheet.meter} 박자 / {sheet.key} 조는 YuE2 악보로 쓸 수 없어요")
    if not any(n.pitch is not None for s in sheet.sections for b in s.bars for n in b.notes):
        raise YueSongError("넘겨준 마디에 음이 하나도 없어요")
    if not open_end and not any(n.pitch is not None for i, s in enumerate(sheet.sections) if i not in ins_sections
                                for b in s.bars for n in b.notes):
        raise YueSongError("곡 전체를 부르려면 보컬이 부르는 구간이 있어야 해요: 넘겨준 구간을 모두 악기가 연주해요")
    if not any(c.start == 0 for c in sheet.sections[0].bars[0].chords):
        raise YueSongError("넘겨줄 마디는 첫 박에 코드가 있어야 해요 (먼저 코드를 붙여 주세요)")
    num, den = (int(x) for x in sheet.meter.split("/"))
    bar_units, sig, flats = num * 16 // den, _signature(sheet.key), KEYS[sheet.key] < 0
    lines = ["X:1", "T:", f"M:{sheet.meter}", "L:1/16", f"Q:1/4={sheet.tempo}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"', 'V: Ins clef=treble name="Ins Melody" snm="Inst."', f"K:{sheet.key}"]
    last = None
    for si, sec in enumerate(sheet.sections):
        if not sec.bars:
            raise YueSongError(f"{si + 1}번째 구간({sec.name})에 마디가 없어요")
        lines.append(f"% {yueimport.SECTION_EN_OF.get(sec.name, 'verse')}")
        played = si in ins_sections
        for at in range(0, len(sec.bars), CHUNK):
            vocal, ins = [], []
            for bar in sec.bars[at:at + CHUNK]:
                chords = {round(c.start / S16): c.symbol for c in bar.chords}
                if 0 not in chords:
                    chords[0] = last                          # YuE2 names the chord in every bar
                last = chords[max(chords)]
                vocal.append(_bar(bar, chords, bar_units, sig, flats, sung=not played))
                if played:
                    ins.append(_bar(bar, {}, bar_units, sig, flats, sung=True))
                for text in (vocal[-1], ins[-1] if played else None):      # a bar off the sixteenth grid would only fail in the runner
                    try:
                        yueimport._read_bar(text, sig, bar_units, "")
                    except yueimport.AbcError:
                        raise YueSongError(f"{si + 1}번째 구간({sec.name})의 {at + len(vocal)}마디가 16분음표 단위에 맞지 않아요") from None
            lines += ["V: Vocal", "|".join(vocal) + "|", "V: Ins", ("|".join(ins) + "|") if played else f"Z{len(vocal)}|"]
    return "\n".join(lines + (["V: Vocal", ""] if open_end else [""]))


def pick_seed(sheet: LeadSheet | str, index=None) -> LeadSheet:
    """Some sections of a song as the given part (same key, meter, tempo, genre).  `index`: one section, a list of
    sections in the song's own order, or "all"; it may be left out only when the song has a single section."""
    if isinstance(sheet, str):
        sheet = parse_leadsheet(sheet)
    if index is None:
        if len(sheet.sections) != 1:
            raise YueSongError(f"이 곡은 구간이 {len(sheet.sections)}개예요: 넘겨줄 구간을 골라 주세요")
        index = [0]
    elif index == "all":
        index = list(range(len(sheet.sections)))
    elif isinstance(index, int):
        index = [index]
    index = list(index)
    if not index or any(not 0 <= i < len(sheet.sections) for i in index):
        raise YueSongError(f"구간 번호 {index}이(가) 맞지 않아요: 이 곡의 구간 번호는 0~{len(sheet.sections) - 1}이에요")
    if any(b <= a for a, b in zip(index, index[1:])):
        raise YueSongError(f"구간 번호 {index}은(는) 곡의 순서대로 한 번씩만 적어 주세요")
    return LeadSheet(genre=sheet.genre, key=sheet.key, meter=sheet.meter, tempo=sheet.tempo,
                     sections=[sheet.sections[i] for i in index], country=sheet.country)


EXTRA_PARTS = ("Chorus", "Bridge", "Outro")     # parts the user may ask to be added to the plan (the page's boxes)


def _tags(full: bool, seconds: float, extra) -> list[str]:
    if any(e not in EXTRA_PARTS for e in extra):
        raise YueSongError(f"더할 수 있는 파트는 {', '.join(EXTRA_PARTS)} 중에서만 고를 수 있어요 (받은 값 {list(extra)})")
    tags = ["Verse", "Chorus"]
    if full:
        tags = ["Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus"]
        for more in ("Verse", "Chorus", "Chorus", "Outro"):
            if seconds >= 150 + 30 * (len(tags) - 6):
                tags.append(more)
    return tags + list(extra)


def ah_lyrics(full: bool = False, seconds: float = 120.0, extra=()) -> str:
    """No words: every syllable "아", four lines to a section.  A verse and a chorus, or the sections of a whole song:
    the model plans about as much song as there are words, so a longer song gets more sections.  `extra`: parts added
    at the end on request (the page's "후렴 추가" boxes)."""
    return "\n\n".join(f"[{tag}]\n" + "\n".join([AH_LINE] * 4) for tag in _tags(full, seconds, extra)) + "\n"


def model_lyrics(full: bool = False, seconds: float = 120.0, extra=()) -> str:
    """The parts only, no lines: the model is left to write the words itself (the page's "랜덤 가사").  Not run on the
    Mac yet: if the model sings nothing for an empty part, this is the one place to change."""
    return "\n\n".join(f"[{tag}]" for tag in _tags(full, seconds, extra)) + "\n"


def words_for(lyrics: str | None, mode: str, full: bool, seconds: float, extra) -> str:
    """The lyrics file: the user's words, else the model's own (mode "model"), else "아"."""
    if lyrics and lyrics.strip():
        return yue.tagged(lyrics)
    return (model_lyrics if mode == "model" else ah_lyrics)(full, seconds, extra)


def style_text(genre: str | None, voice: str, tempo: int, lyrics: str | None = None) -> str:
    if voice not in VOICE:
        raise YueSongError(f"목소리는 {', '.join(VOICE)} 중 하나여야 해요 (받은 값 {voice!r})")
    band = yue.GENRE_STYLE.get(genre or "", "band pop, piano, guitar, bass, drums")
    return f"{yue.language_of(lyrics)}, {band}, {VOICE[voice]}, {tempo} BPM"


def command(req: dict, work, out) -> list[str]:
    """The runner call for a request whose lyrics / prefix were written into `work` (cwd: the MLX port's folder)."""
    low = req.get("low_memory", True)                          # off: no cache cap, one decoder tile (about 18 GB at the peak)
    return [sys.executable, str(RUNNER), "--model", "8bit", "--style", req["style"], "--lyrics-file", str(Path(work) / "lyrics.txt"),
            "--prefix-file", str(Path(work) / "prefix.abc"), "--seed", str(req["seed"]), "--cfg-scale", str(req["cfg_scale"]),
            "--max-seconds", str(req["max_seconds"]), "--cache-gb", str(CACHE_GB) if low else "0", "--tile", str(TILE) if low else "1024",
            "--kv-bits", "8", "--out", str(out)]   # 8-bit KV cache (instmate.yuekv), chosen by ear 2026-10-16


def _same(a, b) -> bool:
    return [(n.pitch, n.start, n.dur, n.tie) for n in a.notes] == [(n.pitch, n.start, n.dur, n.tie) for n in b.notes]


def _kept(given: LeadSheet, ins_sections, sheet: LeadSheet, ins: list[dict]) -> bool:
    """The given bars open the score note for note: sung sections in the Vocal voice, played ones in the Ins voice."""
    got = [b for s in sheet.sections for b in s.bars]
    at = 0
    for si, sec in enumerate(given.sections):
        for bar in sec.bars:
            if at >= len(got):
                return False
            if si in ins_sections:
                want = [(round(n.start / S16), max(1, round(n.dur / S16)), n.pitch) for n in bar.notes if n.pitch is not None]
                have = [(n["start"] // S16, n["dur"] // S16, n["pitch"]) for n in ins if n["bar"] == at]
                if want != have or any(n.pitch is not None for n in got[at].notes):
                    return False
            elif not _same(got[at], bar):
                return False
            at += 1
    return True


def limits(tempo: int, meter: str = "4/4") -> dict:
    """What the page tells the user before anything runs: the app's own caps and how much one YuE2 run can hold at
    this tempo (request, score and audio share the model's window)."""
    num, den = (int(x) for x in meter.split("/"))
    bar_s = 60.0 / tempo * num * 4 / den
    window = int((WINDOW_TOKENS - 400) / (TOKENS_PER_BAR + bar_s * 25))
    return {"max_bars": MAX_BARS, "max_seconds": int(WHOLE_SECONDS), "window_bars": window, "window_seconds": int(window * bar_s)}


def count_bars(abc: str) -> int:
    """Bars written so far in the Vocal voice (works on an open prefix too)."""
    bars, voice = 0, None
    for line in abc.splitlines():
        line = line.strip()
        if line.startswith("V:"):
            voice = line[2:].strip().split()[0]
        elif line and not line.startswith("%") and voice == "Vocal" and "|" in line:
            for seg in (s.strip() for s in line.split("|")):
                m = yueimport.MULTI.fullmatch(seg)
                bars += int(m.group(1) or 1) if m else (1 if seg else 0)
    return bars


def plan_rank(info: dict, target_bars: int = 0) -> tuple:
    return (-abs(info["bars"] - target_bars), info["score"]) if target_bars else (info["score"],)


def plan_score(abc: str, given_bars: int = 0, target_bars: int = 0) -> dict:
    """How usable a plan is, read from the score alone: it must be readable; a sung chorus counts most, then how
    many different bars it sings (a plan that only repeats the given bars scores low), then its length."""
    try:
        sheet, _, info = yueimport.read(abc)
    except yueimport.AbcError as exc:
        return {"ok": False, "score": -1.0, "error": str(exc)}
    if any(not section.bars for section in sheet.sections):
        return {"ok": False, "score": -1.0, "error": "마디가 없는 빈 구간이 있어요"}
    def sung(bars):
        return [tuple((n.pitch, n.start, n.dur) for n in b.notes if n.pitch is not None) for b in bars if any(n.pitch is not None for n in b.notes)]
    every = sung([b for s in sheet.sections for b in s.bars])
    chorus = sung([b for s in sheet.sections if s.name == "후렴" for b in s.bars])
    verse = sung([b for s in sheet.sections if s.name == "벌스" for b in s.bars])
    variety = round(len(set(every)) / len(every), 3)
    mean = lambda rows: sum(p for r in rows for p, _, _ in r) / max(1, sum(len(r) for r in rows))
    higher = bool(chorus and verse and mean(chorus) > mean(verse))
    fit = max(0.0, 1.0 - abs(info["bars"] - target_bars) / target_bars) if target_bars else None
    length = 1.5 * fit if target_bars else 0.5 * min(info["bars"], 32) / 32     # a whole song: the asked length counts most
    score = (2.0 if chorus else 0.0) + variety + length + (0.25 if higher else 0.0)
    return {"ok": True, "score": round(score, 3), **({"length_fit": round(fit, 3)} if target_bars else {}), "has_chorus": bool(chorus), "variety": variety, "chorus_higher": higher,
            "bars": info["bars"], "sections": info["sections"], "given_bars": given_bars}


def repair_generated_rests(abc: str, keep: str = "") -> tuple[str, int]:
    """Fit rests in new model bars to the meter; never alter the prefix or pitched notes."""
    if keep and (not abc.startswith(keep) or not keep.endswith("\n")):
        raise YueSongError("입력 악보를 보존할 수 없는 생성 결과예요")
    meter = re.search(r"^M:(\d+)/(\d+)\s*$", abc, re.M)
    if not meter:
        return abc, 0
    units = int(meter[1]) * 16 // int(meter[2])
    changed, output = 0, []
    for line in abc[len(keep):].splitlines(keepends=True):
        if "|" not in line or line.lstrip().startswith(("%", "V:", "M:", "L:", "Q:", "K:")):
            output.append(line)
            continue
        bars = line.rstrip("\r\n").split("|")
        for i, bar in enumerate(bars):
            if not bar.strip() or yueimport.MULTI.fullmatch(bar.strip()):
                continue
            tokens, pos = [], 0
            while pos < len(bar.rstrip()):
                match = yueimport.TOKEN.match(bar, pos)
                if not match:
                    break
                tokens.append(match)
                pos = match.end()
            if pos != len(bar.rstrip()):
                continue
            total = sum(int(m["len"] or 1) for m in tokens if m["letter"])
            excess = total - units
            rests = sum(int(m["len"] or 1) for m in tokens if m["letter"] == "z")
            if excess == 0 or excess > rests or not rests:
                continue
            if excess < 0:
                rest = next(m for m in reversed(tokens) if m["letter"] == "z")
                bars[i] = bar[:rest.start()] + f'z{int(rest["len"] or 1) - excess}' + bar[rest.end():]
            else:
                parts = []
                for m in tokens:
                    text = m[0]
                    if m["letter"] == "z" and excess:
                        length = int(m["len"] or 1)
                        take = min(excess, length)
                        excess -= take
                        text = f"z{length - take}" if length > take else ""
                    parts.append(text)
                bars[i] = "".join(parts)
            changed += 1
        output.append("|".join(bars) + line[len(line.rstrip("\r\n")):])
    return keep + "".join(output), changed


def repair_plan(abc: str, keep: str = "") -> tuple[str, int]:
    """(a readable score, how many sections were dropped from its end).  A long plan sometimes breaks near its end
    (a stray header field, a short bar): whole sections are dropped from the end until the score reads.  `keep`: the
    start that must stay (the given bars, the first part); a score broken inside it is refused."""
    lines = abc.splitlines()
    marks = [i for i, ln in enumerate(lines) if ln.startswith("% ")]
    floor = len(keep.rstrip("\n").splitlines()) if keep else 0   # a section that starts inside `keep` is never dropped
    cuts = [len(lines)] + [i for i in reversed(marks) if i >= floor and i > 0]
    for dropped, end in enumerate(cuts):
        text = "\n".join(lines[:end]) + "\n"
        try:
            yueimport.read(text)
            return (abc if dropped == 0 else text), dropped
        except yueimport.AbcError:
            continue
    if keep and not keep.endswith("V: Vocal\n"):               # an open start alone is no plan: the model wrote nothing that reads
        try:
            yueimport.read(keep)
            return keep, len(cuts)
        except yueimport.AbcError:
            pass
    raise YueSongError("끝 구간들을 빼도 악보를 읽을 수 없어요")


def first_part(abc: str, given_bars: int = 0) -> str:
    """The score up to the end of the first chorus that starts after the given bars (the first part a listener judges
    the song by; a chorus inside the given bars is the user's, and the cut never falls inside what was given);
    unchanged when there is no such chorus."""
    lines = abc.splitlines()
    marks = [i for i, ln in enumerate(lines) if ln.startswith("% ")]
    for start in (i for i in marks if lines[i] == "% chorus"):
        if count_bars("\n".join(lines[:start])) >= given_bars:
            end = next((i for i in marks if i > start), len(lines))
            return "\n".join(lines[:end]) + "\n"
    return abc


def _run(prefix: str, style: str, words: str, seed: int, max_seconds: float, flags: list[str], progress, said: str, low_memory: bool = True):
    """(wav bytes, the score that was sung, the whole plan, the runner's JSON)."""
    import subprocess
    import tempfile
    reason = yue.unavailable_reason()
    if reason:
        raise yue.YueUnavailable(reason)
    req = {"style": style, "seed": seed, "cfg_scale": 2.0, "max_seconds": float(max_seconds), "low_memory": low_memory}
    with tempfile.TemporaryDirectory(prefix="instmate_yuesong_") as tmp:
        work = Path(tmp)
        (work / "lyrics.txt").write_text(words, encoding="utf-8")
        (work / "prefix.abc").write_text(prefix, encoding="utf-8")
        out = work / "song.wav"
        _cancel.clear()
        for attempt in (0, 1):
            if progress:
                progress(said + (" (악보를 다시 쓰는 중)" if attempt else ""))
            proc = subprocess.Popen(command({**req, "seed": seed + attempt * 100}, work, out) + flags, cwd=yue.model_dir(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)
            limit = max(240.0, yue.TIMEOUT_PER_S * max_seconds)
            waited = 0.0
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    waited += 0.5
                    if _cancel.is_set():
                        proc.kill()
                        proc.communicate()
                        _cancel.clear()
                        raise YueCancelled("취소했어요")
                    if waited >= limit:
                        proc.kill()
                        proc.communicate()
                        raise RuntimeError("YuE2 생성이 제한 시간 안에 끝나지 않았어요") from None
            proc = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
            if proc.returncode != 3 or "--no-continue" in flags:
                break
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"YuE2 생성 실패: {(proc.stderr or proc.stdout or '').strip()[-300:]}")
        abc = out.with_suffix(".abc").read_text(encoding="utf-8")
        whole = out.with_suffix(".full.abc")
        return out.read_bytes(), abc, (whole.read_text(encoding="utf-8") if whole.is_file() else abc), json.loads(proc.stdout.strip().splitlines()[-1])


def _info(abc: str, genre_ko: str, ran: dict, t0: float) -> tuple[LeadSheet, list, dict]:
    import time
    sheet, ins, info = yueimport.read(abc, genre_ko)
    return sheet, ins, {"abc": abc, "leadsheet": serialize_leadsheet(sheet), "bars": info["bars"], "sections": info["sections"], "ins": ins,
                        "seconds": ran["seconds"], "wanted_seconds": ran.get("wanted_seconds"), "cut": bool(ran.get("cut")),
                        "plans": ran.get("plans", []), "plan_seed": ran.get("plan_seed"), "stages": ran["stages"],
                        "elapsed_sec": round(time.time() - t0, 1)}


def make(seed_sheet: LeadSheet | str, genre: str | None = None, voice: str = "female", lyrics: str | None = None,
         seed: int = 1, max_seconds: float | None = None, progress=None, ins_sections=None,
         continue_song: bool = True, plans: int = 1, part: str = "all", key_shift: int = 0,
         low_memory: bool = True, lyrics_mode: str = "ah", extra=()) -> tuple[bytes, dict]:
    """(WAV bytes, info).  info: abc (the score that was sung), leadsheet (the same as the app's text), bars, sections,
    given_bars, given_kept, lyrics ("ah" | "given"), seconds, cut, plans (the candidates' scores), stages (peak memory
    per stage), elapsed_sec.  `plans`: how many scores the model writes before the best is kept (`plan_score`).
    `part` "first": only the score up to the end of the first chorus is sung; the whole plan is in info["full_plan"]."""
    import time
    if part not in ("all", "first") or not 1 <= plans <= 8:
        raise YueSongError(f"part는 all / first, plans는 1~8이어야 해요 (받은 값 {part!r}, {plans})")
    if key_shift not in KEY_SHIFTS:
        raise YueSongError(f"키 낮춤(key_shift)은 {KEY_SHIFTS} 중 하나여야 해요 (받은 값 {key_shift})")
    if isinstance(seed_sheet, str):
        seed_sheet = parse_leadsheet(seed_sheet)
    ins_sections = sorted(set(ins_sections or []))
    prefix = seed_prefix(seed_sheet, ins_sections, open_end=continue_song)
    if max_seconds is None:
        max_seconds = MAX_SECONDS if continue_song else WHOLE_SECONDS
    given = sum(len(s.bars) for s in seed_sheet.sections)
    words = lyrics if lyrics and lyrics.strip() else None
    flags = ["--given-bars", str(given)] + (["--plans", str(plans)] if continue_song else ["--no-continue"]) \
        + (["--cut", "first"] if part == "first" else [])
    t0 = time.time()
    data, abc, whole, ran = _run(prefix, style_text(genre, voice, seed_sheet.tempo, lyrics), words_for(lyrics, lyrics_mode, False, 120.0, extra), seed, max_seconds, flags,
                                 progress, "YuE2가 주어진 마디에 이어 곡을 쓰고 부르는 중", low_memory)
    sheet, ins, info = _info(abc, seed_sheet.genre, ran, t0)
    data, shifted = _voiced(data, voice, sheet.key, key_shift, low_memory)
    return data, {**info, **shifted, "full_plan": whole, "stage": "first" if part == "first" else "all", "given_bars": given,
                  "given_kept": _kept(seed_sheet, ins_sections, sheet, ins), "ins_sections": ins_sections, "continued": continue_song,
                  "lyrics": "given" if words else lyrics_mode, "voice": voice, "seed": ran.get("plan_seed") or seed}


def make_first(seed_sheet: LeadSheet | str, genre: str | None = None, voice: str = "female", lyrics: str | None = None,
               seed: int = 1, plans: int = 3, ins_sections=None, max_seconds: float = 120.0, progress=None,
               check: bool = False, tries: int = 2, key_shift: int = 0, low_memory: bool = True,
               lyrics_mode: str = "ah", extra=()) -> tuple[bytes, dict]:
    """Step one: the model writes `plans` scores from the given bars, the best is kept and its first part (to the end
    of the first chorus) is sung.  The user listens, then `make_full` goes on from info["abc"].
    `check`: the take is compared with its own score (`yuecheck.follow`); below yuecheck.GOOD it is made again with
    the next seed, up to `tries` takes, and the best one is kept (info: follow, follow_tries, tries)."""
    from . import yuecheck
    best, marks = None, []
    for k in range(max(1, tries) if check else 1):               # a new try writes new plans (seeds never overlap)
        wav, info = make(seed_sheet, genre, voice, lyrics, seed + k * plans, max_seconds, progress, ins_sections, True, plans, "first",
                         key_shift, low_memory, lyrics_mode, extra)
        if not check:
            return wav, info
        try:
            if progress:
                progress("부른 것이 악보와 맞는지 확인하는 중")
            mark = yuecheck.follow(wav, info["abc"], shift=info["male_shift"])     # a lowered take is compared in its own key
        except yuecheck.CheckError as exc:
            return wav, {**info, "follow": {"match": None, "skipped": str(exc)}, "follow_tries": [], "tries": k + 1}
        marks.append({"seed": info["seed"], "plans": info["plans"], **mark})
        if best is None or (mark["match"] or 0.0) > (best[2]["match"] or 0.0):
            best = (wav, info, mark)
        if (mark["match"] or 0.0) >= yuecheck.GOOD:
            break
    wav, info, mark = best
    return wav, {**info, "follow": mark, "follow_tries": marks, "tries": len(marks)}


def _voiced(data: bytes, voice: str, key: str, key_shift: int = 0, low_memory: bool = True) -> tuple[bytes, dict]:
    """The whole take lowered by the preset's shift plus the asked key shift; the score stays in its key and
    `sounding_key` says where the take now sounds (`male_shift`: the total, the name is the first version's)."""
    total = VOICES[voice]["shift"] + key_shift
    base = {"key_shift": key_shift, "low_memory": low_memory}
    if not total:
        return data, {**base, "male_shift": 0, "sounding_key": key}
    from .yueblend import lower_key
    return yue.pitch_shift(data, total), {**base, "male_shift": total, "sounding_key": lower_key(key, -total)}


def make_full(first_abc: str, genre: str | None = None, voice: str = "female", lyrics: str | None = None, seed: int = 1,
              target_seconds: float = 210.0, progress=None, sheet_genre: str = "팝", plans: int = 3, key_shift: int = 0,
              low_memory: bool = True, lyrics_mode: str = "ah", extra=()) -> tuple[bytes, dict]:
    """Step two: the whole song.  The first part's score stays as written; the model goes on from its end.  It writes
    The runner ranks valid plans by distance to the target, then quality, and extends an undersized plan for at most
    four rounds. The first part stays intact; an undersized result is saved with an explicit warning."""
    import math
    import time
    try:
        first, _, finfo = yueimport.read(first_abc, sheet_genre)
    except yueimport.AbcError as exc:
        raise YueSongError(f"1절 악보를 읽을 수 없어요: {exc}") from None
    if not 30.0 <= target_seconds <= WHOLE_SECONDS:
        raise YueSongError(f"곡 길이(target_seconds)는 30~{WHOLE_SECONDS:g}초여야 해요 (받은 값 {target_seconds})")
    num, den = (int(x) for x in first.meter.split("/"))
    bar_s = 60.0 / first.tempo * num * 4 / den
    want = min(MAX_BARS, limits(first.tempo, first.meter)["window_bars"], math.ceil(target_seconds / bar_s))
    words = lyrics if lyrics and lyrics.strip() else None
    t0 = time.time()
    data, abc, _, ran = _run(first_abc, style_text(genre, voice, first.tempo, lyrics), words_for(lyrics, lyrics_mode, True, target_seconds, extra), seed,
                             min(WHOLE_SECONDS, target_seconds * 1.3 + 10), ["--plans", str(plans), "--target-bars", str(want)],
                             progress, "YuE2가 1절에 이어 곡 전체를 쓰고 부르는 중", low_memory)
    _, _, info = _info(abc, sheet_genre, ran, t0)
    if key_shift not in KEY_SHIFTS:
        raise YueSongError(f"키 낮춤(key_shift)은 {KEY_SHIFTS} 중 하나여야 해요 (받은 값 {key_shift})")
    data, shifted = _voiced(data, voice, first.key, key_shift, low_memory)
    length_met = info["bars"] >= want * 0.9 and info["seconds"] >= target_seconds * 0.9
    length = {"target_seconds": target_seconds, "length_met": length_met}
    if not length_met:
        length["warning"] = f"목표 길이에 미달했어요. 요청 {target_seconds:g}초·{want}마디, 실제 {info['seconds']:g}초·{info['bars']}마디예요. 짧은 결과로 저장했어요."
    return data, {**info, **shifted, **length, "stage": "full", "first_bars": finfo["bars"], "first_kept": abc.startswith(first_abc), "target_bars": want,
                  "lyrics": "given" if words else lyrics_mode, "voice": voice, "seed": seed}
