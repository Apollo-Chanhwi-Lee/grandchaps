"""A score YuE2 wrote itself (the .abc beside a free or seeded take) -> the app's lead sheet.

YuE2's own dialect: L:1/16, a real key signature in K:, "% section" comments, a "Vocal" voice with chord symbols and an
"Ins" voice, written in alternating chunks of the same number of bars; Z / Zn are whole-bar rests.
Only the Vocal voice and the chords go into the lead sheet; the Ins line is returned beside it as plain notes.
Nothing is guessed: a bar whose lengths do not add up, a chord or section name the app does not know, chunks of
different length or a score nobody sings raise AbcError with the place.
"""
from __future__ import annotations

import re

from .leadsheet import Bar, ChordMark, KEYS, LeadNote, LeadSheet, METERS, SECTION_NAMES, Section, _signature, _valid_chord
from .theory import TPQ
from .yue import SECTION_EN

S16 = TPQ // 4
SECTION_KO = {en: ko for ko, en in SECTION_EN.items()}
# the lead sheet's own spelling wins (yue.SECTION_EN spells the bridge differently; a bridge the model wrote could not be saved)
SECTION_KO.update({en: name for en, ko in list(SECTION_KO.items()) for name in SECTION_NAMES if ko not in SECTION_NAMES and name[0] == ko[0]})
SECTION_EN_OF = {**SECTION_EN, **{ko: en for en, ko in SECTION_KO.items()}}      # every Korean spelling -> YuE2's name
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
TOKEN = re.compile(r'\s*(?:"(?P<chord>[^"]*)"|(?P<acc>\^{1,2}|_{1,2}|=)?(?P<letter>[A-Ga-gz])(?P<oct>[,\']*)(?P<len>\d*)(?P<tie>-?))')
MULTI = re.compile(r"Z(\d*)")


class AbcError(ValueError):
    """The score cannot be read as written (the message names the place)."""


def _bars(text: str) -> list[str]:
    return [b for b in (p.strip() for p in text.strip().split("|")) if b]


def _expand(chunk: list[str]) -> list[str | None]:
    """Bars of one chunk; a multi-bar rest becomes that many None entries."""
    out: list[str | None] = []
    for b in chunk:
        m = MULTI.fullmatch(b)
        out += [None] * int(m.group(1) or 1) if m else [b]
    return out


def _read_bar(text: str | None, sig: dict[str, int], bar_units: int, where: str):
    """(notes as (pitch | None, start unit, units, tie), chords as (symbol, start unit)) of one bar."""
    if text is None:
        return [(None, 0, bar_units, False)], []
    notes, chords, now, pos, local = [], [], 0, 0, {}
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise AbcError(f"{where}: '{text[pos:pos + 12]}'을(를) 읽을 수 없어요")
        pos = m.end()
        if m.group("chord") is not None:
            chords.append((m.group("chord"), now))
            continue
        units = int(m.group("len") or 1)
        letter = m.group("letter")
        if letter == "z":
            notes.append((None, now, units, False))
        else:
            up, octs = letter.upper(), m.group("oct")
            octave = (5 if letter.islower() else 4) + octs.count("'") - octs.count(",")
            acc = m.group("acc")
            if acc is not None:
                local[(up, octave)] = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}[acc]
            shift = local.get((up, octave), sig[up])
            notes.append((12 * (octave + 1) + LETTER_PC[up] + shift, now, units, bool(m.group("tie"))))
        now += units
    if now != bar_units:
        raise AbcError(f"{where}: 마디 길이가 16분음표 {now}개인데 박자에는 {bar_units}개가 필요해요 ('{text}')")
    return notes, chords


def read(text: str, genre: str = "jpop") -> tuple[LeadSheet, list[dict], dict]:
    """(lead sheet of the Vocal voice with chords, Ins notes as {bar, start, dur, pitch} in ticks, info)."""
    head = {m.group(1): m.group(2).strip() for m in re.finditer(r"^([A-Z]):[ \t]*(.*)$", text, re.M) if m.group(1) in "MLQK"}
    for tag in "MLQK":
        if tag not in head:
            raise AbcError(f"머리글 {tag}: 줄이 없어요")
    if head["L"] != "1/16":
        raise AbcError(f"기본 음표 길이 {head['L']}은(는) 쓸 수 없어요 (1/16만 돼요)")
    meter, key = head["M"], head["K"]
    if meter not in METERS:
        raise AbcError(f"{meter} 박자는 쓸 수 없어요")
    if key not in KEYS:
        raise AbcError(f"조 {key}은(는) 앱이 아는 조가 아니에요")
    q = re.fullmatch(r"1/4=(\d+)", head["Q"])
    if not q or not 20 <= int(q.group(1)) <= 400:
        raise AbcError(f"템포 '{head['Q']}'이(가) 1/4=N 꼴(N은 20~400)이 아니에요")
    num, den = (int(x) for x in meter.split("/"))
    bar_units, sig = num * 16 // den, _signature(key)
    sections: list[Section] = []
    ins: list[dict] = []
    name, voice, pending, bar_no = None, None, None, 0
    body = text.split(f"K:{head['K']}", 1)[1] if text.count(f"K:{head['K']}") == 1 else None
    if body is None:
        raise AbcError("K: 줄이 정확히 한 번 있어야 해요")
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            en = line[1:].strip()
            if en not in SECTION_KO:
                raise AbcError(f"모르는 구간 이름 '{en}'이에요")
            if pending is not None:
                raise AbcError(f"구간 '{en}'이(가) 앞 구간의 Ins 묶음보다 먼저 시작해요")
            name = SECTION_KO[en]
            sections.append(Section(name))
            continue
        if line.startswith("V:"):
            voice = line[2:].strip().split()[0]
            if voice not in ("Vocal", "Ins"):
                raise AbcError(f"모르는 성부 '{voice}'이에요")
            continue
        if name is None or voice is None:
            raise AbcError(f"첫 구간 줄이나 성부(V:) 줄보다 앞에 음표가 있어요: '{line[:30]}'")
        chunk = _expand(_bars(line))
        if voice == "Vocal":
            if pending is not None:
                raise AbcError(f"[{name}] Vocal 묶음이 연달아 두 번 나와요")
            pending = chunk
            for i, b in enumerate(chunk):
                notes, chords = _read_bar(b, sig, bar_units, f"[{name}] 보컬 {bar_no + i + 1}마디")
                for sym, _ in chords:
                    if not _valid_chord(sym):
                        raise AbcError(f"[{name}] {bar_no + i + 1}마디: 코드 '{sym}'은(는) 앱이 아는 코드가 아니에요")
                sections[-1].bars.append(Bar([LeadNote(p, s * S16, u * S16, t) for p, s, u, t in notes],
                                             [ChordMark(sym, s * S16) for sym, s in chords]))
        else:
            if pending is None:
                raise AbcError(f"[{name}] 앞에 Vocal 묶음이 없는 Ins 묶음이 있어요")
            if len(chunk) != len(pending):
                raise AbcError(f"[{name}] {bar_no + 1}마디부터: Vocal은 {len(pending)}마디, Ins는 {len(chunk)}마디예요")
            for i, b in enumerate(chunk):
                for p, s, u, _ in _read_bar(b, sig, bar_units, f"[{name}] 악기 {bar_no + i + 1}마디")[0]:
                    if p is not None:
                        ins.append({"bar": bar_no + i, "start": s * S16, "dur": u * S16, "pitch": p})
            bar_no += len(chunk)
            pending = None
    if pending is not None:
        raise AbcError("마지막 Vocal 묶음에 짝이 되는 Ins 묶음이 없어요")
    if not any(n.pitch is not None for s in sections for b in s.bars for n in b.notes):
        raise AbcError("악보에 부르는 음이 하나도 없어요")
    sheet = LeadSheet(genre=genre, key=key, meter=meter, tempo=int(q.group(1)), sections=sections)
    info = {"bars": bar_no, "sections": [(s.name, len(s.bars)) for s in sections], "sung_notes": sum(n.pitch is not None for s in sections for b in s.bars for n in b.notes),
            "ins_notes": len(ins), "chords": sum(len(b.chords) for s in sections for b in s.bars)}
    return sheet, ins, info
