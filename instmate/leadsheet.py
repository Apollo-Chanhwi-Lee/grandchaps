"""Lead sheet text format: header + sections of ABC-subset bars with chord symbols.

Text format (see docs/leadsheet_format.md):

    [장르:발라드] [조:C] [박자:4/4] [템포:72]            (optional last tag: [국가:한국])
    [구조:후렴]
    |"C" E2 G2 c2 B2 | "Am" A4 G4 |

The unit note length is fixed at 1/8 (240 ticks), so `C` alone is an eighth note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from .theory import TPQ, Note

UNIT = TPQ // 2                      # L:1/8 -> 240 ticks
PITCH_MIN, PITCH_MAX = 21, 108
GENRES = ("발라드", "록", "댄스", "클래식", "팝", "민요", "기타", "케이팝", "제이팝", "뉴에이지", "힙합", "피아노 록", "팝펑크",
          "하드록", "감성록", "모던록")
METERS = {"4/4": 4 * TPQ, "3/4": 3 * TPQ, "2/4": 2 * TPQ, "6/8": 3 * TPQ}
SECTION_NAMES = ["인트로", "벌스", "프리", "후렴", "브릿지", "간주", "아웃트로", "구간"]   # 구간 = unlabeled phrase/clip
HEADER_TAGS = ("장르", "조", "박자", "템포")
OPTIONAL_TAGS = ("국가",)            # style origin; absent = no country condition (the v1 data)
COUNTRIES = ("한국", "일본", "중국", "영국", "독일", "서양", "기타")
QUALITIES = {"", "m", "7", "maj7", "m7", "dim", "dim7", "aug", "sus2", "sus4",
             "6", "m6", "9", "add9", "m7b5", "7sus4"}

# key name -> number of sharps (negative = flats)
_MAJOR = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
          "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
_MINOR = {"Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7,
          "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7}
KEYS = {**_MAJOR, **_MINOR}

_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_ORDER, _FLAT_ORDER = "FCGDAEB", "BEADGCF"
_ACC_VAL = {"^^": 2, "^": 1, "=": 0, "_": -1, "__": -2}
_ACC_STR = {2: "^^", 1: "^", 0: "=", -1: "_", -2: "__"}
_SPELL_SHARP = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
                ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
_SPELL_FLAT = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
               ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0)]

_LEN = r"(\d*(?:/\d*)?)"
MAX_NUMBER_DIGITS = 6                # digits allowed in a note length or tempo; checked BEFORE int() (Python's int digit limit)
_TEMPO_RE = re.compile(r"[0-9]{1,%d}" % MAX_NUMBER_DIGITS)
_NOTE_RE = re.compile(r"(\^\^|\^|__|_|=)?([A-Ga-g])([',]*)" + _LEN + r"(-?)")
_REST_RE = re.compile(r"z" + _LEN)
_TAG_RE = re.compile(r"\[([^\[\]:]*):([^\[\]]*)\]")
_SEC_RE = re.compile(r"\[구조:([^\[\]]*)\]")
_CHORD_RE = re.compile(r"([A-G][#b]?)([^/]*)(?:/([A-G][#b]?))?")


class LeadsheetError(ValueError):
    """Raised for any lead-sheet problem; the message (Korean) includes the location."""


@dataclass
class LeadNote:
    pitch: int | None
    start: int                      # tick within the bar
    dur: int
    tie: bool = False


@dataclass
class ChordMark:
    symbol: str
    start: int                      # tick within the bar


@dataclass
class Bar:
    notes: list[LeadNote] = field(default_factory=list)
    chords: list[ChordMark] = field(default_factory=list)


@dataclass
class Section:
    name: str
    bars: list[Bar] = field(default_factory=list)


@dataclass
class LeadSheet:
    genre: str
    key: str
    meter: str
    tempo: int
    sections: list[Section] = field(default_factory=list)
    country: str | None = None


# ---------- small helpers ----------
def _valid_chord(name: str) -> bool:
    m = _CHORD_RE.fullmatch(name)
    return bool(m) and m.group(2) in QUALITIES


def _signature(key: str) -> dict[str, int]:
    """Letter -> accidental (+1/-1/0) implied by the key signature."""
    n = KEYS[key]
    sig = {c: 0 for c in "ABCDEFG"}
    if n > 0:
        for c in _SHARP_ORDER[:n]:
            sig[c] = 1
    elif n < 0:
        for c in _FLAT_ORDER[:-n]:
            sig[c] = -1
    return sig


def _bar_ticks(meter: str) -> int:
    if meter not in METERS:
        raise LeadsheetError(f"[헤더] 지원하지 않는 박자: '{meter}'")
    return METERS[meter]


def _field_error(tag: str, val: str) -> str | None:
    if tag == "국가" and val not in COUNTRIES:
        return f"[헤더] 알 수 없는 국가: '{val}' (가능: {', '.join(COUNTRIES)})"
    if tag == "장르" and val not in GENRES:
        return f"[헤더] 알 수 없는 장르: '{val}' (가능: {', '.join(GENRES)})"
    if tag == "조" and val not in KEYS:
        return f"[헤더] 알 수 없는 조: '{val}' (예: C, G, F#, Bb, Am, F#m, Ebm)"
    if tag == "박자" and val not in METERS:
        return f"[헤더] 알 수 없는 박자: '{val}' (가능: {', '.join(METERS)})"
    if tag == "템포" and not (_TEMPO_RE.fullmatch(val) and 40 <= int(val) <= 220):
        shown = val if len(val) <= 40 else f"{val[:40]}... ({len(val)}자)"
        return f"[헤더] 템포는 40~220 사이의 정수여야 합니다: '{shown}'"
    return None


def _header_error(genre, key, meter, tempo) -> str | None:
    vals = {"장르": genre, "조": key, "박자": meter,
            "템포": str(tempo) if isinstance(tempo, int) and not isinstance(tempo, bool) else repr(tempo)}
    for tag in HEADER_TAGS:
        err = _field_error(tag, vals[tag])
        if err:
            return err
    return None


class _Rep:
    """Error reporter: raises at the first problem (strict) or collects them."""

    def __init__(self, strict: bool):
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        if self.strict:
            raise LeadsheetError(msg)
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------- parsing ----------
def _parse_header(line: str, rep: _Rep) -> dict[str, str]:
    rest = _TAG_RE.sub(" ", line).strip()
    if rest:
        rep.err(f"[헤더] 해석할 수 없는 내용: '{rest}'")
    vals: dict[str, str] = {}
    for m in _TAG_RE.finditer(line):
        tag, val = m.group(1).strip(), m.group(2).strip()
        if tag not in HEADER_TAGS + OPTIONAL_TAGS:
            rep.err(f"[헤더] 알 수 없는 태그: '{tag}' (가능: {', '.join(HEADER_TAGS + OPTIONAL_TAGS)})")
        elif tag in vals:
            rep.err(f"[헤더] 태그가 중복되었습니다: '{tag}'")
        else:
            vals[tag] = val
    for tag in HEADER_TAGS:
        if tag not in vals:
            rep.err(f"[헤더] 필수 태그가 없습니다: '{tag}'")
        else:
            e = _field_error(tag, vals[tag])
            if e:
                rep.err(e)
    for tag in OPTIONAL_TAGS:
        if tag in vals:
            e = _field_error(tag, vals[tag])
            if e:
                rep.err(e)
    return vals


def _split_bars(lines: list[str]) -> list[str]:
    """Concatenate body lines and split on '|' (leading/final bar lines allowed)."""
    buf = ""
    for ln in lines:
        ln = ln.strip()
        if buf.rstrip().endswith("|") and ln.startswith("|") and not ln.startswith("|]"):
            ln = ln[1:]                 # line ends with '|' and next starts with '|': one bar line
        buf = (buf + " " + ln).strip()
    if buf.startswith("|"):
        buf = buf[1:]
    buf = buf.rstrip()
    if buf.endswith("|]") or buf.endswith("||"):
        buf = buf[:-2]
    elif buf.endswith("|"):
        buf = buf[:-1]
    if not buf.strip():
        return []
    return [s.strip() for s in buf.split("|")]


def _length(spec: str) -> Fraction | None:
    """Length in units, or None when the denominator is zero."""
    if not spec:
        return Fraction(1)
    if "/" in spec:
        a, b = spec.split("/")
        den = int(b) if b else 2
        return Fraction(int(a) if a else 1, den) if den else None
    return Fraction(int(spec))


def _parse_bar(sec: str, bi: int, text: str, sig: dict[str, int], bar_len: int | None,
               meter: str | None, rep: _Rep) -> Bar | None:
    loc = f"[{sec} {bi}번째 마디]"
    if not text:
        rep.err(f"{loc} 빈 마디입니다 (연속된 '|' 또는 내용 없음)")
        return None
    n = len(text)
    pos = tick = 0
    notes: list[LeadNote] = []
    chords: list[ChordMark] = []
    pending: list[str] = []
    state: dict[tuple[str, int], int] = {}
    while True:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        ch = text[pos]
        if ch == '"':
            end = text.find('"', pos + 1)
            if end < 0:
                rep.err(f"{loc} 닫히지 않은 코드 따옴표: '{text[pos:]}'")
                return None
            name = text[pos + 1:end].strip()
            if _valid_chord(name):
                pending.append(name)
            else:
                rep.err(f"{loc} 알 수 없는 코드 이름: '{name}'")
            pos = end + 1
            continue
        m = _REST_RE.match(text, pos) if ch == "z" else _NOTE_RE.match(text, pos)
        if not m:
            tok = text[pos:].split(None, 1)[0]
            rep.err(f"{loc} 지원하지 않는 ABC 기호: '{tok}'")
            return None
        is_rest = ch == "z"
        tok = m.group(0)
        spec = m.group(1) if is_rest else m.group(4)
        too_long = [d for d in re.findall(r"\d+", spec) if len(d) > MAX_NUMBER_DIGITS]
        if too_long:                    # int() would raise a bare ValueError for thousands of digits: report it with the location
            shown = tok if len(tok) <= 40 else f"{tok[:40]}... ({len(tok)}자)"
            rep.err(f"{loc} 음 길이의 숫자가 너무 깁니다 (최대 {MAX_NUMBER_DIGITS}자리): '{shown}'")
            return None
        frac = _length(spec)
        ticks = UNIT * frac if frac is not None else Fraction(0)
        if ticks <= 0 or ticks.denominator != 1:
            rep.err(f"{loc} 음 길이가 정수 틱이 아니거나 0입니다: '{tok}' ({ticks} 틱)")
            return None
        ticks = int(ticks)
        pitch = None
        tie = False
        if not is_rest:
            acc_s, letter, marks, _, tie_s = m.groups()
            up = letter.islower()
            letter = letter.upper()
            octv = (5 if up else 4) + marks.count("'") - marks.count(",")
            if acc_s:
                acc = _ACC_VAL[acc_s]
                state[(letter, octv)] = acc
            else:
                acc = state.get((letter, octv), sig[letter])
            pitch = 12 * (octv + 1) + _BASE[letter] + acc
            if not PITCH_MIN <= pitch <= PITCH_MAX:
                rep.err(f"{loc} 음역 범위(MIDI {PITCH_MIN}~{PITCH_MAX})를 벗어났습니다: '{tok}' (MIDI {pitch})")
                return None
            tie = bool(tie_s)
        for c in pending:
            chords.append(ChordMark(c, tick))
        pending = []
        notes.append(LeadNote(pitch, tick, ticks, tie))
        tick += ticks
        pos = m.end()
    if pending:
        rep.err(f"{loc} 코드 '{pending[-1]}' 뒤에 음표나 쉼표가 없습니다")
        return None
    if bar_len is not None and tick != bar_len:
        if tick > bar_len:
            rep.err(f"{loc} 마디가 너무 깁니다: {tick}틱 (박자 {meter} 기준 {bar_len}틱), 마디 내용 '{text}'")
        else:
            rep.err(f"{loc} 마디가 너무 짧습니다: {tick}틱 (박자 {meter} 기준 {bar_len}틱), 마디 내용 '{text}'")
        return None
    return Bar(notes, chords)


def _parse(text: str, rep: _Rep) -> LeadSheet | None:
    if not isinstance(text, str):
        rep.err("입력이 문자열이 아닙니다")
        return None
    if text.startswith("\ufeff"):
        text = text[1:]                         # strip one leading UTF-8 BOM
    lines = [(i + 1, s.strip()) for i, s in enumerate(text.splitlines()) if s.strip()]
    if not lines:
        rep.err("내용이 비어 있습니다")
        return None
    if lines[0][1].startswith("[구조:"):
        hdr = _parse_header("", rep)            # reports every missing tag
        idx = 0
    else:
        hdr = _parse_header(lines[0][1], rep)
        idx = 1
    genre = hdr.get("장르", "")
    key = hdr.get("조", "")
    meter = hdr.get("박자", "")
    tempo_s = hdr.get("템포", "")
    tempo = int(tempo_s) if _TEMPO_RE.fullmatch(tempo_s) else 0
    sig = _signature(key) if key in KEYS else _signature("C")
    bar_len = METERS.get(meter)

    raw: list[list] = []                        # [name, body_lines]
    stray = False
    for ln, l in lines[idx:]:
        if l.startswith("[구조:"):
            m = _SEC_RE.fullmatch(l)
            if not m:
                rep.err(f"줄 {ln}: 구조 선언 형식 오류: '{l}' ([구조:이름] 형태여야 합니다)")
                raw.append([l, []])
                continue
            name = m.group(1).strip()
            if name not in SECTION_NAMES:
                rep.err(f"줄 {ln}: 알 수 없는 구조 이름: '{name}' (가능: {', '.join(SECTION_NAMES)})")
            raw.append([name, []])
        elif not raw:
            if not stray:
                rep.err(f"줄 {ln}: 구조 선언([구조:이름]) 이전의 내용: '{l}'")
                stray = True
        else:
            raw[-1][1].append(l)
    if not raw:
        rep.err("구조가 하나도 없습니다 ([구조:이름] 줄이 필요합니다)")

    sections: list[Section] = []
    for name, body in raw:
        bar_texts = _split_bars(body)
        if not bar_texts:
            rep.err(f"[{name}] 마디가 하나도 없습니다")
        bars = []
        for bi, bt in enumerate(bar_texts, 1):
            b = _parse_bar(name, bi, bt, sig, bar_len, meter, rep)
            if b is not None:
                bars.append(b)
        sections.append(Section(name, bars))
        flat = [(bi, n) for bi, b in enumerate(bars, 1) for n in b.notes]
        for i, (bi, n) in enumerate(flat):
            if n.tie and (i + 1 >= len(flat) or flat[i + 1][1].pitch != n.pitch):
                rep.warn(f"[{name} {bi}번째 마디] 붙임줄 뒤에 같은 높이의 음이 이어지지 않습니다")
    if sections and not any(b.chords for s in sections for b in s.bars):
        rep.warn("코드 기호가 하나도 없습니다")
    country = hdr.get("국가")
    return LeadSheet(genre, key, meter, tempo, sections, country if country in COUNTRIES else None)


def parse_leadsheet(text: str) -> LeadSheet:
    """Parse lead-sheet text; raises LeadsheetError at the first problem."""
    sheet = _parse(text, _Rep(strict=True))
    assert sheet is not None
    return sheet


def validate_leadsheet(text: str) -> dict:
    """Never raises for bad input; collects as many independent errors as possible."""
    rep = _Rep(strict=False)
    try:
        sheet = _parse(text, rep)
    except LeadsheetError as exc:               # defensive; strict=False should not raise
        rep.errors.append(str(exc))
        sheet = None
    except Exception as exc:                    # validate must never raise
        rep.errors.append(f"내부 오류: {type(exc).__name__}: {exc}")
        sheet = None
    secs = sheet.sections if sheet else []
    stats = {
        "sections": len(secs),
        "bars": sum(len(s.bars) for s in secs),
        "notes": sum(1 for s in secs for b in s.bars for n in b.notes if n.pitch is not None),
    }
    return {"ok": not rep.errors, "errors": rep.errors, "warnings": rep.warnings, "stats": stats}


# ---------- serializing ----------
def _spell(midi: int, sig: dict[str, int], flats: bool) -> tuple[str, int, int]:
    """(letter, accidental, written octave) for a MIDI pitch, preferring the key's own spelling."""
    pc = midi % 12
    for letter in "CDEFGAB":
        if (_BASE[letter] + sig[letter]) % 12 == pc:
            acc = sig[letter]
            break
    else:
        letter, acc = (_SPELL_FLAT if flats else _SPELL_SHARP)[pc]
    octv = (midi - _BASE[letter] - acc) // 12 - 1
    return letter, acc, octv


def _len_str(dur: int) -> str:
    f = Fraction(dur, UNIT)
    n, d = f.numerator, f.denominator
    if d == 1:
        return "" if n == 1 else str(n)
    if n == 1:
        return "/" if d == 2 else f"/{d}"
    return f"{n}/{d}"


def _note_str(n: LeadNote, sig, flats, state, loc) -> str:
    if n.dur <= 0:
        raise LeadsheetError(f"{loc} 음 길이가 0 이하입니다")
    if n.pitch is None:
        if n.tie:
            raise LeadsheetError(f"{loc} 쉼표에는 붙임줄을 쓸 수 없습니다")
        return "z" + _len_str(n.dur)
    if not PITCH_MIN <= n.pitch <= PITCH_MAX:
        raise LeadsheetError(f"{loc} 음역 범위를 벗어났습니다: MIDI {n.pitch}")
    letter, acc, octv = _spell(n.pitch, sig, flats)
    accs = ""
    if state.get((letter, octv), sig[letter]) != acc:
        accs = _ACC_STR[acc]
        state[(letter, octv)] = acc
    if octv >= 5:
        name = letter.lower() + "'" * (octv - 5)
    else:
        name = letter + "," * (4 - octv)
    return accs + name + _len_str(n.dur) + ("-" if n.tie else "")


def serialize_leadsheet(sheet: LeadSheet) -> str:
    """Canonical text. Raises LeadsheetError if the sheet cannot be represented."""
    err = _header_error(sheet.genre, sheet.key, sheet.meter, sheet.tempo)
    if err:
        raise LeadsheetError(err)
    bar_len = METERS[sheet.meter]
    sig = _signature(sheet.key)
    flats = KEYS[sheet.key] < 0
    if sheet.country is not None and sheet.country not in COUNTRIES:
        raise LeadsheetError(f"[헤더] 알 수 없는 국가: '{sheet.country}'")
    out = [f"[장르:{sheet.genre}] [조:{sheet.key}] [박자:{sheet.meter}] [템포:{sheet.tempo}]"
           + (f" [국가:{sheet.country}]" if sheet.country else "")]
    for sec in sheet.sections:
        if sec.name not in SECTION_NAMES:
            raise LeadsheetError(f"알 수 없는 구조 이름: '{sec.name}'")
        if not sec.bars:
            raise LeadsheetError(f"[{sec.name}] 마디가 하나도 없습니다")
        out.append(f"[구조:{sec.name}]")
        bar_strs = []
        for bi, bar in enumerate(sec.bars, 1):
            loc = f"[{sec.name} {bi}번째 마디]"
            starts = {n.start for n in bar.notes}
            end = 0
            for n in bar.notes:
                if n.start < 0 or n.start < end:
                    raise LeadsheetError(f"{loc} 음표가 겹치거나 순서가 맞지 않습니다 ({n.start}틱)")
                end = n.start + n.dur
            if end != bar_len or sum(n.dur for n in bar.notes) != bar_len:
                raise LeadsheetError(
                    f"{loc} 마디 길이가 박자 {sheet.meter}({bar_len}틱)와 다릅니다: 음표 합계 "
                    f"{sum(n.dur for n in bar.notes)}틱")
            for c in bar.chords:
                if not _valid_chord(c.symbol):
                    raise LeadsheetError(f"{loc} 알 수 없는 코드 이름: '{c.symbol}'")
                if c.start not in starts:
                    raise LeadsheetError(
                        f"{loc} 코드 '{c.symbol}'가 음표의 시작 위치가 아닌 {c.start}틱에 있어 표현할 수 없습니다")
            marks = sorted(bar.chords, key=lambda c: c.start)     # stable for equal starts
            state: dict[tuple[str, int], int] = {}
            toks: list[str] = []
            pos = 0
            for n in bar.notes:
                if n.start != pos:
                    raise LeadsheetError(f"{loc} 음표가 연속적이지 않습니다 (기대 {pos}틱, 실제 {n.start}틱)")
                toks += [f'"{c.symbol}"' for c in marks if c.start == n.start]
                toks.append(_note_str(n, sig, flats, state, loc))
                pos += n.dur
            if not toks:
                raise LeadsheetError(f"{loc} 빈 마디입니다")
            bar_strs.append(" ".join(toks))
        for i in range(0, len(bar_strs), 4):
            out.append("| " + " | ".join(bar_strs[i:i + 4]) + " |")
    return "\n".join(out) + "\n"


# ---------- conversion to/from absolute-tick notes ----------
def to_notes(sheet: LeadSheet) -> tuple[list[Note], list[str]]:
    """Flatten to absolute ticks (sections back to back) plus one chord symbol per bar.

    The chord per bar is the one in effect at the bar start ("N" before any chord).
    Tied notes (same pitch, contiguous, within a section) are merged into one Note.
    """
    bl = _bar_ticks(sheet.meter)
    notes: list[Note] = []
    chords: list[str] = []
    cur = "N"
    offset = 0
    for sec in sheet.sections:
        merged: list[list] = []
        last_tie = False
        for bi, bar in enumerate(sec.bars):
            marks = sorted(bar.chords, key=lambda c: c.start)
            for m in marks:
                if m.start <= 0:
                    cur = m.symbol
            chords.append(cur)
            for m in marks:
                if m.start > 0:
                    cur = m.symbol
            for n in bar.notes:
                start = offset + bi * bl + n.start
                if (merged and last_tie and n.pitch is not None and merged[-1][0] == n.pitch
                        and merged[-1][1] + merged[-1][2] == start):
                    merged[-1][2] += n.dur
                else:
                    merged.append([n.pitch, start, n.dur])
                last_tie = n.tie and n.pitch is not None
        notes += [Note(p, s, d) for p, s, d in merged]
        offset += len(sec.bars) * bl
    return notes, chords


def from_notes(genre: str, key: str, meter: str, tempo: int,
               sections: list[tuple[str, list[Note], list[str]]]) -> LeadSheet:
    """Build a LeadSheet from per-section absolute-tick notes (start 0 per section).

    chords: one symbol per bar, placed at tick 0 of that bar; "N"/""/None means no chord.
    Notes crossing bar lines are split into tied pieces; gaps and the bar tail become rests.
    """
    err = _header_error(genre, key, meter, tempo)
    if err:
        raise LeadsheetError(err)
    bl = METERS[meter]
    out: list[Section] = []
    for name, notes, chord_list in sections:
        if name not in SECTION_NAMES:
            raise LeadsheetError(f"알 수 없는 구조 이름: '{name}'")
        pieces: list[tuple[int | None, int, int]] = []
        cur = 0
        for n in sorted(notes, key=lambda x: x.start):
            if n.start < 0 or n.dur <= 0:
                raise LeadsheetError(f"[{name}] 잘못된 음표 (start={n.start}, dur={n.dur})")
            if n.start < cur:
                raise LeadsheetError(f"[{name}] 음표가 겹칩니다 ({n.start}틱)")
            if n.pitch is not None and not PITCH_MIN <= n.pitch <= PITCH_MAX:
                raise LeadsheetError(f"[{name}] 음역 범위를 벗어났습니다: MIDI {n.pitch}")
            if n.start > cur:
                pieces.append((None, cur, n.start - cur))
            pieces.append((n.pitch, n.start, n.dur))
            cur = n.start + n.dur
        nbars = max(-(-cur // bl), len(chord_list), 1)
        if cur < nbars * bl:
            pieces.append((None, cur, nbars * bl - cur))
        bars = [Bar() for _ in range(nbars)]
        for pitch, s, d in pieces:
            while d > 0:
                bi, off = divmod(s, bl)
                take = min(d, bl - off)
                bars[bi].notes.append(LeadNote(pitch, off, take, tie=(d > take and pitch is not None)))
                s += take
                d -= take
        for bi, sym in enumerate(chord_list):
            if sym in (None, "", "N"):
                continue
            if not _valid_chord(sym):
                raise LeadsheetError(f"[{name} {bi + 1}번째 마디] 알 수 없는 코드 이름: '{sym}'")
            bars[bi].chords.append(ChordMark(sym, 0))
        out.append(Section(name, bars))
    return LeadSheet(genre, key, meter, tempo, out)
