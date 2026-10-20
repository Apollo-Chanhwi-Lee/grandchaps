"""Band pack: what a band takes to the rehearsal room to play the song themselves (one zip).

    chord_chart.html   printable chart: key, tempo, form, key changes, and the chords of every bar (with the beat of
                       a chord change inside a bar), section by section, with song-wide bar numbers
    parts_guide.md     per part and section: where it plays, its rhythm on a 16th-note grid (start / held / silent),
                       range, sound; guitar chord shapes and tabs (lead line, arpeggio), bass tab, drum rows
    parts/<part>.mid   one MIDI file per part (tempo track + that part, padded to the song's length)
    song.mid           all parts
    melody_chords.musicxml, leadsheet.txt   the melody with chords, as notation and as the app's text format
    README.txt         what each file is, what is not included, and how the melody outside the chorus was made

Everything is read from a finished result (sections, bar_chords, midi_b64, ...): nothing is composed here.  Tabs and
chord shapes are suggestions computed from the notes (standard tuning, no capo), not a checked fingering.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import zipfile

import mido

from . import instruments
from .genres import PROFILES
from .theory import TPQ as CHART_TPQ

PART_KO = {"melody": "멜로디(보컬)", "lead": "리드 기타", "piano": "피아노", "keys": "멜로디 피아노", "bass": "베이스",
           "guitar": "기타", "strings": "스트링", "drums": "드럼", "harmony": "화음(두 번째 목소리)"}
PART_ORDER = ("melody", "harmony", "guitar", "lead", "piano", "keys", "bass", "strings", "drums")
DRUM_ROWS = (("킥", (35, 36)), ("스네어", (37, 38, 39, 40)), ("하이햇", (42, 44, 46)), ("라이드", (51, 53, 59)),
             ("크래시", (49, 52, 55, 57)), ("탐", (41, 43, 45, 47, 48, 50)))
DRUM_OTHER = {54: "탬버린", 56: "카우벨", 69: "카바사", 70: "마라카스", 75: "클라베", 39: "클랩"}
TOMS = (41, 43, 45, 47, 48, 50)
FILL_HITS = 3                # this many tom hits in a bar = a tom fill candidate
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
BARS_PER_LINE = 4
GUITAR_TUNING = (64, 59, 55, 50, 45, 40)      # strings 1-6: e B G D A E
BASS_TUNING = (43, 38, 33, 28)                # strings 1-4: G D A E
STRING_NAMES = {6: ("e", "B", "G", "D", "A", "E"), 4: ("G", "D", "A", "E")}
MAX_FRET = 15
SOURCE_KO = {"ai": "후렴 밖의 멜로디는 학습 모델(AI)이 제안했습니다. 이 모델의 학습 자료에는 비상업 조건이 있어, 연습·합주·수업 발표용으로 "
                   "쓰고 발매곡에는 이 멜로디를 그대로 쓰지 마세요.",
             "own": "후렴 밖의 멜로디는 입력한 후렴 멜로디를 규칙으로 변형한 것입니다(학습 모델 미사용). 코드와 편곡도 음악 규칙으로 "
                    "만들었습니다. 발매 전에는 기존 곡과 비슷하지 않은지 직접 확인하세요.",
             None: "코드와 편곡은 음악 규칙으로 만들었습니다."}


class BandPackError(ValueError):
    """The result cannot be exported (the message says what is missing)."""


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


# ---------------------------------------------------------------- reading the MIDI
def _tracks(midi: bytes) -> tuple[dict[str, list[tuple[int, int, int, int]]], int, int]:
    """({track: [(start tick, pitch, velocity, end tick)]}, ticks per beat, ticks per bar)."""
    mid = mido.MidiFile(file=io.BytesIO(midi))
    tpq = mid.ticks_per_beat
    num, den = next(((m.numerator, m.denominator) for m in mid.tracks[0] if m.type == "time_signature"), (4, 4))
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    for tr in mid.tracks:
        name = next((m.name for m in tr if m.type == "track_name"), None)
        if name is None:
            continue
        now, notes, open_ = 0, [], {}
        for m in tr:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                open_.setdefault(m.note, []).append(len(notes))
                notes.append([now, m.note, m.velocity, now])
            elif m.type in ("note_off", "note_on") and open_.get(m.note):
                notes[open_[m.note].pop(0)][3] = now
        if notes:
            out[name] = [tuple(n) for n in notes]
    return out, tpq, tpq * 4 * num // den


def sounding(notes: list[tuple], bar: int, bar_len: int) -> bool:
    """A note starts in the bar or is still held into it."""
    a, b = bar * bar_len, (bar + 1) * bar_len
    return any(s < b and e > a for s, _, _, e in notes)


def grid(notes: list[tuple], bar: int, bar_len: int, tpq: int, pitches: tuple[int, ...] | None = None,
         holds: bool = True) -> str:
    """One bar on a 16th-note grid, beats separated by spaces: "x" a note starts, "-" a note is still held,
    "." silence.  `holds=False` (drums): only starts."""
    step = tpq / 4
    steps = int(bar_len / step)
    cells = ["."] * steps
    for s, p, _, e in notes:
        if pitches is not None and p not in pitches:
            continue
        a, b = (s - bar * bar_len) / step, (e - bar * bar_len) / step
        if holds:
            for i in range(max(0, int(a + 0.5) + 1), min(steps, int(b + 0.5))):
                if cells[i] == ".":
                    cells[i] = "-"
        if 0 <= int(a + 0.5) < steps and s >= bar * bar_len - step / 2:
            cells[int(a + 0.5)] = "x"
    return " ".join("".join(cells[i:i + 4]) for i in range(0, steps, 4))


def strum_marks(notes: list[tuple], bar: int, bar_len: int, tpq: int) -> str | None:
    """Down/up strokes of a strummed bar on the eighth-note grid ("D . D U . U D U"), None if it is not strummed."""
    by_start: dict[int, int] = {}
    for n in notes:
        t = n[0]
        if bar * bar_len <= t < (bar + 1) * bar_len:
            by_start[t - bar * bar_len] = by_start.get(t - bar * bar_len, 0) + 1
    if not by_start or sum(by_start.values()) < 3 * len(by_start) or any(t % (tpq // 2) for t in by_start):
        return None
    return " ".join(("D" if i % 2 == 0 else "U") if i * tpq // 2 in by_start else "." for i in range(bar_len * 2 // tpq))


# ---------------------------------------------------------------- guitar chord shapes (standard tuning, no capo)
OPEN_SHAPES = {"C": "x32010", "A": "x02220", "G": "320003", "E": "022100", "D": "xx0232", "Am": "x02210", "Em": "022000",
               "Dm": "xx0231", "G7": "320001", "C7": "x32310", "D7": "xx0212", "E7": "020100", "A7": "x02020", "B7": "x21202",
               "Am7": "x02010", "Em7": "022030", "Dm7": "xx0211", "Cmaj7": "x32000", "Fmaj7": "xx3210", "Gmaj7": "320002",
               "Amaj7": "x02120", "Dmaj7": "xx0222", "Emaj7": "021100", "Asus4": "x02230", "Dsus4": "xx0233", "Esus4": "022200"}
E_SHAPE = {"": (0, 2, 2, 1, 0, 0), "m": (0, 2, 2, 0, 0, 0), "7": (0, 2, 0, 1, 0, 0), "m7": (0, 2, 0, 0, 0, 0),
           "maj7": (0, 2, 1, 1, 0, 0), "sus4": (0, 2, 2, 2, 0, 0)}                     # strings 6..1, root on string 6
A_SHAPE = {"": (None, 0, 2, 2, 2, 0), "m": (None, 0, 2, 2, 1, 0), "7": (None, 0, 2, 0, 2, 0), "m7": (None, 0, 2, 0, 1, 0),
           "maj7": (None, 0, 2, 1, 2, 0), "sus4": (None, 0, 2, 2, 3, 0)}               # root on string 5
ROOT_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_CHORD = re.compile(r"^([A-G])([#b]?)(maj7|m7|m|7|sus4)?(?:/([A-G][#b]?))?$")


def chord_shape(symbol: str) -> str | None:
    """A common guitar shape, strings 6 to 1 ("x32010"; frets above 9 are written "x-4-6-6-6-4"): the open shape when
    there is one, else the lower of the two barre shapes.  None for chord types without a table entry here."""
    m = _CHORD.match(symbol)
    if not m:
        return None
    letter, acc, quality, _bass = m.groups()
    quality = quality or ""
    base = f"{letter}{acc}{quality}"
    if base in OPEN_SHAPES:
        return OPEN_SHAPES[base]
    pc = (ROOT_PC[letter] + {"#": 1, "b": -1, "": 0}[acc]) % 12
    options = []
    for shape, open_pc in ((E_SHAPE, 4), (A_SHAPE, 9)):
        fret = (pc - open_pc) % 12
        if fret == 0:
            fret = 12
        options.append((fret, [None if f is None else f + fret for f in shape[quality]]))
    fret, frets = min(options, key=lambda o: o[0])
    if max(f for f in frets if f is not None) > 9:
        return "-".join("x" if f is None else str(f) for f in frets)
    return "".join("x" if f is None else str(f) for f in frets)


# ---------------------------------------------------------------- tabs
HIGH_FRET_COST = 0.35         # per fret: prefer low positions (and open strings) when the hand does not have to move
STRING_JUMP_COST = 0.4        # per string skipped between two notes


def place(notes: list[tuple], tuning: tuple[int, ...]) -> list[tuple[int, int, int]] | None:
    """[(start, string index, fret)] for a one-note-at-a-time line: the fingering with the least hand movement over
    the whole line (dynamic programming), preferring low frets and open strings.  None when a note is out of range."""
    line = sorted(notes)
    spots = [[(i, n[1] - open_) for i, open_ in enumerate(tuning) if 0 <= n[1] - open_ <= MAX_FRET] for n in line]
    if not line or any(not sp for sp in spots):
        return None
    cost = [{sp: HIGH_FRET_COST * sp[1] for sp in spots[0]}]
    back: list[dict] = [{}]
    for k in range(1, len(line)):
        cur, prev_of = {}, {}
        for sp in spots[k]:
            best = None
            for prev, c in cost[-1].items():
                move = abs(sp[1] - prev[1]) if sp[1] and prev[1] else 0       # an open string frees the hand
                step = c + move + STRING_JUMP_COST * abs(sp[0] - prev[0]) + HIGH_FRET_COST * sp[1]
                if best is None or step < best[0]:
                    best = (step, prev)
            cur[sp], prev_of[sp] = best
        cost.append(cur)
        back.append(prev_of)
    sp = min(cost[-1], key=cost[-1].get)
    path = [sp]
    for k in range(len(line) - 1, 0, -1):
        sp = back[k][sp]
        path.append(sp)
    return [(n[0], i, fret) for n, (i, fret) in zip(line, reversed(path))]


def tab(notes: list[tuple], bars: list[int], bar_len: int, tpq: int, tuning: tuple[int, ...]) -> list[str] | None:
    """ASCII tab of `bars` (16th-note columns): one line per string, highest string first.  None for chords, notes off
    the grid or out of range."""
    step = tpq // 4
    steps = bar_len // step
    inside = [n for n in notes if any(b * bar_len <= n[0] < (b + 1) * bar_len for b in bars)]
    starts = [n[0] for n in inside]
    if not inside or len(set(starts)) != len(starts) or any(t % step for t in starts):
        return None
    placed = place(inside, tuning)
    if placed is None:
        return None
    at = {(s, i): fret for s, i, fret in placed}
    names = STRING_NAMES[len(tuning)]
    lines = []
    for i, name in enumerate(names):
        cells = []
        for b in bars:
            row = "".join(f"{at.get((b * bar_len + k * step, i), '-'):-<2}" for k in range(steps))
            cells.append(row)
        lines.append(f"{name}|" + "|".join(cells) + "|")
    return lines


# ---------------------------------------------------------------- key changes
def key_note(result: dict, sec: dict, bars: list[list]) -> str | None:
    """Text for a section played in another key than the song's ("반음 위로 전조: 조 Db"), from the section's own
    recorded shift (never guessed from the chords); the spelling follows the section's chord symbols."""
    shift = sec.get("shift") or 0
    if not shift:
        return None
    key = result.get("key") or {}
    if "tonic" not in key:
        return f"{shift:+d}반음 전조"
    flats = any(len(sym) > 1 and sym[1] == "b" for bar in bars for _, sym in bar)
    tonic = (FLAT_NAMES if flats else NOTE_NAMES)[(key["tonic"] + shift) % 12]
    mode = "m" if key.get("mode") == "minor" else ""
    how = "반음 위로" if shift == 1 else f"{shift:+d}반음"
    return f"{how} 전조: 조 {tonic}{mode}"


# ---------------------------------------------------------------- documents
def part_guide(result: dict, band: dict | None = None) -> str:
    """Markdown: per part, per section, what it plays.  `band`: {name, members, guitarists, vocal_low, vocal_high}
    (MIDI note numbers) adds checks against that band's line-up."""
    midi = base64.b64decode(result["midi_b64"])
    tracks, tpq, bar_len = _tracks(midi)
    genre = result.get("genre")
    lineup = instruments.api_lineup(genre) if genre in PROFILES else None
    sound = {p["part"]: p["label"] for p in lineup["parts"]} if lineup else {}
    L = [f"# 파트별 연주 안내: {result.get('title', '작곡메이트')}", "",
         "- 리듬 격자는 한 마디를 16분음 단위로 나눈 것입니다(박 단위로 띄어 씀). `x` = 새 음을 치는 자리, `-` = 앞 음을 계속 "
         "누르고 있는 자리, `.` = 소리 없음. 드럼은 치는 자리(`x`)만 표시합니다.",
         "- 각 구간에서 그 파트가 처음 소리 내는 두 마디를 보여 주고, 그것이 곡 전체의 몇 번째 마디인지 적었습니다. "
         "나머지 마디와 정확한 음은 `parts/` 폴더의 파트별 MIDI를 여세요.",
         "- 타브와 코드 폼은 표준 튜닝·카포 없음·실음 기준으로 음높이에서 자동으로 고른 **추천 운지**입니다. 연주해 보고 편한 "
         "자리로 바꾸세요.", ""]
    checks = lineup_notes(result, tracks, bar_len, band)
    if checks:
        L += ["## 우리 밴드 편성 확인", ""] + checks + [""]
    members = result.get("member_parts") or []
    edited = {m["part"]: sorted({b for e in m.get("edits", []) for b in range(e["start_bar"], e["start_bar"] + e["bars"])})
              for m in members if m.get("edits")}
    if members:
        who = ", ".join(PART_KO.get(m["part"], m["part"]) + (f" ({m.get('last_editor') or m.get('author')})"
                                                            if m.get("last_editor") or m.get("author") else "") for m in members)
        L += [f"- 멤버가 만든 파트: {who}. 이 파트는 멤버의 MIDI 그대로입니다(음, 세기, 페달·벤드 유지). 아래 안내와 타브는 "
              "그 MIDI에서 읽은 것입니다.", ""]
    role = result.get("melody_role") or "vocal"
    for part in [p for p in PART_ORDER if p in tracks]:
        notes = tracks[part]
        title = PART_KO.get(part, part)
        if part == "melody" and role != "vocal":           # the melody is an instrument's line in this song
            title = f"멜로디({result.get('melody_role_label') or role})"
        L += [f"## {title}", ""]
        if part != "drums":
            lo, hi = min(n[1] for n in notes), max(n[1] for n in notes)
            L.append(f"- 음역: {note_name(lo)} ~ {note_name(hi)}")
        if part == "melody" and role == "vocal":
            L.append("- 가사가 없는 멜로디입니다. 보컬 음역에 맞지 않으면 앱에서 조를 바꿔 다시 만드세요.")
        elif part == "melody":
            L.append(f"- 이 곡의 멜로디는 {result.get('melody_role_label') or role}가 연주합니다. 앱은 타브를 만들지 않으니 "
                     "MIDI와 악보를 보고 운지를 정하세요.")
        if part in sound:
            L.append(f"- 추천 소리: {sound[part]}")
            if part == "guitar":
                L += [f"- {sec['kind_label']}에서는 {sec['guitar']['label']}" for sec in lineup["sections"] if "guitar" in sec]
        if part == "guitar":
            L += _shape_lines(result)
        if part == "drums":
            extra = sorted({n[1] for n in notes} - {p for _, ps in DRUM_ROWS for p in ps})
            if extra:
                L.append("- 아래 줄에 없는 타악기: " + ", ".join(DRUM_OTHER.get(p, f"MIDI {p}") for p in extra)
                         + " (드럼 MIDI에 들어 있음)")
        L.append("")
        shown: dict[str, str] = {}                 # what was printed -> the section that printed it first
        for sec in result["sections"]:
            bars = list(range(sec["start_bar"], sec["start_bar"] + sec["bars"]))
            playing = [b for b in bars if sounding(notes, b, bar_len)]
            span = f"전체 {bars[0] + 1}~{bars[-1] + 1}마디"
            if not playing:
                L.append(f"- **{sec['label']}** ({span}): 쉼")
                continue
            L.append(f"- **{sec['label']}** ({span}, {len(playing)}/{sec['bars']}마디 연주)")
            show = playing[:2]
            where = "전체 " + "·".join(str(b + 1) for b in show) + "마디 (구간의 " + "·".join(str(b - bars[0] + 1) for b in show) + "번째)"
            if part == "drums":
                body = []
                for name, pitches in DRUM_ROWS:
                    rows = [grid(notes, b, bar_len, tpq, pitches, holds=False) for b in show]
                    if any("x" in r for r in rows):
                        body.append(f"  - {name}: `{'` | `'.join(rows)}`")
                L += _once(shown, body, sec["label"], where, "치는 자리가")
                L += _edited_lines(edited.get(part, []), sec, notes, bar_len, tpq, drums=True)
                fills = [b + 1 for b in bars if sum(1 for n in notes if b * bar_len <= n[0] < (b + 1) * bar_len and n[1] in TOMS) >= FILL_HITS]
                if fills:
                    L.append(f"  - 탐 필 후보: 전체 {', '.join(str(b) for b in fills)}마디 (탐이 {FILL_HITS}번 이상 나오는 마디만 찾은 것)")
                continue
            body = [f"  - 리듬: `{'` | `'.join(grid(notes, b, bar_len, tpq) for b in show)}`"]
            if part == "guitar":
                marks = strum_marks(notes, show[0], bar_len, tpq)
                if marks:
                    body.append(f"  - 스트럼 제안 (리듬에서 정한 것: 정박 다운 D, 엇박 업 U, 8분음 기준): `{marks}`")
            tuning = BASS_TUNING if part == "bass" else GUITAR_TUNING if part in ("guitar", "lead") else None
            if tuning:
                rows = tab(notes, playing if part == "lead" else show, bar_len, tpq, tuning)
                if rows:
                    body += ["  - 타브 (추천 운지, 16분음 한 칸):", "", "    ```"] + [f"    {r}" for r in rows] + ["    ```", ""]
            pitches = " ".join(str(n[1]) for b in show for n in sorted(notes) if b * bar_len <= n[0] < (b + 1) * bar_len)
            L += _once(shown, body, sec["label"], where, "리듬과 음이", extra=pitches)
            L += _edited_lines(edited.get(part, []), sec, notes, bar_len, tpq, drums=False)
        L.append("")
    return "\n".join(L) + "\n"


def lineup_notes(result: dict, tracks: dict, bar_len: int, band: dict | None) -> list[str]:
    """Checks against the band's own line-up (number of guitarists, the singer's range). Empty without band info."""
    if not band:
        return []
    out = []
    if band.get("name"):
        out.append(f"- 밴드: {band['name']}" + (f" ({band['members']})" if band.get("members") else ""))
    total = sum(s["bars"] for s in result["sections"])
    if band.get("guitarists") == 1 and "guitar" in tracks and "lead" in tracks:
        both = [b + 1 for b in range(total) if sounding(tracks["guitar"], b, bar_len) and sounding(tracks["lead"], b, bar_len)]
        if both:
            out.append(f"- 기타가 한 명: 전체 {_ranges(both)}마디에서 리드와 반주 기타가 겹칩니다. 리드를 치고 그 마디의 반주는 "
                       "건반에 맡기거나, 리드를 건반·보컬로 옮기세요.")
    if band.get("guitarists") == 0 and ("guitar" in tracks or "lead" in tracks):
        out.append("- 기타가 없음: 기타·리드 파트는 건반이 대신하거나 빼고 다시 만드세요(앱의 악기 선택에서 기타를 끄면 됩니다).")
    lo, hi = band.get("vocal_low"), band.get("vocal_high")
    if lo is not None and hi is not None and "melody" in tracks:
        m_lo, m_hi = min(n[1] for n in tracks["melody"]), max(n[1] for n in tracks["melody"])
        if m_lo >= lo and m_hi <= hi:
            out.append(f"- 보컬 음역 {note_name(lo)}~{note_name(hi)}: 멜로디({note_name(m_lo)}~{note_name(m_hi)})가 안에 들어옵니다.")
        else:
            fits = [k for k in range(-11, 12) if m_lo + k >= lo and m_hi + k <= hi]
            tip = (f" {min(fits, key=abs):+d}반음 옮기면 들어옵니다(앱에서 조를 바꿔 다시 만드세요)." if fits
                   else " 멜로디 폭이 음역보다 넓어 조를 옮겨도 다 들어오지 않습니다. 높은 구간을 고치거나 옥타브를 나눠 부르세요.")
            out.append(f"- 보컬 음역 {note_name(lo)}~{note_name(hi)}: 멜로디({note_name(m_lo)}~{note_name(m_hi)})가 벗어납니다." + tip)
    return out


def _ranges(bars: list[int]) -> str:
    """[1, 2, 3, 7] -> "1~3, 7"."""
    out, start, prev = [], bars[0], bars[0]
    for b in bars[1:] + [None]:
        if b != prev + 1:
            out.append(str(start) if start == prev else f"{start}~{prev}")
            start = b
        prev = b
    return ", ".join(out)


EDITED_BARS_SHOWN = 16         # per part and section; beyond that the guide points to the MIDI


def _edited_lines(edited: list[int], sec: dict, notes: list[tuple], bar_len: int, tpq: int, drums: bool) -> list[str]:
    """Every bar of this section that a member edited in the app, shown in full (not only the two example bars)."""
    bars = [b for b in edited if sec["start_bar"] <= b < sec["start_bar"] + sec["bars"]]
    if not bars:
        return []
    out = [f"  - 직접 고친 마디 (전체 {_ranges([b + 1 for b in bars])}마디):"]
    for b in bars[:EDITED_BARS_SHOWN]:
        if drums:
            rows = [f"{name} `{grid(notes, b, bar_len, tpq, pitches, holds=False)}`" for name, pitches in DRUM_ROWS
                    if "x" in grid(notes, b, bar_len, tpq, pitches, holds=False)]
            out.append(f"    - {b + 1}마디: " + (" · ".join(rows) if rows else "쉼"))
        else:
            names = " ".join(note_name(n[1]) for n in sorted(notes) if b * bar_len <= n[0] < (b + 1) * bar_len)
            out.append(f"    - {b + 1}마디: `{grid(notes, b, bar_len, tpq)}`" + (f" 음: {names}" if names else " 쉼"))
    if len(bars) > EDITED_BARS_SHOWN:
        out.append(f"    - 나머지 {len(bars) - EDITED_BARS_SHOWN}마디는 파트 MIDI에서 확인하세요")
    return out


def _once(shown: dict[str, str], body: list[str], label: str, where: str, what: str, extra: str = "") -> list[str]:
    """The lines of a section's example, or one line pointing at the earlier section whose shown bars are the same.
    `extra` (the pitches) is compared too, so a transposed repeat is not called the same; `what` says what matched."""
    key = "\n".join(body) + "\n" + extra
    if key in shown:
        return [f"  - 보기 {where}: 이 두 마디는 {shown[key]}의 보기와 {what} 같음 (나머지 마디는 MIDI로 확인)"]
    shown[key] = label
    return [f"  - 보기: {where}"] + body


def _shape_lines(result: dict) -> list[str]:
    seen, known, unknown = set(), [], []
    for bar in result["bar_chords"]:
        for _, sym in bar:
            if sym in seen:
                continue
            seen.add(sym)
            shape = chord_shape(sym)
            (known if shape else unknown).append(f"{sym} `{shape}`" if shape else sym)
    out = ["- 튜닝: 표준(E A D G B E), 카포 없음, 코드는 실음 기준"]
    if known:
        out.append("- 코드 폼 (6번 줄 → 1번 줄, x = 치지 않음, 추천): " + " · ".join(known))
    if unknown:
        out.append("- 폼 표에 없는 코드: " + ", ".join(unknown))
    if any("/" in s for s in seen):
        out.append("- `C/E` 같은 코드는 앞의 코드를 잡고, 뒤의 음(베이스 음)은 베이스가 칩니다.")
    return out


def chord_chart(result: dict, melody_source: str | None = None) -> str:
    """Standalone printable HTML: header facts and the chords of every bar, four bars per line."""
    e = html.escape
    key = result.get("key", {}).get("name", "")
    secs = result["sections"]
    chords = result["bar_chords"]
    tracks, _, bar_len = _tracks(base64.b64decode(result["midi_b64"]))
    rows, moved = [], False
    for sec in secs:
        first = sec["start_bar"]
        bars = chords[first:first + sec["bars"]]
        cells = []
        for i, bar in enumerate(bars):
            if len(bar) > 1:        # a change inside the bar: say on which beat
                text = " &nbsp; ".join(f'{e(sym)}<sup>{pos // CHART_TPQ + 1}박</sup>' for pos, sym in bar)
            else:
                text = e(bar[0][1]) if bar else "&nbsp;"
            cells.append(f'<td><span class="n">{first + i + 1}</span>{text}</td>')
        lines = "".join("<tr>" + "".join(cells[i:i + BARS_PER_LINE]) + "</tr>" for i in range(0, len(cells), BARS_PER_LINE))
        playing = [p for p in PART_ORDER if p in tracks
                   and any(sounding(tracks[p], b, bar_len) for b in range(first, first + sec["bars"]))]
        inst = " · ".join(PART_KO.get(p, p) for p in playing)
        note = key_note(result, sec, bars)
        if note is None and moved:
            note = f"원래 조({key})로 복귀"
        moved = bool(sec.get("shift"))
        flag = f' <b class="key">{e(note)}</b>' if note else ""
        rows.append(f'<section><h2>{e(sec["label"])} <small>전체 {first + 1}~{first + sec["bars"]}마디 ({sec["bars"]}마디) · '
                    f'{e(inst)}</small>{flag}</h2><table>{lines}</table></section>')
    form = " → ".join(e(s["label"]) for s in secs)
    total = round(result.get("duration_sec", 0))
    genre = PROFILES[result["genre"]]["label"] if result.get("genre") in PROFILES else result.get("genre", "")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{e(result.get('title', '작곡메이트'))} 코드 차트</title>
<style>
body {{ font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; margin: 24px; color: #111; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} .facts {{ margin: 0 0 6px; }} .form, .note {{ color: #444; font-size: 13px; margin: 0 0 8px; }}
section {{ break-inside: avoid; margin-bottom: 14px; }} h2 {{ font-size: 15px; margin: 0 0 4px; }} h2 small {{ font-weight: 400; color: #555; }}
.key {{ font-size: 12px; background: #111; color: #fff; padding: 1px 6px; border-radius: 3px; margin-left: 6px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
td {{ border: 1px solid #333; padding: 12px 8px 12px; font-size: 18px; font-weight: 600; position: relative; width: 25%; }}
td sup {{ font-size: 10px; font-weight: 400; color: #555; margin-left: 1px; }}
.n {{ position: absolute; top: 2px; left: 4px; font-size: 9px; font-weight: 400; color: #777; }}
@media print {{ body {{ margin: 10mm; }} }}
</style></head><body>
<h1>{e(result.get('title', '작곡메이트'))} · 합주용 코드 차트</h1>
<p class="facts"><b>조</b> {e(key)} &nbsp; <b>템포</b> {result.get('tempo', '')} BPM &nbsp; <b>박자</b> {e(result.get('meter', '4/4'))}
 &nbsp; <b>장르</b> {e(str(genre))} &nbsp; <b>길이</b> {total // 60}:{total % 60:02d} ({result.get('bars', '')}마디)</p>
<p class="form">구성: {form}</p>
<p class="note">칸 왼쪽 위의 작은 숫자는 곡 전체의 마디 번호입니다. 한 마디에 코드가 둘 이상이면 바뀌는 박을 작게 적었습니다.
코드는 실음 기준(카포 없음)입니다.</p>
<p class="note">합주 제안(악보에 들어 있는 것은 아님): 한 마디 카운트 후 시작하고, 마지막 코드는 함께 길게 끌다가 신호에 맞춰 끊으세요.</p>
<p class="note">{e(SOURCE_KO.get(melody_source, SOURCE_KO[None]))} 이 차트는 초안입니다. 합주하면서 자유롭게 고치세요.</p>
{''.join(rows)}
</body></html>
"""


README = """작곡메이트 밴드 팩
==================

chord_chart.html        합주용 코드 차트 (브라우저에서 열어 인쇄). 전체 마디 번호, 전조 위치, 구간별 연주 파트 표시
parts_guide.md          파트별 연주 안내 (리듬 격자, 음역, 추천 소리, 기타 코드 폼, 리드·아르페지오·베이스 타브, 드럼 줄)
parts/<part>.mid        파트별 MIDI (melody 멜로디, guitar 기타, lead 리드 기타, piano 피아노, keys 멜로디 피아노,
                        bass 베이스, strings 스트링, drums 드럼). 모든 파일이 곡의 0초에서 시작하고 길이가 같으므로
                        DAW에서는 모두 같은 위치(맨 앞)에 놓으세요.
song.mid                전체 MIDI. 악기 번호는 General MIDI 기준이라 프로그램마다 소리가 다르게 날 수 있습니다. 멜로디는
                        보컬 계열 소리로 적혀 있지만 실제 노래가 아니라 음높이 안내입니다. 앱에서 효과나 플러그인으로 내는
                        소리(크런치·디스토션 기타, 신스 패드)는 가장 가까운 General MIDI 소리로 적었습니다.
scores/<part>.musicxml  파트별 오선보 (MuseScore 등에서 열어 인쇄). 앱의 "파트별 악보·연습"에서 보는 것과 같은 악보입니다.
                        기타·베이스는 옥타브 음자리표, 피아노는 두 단, 드럼은 타악기 보표로 적었고, 코드 이름과 구간 이름이
                        위에 적혀 있습니다. 같이 시작하는 음은 화음 하나로 묶은 한 줄 리듬입니다.
melody_chords.musicxml  멜로디와 코드 악보 (MuseScore 등에서 열기)
leadsheet.txt           작곡메이트 리드시트 텍스트 (앱의 "리드시트에서 곡 만들기"에 붙여 넣어 다시 편집)
manifest.json           이 팩의 파일별 SHA-256, 만든 방식, 저장된 곡 기록과의 대조 결과 (나중에 어느 버전인지 확인용)

들어 있지 않은 것
-----------------
PDF와 검증된 전곡 타브는 없습니다. 타브와 코드 폼은 음높이에서 자동으로 고른 추천 운지입니다.
앱이 들려주는 음원(가이드 보컬, AI 보컬 데모 포함)은 시안이며 이 팩에 들어 있지 않습니다.

만든 방식
---------
{source}

이 자료는 밴드가 고쳐서 직접 연주하기 위한 초안입니다.
"""


def part_midi(midi: bytes, part: str, total_ticks: int) -> bytes:
    """The tempo track plus one part, ending at the song's end so every part file has the same length."""
    mid = mido.MidiFile(file=io.BytesIO(midi))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for i, tr in enumerate(mid.tracks):
        name = next((m.name for m in tr if m.type == "track_name"), None)
        if i != 0 and name != part:
            continue
        msgs = [m for m in tr if m.type != "end_of_track"]
        length = sum(m.time for m in msgs)
        copy = mido.MidiTrack(msgs)
        copy.append(mido.MetaMessage("end_of_track", time=max(0, total_ticks - length)))
        out.tracks.append(copy)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def pick_tracks(midi: bytes, only: str | None = None, without: str | None = None) -> bytes:
    """The song's MIDI with one part alone (`only`) or left out (`without`), for practising against the rest.  The
    tempo track and every other message stay as they are; the length is the song's."""
    if (only is None) == (without is None):
        raise BandPackError("only와 without 중 하나만 골라 주세요")
    mid = mido.MidiFile(file=io.BytesIO(midi))
    names = [next((m.name for m in tr if m.type == "track_name"), None) for tr in mid.tracks]
    target = only if only is not None else without
    if names.count(target) != 1:
        raise BandPackError(f"이 곡에 {target} 파트가 {names.count(target)}개 있어요 (있는 파트: "
                            f"{', '.join(n for n in names if n)})")
    total = max(sum(m.time for m in tr) for tr in mid.tracks)
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for i, (tr, name) in enumerate(zip(mid.tracks, names)):
        keep = i == 0 or name is None or (name == only if only is not None else name != without)
        if not keep:
            continue
        msgs = [m for m in tr if m.type != "end_of_track"]
        copy = mido.MidiTrack(msgs)
        copy.append(mido.MetaMessage("end_of_track", time=max(0, total - sum(m.time for m in msgs))))
        out.tracks.append(copy)
    if len(out.tracks) < 2:
        raise BandPackError("남는 파트가 없어요")
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def build(result: dict, melody_source: str | None = None, band: dict | None = None,
          provenance: dict | None = None) -> bytes:
    """The zip.  `result`: a finished song (needs sections, bar_chords, midi_b64; musicxml and leadsheet if present).
    `band`: the band's line-up for the guide's checks.  `provenance`: facts the caller knows about where the song
    came from (stored song id, whether the content matches the stored record...); written to manifest.json together
    with the SHA-256 of every file, so a pack can later be matched to the version it was made from."""
    missing = [k for k in ("sections", "bar_chords", "midi_b64") if not result.get(k)]
    if missing:
        raise BandPackError(f"곡 결과에 {', '.join(missing)} 값이 없어 밴드 팩을 만들 수 없어요")
    if melody_source not in SOURCE_KO:
        raise BandPackError(f"melody_source must be ai, own or None, got {melody_source!r}")
    try:
        midi = base64.b64decode(result["midi_b64"], validate=True)
        tracks, _, bar_len = _tracks(midi)
    except Exception as exc:
        raise BandPackError(f"MIDI를 읽지 못했어요: {exc}") from None
    total = sum(s["bars"] for s in result["sections"])
    if len(result["bar_chords"]) < total:
        raise BandPackError(f"코드가 {len(result['bar_chords'])}마디뿐인데 곡은 {total}마디예요")
    absent = ""
    if not result.get("musicxml") or not result.get("leadsheet"):
        gone = [n for n, k in (("melody_chords.musicxml", "musicxml"), ("leadsheet.txt", "leadsheet")) if not result.get(k)]
        why = result.get("stale_scores") or "곡 결과에 이 자료가 없어요"
        absent = f"\n이 팩에 없는 파일: {', '.join(gone)}\n  이유: {why}\n"
    files: dict[str, bytes] = {
        "README.txt": (README.format(source=SOURCE_KO[melody_source]) + absent).encode("utf-8"),
        "chord_chart.html": chord_chart(result, melody_source).encode("utf-8"),
        "parts_guide.md": part_guide(result, band).encode("utf-8"),
        "song.mid": midi,
    }
    from . import partscore                   # here, not at the top: partscore reads the MIDI with this module's reader
    for part in [p for p in PART_ORDER if p in tracks]:
        files[f"parts/{part}.mid"] = part_midi(midi, part, total * bar_len)
        try:                                   # the same sheet the app shows for practising
            files[f"scores/{part}.musicxml"] = partscore.musicxml(result, part).encode("utf-8")
        except Exception:                      # a sheet that cannot be drawn never costs the band its pack
            continue
    if result.get("musicxml"):
        files["melody_chords.musicxml"] = result["musicxml"].encode("utf-8")
    if result.get("leadsheet"):
        files["leadsheet.txt"] = result["leadsheet"].encode("utf-8")
    manifest = {"made_by": "작곡메이트 band pack", "melody_source": melody_source,
                "melody_source_meaning": "own = no trained model was called; it says nothing about who owns the input melody",
                "member_parts": list(result.get("member_parts") or []), "band": band or None,
                "provenance": provenance or {"source": "request", "note": "not tied to a stored song record"},
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()
