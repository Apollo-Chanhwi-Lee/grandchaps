"""One instrument's part of a finished song as sheet music (MusicXML), for practising.

    musicxml(result, part, start_bar=0, bars=None) -> str
    parts(result) -> [part names that have notes]

The notes are read from the song's own MIDI (what is heard and exported), so the sheet shows what the app plays:
nothing is composed or changed here.  Notes that start together are written as one chord; each staff is a single
rhythmic line (a note is written until the next one starts), which is how a player reads a band part.  Chord symbols
and section names are written above the staff so the part can be followed in rehearsal.

Clefs follow each instrument's reading habit: guitar and bass on octave clefs (they sound an octave below where they
are written), the piano on two staves split at middle C, drums on a percussion staff with the usual positions.
The sheet is on the 16th-note grid; anything finer in a member's own part is rounded for reading only (the MIDI
keeps what was played).
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from music21 import articulations, chord, clef, expressions, harmony, layout, metadata, meter as m21meter, note, percussion, pitch, stream, tempo
from music21 import key as m21key
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .bandpack import PART_ORDER, _tracks
from .export import _m21_symbol

NOTE_NAMES = ("C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")
TITLE = {"melody": "멜로디", "lead": "리드 기타", "keys": "멜로디 피아노", "piano": "피아노", "bass": "베이스", "guitar": "기타",
         "strings": "스트링", "drums": "드럼", "harmony": "화음"}
SPLIT = 60                                   # piano: middle C and above on the upper staff
LOW_MELODY = 57                              # a melody centred below A3 is a man's line (a woman's sits around E4-A4)
# General MIDI drum note -> (staff position step, octave, notehead) on a five-line percussion staff
DRUMS = {35: ("F", 4, "normal"), 36: ("F", 4, "normal"), 37: ("C", 5, "x"), 38: ("C", 5, "normal"), 39: ("C", 5, "x"),
         40: ("C", 5, "normal"), 41: ("A", 4, "normal"), 43: ("A", 4, "normal"), 45: ("D", 5, "normal"),
         47: ("D", 5, "normal"), 48: ("E", 5, "normal"), 50: ("E", 5, "normal"), 42: ("G", 5, "x"), 44: ("D", 4, "x"),
         46: ("G", 5, "x"), 49: ("A", 5, "x"), 57: ("A", 5, "x"), 51: ("F", 5, "x"), 59: ("F", 5, "x"),
         54: ("B", 5, "triangle"), 56: ("E", 5, "triangle")}
DRUM_DEFAULT = ("B", 4, "normal")
OPEN_HAT = 46


class PartScoreError(ValueError):
    """The part cannot be written as a sheet (Korean message)."""


def parts(result: dict) -> list[str]:
    tracks, _, _ = _tracks(base64.b64decode(result["midi_b64"]))
    return [p for p in PART_ORDER if p in tracks]


def _clef(part: str, staff: str = "upper", low: bool = False):
    if part == "drums":
        return clef.PercussionClef()
    if part == "bass":
        return clef.Bass8vbClef()
    if part in ("guitar", "lead") or low:        # `low`: a man's vocal line, written an octave above where it sounds
        return clef.Treble8vbClef()
    if part in ("piano", "keys") and staff == "lower":
        return clef.BassClef()
    return clef.TrebleClef()


def _line(notes: list[tuple], lo: int, hi: int, grid: int) -> list[tuple[int, int, list[int]]]:
    """[(start, length, [pitches])] of the notes starting in lo..hi, on the grid, as one rhythmic line: notes starting
    together form a chord, and a chord is written until the next one starts (or its own longest note ends)."""
    groups: dict[int, list[tuple[int, int]]] = {}
    for start, pitch, _vel, end in notes:
        s = int(round(start / grid)) * grid
        if lo <= s < hi:
            e = max(s + grid, int(round(end / grid)) * grid)
            groups.setdefault(s, []).append((pitch, min(e, hi)))
    out, starts = [], sorted(groups)
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else hi
        length = min(nxt, max(e for _, e in groups[s])) - s
        out.append((s - lo, max(grid, length), sorted({p for p, _ in groups[s]})))
    return out


def _spell(midi: int, scale: frozenset[str]):
    """A MIDI pitch spelled for the key (a flat key writes Bb, not A#); a natural the key already implies is not drawn."""
    p = pitch.Pitch(midi=midi)
    for cand in (p, p.getEnharmonic()):
        if cand.name in scale:
            p = cand
            break
    if p.accidental is not None and p.accidental.alter == 0:
        p.accidental = None
    return p


def _element(part: str, pitches: list[int], scale: frozenset[str] = frozenset()):
    if part == "drums":
        hits = []
        for p in pitches:
            step, octave, head = DRUMS.get(p, DRUM_DEFAULT)
            u = note.Unpitched()
            u.displayStep, u.displayOctave, u.notehead = step, octave, head
            hits.append(u)
        el = hits[0] if len(hits) == 1 else percussion.PercussionChord(hits)
        if OPEN_HAT in pitches:                                 # the open hi-hat: an x with the small circle above it
            el.articulations.append(articulations.OpenString())
        return el
    spelled = [_spell(p, scale) for p in pitches]
    return note.Note(spelled[0]) if len(spelled) == 1 else chord.Chord(spelled)


def _staff(part: str, line: list[tuple[int, int, list[int]]], total: int, tpq: int, staff: str, first: list,
           scale: frozenset[str] = frozenset(), low: bool = False) -> stream.Part:
    p = stream.PartStaff() if part in ("piano", "keys") else stream.Part()
    p.partName = TITLE.get(part, part)
    p.insert(0, _clef(part, staff, low))
    for el in first:
        p.insert(0, el)
    at = 0
    for start, length, pitches in line:
        if start > at:
            r = note.Rest()
            r.quarterLength = (start - at) / tpq
            p.insert(at / tpq, r)
        el = _element(part, pitches, scale)
        el.quarterLength = length / tpq
        p.insert(start / tpq, el)
        at = start + length
    if at < total:
        r = note.Rest()
        r.quarterLength = (total - at) / tpq
        p.insert(at / tpq, r)
    return p


def musicxml(result: dict, part: str, start_bar: int = 0, bars: int | None = None) -> str:
    """See the module docstring.  `start_bar`, `bars`: a stretch of the song (default: all of it)."""
    try:
        tracks, tpq, bar_len = _tracks(base64.b64decode(result["midi_b64"], validate=True))
    except Exception as exc:
        raise PartScoreError(f"곡의 MIDI를 읽지 못했어요: {exc}") from None
    if part not in tracks:
        raise PartScoreError(f"이 곡에는 {TITLE.get(part, part)} 파트가 없어요 (있는 파트: "
                             f"{', '.join(TITLE.get(p, p) for p in PART_ORDER if p in tracks)})")
    song_bars = sum(s["bars"] for s in result.get("sections") or []) or -(-max(n[3] for n in tracks[part]) // bar_len)
    if type(start_bar) is not int or start_bar < 0 or start_bar >= song_bars:
        raise PartScoreError(f"시작 마디가 곡 밖이에요 (곡은 {song_bars}마디)")
    n_bars = song_bars - start_bar if bars is None else bars
    if type(n_bars) is not int or n_bars < 1 or start_bar + n_bars > song_bars:
        raise PartScoreError(f"마디 범위가 곡 밖이에요 (곡은 {song_bars}마디)")
    lo, hi, grid = start_bar * bar_len, (start_bar + n_bars) * bar_len, tpq // 4
    total = hi - lo
    num = bar_len * 4 // tpq                                     # quarter-note beats per bar x 4 / 4
    sig = m21meter.TimeSignature(result.get("meter") or f"{num // 4 if num % 4 == 0 else num}/4")
    first = [sig]
    if part != "drums" and result.get("key"):
        k = result["key"]
        first.append(m21key.Key(NOTE_NAMES[int(k["tonic"]) % 12], "minor" if k.get("mode") == "minor" else "major"))
    if result.get("tempo"):
        first.append(tempo.MetronomeMark(number=int(result["tempo"])))
    notes = tracks[part]
    song_key = next((e for e in first if isinstance(e, m21key.Key)), m21key.Key("C"))
    scale = frozenset(sp.name for sp in song_key.getScale().getPitches())
    if part in ("piano", "keys"):
        upper = _staff(part, _line([n for n in notes if n[1] >= SPLIT], lo, hi, grid), total, tpq, "upper", first, scale)
        lower = _staff(part, _line([n for n in notes if n[1] < SPLIT], lo, hi, grid), total, tpq, "lower",
                       [m21meter.TimeSignature(sig.ratioString)] + [type(e)(e.tonic.name, e.mode) for e in first if isinstance(e, m21key.Key)],
                       scale)
        staves = [upper, lower]
    else:
        # a melody that lies below middle C (a man's key) is written like a tenor part: treble clef, an octave up, so
        # it reads in the staff instead of hanging on ledger lines.  Decided on the whole part, not on the stretch.
        pitches = sorted(n[1] for n in notes)
        male = (result.get("voice_fit") or {}).get("voice") == "male"
        low = part == "melody" and bool(pitches) and (male or pitches[len(pitches) // 2] < LOW_MELODY)
        staves = [_staff(part, _line(notes, lo, hi, grid), total, tpq, "upper", first, scale, low)]
    top = staves[0]
    for sec in result.get("sections") or []:                    # section names where they begin
        if start_bar <= sec["start_bar"] < start_bar + n_bars:
            # bold words above the staff, not a boxed rehearsal mark: the engraver measures Korean text too narrow for
            # a box and stacks plain words clear of the chord names
            words = expressions.TextExpression(sec["label"])
            words.placement, words.style.fontWeight, words.style.fontStyle = "above", "bold", "normal"
            top.insert((sec["start_bar"] - start_bar) * bar_len / tpq, words)
    if part != "drums":
        last = None
        for i, bar in enumerate((result.get("bar_chords") or [])[start_bar:start_bar + n_bars]):
            for tick, name in bar:
                if name == last:
                    continue                                     # a chord is named when it changes
                last = name
                try:
                    cs = harmony.ChordSymbol(_m21_symbol(name))
                except Exception:
                    continue                                     # a name music21 cannot read is left out of the sheet
                cs.writeAsChord = False
                top.insert((i * bar_len + int(tick)) / tpq, cs)
    made = [s.makeMeasures().makeTies() for s in staves]
    for staff_part in made:                                     # bar numbers are the song's own (a chorus starts at bar 33)
        for m in staff_part.getElementsByClass(stream.Measure):
            m.number += start_bar
    score = stream.Score(made)
    if len(made) == 2:
        score.insert(0, layout.StaffGroup(made, name=TITLE.get(part, part), symbol="brace"))
    score.metadata = metadata.Metadata(title=TITLE.get(part, part), composer="")
    xml = GeneralObjectExporter(score).parse().decode("utf-8")
    # music21 writes a random id per part: replaced by fixed ones, so the same song always gives the same file (the
    # band pack's manifest records each file's hash)
    for i, random_id in enumerate(dict.fromkeys(re.findall(r'<score-part id="([^"]+)"', xml)), 1):
        xml = xml.replace(f'"{random_id}"', f'"P{i}"')
    return xml


__all__ = ["musicxml", "parts", "PartScoreError", "TITLE"]


# ---------------------------------------------------------------- tablature (guitar, bass) and a drum chart
TAB_PARTS = {"guitar": "guitar", "lead": "guitar", "bass": "bass"}
MAX_SPAN = 4                                 # frets one hand position covers
KEEP = 10                                    # fingerings kept per chord while choosing a path
# drum chart rows, top to bottom: label, {MIDI note: mark}
DRUM_ROWS = (("CC", {49: "x", 57: "x"}), ("RD", {51: "x", 59: "x"}), ("HH", {42: "x", 46: "o", 44: "x"}),
             ("TB", {54: "x", 56: "x"}), ("T1", {48: "o", 50: "o"}), ("T2", {45: "o", 47: "o"}),
             ("SD", {38: "o", 40: "o", 37: "x", 39: "x"}), ("FT", {41: "o", 43: "o"}), ("BD", {35: "o", 36: "o"}))
DRUM_NAMES = {"CC": "크래시", "RD": "라이드", "HH": "하이햇 (x 닫힘, o 열림)", "TB": "탬버린", "T1": "탐 높음", "T2": "탐 중간",
              "SD": "스네어 (o 스네어, x 림·클랩)", "FT": "플로어 탐", "BD": "킥"}


def _shapes(pitches: list[int], tuning: tuple[int, ...], max_fret: int) -> list[tuple[tuple[int, int], ...]]:
    """Every way to hold the pitches at once: ((string index, fret), ...) with one note per string, higher notes on
    higher strings, and the fretted notes within one hand position.  Best (lowest, most open) first."""
    want = sorted(set(pitches), reverse=True)
    out: list[tuple[tuple[int, int], ...]] = []

    def go(k: int, string: int, taken: tuple):
        if k == len(want):
            frets = [f for _, f in taken if f]
            if not frets or max(frets) - min(frets) <= MAX_SPAN:
                out.append(taken)
            return
        for s in range(string, len(tuning)):
            fret = want[k] - tuning[s]
            if 0 <= fret <= max_fret:
                go(k + 1, s + 1, taken + ((s, fret),))

    go(0, 0, ())
    out.sort(key=lambda sh: (sum(f for _, f in sh), max((f for _, f in sh), default=0)))
    return out[:KEEP]


def _position(shape) -> int | None:
    frets = [f for _, f in shape if f]
    return min(frets) if frets else None      # all open strings: the hand is free


def fingering(groups: list[tuple[int, list[int]]], tuning: tuple[int, ...], max_fret: int = 15):
    """[(start, ((string index, fret), ...) or None)] for chords in time order: the path with the least hand movement
    that stays low on the neck.  None for a chord that cannot be held (out of range, more notes than strings, or
    wider than a hand); it is reported, never silently dropped."""
    options = [_shapes(p, tuning, max_fret) for _, p in groups]
    cost: dict = {}
    back: list[dict] = []
    prev_k = None
    for k, shapes in enumerate(options):
        cur, link = {}, {}
        for sh in shapes:
            own = 0.15 * sum(f for _, f in sh) + 0.6 * max((f for _, f in sh), default=0)
            if prev_k is None:
                cur[sh], link[sh] = own, None
                continue
            best = None
            for prev, c in cost.items():
                a, b = _position(prev), _position(sh)
                step = c + own + (abs(a - b) if a is not None and b is not None else 0)
                if best is None or step < best[0]:
                    best = (step, prev)
            cur[sh], link[sh] = best
        back.append(link)
        if shapes:
            cost, prev_k = cur, k
    out: list = [None] * len(groups)
    if prev_k is None:
        return [(s, None) for s, _ in groups]
    sh = min(cost, key=cost.get)
    k = prev_k
    while sh is not None:
        out[k] = sh
        sh = back[k][sh]
        k -= 1
        while k >= 0 and not options[k]:
            k -= 1
    return [(groups[i][0], out[i]) for i in range(len(groups))]


def tab(result: dict, part: str, start_bar: int = 0, bars: int | None = None) -> dict:
    """A stretch of one part as tablature data for the page to draw.

    guitar / lead / bass: {"kind": "strings", "lines": string names top to bottom, "bars": [{"number", "chords",
    "cells": [[column, line, text]]}], "steps": columns per bar, "unplayable": chords left out, "note"}.
    drums: the same shape with one line per drum that plays ("kind": "drums", cells hold "x" / "o").
    Fingerings are suggestions computed from the pitches (low on the neck, little hand movement), not a checked
    transcription of how a guitarist would play it."""
    try:
        tracks, tpq, bar_len = _tracks(base64.b64decode(result["midi_b64"], validate=True))
    except Exception as exc:
        raise PartScoreError(f"곡의 MIDI를 읽지 못했어요: {exc}") from None
    if part not in TAB_PARTS and part != "drums":
        raise PartScoreError("타브는 기타·리드 기타·베이스·드럼 파트에만 있어요")
    if part not in tracks:
        raise PartScoreError(f"이 곡에는 {TITLE.get(part, part)} 파트가 없어요")
    song_bars = sum(s["bars"] for s in result.get("sections") or []) or -(-max(n[3] for n in tracks[part]) // bar_len)
    n_bars = song_bars - start_bar if bars is None else bars
    if type(start_bar) is not int or type(n_bars) is not int or start_bar < 0 or n_bars < 1 or start_bar + n_bars > song_bars:
        raise PartScoreError(f"마디 범위가 곡 밖이에요 (곡은 {song_bars}마디)")
    grid = tpq // 4
    steps = bar_len // grid
    lo, hi = start_bar * bar_len, (start_bar + n_bars) * bar_len
    line = _line(tracks[part], lo, hi, grid)                    # [(start from lo, length, [pitches])]
    chords = (result.get("bar_chords") or [])[start_bar:start_bar + n_bars]
    labels = {s["start_bar"]: s["label"] for s in result.get("sections") or []}
    out_bars = [{"number": start_bar + i + 1, "label": labels.get(start_bar + i),
                 "chords": [[int(t) // grid, name] for t, name in (chords[i] if i < len(chords) else [])] if part != "drums" else [],
                 "cells": []} for i in range(n_bars)]
    unplayable = 0
    if part == "drums":
        used = [(label, marks) for label, marks in DRUM_ROWS if any(p in marks for _, _, ps in line for p in ps)]
        other = sorted({p for _, _, ps in line for p in ps if not any(p in m for _, m in DRUM_ROWS)})
        lines = [label for label, _ in used] + (["기타"] if other else [])
        for start, _, pitches in line:
            b, col = start // bar_len, (start % bar_len) // grid
            for p in pitches:
                row = next((i for i, (_, marks) in enumerate(used) if p in marks), len(used))
                out_bars[b]["cells"].append([col, row, used[row][1][p] if row < len(used) else "x"])
        legend = " · ".join(f"{label} {DRUM_NAMES[label]}" for label, _ in used)
        return {"kind": "drums", "part": part, "title": TITLE[part], "lines": lines, "steps": steps, "bars": out_bars,
                "unplayable": 0, "note": legend}
    kind = TAB_PARTS[part]
    from .bandpack import BASS_TUNING, GUITAR_TUNING, STRING_NAMES
    tuning = GUITAR_TUNING if kind == "guitar" else BASS_TUNING
    for (start, _, _), (_, shape) in zip(line, fingering([(s, p) for s, _, p in line], tuning)):
        b, col = start // bar_len, (start % bar_len) // grid
        if shape is None:
            unplayable += 1
            out_bars[b]["cells"].append([col, len(tuning) // 2, "?"])
            continue
        out_bars[b]["cells"] += [[col, string, str(fret)] for string, fret in shape]
    note_text = ("숫자는 누를 프렛, 0은 개방현이에요. 운지는 음높이에서 자동으로 고른 추천이에요(낮은 프렛, 손 이동이 적은 쪽)."
                 + (f" 한 손에 잡히지 않는 화음 {unplayable}곳은 ?로 표시했어요. 오선보를 보세요." if unplayable else ""))
    return {"kind": "strings", "part": part, "title": TITLE[part], "lines": list(STRING_NAMES[len(tuning)]), "steps": steps,
            "bars": out_bars, "unplayable": unplayable, "note": note_text}


# ---------------------------------------------------------------- engraved pages (Verovio) and the print document
A4 = (2100, 2970)                             # tenths of a millimetre
MARGIN = {"top": 300, "bottom": 230, "side": 160}     # room for the page's own title line and footer
PRINT_CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; background: #e9e9e4; color: #111; font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', system-ui, sans-serif; }
.page { position: relative; width: 210mm; height: 297mm; margin: 8mm auto; background: #fff; overflow: hidden; page-break-after: always; break-after: page; box-shadow: 0 1px 6px rgba(0,0,0,.25); }
.page > svg { position: absolute; left: 0; top: 0; width: 210mm; height: 297mm; }
.page header { position: absolute; left: 16mm; right: 16mm; top: 9mm; display: flex; justify-content: space-between; align-items: baseline; }
.page header h1 { margin: 0; font-size: 17pt; font-weight: 700; }
.page header .part { font-size: 12pt; font-weight: 700; }
.page header .meta { font-size: 9pt; color: #444; }
.page header.small h1 { font-size: 10pt; font-weight: 600; color: #333; }
.page footer { position: absolute; left: 16mm; right: 16mm; bottom: 7mm; display: flex; justify-content: space-between; font-size: 8pt; color: #555; }
.page .tabnote { position: absolute; left: 16mm; right: 16mm; top: 22mm; font-size: 8.5pt; color: #444; }
.page .tab { position: absolute; left: 12mm; right: 12mm; top: 30mm; }
.page .tab svg { width: 100%; height: auto; display: block; }
.hint { max-width: 210mm; margin: 6mm auto 0; font-size: 10pt; color: #333; }
@media print { body { background: #fff; } .page { margin: 0; box-shadow: none; } .hint { display: none; } }
"""


class EngraverMissing(PartScoreError):
    """The engraving library is not installed; the page falls back to its own simple drawing."""


def _toolkit():
    try:
        import verovio
    except ImportError:
        raise EngraverMissing("악보 조판 엔진(verovio)이 설치돼 있지 않아요. `pip install verovio` 후 앱을 다시 켜 주세요. "
                              "그동안은 간단한 화면 악보를 써요.") from None
    # the fonts are named explicitly: a toolkit made in a request thread does not find its default resource folder
    tk = verovio.toolkit(False)
    data = Path(verovio.__file__).resolve().parent / "data"
    if not tk.setResourcePath(str(data)):
        raise PartScoreError(f"악보 조판 엔진의 글꼴 자료를 찾지 못했어요: {data}")
    tk.setOptions({"pageWidth": A4[0], "pageHeight": A4[1], "pageMarginTop": MARGIN["top"], "pageMarginBottom": MARGIN["bottom"],
                   "pageMarginLeft": MARGIN["side"], "pageMarginRight": MARGIN["side"], "scale": 40, "header": "none",
                   "footer": "none", "breaks": "auto", "justifyVertically": True, "svgViewBox": True,
                   "xmlIdSeed": 1})          # fixed ids: the same song always gives the same pages
    return tk


def engrave(result: dict, part: str, start_bar: int = 0, bars: int | None = None) -> list[str]:
    """The part's sheet laid out on A4 pages by an engraving engine (Verovio): proportional spacing, aligned systems,
    page breaks between systems.  One SVG per page.  The notes are those of `musicxml`: only the drawing differs."""
    xml = musicxml(result, part, start_bar, bars)
    tk = _toolkit()
    if not tk.loadData(xml):
        raise PartScoreError(f"악보를 조판하지 못했어요: {tk.getLog()[:200]}")
    return [tk.renderToSVG(i + 1) for i in range(tk.getPageCount())]


def sounds_octave_lower(result: dict) -> bool:
    """True when the melody sheet is written on the octave treble clef (a man's key): the same rule `musicxml` uses."""
    tracks, _, _ = _tracks(base64.b64decode(result["midi_b64"]))
    pitches = sorted(n[1] for n in tracks.get("melody", []))
    male = (result.get("voice_fit") or {}).get("voice") == "male"
    return bool(pitches) and (male or pitches[len(pitches) // 2] < LOW_MELODY)


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def tab_svg(data: dict, per_row: int = 4) -> str:
    """Tablature / drum chart data (`tab`) drawn as one SVG: bars in rows of `per_row`."""
    cell, gap, left, pad, head = 15, 16, 34, 12, 30
    lines, steps, bars = len(data["lines"]), data["steps"], data["bars"]
    bar_w, row_h = steps * cell + 10, head + (lines - 1) * gap + 30
    rows = -(-len(bars) // per_row)
    w, h = left + per_row * bar_w + pad, rows * row_h + pad
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" font-family="ui-monospace, Menlo, Consolas, monospace" '
         f'font-size="12" fill="#111"><rect width="{w}" height="{h}" fill="#fff"/>']
    for i, bar in enumerate(bars):
        row, col = divmod(i, per_row)
        x0, y0 = left + col * bar_w, row * row_h + head + pad
        if col == 0:
            s += [f'<text x="{left - 8}" y="{y0 + n * gap + 4}" text-anchor="end" fill="#555">{_esc(name)}</text>'
                  for n, name in enumerate(data["lines"])]
        s += [f'<line x1="{x0}" y1="{y0 + n * gap}" x2="{x0 + bar_w}" y2="{y0 + n * gap}" stroke="#999" stroke-width="1"/>' for n in range(lines)]
        bottom = y0 + (lines - 1) * gap
        s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{bottom}" stroke="#333" stroke-width="{2 if col == 0 else 1}"/>')
        if col == per_row - 1 or i == len(bars) - 1:
            s.append(f'<line x1="{x0 + bar_w}" y1="{y0}" x2="{x0 + bar_w}" y2="{bottom}" stroke="#333" stroke-width="1"/>')
        s += [f'<line x1="{x0 + 5 + b * 4 * cell}" y1="{bottom + 4}" x2="{x0 + 5 + b * 4 * cell}" y2="{bottom + 9}" stroke="#bbb"/>'
              for b in range(1, steps // 4)]
        s.append(f'<text x="{x0 + 3}" y="{y0 - 18}" font-size="10" fill="#777">{bar["number"]}</text>')
        if bar.get("label"):
            s.append(f'<text x="{x0 + 22}" y="{y0 - 18}" font-size="11" font-weight="700" fill="#2f6b1f" '
                     f'font-family="system-ui, sans-serif">{_esc(bar["label"])}</text>')
        last_x = -99
        for at, name in bar.get("chords") or []:
            x = max(x0 + 5 + at * cell, last_x + 26)
            s.append(f'<text x="{x}" y="{y0 - 5}" font-weight="700" font-family="system-ui, sans-serif">{_esc(name)}</text>')
            last_x = x
        for at, line, text in bar["cells"]:
            x, y, n = x0 + 5 + at * cell + cell / 2, y0 + line * gap, len(str(text))
            s.append(f'<rect x="{x - 4 * n - 1}" y="{y - 6}" width="{8 * n + 2}" height="12" fill="#fff"/>'
                     f'<text x="{x}" y="{y + 4}" text-anchor="middle"{" fill=" + chr(34) + "#b00" + chr(34) if text == "?" else ""}>{_esc(text)}</text>')
    return "".join(s) + "</svg>"


def tab_pages(data: dict, per_row: int = 4) -> list[str]:
    """`tab_svg` cut into printed pages: whole rows only, never a row split across pages."""
    # how many rows fit the printed page: the drawing is 186 mm wide on paper and about 245 mm of height is free
    lines = len(data["lines"])
    width_px, row_px = 34 + per_row * (data["steps"] * 15 + 10) + 12, 30 + (lines - 1) * 16 + 30
    rows = max(1, int(245 / (row_px * 186 / width_px)))
    size = rows * per_row
    return [tab_svg({**data, "bars": data["bars"][i:i + size]}, per_row) for i in range(0, len(data["bars"]), size)]


def _meta(result: dict) -> str:
    key = (result.get("key") or {}).get("label") or (result.get("key") or {}).get("name") or ""
    return " · ".join(x for x in (key, f"{result['tempo']} BPM" if result.get("tempo") else "", f"{result.get('bars')}마디" if result.get("bars") else "") if x)


def print_html(result: dict, parts_wanted: list[str], start_bar: int = 0, bars: int | None = None, title: str = "",
               tabs: bool = True) -> dict:
    """A complete document ready for the browser's print: every wanted part on A4 pages with the song title, the part
    name, key / tempo, the stretch, a footer and page numbers; the part's tablature (guitar, bass) or drum chart on
    pages of its own after its sheet.  Returns {"html", "pages", "parts"}."""
    if not parts_wanted:
        raise PartScoreError("인쇄할 파트를 골라 주세요")
    title = (title or "").strip()[:80] or "작곡메이트 곡"
    song_bars = sum(s["bars"] for s in result.get("sections") or [])
    n_bars = (song_bars - start_bar) if bars is None else bars
    stretch = "곡 전체" if start_bar == 0 and n_bars == song_bars else f"{start_bar + 1}~{start_bar + n_bars}마디"
    blocks: list[tuple[str, str, str]] = []                       # (part title, kind, inner html)
    for part in parts_wanted:
        name = TITLE.get(part, part)
        if part == "melody":                    # say which voice the sheet is for: the two look alike on paper
            voice = (result.get("voice_fit") or {}).get("voice")
            if sounds_octave_lower(result):
                name += " · 남성 (적힌 것보다 한 옥타브 아래로 불러요)"
            elif voice == "female":
                name += " · 여성 (적힌 높이 그대로)"
        for svg in engrave(result, part, start_bar, bars):
            blocks.append((name, "sheet", svg))
        if tabs and (part in TAB_PARTS or part == "drums"):
            data = tab(result, part, start_bar, bars)
            what = "드럼 차트" if data["kind"] == "drums" else "타브"
            for svg in tab_pages(data):
                blocks.append((f"{name} {what}", "tab", f'<p class="tabnote">{_esc(data["note"])}</p><div class="tab">{svg}</div>'))
    total = len(blocks)
    pages, seen = [], set()
    for n, (name, kind, inner) in enumerate(blocks, 1):
        first = name not in seen
        seen.add(name)
        head = (f'<header><h1>{_esc(title)}</h1><span class="part">{_esc(name)}</span><span class="meta">{_esc(_meta(result))} · {stretch}</span></header>'
                if first else f'<header class="small"><h1>{_esc(title)} · {_esc(name)}</h1><span class="meta">{stretch}</span></header>')
        pages.append(f'<section class="page">{inner if kind == "sheet" else ""}{head}{inner if kind == "tab" else ""}'
                     f'<footer><span>작곡메이트로 만든 연습용 악보</span><span>{n} / {total}</span></footer></section>')
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{_esc(title)} · '
            f'{_esc(", ".join(TITLE.get(p, p) for p in parts_wanted))}</title><style>{PRINT_CSS}</style></head><body>'
            '<p class="hint">인쇄 창에서 종이로 뽑거나, 대상에서 "PDF로 저장"을 고르세요. 용지 A4, 여백 "없음" 또는 "기본값", 배율 100%가 맞아요.</p>'
            + "".join(pages) + "</body></html>")
    return {"html": html, "pages": total, "parts": [TITLE.get(p, p) for p in parts_wanted]}
