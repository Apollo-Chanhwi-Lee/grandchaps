"""Compose option "melody_shape" (off unless asked for): shape the sections the app wrote so that the chorus arrives.

Applied to a finished lead sheet, after instmate/melodyfix.py, and only to sections the caller does not protect (the
user's own melody is never touched, and neither is the verse):
- pre-chorus, when a chorus follows: the last long note of its last two bars moves to the nearest 2nd or 5th degree,
  the dominant chord sounds from there to the end, and the last beat becomes a two-note pickup that walks by step to
  the chorus's first note (a rest when no such pickup fits the singer's range).  The section ends open instead of
  arriving on the tonic before the chorus does.
- a pre-chorus whose second half repeats its first half note for note: the second half moves up a third (a second
  when a third would reach the chorus's highest note), same rhythm.
- bridge: when three or more of its bars are bars of a verse or pre-chorus (transposition ignored), it is written
  anew: chords that start on IV (VI in minor) and end on the dominant, a line that starts low and rises, an open ending.

No model, no randomness.  Every choice here (which degrees, how many shared bars, the rhythm cells, the range of the
new bridge) is a design hypothesis to be judged by ear (work/listening), not a rule taken from a source.  4/4 only.
"""
from __future__ import annotations

import copy

from .leadsheet import Bar, ChordMark, LeadNote, LeadSheet, Section
from .render import KIND, key_of, parse_chord
from .theory import TPQ, Key

BAR = 4 * TPQ
SHARED_BARS = 3                     # a bridge with this many verse / pre-chorus bars is written anew
SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"}
# the new bridge: one rhythm per bar (quarter notes), cycled; the last bar is one long note before the pickup
CELLS = ([1, 0.5, 0.5, 1, 1], [1.5, 0.5, 2], [0.5, 0.5, 1, 1, 1], [1.5, 0.5, 2])
BRIDGE_FLOOR, BRIDGE_SPAN = 57, 12  # lowest start of the new bridge's line and how far it may rise (semitones)


def _kind(sec: Section) -> str | None:
    return KIND.get(sec.name.split()[0]) if sec.name.split() else None


def _scale(key: Key) -> list[int]:
    steps = [0, 2, 4, 5, 7, 9, 11] if key.mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    return sorted((key.tonic + s) % 12 for s in steps)


def _step(key: Key, pitch: int, steps: int) -> int:
    """`pitch` moved by `steps` scale steps (a pitch outside the scale counts as its nearest scale note)."""
    scale = _scale(key)
    octave, pc = divmod(pitch, 12)
    i = min(range(7), key=lambda j: abs(scale[j] - pc)) + steps
    return (octave + i // 7) * 12 + scale[i % 7]


def _nearest(target: float, pcs: set[int], lo: int, hi: int, avoid: int | None = None) -> int | None:
    cands = [p for p in range(lo, hi + 1) if p % 12 in pcs and p != avoid]
    return min(cands, key=lambda p: (abs(p - target), p)) if cands else None


def _names(sheet: LeadSheet) -> dict[str, str]:
    """Chord symbols of this key by function: dominant, the bridge's opening chord and its second chord."""
    key = key_of(sheet.key)
    name = (FLATS if sheet.key in FLAT_KEYS else SHARPS)
    at = lambda semis: name[(key.tonic + semis) % 12]
    if key.mode == "major":
        return {"V": at(7), "open": at(5), "second": at(9) + "m"}       # IV and vi
    return {"V": at(7), "open": at(8), "second": at(5) + "m"}           # VI and iv


def _signature(bar: Bar) -> tuple | None:
    """A bar's melody with transposition taken out; None for a bar with no pitched note."""
    first = next((n.pitch for n in sorted(bar.notes, key=lambda n: n.start) if n.pitch is not None), None)
    if first is None:
        return None
    return tuple((None if n.pitch is None else n.pitch - first, n.start, n.dur) for n in sorted(bar.notes, key=lambda n: n.start))


def shared_bars(sec: Section, others: list[Section]) -> int:
    """How many bars of `sec` are, up to transposition, a bar of one of `others` (bars without notes do not count)."""
    known = {_signature(b) for o in others for b in o.bars} - {None}
    return sum(1 for b in sec.bars if _signature(b) in known)


def _fill(bar: Bar) -> None:
    """Sort the bar's notes and put rests into the gaps so that the bar is full again."""
    notes, at = [], 0
    for n in sorted(bar.notes, key=lambda n: n.start):
        if n.start > at:
            notes.append(LeadNote(None, at, n.start - at))
        notes.append(n)
        at = n.start + n.dur
    if at < BAR:
        notes.append(LeadNote(None, at, BAR - at))
    bar.notes = notes


def _first_pitch(sec: Section) -> int | None:
    return next((n.pitch for b in sec.bars for n in sorted(b.notes, key=lambda n: n.start) if n.pitch is not None), None)


def _top(sections: list[Section]) -> int | None:
    ps = [n.pitch for s in sections for b in s.bars for n in b.notes if n.pitch is not None]
    return max(ps) if ps else None


def _open_ending(sec: Section, sheet: LeadSheet, target: int | None, lo: int | None, hi: int | None) -> str | None:
    """The section's ending made open (see the module docstring); what was done, or None when nothing fits."""
    key, v = key_of(sheet.key), _names(sheet)["V"]
    try:
        parse_chord(v, key)
    except ValueError:
        return None
    cut = BAR - TPQ                                   # the pickup takes the last beat of the last bar
    tail = sec.bars[-2:]
    found = None                                      # the last long note that starts before the pickup
    for bar in tail:
        for n in sorted(bar.notes, key=lambda n: n.start):
            if n.pitch is not None and n.dur >= TPQ and not n.tie and (bar is not sec.bars[-1] or n.start < cut):
                found = (bar, n)
    done = []
    if found:
        bar, n = found
        open_pcs = {(key.tonic + 2) % 12, (key.tonic + 7) % 12}
        to = _nearest(n.pitch, open_pcs, max(lo or 0, n.pitch - 6), min(hi or 127, n.pitch + 6))
        if to is not None:
            if to != n.pitch:
                done.append(f"긴 음 {n.pitch}->{to}")
            n.pitch = to
            beat = n.start                             # a chord symbol can only stand where a note starts
            for b in tail[tail.index(bar):]:
                b.chords = [c for c in b.chords if b is bar and c.start < beat] + [ChordMark(v, beat if b is bar else 0)]
            done.append(f"끝 화음 {v}")
    last = sec.bars[-1]
    before = [n for n in last.notes if n.start < cut]
    for n in before:
        if n.start + n.dur > cut:
            n.dur, n.tie = cut - n.start, False
    held = next((n.pitch for n in reversed(sorted(before, key=lambda n: n.start)) if n.pitch is not None), None)
    pickup: list[LeadNote] = []
    if target is not None:
        for d in ([1, -1] if held is None or held <= target else [-1, 1]):      # walk up to the chorus, or down to it
            ps = [_step(key, target, -2 * d), _step(key, target, -d)]
            if all((lo is None or p >= lo) and (hi is None or p <= hi) for p in ps):
                pickup = [LeadNote(ps[0], cut, TPQ // 2), LeadNote(ps[1], cut + TPQ // 2, TPQ // 2)]
                break
    last.notes = before + pickup
    _fill(last)
    done.append("못갖춘마디 2음" if pickup else "끝 박 쉼")
    return ", ".join(done)


def _vary_second_half(sec: Section, key: Key, ceiling: int | None, hi: int | None) -> str | None:
    n = len(sec.bars)
    if n < 4 or n % 2:
        return None
    half = n // 2
    same = all(_signature(a) is not None and [(x.pitch, x.start, x.dur) for x in a.notes] == [(x.pitch, x.start, x.dur) for x in b.notes]
               for a, b in zip(sec.bars[:half], sec.bars[half:]))
    if not same:
        return None
    limit = min(x for x in (ceiling - 1 if ceiling is not None else None, hi, 127) if x is not None)
    for steps in (2, 1):
        moved = [[None if x.pitch is None else _step(key, x.pitch, steps) for x in b.notes] for b in sec.bars[half:]]
        if all(p is None or p <= limit for row in moved for p in row):
            for b, row in zip(sec.bars[half:], moved):
                for x, p in zip(b.notes, row):
                    x.pitch = p
            return f"뒤 {half}마디를 {steps + 1}도 위로"
    return None


def _new_bridge(sec: Section, sheet: LeadSheet, others: list[Section], ceiling: int | None,
                lo: int | None, hi: int | None) -> None:
    key, names = key_of(sheet.key), _names(sheet)
    n = len(sec.bars)
    plan = [names["open"], names["open"], names["second"], names["second"], names["open"], names["open"], names["V"], names["V"]]
    low = max(lo or BRIDGE_FLOOR, BRIDGE_FLOOR)
    high = min(x for x in (hi, ceiling - 1 if ceiling is not None else None, 84) if x is not None)
    start = min(low + 3, high)
    end = min(high - 2, start + BRIDGE_SPAN) if high - 2 > start else high
    scale = set(_scale(key))
    known = {_signature(b) for o in others for b in o.bars} - {None}
    prev = None
    bars: list[Bar] = []
    for bi in range(n):
        symbol = plan[bi * 8 // n]
        chord = set(parse_chord(symbol, key).pcs)
        cell = [4] if bi == n - 1 else CELLS[bi % len(CELLS)]
        for attempt in range(3):                      # a bar that happens to equal a verse bar gets another cell
            notes, at = [], 0
            walk = prev
            for q in (cell if attempt == 0 else CELLS[(bi + attempt) % len(CELLS)]):
                dur = int(q * TPQ)
                where = (bi * BAR + at) / max(1, n * BAR - 1)
                rests_here = at % TPQ == 0 and dur >= TPQ
                p = _nearest(start + (end - start) * where, chord if rests_here else scale, low, high, avoid=walk)
                notes.append(LeadNote(p, at, dur))
                walk, at = p, at + dur
            bar = Bar(notes, [ChordMark(symbol, 0)])
            if _signature(bar) not in known or bi == n - 1:
                break
        prev = walk
        bars.append(bar)
    sec.bars = bars


def shape(sheet: LeadSheet, keep: set[int] = frozenset(), lo: int | None = None, hi: int | None = None
          ) -> tuple[LeadSheet, list[dict]]:
    """A copy of `sheet` with its app-written pre-choruses and bridges shaped, and what was done:
    [{"section": index, "name": ..., "what": "pre_variation" | "pre_ending" | "bridge_new" | "bridge_ending", "detail": ...}].
    `keep`: indices of sections that stay exactly as they are (the user's melody).  `lo`, `hi`: the singer's range in
    the sheet's key; a note this option writes stays inside it."""
    out = copy.deepcopy(sheet)
    for sec in out.sections:                          # a repeated bar may be one object twice: give every bar its own
        sec.bars = [copy.deepcopy(b) for b in sec.bars]
    changes: list[dict] = []
    if sheet.meter != "4/4":
        return out, changes
    key = key_of(sheet.key)
    kinds = [_kind(s) for s in out.sections]
    ceiling = _top([s for s, k in zip(out.sections, kinds) if k == "chorus"])
    others = [copy.deepcopy(s) for s, k in zip(out.sections, kinds) if k in ("verse", "pre")]
    note = lambda i, what, detail: changes.append({"section": i, "name": out.sections[i].name, "what": what, "detail": detail})
    for i, sec in enumerate(out.sections):
        if i in keep or not sec.bars:
            continue
        nxt = out.sections[i + 1] if i + 1 < len(out.sections) and kinds[i + 1] == "chorus" else None
        if kinds[i] == "pre" and nxt is not None:
            varied = _vary_second_half(sec, key, ceiling, hi)
            if varied:
                note(i, "pre_variation", varied)
            ended = _open_ending(sec, out, _first_pitch(nxt), lo, hi)
            if ended:
                note(i, "pre_ending", ended)
        elif kinds[i] == "bridge":
            shared = shared_bars(sec, others)
            if shared >= SHARED_BARS:
                _new_bridge(sec, out, others, ceiling, lo, hi)
                note(i, "bridge_new", f"벌스·프리와 같은 마디 {shared}개 → 새로 씀")
                if nxt is not None:
                    ended = _open_ending(sec, out, _first_pitch(nxt), lo, hi)
                    if ended:
                        note(i, "bridge_ending", ended)
    return out, changes
