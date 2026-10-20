"""Settle the melodies the app wrote onto their chords (no model, no randomness).

What it changes, and only in sections the caller does not protect (the user's own melody is never touched):
- a note the ear rests on (one beat or more on a strong beat: beats 1 and 3 in 4/4, the bar's first beat otherwise;
  or two beats or more on any beat) that is not a note of the chord sounding there moves to the nearest chord note,
  at most two semitones away;
- the last note of the section does the same, so a section does not end on a note outside its chord.
Short notes, off-beat notes and one-beat notes on weak beats stay: passing and neighbour notes are what makes a line move.  Tied notes stay too
(moving one half of a tie would split one sound into two pitches).  A move never leaves [lo, hi] when a range is given.
"""
from __future__ import annotations

import copy

from .leadsheet import LeadSheet
from .render import key_of, parse_chord
from .theory import TPQ

MAX_MOVE = 2            # semitones: a step, not a leap; a note further from every chord note is left as written


def _nearest_chord_note(pitch: int, pcs: set[int], prev: int | None, lo: int | None, hi: int | None) -> int | None:
    """The chord note closest to `pitch` within MAX_MOVE; ties go to the one closer to the previous note."""
    cands = [pitch + d for d in range(-MAX_MOVE, MAX_MOVE + 1)
             if d and (pitch + d) % 12 in pcs and (lo is None or pitch + d >= lo) and (hi is None or pitch + d <= hi)]
    if not cands:
        return None
    return min(cands, key=lambda p: (abs(p - pitch), abs(p - prev) if prev is not None else 0, p))


def settle(sheet: LeadSheet, keep: set[int] = frozenset(), lo: int | None = None, hi: int | None = None
           ) -> tuple[LeadSheet, list[dict]]:
    """A copy of `sheet` with the app-written sections settled, and the list of moves
    [{"section": index, "name": ..., "bar": bar in section (0-based), "from": pitch, "to": pitch, "why": "long"|"ending"}].
    `keep`: indices of sections that must stay exactly as they are (the user's melody)."""
    out = copy.deepcopy(sheet)
    key = key_of(sheet.key)
    strong = 2 * TPQ if sheet.meter == "4/4" else None          # None: only the first beat of the bar is strong
    moves: list[dict] = []
    for si, sec in enumerate(out.sections):
        if si in keep:
            continue
        pcs: set[int] | None = None                 # the chord carries over bar lines until the next symbol
        prev: int | None = None
        last = None                                 # (bar index, note, chord notes) of the section's final pitched note
        tied_into = False
        for bi, bar in enumerate(sec.bars):
            marks = sorted(bar.chords, key=lambda c: c.start)
            for n in sorted(bar.notes, key=lambda n: n.start):
                for c in marks:
                    if c.start <= n.start:
                        try:
                            pcs = set(parse_chord(c.symbol, key).pcs)
                        except ValueError:
                            pcs = None              # "N" or an unknown symbol: nothing to settle against
                if n.pitch is None:
                    tied_into = False
                    continue
                free = not n.tie and not tied_into
                on_strong = n.start % strong == 0 if strong else n.start == 0
                rests = n.start % TPQ == 0 and (n.dur >= 2 * TPQ or (n.dur >= TPQ and on_strong))
                if free and pcs and rests and n.pitch % 12 not in pcs:
                    to = _nearest_chord_note(n.pitch, pcs, prev, lo, hi)
                    if to is not None:
                        moves.append({"section": si, "name": sec.name, "bar": bi, "from": n.pitch, "to": to, "why": "long"})
                        n.pitch = to
                last = (bi, n, set(pcs) if pcs else None, free)
                prev, tied_into = n.pitch, n.tie
        if last and last[2] and last[3] and last[1].pitch % 12 not in last[2]:
            bi, n, chord, _ = last
            to = _nearest_chord_note(n.pitch, chord, None, lo, hi)
            if to is not None:
                moves.append({"section": si, "name": sec.name, "bar": bi, "from": n.pitch, "to": to, "why": "ending"})
                n.pitch = to
    return out, moves
