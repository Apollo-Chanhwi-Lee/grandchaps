"""Sections of a whole imported score, found from what repeats in its melody.

    find(score) -> [{"name", "start_bar", "bars"}] | None

The score is cut into 4-bar blocks and every block is read beat by beat (the pitch sounding on each beat), so a
repeat sung with other words, and so a slightly different rhythm, still counts.  Two blocks are the same phrase when
most of their beats agree (a block may sit up to two semitones higher, as a raised last chorus does).  Among the
phrases that come back, the chorus is the one that is both high and frequent.  What leads up to a chorus is a verse
(its last block a pre-chorus when the run is long), a run heard only once before the last chorus is a bridge, blocks
without a note are intro, interlude or outro.  None when no phrase comes back: the caller keeps its own split
(`scoreimport.default_sections`).

A rule on the notes, no trained model.  "What repeats is the chorus" is wrong for some songs, so the result is
shown to the user and this is used only when asked for.
"""
from __future__ import annotations

BLOCK = 4               # bars per block
BEATS = 4               # beats read per bar
SAME = 0.7              # share of agreeing beats for two blocks to be the same phrase
MIN_BEATS = 8           # beats with a note that a comparison needs
SHIFTS = (0, 1, 2)      # semitones a repeat may sit higher
PER_REPEAT = 1.5        # semitones one more repeat is worth when the chorus is chosen
NEAR = 3.0              # a chorus grows only over phrases at most this many semitones lower than its core
PRE_FROM = 3            # blocks in a run before a chorus from which its last block is called a pre-chorus
CHORUS_MAX = 4          # blocks one chorus may span
OUTRO_MAX = 8           # bars a sung ending after the last chorus may have to be called an outro


def beat_pitches(score: dict) -> list[int | None]:
    """The pitch sounding at the start of every beat of the song (None for silence)."""
    beat = score["bar_ticks"] // BEATS
    out: list[int | None] = [None] * (int(score["bars"]) * BEATS)
    for x in score["notes"]:
        for k in range(-(-x["start"] // beat), -(-(x["start"] + x["dur"]) // beat)):
            if k < len(out) and out[k] is None:
                out[k] = x["pitch"]
    return out


def similarity(a: list, b: list) -> float:
    """Share of beats on which two blocks agree (0..1), the later one allowed to sit a little higher."""
    both = [(x, y) for x, y in zip(a, b) if x is not None or y is not None]
    if len(both) < MIN_BEATS:
        return 0.0
    return max(sum(1 for x, y in both if x is not None and y is not None and y - x == s) for s in SHIFTS) / len(both)


def _blocks(beats: list, first_bar: int) -> list[list]:
    size = BLOCK * BEATS
    return [beats[i:i + size] for i in range(first_bar * BEATS, len(beats) - size + 1, size)]


def labels(blocks: list[list]) -> list[int | None]:
    """A phrase number per block: the first earlier block it matches, else a new one; None for an empty block."""
    out: list[int | None] = []
    for i, b in enumerate(blocks):
        if all(x is None for x in b):
            out.append(None)
            continue
        hit = next((out[j] for j in range(i) if out[j] is not None and similarity(blocks[j], b) >= SAME), None)
        out.append(hit if hit is not None else i)
    return out


def _mean_pitch(blocks: list[list], lab: list, phrase: int) -> float:
    pitches = [x for b, l in zip(blocks, lab) if l == phrase for x in b if x is not None]
    return sum(pitches) / len(pitches)


def _chorus(blocks: list[list], lab: list) -> tuple[list[int], int] | None:
    """(the phrases of one chorus in order, which of them is its core), or None when nothing comes back."""
    counts: dict[int, int] = {}
    for l in lab:
        if l is not None:
            counts[l] = counts.get(l, 0) + 1
    again = [l for l, c in counts.items() if c >= 2]
    if not again:
        return None
    height = {l: _mean_pitch(blocks, lab, l) for l in again}
    core = max(again, key=lambda l: height[l] + PER_REPEAT * counts[l])
    pattern = [core]

    def neighbour(step: int, edge: int):
        """The one phrase next to every occurrence of the pattern's edge phrase, when it is always the same, comes
        back as often and is not clearly lower than the core."""
        at = [i + step for i, l in enumerate(lab) if l == edge]
        seen = {lab[i] if 0 <= i < len(lab) else None for i in at}
        if len(seen) != 1:
            return None
        got = seen.pop()
        ok = got is not None and got not in pattern and counts[got] == counts[core] and height[got] >= height[core] - NEAR
        return got if ok else None

    while len(pattern) < CHORUS_MAX:
        nxt = neighbour(1, pattern[-1])
        if nxt is None:
            break
        pattern.append(nxt)
    while len(pattern) < CHORUS_MAX:
        prev = neighbour(-1, pattern[0])
        if prev is None:
            break
        pattern.insert(0, prev)
    return pattern, pattern.index(core)


def _occurrences(beats: list, core: list, bars: int) -> list[int]:
    """First bars of every stretch that is the chorus's core block again, at any bar (a song's 4-bar grid often
    slips, and a later chorus often keeps only its core)."""
    window = lambda k: beats[k * BEATS:(k + BLOCK) * BEATS]
    out, t = [], 0
    while t + BLOCK <= bars:
        if similarity(core, window(t)) >= SAME:
            t = max(range(t, min(t + BLOCK, bars - BLOCK + 1)), key=lambda k: (similarity(core, window(k)), -k))
            out.append(t)
            t += BLOCK
        else:
            t += 1
    return out


def find(score: dict) -> list[dict] | None:
    beats, bars = beat_pitches(score), int(score["bars"])
    best = None                                            # the block grid (its first bar) on which most blocks repeat
    for first in range(BLOCK):
        blocks = _blocks(beats, first)
        lab = labels(blocks)
        repeats = sum(1 for i, l in enumerate(lab) if l is not None and l != i)
        if best is None or repeats > best[0]:
            best = (repeats, first, blocks, lab)
    _, first, blocks, lab = best
    got = _chorus(blocks, lab)
    if got is None:
        return None
    pattern, pos = got
    size = len(pattern) * BLOCK
    starts = []                                            # every chorus: the core's place, minus what leads into it
    for t in _occurrences(beats, blocks[lab.index(pattern[pos])], bars):
        t = max(t - pos * BLOCK, starts[-1] + size if starts else 0)
        if t + size <= bars:
            starts.append(t)
    if len(starts) < 2:
        return None
    sung = [any(x is not None for x in beats[k * BEATS:(k + 1) * BEATS]) for k in range(bars)]
    if not any(sung):
        return None
    first_sung, last_sung = sung.index(True), bars - 1 - sung[::-1].index(True)
    names: list[str | None] = [None] * bars
    for t in starts:
        names[t:t + size] = ["후렴"] * size
    for k in range(bars):                                  # silence: before the first note, after the last, or a long gap
        if names[k] is None and not sung[k]:
            lo = hi = k
            while lo > 0 and not sung[lo - 1] and names[lo - 1] != "후렴":
                lo -= 1
            while hi < bars - 1 and not sung[hi + 1] and names[hi + 1] != "후렴":
                hi += 1
            if k < first_sung:
                names[k] = "인트로"
            elif k > last_sung:
                names[k] = "아웃트로"
            elif hi - lo + 1 >= BLOCK:
                names[k] = "간주"
    runs, k = [], 0                                        # stretches that are neither chorus nor silence
    while k < bars:
        if names[k] is None:
            j = k
            while j < bars and names[j] is None:
                j += 1
            runs.append((k, j))
            k = j
        else:
            k += 1
    head = lambda r: beats[r[0] * BEATS:(r[0] + BLOCK) * BEATS]
    for n, (lo, hi) in enumerate(runs):
        before, after = [t for t in starts if t < lo], [t for t in starts if t >= hi]
        if not after:
            name = "아웃트로" if before and hi - lo <= OUTRO_MAX else "벌스"      # a long sung ending is more song, not an outro
        elif before and len(starts) >= 3 and not any(t >= r[1] for r in runs[n + 1:] for t in starts) \
                and all(similarity(head(r), head((lo, hi))) < SAME for r in runs[:n]):
            name = "브릿지"                                  # heard once, right before the last chorus
        else:
            name = "벌스"
        names[lo:hi] = [name] * (hi - lo)
        if name == "벌스" and after and hi == after[0] and hi - lo >= PRE_FROM * BLOCK:
            names[hi - BLOCK:hi] = ["프리"] * BLOCK
    out: list[dict] = []
    for k, name in enumerate(names):
        if out and out[-1]["name"] == name and k not in starts:
            out[-1]["bars"] += 1
        else:
            out.append({"name": name, "start_bar": k, "bars": 1})
    return out


def describe(sections: list[dict]) -> str:
    """"벌스 1~16마디, 후렴 17~24마디, ..." for the user to check."""
    return ", ".join(f"{s['name']} {s['start_bar'] + 1}~{s['start_bar'] + s['bars']}마디" for s in sections)
