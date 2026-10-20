"""Self-check: melody/chord clash rate, instrument ranges, bar lengths."""
from __future__ import annotations

from .arrange import RANGES, Event
from .chords import ChordDef, note_role
from .theory import BAR, METER_TICKS, TPQ, Key, Note, meter_beat_weight, split_by_bar

CLASH_MAX = 0.34   # a bar above this share of clashing strong-beat melody is regenerated
BEAT3_WEIGHT = 0.5  # beat 1 counts fully; beat 3 half, so a lone passing tone on beat 3 (1/3) passes


def clash_rates(key: Key, melody: list[Note], chords: list, meter: str = "4/4") -> list[float]:
    """Per bar: weighted share of strong-beat (1 and 3) melody that is neither chord tone nor allowed tension.

    `chords[i]` is one ChordDef for the whole bar, or a list of (tick in bar, ChordDef) for a bar with chord
    changes; a note is judged against the chord in effect at its start. `meter` sets bar length and accents.
    """
    bl = METER_TICKS[meter]
    bars = split_by_bar(melody, bl)
    if len(bars) != len(chords):
        raise ValueError(f"{len(bars)} bars but {len(chords)} chords")
    rates = []
    for notes, cs in zip(bars, chords):
        segs = cs if isinstance(cs, list) else [(0, cs)]
        total = bad = 0.0
        for n in notes:
            c = [x for t, x in segs if t <= n.start % bl][-1]
            bw = meter_beat_weight(n.start % bl, meter)
            if n.pitch is None or bw < 0.8:
                continue
            w = n.dur / TPQ * (1.0 if bw == 1.0 else BEAT3_WEIGHT)
            total += w
            if note_role(key, c, n.pitch % 12) in ("scale", "chromatic"):
                bad += w
        rates.append(round(bad / total, 3) if total else 0.0)
    return rates


def range_violations(events: list[Event]) -> list[dict]:
    out = []
    for e in events:
        if e.track in RANGES:
            lo, hi = RANGES[e.track]
            if not lo <= e.pitch <= hi:
                out.append({"track": e.track, "pitch": e.pitch, "start": e.start, "range": [lo, hi]})
    return out


def length_violations(events: list[Event], n_bars: int, bar_len: int = BAR) -> list[dict]:
    end = n_bars * bar_len
    return [{"track": e.track, "start": e.start, "end": e.start + e.dur}
            for e in events if e.start < 0 or e.dur <= 0 or e.start + e.dur > end]


def report(key: Key, melody: list[Note], chords: list[ChordDef], events: list[Event]) -> dict:
    rates = clash_rates(key, melody, chords)
    return {
        "clash_rates": rates,
        "problem_bars": [i for i, r in enumerate(rates) if r > CLASH_MAX],
        "mean_clash": round(sum(rates) / len(rates), 3),
        "range_violations": range_violations(events),
        "length_violations": length_violations(events, len(chords)),
        "threshold": CLASH_MAX,
    }
