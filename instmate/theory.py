"""Basic music theory: note model, number-notation input, key analysis."""
from __future__ import annotations

from dataclasses import dataclass, asdict

TPQ = 480                 # ticks per quarter note
BAR = TPQ * 4             # 4/4 only in MVP
MAX_BARS = 8

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]

# Krumhansl-Kessler key profiles
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

AMBIGUITY_MARGIN = 0.08   # score gap below which the agent asks the user


@dataclass
class Note:
    pitch: int | None     # MIDI number, None = rest
    start: int            # ticks
    dur: int              # ticks

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Key:
    tonic: int            # pitch class 0-11
    mode: str             # "major" | "minor"

    @property
    def name(self) -> str:
        return NOTE_NAMES[self.tonic] + ("m" if self.mode == "minor" else "")

    @property
    def label_ko(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {'장조' if self.mode == 'major' else '단조'}"

    def to_dict(self):
        return {"tonic": self.tonic, "mode": self.mode, "name": self.name, "label": self.label_ko}


def _token_int(tok: dict, index: int, field: str, default: int) -> int:
    try:
        return int(tok.get(field, default))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"token {index}: {field} must be numeric") from exc


def parse_number_input(tokens: list[dict], input_tonic: int = 0) -> list[Note]:
    """Convert number-notation tokens to notes.

    token: {"degree": 0-7 (0 = rest), "sharp": bool, "octave": int, "dur": ticks}
    Degrees are movable-do relative to `input_tonic` major scale (minor uses 6 as tonic).
    """
    notes: list[Note] = []
    t = 0
    for i, tok in enumerate(tokens):
        degree = _token_int(tok, i, "degree", 0)
        dur = _token_int(tok, i, "dur", TPQ)
        if not 0 <= degree <= 7:
            raise ValueError(f"token {i}: degree must be 0-7, got {degree}")
        if dur <= 0 or dur % (TPQ // 4):
            raise ValueError(f"token {i}: dur must be a positive multiple of {TPQ // 4} ticks, got {dur}")
        if degree == 0:
            notes.append(Note(None, t, dur))
        else:
            octave = _token_int(tok, i, "octave", 4)
            pitch = 12 * (octave + 1) + input_tonic + MAJOR_STEPS[degree - 1] + (1 if tok.get("sharp") else 0)
            if not 21 <= pitch <= 108:
                raise ValueError(f"token {i}: pitch {pitch} outside piano range")
            notes.append(Note(pitch, t, dur))
        t += dur
    if t == 0:
        raise ValueError("melody is empty")
    if t > BAR * MAX_BARS:
        raise ValueError(f"melody longer than {MAX_BARS} bars")
    return notes


# Meters of the lead sheet format: bar length in ticks and the pulse (beat) length used for accents and chord changes
METER_TICKS = {"4/4": BAR, "3/4": 3 * TPQ, "2/4": 2 * TPQ, "6/8": 3 * TPQ}
METER_PULSE = {"4/4": TPQ, "3/4": TPQ, "2/4": TPQ, "6/8": 3 * TPQ // 2}


def num_bars(notes: list[Note], bar: int = BAR) -> int:
    end = max(n.start + n.dur for n in notes)
    return max(1, -(-end // bar))


def split_by_bar(notes: list[Note], bar: int = BAR) -> list[list[Note]]:
    """Notes per bar; a note crossing a bar line is cut into pieces (piece keeps absolute start)."""
    bars: list[list[Note]] = [[] for _ in range(num_bars(notes, bar))]
    for n in notes:
        s, e = n.start, n.start + n.dur
        while s < e:
            b = s // bar
            piece_end = min(e, (b + 1) * bar)
            bars[b].append(Note(n.pitch, s, piece_end - s))
            s = piece_end
    return bars


def beat_weight(tick_in_bar: int) -> float:
    """Metric weight in 4/4: beat 1 > beat 3 > beats 2,4 > offbeats."""
    if tick_in_bar == 0:
        return 1.0
    if tick_in_bar == 2 * TPQ:
        return 0.8
    if tick_in_bar % TPQ == 0:
        return 0.6
    return 0.35


def meter_beat_weight(tick_in_bar: int, meter: str = "4/4") -> float:
    """Metric weight: 4/4 as `beat_weight`; 3/4 and 2/4 accent beat 1 only; 6/8 accents pulses 1 (1.0) and 2 (0.8)."""
    if meter == "4/4":
        return beat_weight(tick_in_bar)
    if tick_in_bar == 0:
        return 1.0
    if meter == "6/8" and tick_in_bar == METER_PULSE["6/8"]:
        return 0.8
    return 0.6 if tick_in_bar % TPQ == 0 else 0.35


def pitch_class_histogram(notes: list[Note]) -> list[float]:
    h = [0.0] * 12
    for n in notes:
        if n.pitch is not None:
            h[n.pitch % 12] += n.dur * (1.0 + 0.5 * beat_weight(n.start % BAR))
    return h


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / 12, sum(b) / 12
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def key_score(notes: list[Note], key: Key) -> float:
    hist = pitch_class_histogram(notes)
    profile = KK_MAJOR if key.mode == "major" else KK_MINOR
    rotated = [profile[(pc - key.tonic) % 12] for pc in range(12)]
    score = _corr(hist, rotated)
    pitched = [n for n in notes if n.pitch is not None]
    if pitched:
        if pitched[-1].pitch % 12 == key.tonic:   # ending on the tonic is strong evidence
            score += 0.12
        if pitched[0].pitch % 12 == key.tonic:
            score += 0.04
    return round(score, 4)


def analyze_key(notes: list[Note], input_tonic: int = 0) -> dict:
    """Compare the input major key with its relative minor; flag ambiguity.

    Also reports the best of all 24 keys so a mistyped input key can be noticed.
    """
    if not any(n.pitch is not None for n in notes):
        raise ValueError("melody has no pitched notes")
    major = Key(input_tonic, "major")
    minor = Key((input_tonic + 9) % 12, "minor")
    cands = sorted(
        [{"key": k.to_dict(), "score": key_score(notes, k)} for k in (major, minor)],
        key=lambda c: -c["score"],
    )
    best_all = max(
        (Key(pc, m) for pc in range(12) for m in ("major", "minor")),
        key=lambda k: key_score(notes, k),
    )
    gap = cands[0]["score"] - cands[1]["score"]
    return {
        "candidates": cands,
        "ambiguous": gap < AMBIGUITY_MARGIN,
        "gap": round(gap, 4),
        "best_overall": best_all.to_dict(),
        "question": (
            f"{cands[0]['key']['label']}와 {cands[1]['key']['label']} 중 어느 느낌인가요?"
            if gap < AMBIGUITY_MARGIN else None
        ),
    }
