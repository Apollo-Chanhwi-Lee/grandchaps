"""Rule-based chord engine: per-bar melody fit x genre transition preferences -> Viterbi."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .theory import BAR, NOTE_NAMES, TPQ, Key, Note, beat_weight, split_by_bar

from .genres import GENRES as _GENRES, PROFILES as _PROFILES

GENRES = list(_GENRES)
GENRE_KO = {g: p["label"] for g, p in _PROFILES.items()}

QUALITY_TONES = {"maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "dom7": [0, 4, 7, 10],
                 "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10]}
QUALITY_TENSIONS = {"maj": {2, 9, 11}, "min": {2, 5, 10}, "dim": set(), "dom7": {2, 9}, "maj7": {2, 9}, "min7": {2, 5}}
QUALITY_SUFFIX = {"maj": "", "min": "m", "dim": "dim", "dom7": "7", "maj7": "maj7", "min7": "m7"}
SEVENTHS = ("dom7", "maj7", "min7")          # voiced as root, third, seventh


@dataclass(frozen=True)
class ChordDef:
    roman: str
    offset: int           # semitones above key tonic
    quality: str
    dark: bool = False    # counts as a "darker" choice
    bright: bool = False  # counts as a "brighter" choice


@dataclass(frozen=True)
class SetChord(ChordDef):
    """Chord given as explicit pitch classes (from a lead-sheet symbol such as 'Am7b5' or 'C/E').

    `pcs` are absolute pitch classes, root first (a slash bass outside the chord is appended last);
    `voice` is the 3-note subset the arrangement voices; `bass` is the bass pitch class;
    `tension_ivs` are intervals above the root that do not count as clashes.
    """
    pcs: tuple = ()
    voice: tuple = ()
    bass: int = 0
    tension_ivs: frozenset = frozenset()
    symbol: str = ""


MAJOR_VOCAB = [
    ChordDef("I", 0, "maj", bright=True), ChordDef("ii", 2, "min", dark=True),
    ChordDef("iii", 4, "min", dark=True), ChordDef("IV", 5, "maj", bright=True),
    ChordDef("V", 7, "maj", bright=True), ChordDef("V7", 7, "dom7"),
    ChordDef("vi", 9, "min", dark=True), ChordDef("vii°", 11, "dim", dark=True),
    ChordDef("iv", 5, "min", dark=True), ChordDef("bVI", 8, "maj", dark=True),
    ChordDef("bVII", 10, "maj"),
]
MINOR_VOCAB = [
    ChordDef("i", 0, "min", dark=True), ChordDef("ii°", 2, "dim", dark=True),
    ChordDef("III", 3, "maj", bright=True), ChordDef("iv", 5, "min", dark=True),
    ChordDef("v", 7, "min", dark=True), ChordDef("V", 7, "maj"),
    ChordDef("V7", 7, "dom7"), ChordDef("VI", 8, "maj", bright=True),
    ChordDef("VII", 10, "maj", bright=True), ChordDef("IV", 5, "maj", bright=True),
    ChordDef("I", 0, "maj", bright=True),
]

# Extended chords: secondary dominants (V7/x resolves to x) and diatonic sevenths. They are in a genre's search
# space only when that genre's STYLE prior names them (`vocab_for`), so the original four genres are unchanged.
# The royal road IVmaj7-V7-iii7-vi and its III7 (= V7/vi) variant: en.wikipedia.org/wiki/Royal_road_progression.
MAJOR_EXT = [
    ChordDef("V7/V", 2, "dom7"), ChordDef("V7/vi", 4, "dom7"), ChordDef("V7/ii", 9, "dom7"), ChordDef("V7/IV", 0, "dom7"),
    ChordDef("Imaj7", 0, "maj7", bright=True), ChordDef("IVmaj7", 5, "maj7", bright=True),
    ChordDef("ii7", 2, "min7", dark=True), ChordDef("iii7", 4, "min7", dark=True), ChordDef("vi7", 9, "min7", dark=True),
]
MINOR_EXT = [
    ChordDef("V7/iv", 0, "dom7"), ChordDef("V7/III", 10, "dom7"), ChordDef("i7", 0, "min7", dark=True),
    ChordDef("iv7", 5, "min7", dark=True), ChordDef("v7", 7, "min7", dark=True),
    ChordDef("VImaj7", 8, "maj7", bright=True), ChordDef("IIImaj7", 3, "maj7", bright=True),
]
EXT_ROMANS = {"major": {c.roman for c in MAJOR_EXT}, "minor": {c.roman for c in MINOR_EXT}}
EXT_LAST_BAR_PENALTY = 0.8
SECONDARY_TARGET = {"V7/V": "V", "V7/vi": "vi", "V7/ii": "ii", "V7/IV": "IV", "V7/iv": "iv", "V7/III": "III"}

# Genre style: chord prior (how idiomatic a chord is) and favored transitions, from the genre profiles
# (instmate/genres.py), where shared habits are inherited between genres.
STYLE = {mode: {g: {"prior": dict(p["chords"][mode]["prior"]), "trans": dict(p["chords"][mode]["trans"])}
                for g, p in _PROFILES.items()} for mode in ("major", "minor")}

class NoCandidate(ValueError):
    """Constraints (banned/forced) leave some bar with no chord a tool can return."""


FIT_WEIGHT = 2.0
REPEAT_PENALTY = -0.15


def vocab(key: Key) -> list[ChordDef]:
    return MAJOR_VOCAB if key.mode == "major" else MINOR_VOCAB


def ext_vocab(key: Key) -> list[ChordDef]:
    return MAJOR_EXT if key.mode == "major" else MINOR_EXT


def vocab_for(key: Key, genre: str) -> list[ChordDef]:
    """Search space of a genre: the basic vocabulary plus the extended chords its STYLE prior names."""
    prior = STYLE[key.mode][genre]["prior"]
    return vocab(key) + [c for c in ext_vocab(key) if c.roman in prior]


ROMAN_BY_OFFSET = {
    "major": {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI",
              10: "bVII", 11: "VII"},
    "minor": {0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV", 6: "#IV", 7: "V", 8: "VI", 9: "#VI",
              10: "VII", 11: "#VII"},
}


def chord_from_offset(key: Key, offset: int, quality: str) -> ChordDef:
    """Vocabulary chord with this offset/quality, or a generic one (used by learned models)."""
    offset %= 12
    for c in vocab(key):
        if c.offset == offset and c.quality == quality:
            return c
    base = ROMAN_BY_OFFSET[key.mode][offset]
    numeral = base.lstrip("b#")
    acc = base[: len(base) - len(numeral)]
    roman = acc + (numeral.lower() if quality in ("min", "dim") else numeral) + {"dim": "°", "dom7": "7"}.get(quality, "")
    return ChordDef(roman, offset, quality, dark=quality in ("min", "dim"), bright=quality == "maj")


def chord_by_roman(key: Key, roman: str) -> ChordDef:
    for c in vocab(key) + ext_vocab(key):        # before the generic pattern, which would read "ii7" as a dominant
        if c.roman == roman:
            return c
    m = re.fullmatch(r"([b#]?)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)(°|7)?", roman)
    if m:
        acc, num, suf = m.groups()
        base = acc + num.upper()
        offsets = [o for o, r in ROMAN_BY_OFFSET[key.mode].items() if r == base]
        if offsets:
            quality = "dim" if suf == "°" else "dom7" if suf == "7" else "min" if num.islower() else "maj"
            c = chord_from_offset(key, offsets[0], quality)
            if c.roman == roman:
                return c
    raise ValueError(f"unknown chord {roman} for {key.mode} key")


def chord_pcs(key: Key, c: ChordDef) -> list[int]:
    if isinstance(c, SetChord):
        return list(c.pcs)
    root = (key.tonic + c.offset) % 12
    return [(root + i) % 12 for i in QUALITY_TONES[c.quality]]


def chord_symbol(key: Key, c: ChordDef) -> str:
    if isinstance(c, SetChord):
        return c.symbol
    return NOTE_NAMES[(key.tonic + c.offset) % 12] + QUALITY_SUFFIX[c.quality]


def key_scale(key: Key) -> set[int]:
    steps = [0, 2, 4, 5, 7, 9, 11] if key.mode == "major" else [0, 2, 3, 5, 7, 8, 10, 11]  # minor incl. leading tone
    return {(key.tonic + s) % 12 for s in steps}


def note_role(key: Key, c: ChordDef, pc: int) -> str:
    """'chord' | 'tension' | 'scale' | 'chromatic' for a melody pitch class against a chord."""
    root = (key.tonic + c.offset) % 12
    interval = (pc - root) % 12
    if isinstance(c, SetChord):
        if pc in c.pcs:
            return "chord"
        if interval in c.tension_ivs:
            return "tension"
        return "scale" if pc in key_scale(key) else "chromatic"
    if interval in QUALITY_TONES[c.quality]:
        return "chord"
    if interval in QUALITY_TENSIONS[c.quality]:
        return "tension"
    return "scale" if pc in key_scale(key) else "chromatic"


def bar_fit(key: Key, c: ChordDef, bar_notes: list[Note]) -> float:
    """Duration- and metre-weighted agreement between a bar's melody and a chord, in [-1, 1]."""
    total = score = 0.0
    for n in bar_notes:
        if n.pitch is None:
            continue
        bw = beat_weight(n.start % BAR)
        w = (n.dur / TPQ) * bw
        role = note_role(key, c, n.pitch % 12)
        # downbeat carries the harmony; a non-chord tone on beat 3 is often a passing tone
        scale_pen = -0.6 if bw == 1.0 else -0.35 if bw >= 0.8 else -0.2
        score += w * {"chord": 1.0, "tension": 0.1 if bw >= 0.8 else 0.3,
                      "scale": scale_pen, "chromatic": -0.8}[role]
        total += w
    return score / total if total else 0.0


def _local(key, genre, bars, i, c, n_bars):
    style = STYLE[key.mode][genre]
    s = FIT_WEIGHT * bar_fit(key, c, bars[i]) + style["prior"].get(c.roman, 0)
    tonic = "I" if key.mode == "major" else "i"
    if i == 0 and c.roman == tonic:
        s += 0.2
    if i == n_bars - 1:                          # cadence: end on the tonic (its seventh counts); no open chord
        if c.roman in (tonic, tonic + "maj7", tonic + "7"):
            s += 0.5
        elif c.quality == "dim":
            s -= 0.3
        elif c.roman in EXT_ROMANS[key.mode]:
            s -= EXT_LAST_BAR_PENALTY
    if i == n_bars - 2 and n_bars >= 3 and c.roman in ("V", "V7") and genre in ("classical", "ballad"):
        s += 0.2
    return s


SECONDARY_RESOLVE, SECONDARY_ASTRAY = 0.6, -0.5


def _trans(key, genre, a: ChordDef, b: ChordDef) -> float:
    t = STYLE[key.mode][genre]["trans"].get((a.roman, b.roman), 0.0)
    target = SECONDARY_TARGET.get(a.roman)
    if target is not None:                       # a secondary dominant should resolve to its target (or its 7th)
        t += SECONDARY_RESOLVE if b.roman.rstrip("7").replace("maj", "") == target else SECONDARY_ASTRAY
    return t + (REPEAT_PENALTY if a.roman == b.roman else 0.0)


def progression(key: Key, genre: str, melody: list[Note],
                banned: dict[int, set[str]] | None = None,
                forced: dict[int, str] | None = None,
                bias: dict[str, float] | None = None) -> list[ChordDef]:
    """Best chord per bar via Viterbi. `banned`/`forced` map bar index -> roman(s); `bias` adds per-roman prefs."""
    if genre not in GENRES:
        raise ValueError(f"unknown genre {genre}")
    banned, forced, bias = banned or {}, forced or {}, bias or {}
    bars = split_by_bar(melody)
    n = len(bars)
    V = vocab_for(key, genre)

    def allowed(i):
        if i in forced:
            return [chord_by_roman(key, forced[i])]
        cands = [c for c in V if c.roman not in banned.get(i, set())]
        if not cands:
            raise NoCandidate(f"bar {i + 1}: every chord was banned")
        return cands

    layers = [allowed(i) for i in range(n)]
    score = [{c.roman: _local(key, genre, bars, 0, c, n) + bias.get(c.roman, 0) for c in layers[0]}]
    back: list[dict[str, str]] = [{}]
    for i in range(1, n):
        score.append({})
        back.append({})
        for c in layers[i]:
            loc = _local(key, genre, bars, i, c, n) + bias.get(c.roman, 0)
            prev_best = max(layers[i - 1],
                            key=lambda p: score[i - 1][p.roman] + _trans(key, genre, p, c))
            score[i][c.roman] = score[i - 1][prev_best.roman] + _trans(key, genre, prev_best, c) + loc
            back[i][c.roman] = prev_best.roman
    last = max(layers[-1], key=lambda c: score[-1][c.roman])
    path = [last.roman]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return [chord_by_roman(key, r) for r in path]


def explain_bar(key: Key, c: ChordDef, bar_notes: list[Note]) -> str:
    """Plain-language reason for a chord choice (no LLM needed)."""
    strong = [n for n in bar_notes if n.pitch is not None and beat_weight(n.start % BAR) >= 0.8]
    sym = chord_symbol(key, c)
    if not any(n.pitch is not None for n in bar_notes):
        return f"{c.roman}({sym}): 쉬는 마디라 앞뒤 코드 흐름만 보고 골랐어요."
    if not strong:
        return f"{c.roman}({sym}): 1박과 3박에 멜로디 음이 없어, 약박 음과 앞뒤 코드 흐름을 보고 골랐어요."
    in_chord = sorted({NOTE_NAMES[n.pitch % 12] for n in strong if note_role(key, c, n.pitch % 12) == "chord"})
    # strong-beat notes outside the chord (and its allowed tensions) are disclosed, never hidden
    clash = []
    for n in strong:
        if note_role(key, c, n.pitch % 12) in ("scale", "chromatic"):
            label = f"{'1박' if n.start % BAR == 0 else '3박'}의 {NOTE_NAMES[n.pitch % 12]}"
            if label not in clash:
                clash.append(label)
    tail = f" 다만 {', '.join(clash)}은(는) 코드 밖 음이라 부딪혀요." if clash else ""
    if in_chord:
        return f"{c.roman}({sym}): 강박의 멜로디 음 {', '.join(in_chord)}이(가) 코드 구성음이에요.{tail}"
    if clash:
        return f"{c.roman}({sym}): 강박 음이 코드 구성음이 아니에요.{tail}"
    return f"{c.roman}({sym}): 강박 음과 완전히 겹치진 않지만(허용 텐션), 장르의 코드 흐름을 우선해 골랐어요."


def alternatives(key: Key, genre: str, melody: list[Note], bar: int, top: int = 3) -> list[dict]:
    bars = split_by_bar(melody)
    ranked = sorted(vocab_for(key, genre), key=lambda c: -_local(key, genre, bars, bar, c, len(bars)))
    return [{"roman": c.roman, "symbol": chord_symbol(key, c)} for c in ranked[:top]]
