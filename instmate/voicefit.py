"""Fit a melody's key to the singer: the tune keeps its shape (every interval, every rhythm) and moves as a whole.

    suggest(low, high, voice) -> semitones     how far to move a melody whose lowest / highest notes are given
    describe(...)                              the Korean line shown to the user and written into the song's trace

A melody inside the voice's comfortable range is not moved.  Otherwise it moves by the smallest number of semitones
that brings it inside; a melody wider than the range is centred in it.  Moving by semitones changes the key (the
chords move with it), so this is a transposition the song records and shows, never a silent change.
The ranges are everyday pop ranges for an untrained singer, not limits: the user can switch the fit off.
"""
from __future__ import annotations

import math

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# comfortable melody range (MIDI): low note that still carries, high note that is not strained
# The range asked of a singer, set by the user on 2026-10-07: a woman G3-G5, a man G2-G4 (two octaves each, the man an
# octave below the woman).  Classical amateur voice types for comparison:
#   bass F2-D4 · baritone G2-E4 · tenor C3-G4 · contralto F3-D5 · mezzo-soprano G3-F5 · soprano C4-G5
# A tune moves only when it leaves this range, and only as far as its top (or bottom) note needs.  A man's part is
# often WRITTEN an octave above where it sounds; such a tune comes down an octave here, and the description says
# "an octave lower" rather than a large number of semitones.
RANGES = {"female": (55, 79),      # G3 .. G5
          "male": (43, 67)}        # G2 .. G4
VOICE_KO = {"female": "여성", "male": "남성"}
MAX_SHIFT = 24                     # beyond two octaves something is wrong with the input, not the key


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def suggest(low: int, high: int, voice: str) -> int:
    """Semitones to move a melody spanning low..high (MIDI) for `voice`; 0 when it already fits."""
    if voice not in RANGES:
        raise ValueError(f"voice must be one of {', '.join(RANGES)}, got {voice!r}")
    if low > high:
        raise ValueError("low must not be above high")
    lo, hi = RANGES[voice]
    if high - low > hi - lo:                       # wider than the range: put its middle on the range's middle
        # half a semitone always goes the same way (Python's round() sends -5.5 and -10.5 to different sides, which
        # made the male fit four, not five, semitones below the female one)
        shift = math.floor((lo + hi) / 2 - (low + high) / 2 + 0.5)
    elif low < lo:
        shift = lo - low
    elif high > hi:
        shift = hi - high
    else:
        shift = 0
    return max(-MAX_SHIFT, min(MAX_SHIFT, shift))


def describe(low: int, high: int, voice: str, shift: int, key_from: str, key_to: str) -> dict:
    """What was done, for the song's record and the screen."""
    lo, hi = RANGES[voice]
    if shift == 0:
        text = (f"음역 확인: 멜로디({note_name(low)}~{note_name(high)})가 {VOICE_KO[voice]} 보컬 음역"
                f"({note_name(lo)}~{note_name(hi)}) 안에 있어 조를 옮기지 않았어요.")
    else:
        fits = lo <= low + shift and high + shift <= hi
        octaves, rest = divmod(abs(shift), 12)
        way, side = ("올렸어요", "위로") if shift > 0 else ("내렸어요", "아래로")
        if not octaves:
            moved = f"{abs(shift)}반음 {way}"
        elif not rest:                             # the same key, another octave
            moved = f"{'한' if octaves == 1 else '두'} 옥타브 {side} 옮겼어요(조는 그대로)"
        else:                                      # e.g. a man's part written an octave high: the octave, then the key
            moved = f"{'한' if octaves == 1 else '두'} 옥타브 {side} 옮기고 조를 {rest}반음 더 {way}"
        text = (f"음역 맞춤: {VOICE_KO[voice]} 보컬 음역({note_name(lo)}~{note_name(hi)})에 맞게 곡 전체를 "
                f"{moved} ({key_from} → {key_to}). "
                f"멜로디 {note_name(low)}~{note_name(high)} → {note_name(low + shift)}~{note_name(high + shift)}. "
                "멜로디의 흐름(음 사이의 간격과 리듬)은 그대로예요."
                + ("" if fits else " 멜로디가 음역에 다 들어가지는 않아 가장 가깝게 맞췄어요."))
    return {"voice": voice, "shift": shift, "key_from": key_from, "key_to": key_to, "low": low + shift, "high": high + shift,
            "range": [lo, hi], "text": text}


def shift_key_name(name: str, shift: int) -> str:
    """'C' moved 7 semitones -> 'G'; 'Am' moved 3 -> 'Cm'.  Only names of `leadsheet.KEYS` come back."""
    from .leadsheet import KEYS
    from .render import key_of
    from .scoreimport import _key_name
    if name not in KEYS:
        raise ValueError(f"알 수 없는 조: {name!r}")
    key = key_of(name)
    return _key_name((key.tonic + shift) % 12, "min" if key.mode == "minor" else "maj")


def _move_chords(chords, shift: int, key_name: str):
    """Chord symbols moved with the melody, spelled for the new key.  None entries (same chord as before) stay."""
    from .leadsheet import KEYS
    from .render import transpose_symbol
    if chords is None:
        return None
    signature = KEYS[key_name]
    flats = None if signature == 0 else signature < 0
    return [c if c is None else transpose_symbol(c, shift, flats) for c in chords]


def fit_score(score: dict, voice: str, key: str | None = None, chords=None) -> tuple[dict, str, list | None, dict]:
    """An imported score (scoreimport.read) moved as a whole into `voice`'s range.

    Returns (score, key name, chords, info): a copy of the score with every note moved by the same number of
    semitones, its own chord symbols and the given `chords` moved with it, and `describe`'s record.  `key`: the key
    the user chose instead of the file's.  Nothing but pitch changes: starts and lengths are the input's."""
    low, high = min(n["pitch"] for n in score["notes"]), max(n["pitch"] for n in score["notes"])
    shift = suggest(low, high, voice)
    key_from = key or score.get("key") or "C"
    key_to = shift_key_name(key_from, shift)
    out = dict(score)
    if shift:
        out["notes"] = [{**n, "pitch": n["pitch"] + shift} for n in score["notes"]]
        out["key"] = key_to
        if isinstance(score.get("tonic"), int):
            out["tonic"] = (score["tonic"] + shift) % 12
        out["range"] = [low + shift, high + shift]
        if isinstance(score.get("chords"), list):
            out["chords"] = _move_chords(score["chords"], shift, key_to)
        chords = _move_chords(chords, shift, key_to)
    return out, key_to, chords, describe(low, high, voice, shift, key_from, key_to)


def fit_sections(sheet, lo: int, hi: int, offset: int, keep: set[int]):
    """A copy of a lead sheet whose sections sit in the singer's range: each section not in `keep` (the user's own
    melody) is moved by whole octaves, as a block, to where the fewest of its notes fall outside lo..hi; among equal
    choices the one nearest the user's melody wins, so the song does not jump registers between sections.
    `offset`: semitones between the sheet's pitches and the sounding ones (sounding = written - offset).
    Chords are untouched (an octave does not change a chord).  Returns (sheet, [{"section", "shift"}] for moved ones)."""
    import copy
    out, log = copy.deepcopy(sheet), []
    own = sorted(n.pitch - offset for i, sec in enumerate(out.sections) if i in keep for b in sec.bars for n in b.notes
                 if n.pitch is not None)
    centre = own[len(own) // 2] if own else (lo + hi) // 2
    for i, sec in enumerate(out.sections):
        ps = sorted(n.pitch - offset for b in sec.bars for n in b.notes if n.pitch is not None)
        if i in keep or not ps:
            continue
        mid = ps[len(ps) // 2]
        best = min((-24, -12, 0, 12, 24),
                   key=lambda k: (sum(not lo <= p + k <= hi for p in ps), abs(mid + k - centre), abs(k)))
        if best:
            for b in sec.bars:
                for n in b.notes:
                    if n.pitch is not None:
                        n.pitch += best
            log.append({"section": sec.name, "index": i, "shift": best})
    return out, log


def fit_tokens(tokens: list[dict], input_tonic: int, voice: str, key_choice: dict | None = None,
               seed_chords=None) -> tuple[list[dict], int, dict | None, list | None, dict]:
    """A typed melody (number notation relative to `input_tonic`) moved as a whole into `voice`'s range.

    Returns (tokens, input_tonic, key_choice, seed_chords, info).  The degrees and lengths are untouched: the tonic
    moves, and every note's octave follows when the move crosses a C.  `key_choice` ({"tonic", "mode"}) and the
    chords chosen for the seed's bars move with the melody."""
    from .scoreimport import _key_name
    from .theory import analyze_key, parse_number_input
    pitches = [n.pitch for n in parse_number_input(tokens, input_tonic) if n.pitch is not None]
    if not pitches:
        raise ValueError("melody has no notes")
    low, high = min(pitches), max(pitches)
    # (a last chorus raised a semitone may pass the top note by that semitone: that lift is what the key change is
    # for, and keeping it out of range would cost the whole song a plainer key, e.g. B instead of C)
    shift = suggest(low, high, voice)
    if key_choice:
        tonic, mode = int(key_choice["tonic"]), key_choice["mode"]
    else:
        top = analyze_key(parse_number_input(tokens, input_tonic), input_tonic)["candidates"][0]["key"]
        tonic, mode = int(top["tonic"]), top["mode"]
    short = "min" if mode == "minor" else "maj"
    key_from, key_to = _key_name(tonic, short), _key_name((tonic + shift) % 12, short)
    info = describe(low, high, voice, shift, key_from, key_to)
    if not shift:
        return tokens, input_tonic, key_choice, seed_chords, info
    carry, new_tonic = divmod(input_tonic + shift, 12)
    moved = [tok if not tok.get("degree") else {**tok, "octave": int(tok.get("octave", 4)) + carry} for tok in tokens]
    choice = {**key_choice, "tonic": (tonic + shift) % 12} if key_choice else None
    return moved, new_tonic, choice, _move_chords(seed_chords, shift, key_to), info
