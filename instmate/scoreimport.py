"""Bring the user's own song in from a score file (MIDI or MusicXML exported from a notation program).

    read(path)                      -> the melody on the app's tick grid (480 per quarter note), tempo, meter, key guess
    to_tokens(score, start_bar, n)  -> number-notation tokens for the melody input (a seed of up to 8 bars)
    to_leadsheet(score, sections)   -> the whole song as the app's lead sheet text: the user's melody in the sections
                                       they name; chords are left empty so the chord tool harmonizes them

A notation program writes notes a little short (a quarter note lasts 0.95 beat): starts and ends are snapped to the
16th-note grid, and what that changed is reported.  A played (unquantized) file is snapped the same way, so the report
matters: the user sees how many notes moved.  The file itself is deleted by the caller; nothing is stored.
"""
from __future__ import annotations

from pathlib import Path

from . import humtrans, lakh, scorecheck
from .leadsheet import KEYS, SECTION_NAMES, _valid_chord, from_notes, serialize_leadsheet
from .theory import TPQ, Note

GRID = TPQ // 4                 # 16th note
SEED_MAX_BARS = 8
# the words people use for the lead sheet's section names; 후크 (hook) is the part that lifts into the chorus: 프리
SECTION_ALIAS = {"도입부": "인트로", "도입": "인트로", "전주": "인트로", "후크": "프리", "훅": "프리", "프리코러스": "프리",
                 "코러스": "후렴", "사비": "후렴", "하이라이트": "후렴", "브리지": "브릿지", "엔딩": "아웃트로", "후주": "아웃트로"}
MAX_SONG_BARS = 400             # a whole song with repeats written out (raised from 160 on 2026-10-10: a 227-bar file was refused)
MOVED_TICKS = 30                # a note start that moved more than this by snapping is counted as "moved"
METERS = {(4, 4): "4/4", (3, 4): "3/4", (2, 4): "2/4", (6, 8): "6/8"}
FLAT_KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
SHARP_KEYS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class ScoreImportError(ValueError):
    """The score cannot be imported (Korean message)."""


def _key_name(tonic: int, mode: str) -> str:
    for names in (FLAT_KEYS, SHARP_KEYS):
        name = names[tonic] + ("m" if mode == "min" else "")
        if name in KEYS:
            return name
    raise ScoreImportError(f"조를 정하지 못했어요 (으뜸음 {tonic}, {mode})")


def guess_key(notes: list[tuple[int, int]]) -> tuple[int, str]:
    """(tonic pitch class, "maj" | "min") for a written melody [(duration, pitch)]: first the scales that need the
    fewest accidentals (a score's notes are deliberate, unlike a hummed take), then, among those, the best
    profile fit with a bonus for ending or starting on the tonic."""
    import numpy as np
    hist = np.zeros(12)
    for d, p in notes:
        hist[p % 12] += d
    total = hist.sum() or 1.0
    first, last = notes[0][1] % 12, notes[-1][1] % 12
    rows = []
    for tonic in range(12):
        for mode, prof, scale in (("maj", lakh.KK_MAJOR, humtrans.MAJOR_SCALE), ("min", lakh.KK_MINOR, humtrans.MINOR_SCALE)):
            out = sum(hist[(tonic + i) % 12] for i in range(12) if i not in scale) / total
            fit = float(np.corrcoef(hist, np.roll(prof, tonic))[0, 1]) if np.ptp(hist) > 0 else 0.0
            rows.append((out, -(fit + 0.35 * (last == tonic) + 0.15 * (first == tonic)), tonic, mode))
    least = min(r[0] for r in rows)
    _, _, tonic, mode = min(r for r in rows if r[0] <= least + 0.01)
    return tonic, mode


def _later_track_meta(path: Path) -> tuple[list, list, list, list] | None:
    """Tempo and time-signature events of a MIDI file, split into the first track's and the later tracks':
    (tempo first, tempo later, meter first, meter later), each [(tick, value)] in time order, tempo as bpm, meter as
    (numerator, denominator).  The file is read through pretty_midi, which takes both from the first track only;
    this is what it does not see.  None for anything that is not a MIDI file or cannot be read this way."""
    if Path(path).suffix.lower() not in (".mid", ".midi"):
        return None
    try:
        import mido
        mid = mido.MidiFile(str(path))
    except Exception:
        return None
    tempo, meter = ([], []), ([], [])
    for i, track in enumerate(mid.tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo[min(i, 1)].append((tick * TPQ / mid.ticks_per_beat, mido.tempo2bpm(msg.tempo)))
            elif msg.type == "time_signature":
                meter[min(i, 1)].append((tick * TPQ / mid.ticks_per_beat, (msg.numerator, msg.denominator)))
    return sorted(tempo[0]), sorted(tempo[1]), sorted(meter[0]), sorted(meter[1])


def read(path: Path) -> dict:
    """{"notes": [{"start", "dur", "pitch"}] (ticks, monophonic, in time order), "bpm", "meter", "bar_ticks", "bars",
    "key" (e.g. "C", "Am"), "tonic" (pitch class of the relative major), "snapped": notes whose start moved,
    "dropped": notes left out because they overlapped the melody line, "range": [lowest, highest]}."""
    try:
        pm = scorecheck._load(Path(path))
    except scorecheck.ScoreError as e:
        raise ScoreImportError(str(e)) from None
    sigs = [(s.numerator, s.denominator) for s in pm.time_signature_changes]
    later = _later_track_meta(Path(path))
    track_notes = []                                       # tracks that disagree: told to the user, the first track counts
    if later and later[3]:
        if not later[2]:                                   # the meter is written in a later track only: that is the meter
            sigs = [v for _, v in later[3]]
        elif later[3][0][1] != later[2][0][1]:
            a, b = later[2][0][1], later[3][0][1]
            track_notes.append(f"박자 표기가 트랙마다 달라요({a[0]}/{a[1]}, {b[0]}/{b[1]}). 첫 트랙의 {a[0]}/{a[1]}을(를) 썼어요.")
    sig = sigs[0] if sigs else (4, 4)
    if len(set(sigs)) > 1:
        raise ScoreImportError("박자가 중간에 바뀌는 악보는 아직 가져올 수 없어요")
    if sig not in METERS:
        raise ScoreImportError(f"지원하지 않는 박자예요: {sig[0]}/{sig[1]} (가능: {', '.join(METERS.values())})")
    tempo_times, tempi = pm.get_tempo_changes()
    bpm = int(round(float(tempi[0]))) if len(tempi) else 120
    first_tempo = float(tempi[0]) if len(tempi) else None
    later_tempi = None
    if later and later[1]:
        if not later[0]:                                   # the tempo is written in a later track only: that is the tempo
            later_tempi = [x for i, x in enumerate(later[1]) if i == 0 or abs(x[1] - later[1][i - 1][1]) >= 0.05]
            bpm, first_tempo = int(round(later_tempi[0][1])), float(later_tempi[0][1])
        elif int(round(later[1][0][1])) != int(round(later[0][0][1])):
            track_notes.append(f"템포 표기가 트랙마다 달라요({later[0][0][1]:g}, {later[1][0][1]:g}). 첫 트랙의 {later[0][0][1]:g}을(를) 썼어요.")
    try:
        mel, others = scorecheck._split_melody(pm)
    except scorecheck.ScoreError as e:
        raise ScoreImportError(str(e)) from None
    scale = TPQ / pm.resolution
    raw = sorted(((float(s) * scale, float(e) * scale, int(p)) for s, e, p in zip(mel["s"], mel["e"], mel["p"])),
                 key=lambda x: (x[0], -x[2]))                  # at one moment the highest note is the melody
    bar = TPQ * 4 * sig[0] // sig[1]
    notes, snapped, dropped = [], 0, 0
    harmony: list[dict] = []              # the second voice: notes written under the melody (single-part files)
    kept: list[dict] = []                 # per kept note: what the file said (start, end)
    changes: list[dict] = []              # every difference between the file's melody and what the app will use
    for s, e, p in raw:
        start = int(round(s / GRID)) * GRID
        end = max(start + GRID, int(round(e / GRID)) * GRID)
        if notes and start < notes[-1]["start"] + notes[-1]["dur"]:
            if start <= notes[-1]["start"]:                 # same slot after snapping: keep the first, count the other
                dropped += 1
                changes.append(_change(bar, s, p, "dropped", f"같은 칸에 다른 음이 있어 뺐어요 (원래 길이 {_beats(e - s)}박)"))
                continue
            notes[-1]["dur"] = start - notes[-1]["start"]   # a melody sings one note at a time
        snapped += abs(start - s) > MOVED_TICKS
        notes.append({"start": start, "dur": end - start, "pitch": p})
        kept.append({"s": s, "e": e})
    if not notes:
        raise ScoreImportError("멜로디로 쓸 음을 찾지 못했어요")
    for n, src in zip(notes, kept):       # compared after the loop: a later note may have shortened an earlier one
        moved, length = n["start"] - src["s"], n["dur"] - (src["e"] - src["s"])
        if abs(moved) > MOVED_TICKS:
            changes.append(_change(bar, src["s"], n["pitch"], "moved",
                                   f"시작이 {_beats(abs(moved))}박 {'늦게' if moved > 0 else '빠르게'} 옮겨졌어요"))
        if abs(length) > max(MOVED_TICKS, 0.25 * (src["e"] - src["s"])):
            changes.append(_change(bar, src["s"], n["pitch"], "length",
                                   f"길이가 {_beats(src['e'] - src['s'])}박 → {_beats(n['dur'])}박"))
    if sum(1 for inst in pm.instruments if not inst.is_drum and inst.notes) == 1:
        # one part with notes sounding together: only the top one is the melody.  The others are not kept, and the
        # user is told which (they used to vanish without a word).
        # A note that starts with a melody note stays as the second voice (harmony.py); the rest are told.
        from . import harmony as harmony_mod
        grid = lambda t: int(round(float(t) * scale / GRID)) * GRID
        under_all = [(grid(s), max(grid(s) + GRID, grid(e)), int(p)) for under in others
                     for s, e, p in zip(under["s"], under["e"], under["p"])]
        harmony, left = harmony_mod.from_under(notes, under_all)
        for s, e, p in left:
            dropped += 1
            changes.append(_change(bar, s, p, "dropped",
                                   "같은 순간에 더 높은 음이 있어 뺐어요 (멜로디 한 음과 화음 한 음까지만 남겨요)"))
    changes.sort(key=lambda c: (c["bar"], c["beat"]))
    tempo_changes = []
    if later_tempi is not None:
        for tick, v in later_tempi[1:]:
            tempo_changes.append({"bar": int(tick // bar) + 1, "bpm": round(float(v), 1)})
    for t, v in list(zip(tempo_times, tempi))[1:] if later_tempi is None else []:
        tick = pm.time_to_tick(float(t)) * scale
        tempo_changes.append({"bar": int(tick // bar) + 1, "bpm": round(float(v), 1)})
    tempo_note = None
    clamped = max(40, min(200, bpm))
    if tempo_changes:
        shown = ", ".join(f"{c['bar']}마디 {c['bpm']:g}" for c in tempo_changes[:4])
        tempo_note = (f"악보의 템포가 {len(tempo_changes)}번 바뀌지만({shown}{' ...' if len(tempo_changes) > 4 else ''}) "
                      f"앱은 첫 템포 {clamped} 하나로 만들어요.")
    elif first_tempo is not None and (clamped != bpm or abs(first_tempo - bpm) >= 0.05):
        tempo_note = f"악보의 템포 {round(first_tempo, 2):g}을(를) {clamped}(으)로 맞췄어요 (앱은 40~200의 정수 템포만 써요)."
    if track_notes:
        tempo_note = " ".join(([tempo_note] if tempo_note else []) + track_notes)
    bars = -(-(notes[-1]["start"] + notes[-1]["dur"]) // bar)
    if bars > MAX_SONG_BARS:
        raise ScoreImportError(f"곡이 너무 길어요 ({bars}마디, 한도 {MAX_SONG_BARS}마디)")
    written = pm.key_signature_changes
    if written:                                  # the score says its key: believe it
        number = int(written[0].key_number)
        tonic, mode, key_from = number % 12, ("min" if number >= 12 else "maj"), "악보에 적힌 조예요."
    else:
        tonic, mode = guess_key([(n["dur"], n["pitch"]) for n in notes])
        key_from = "조는 멜로디에서 추정한 것이에요. 다르면 직접 골라 주세요."
    major = tonic if mode == "maj" else (tonic + 3) % 12
    chords, chords_note = xml_chords(Path(path), int(bars), sig) if Path(path).suffix.lower() in (".musicxml", ".xml", ".mxl") \
        else (None, "MIDI 파일에는 코드 기호가 없어요. 코드는 앱이 붙여요.")
    # exact = nothing in the melody or the tempo differs from the file; otherwise this is a converted version that the
    # user has to accept before anything is made from it (course doc 4차관문_검토: no change without consent)
    return {"exact": not changes and tempo_note is None, "changes": changes[:MAX_CHANGES], "changes_total": len(changes),
            "tempo_changes": tempo_changes[:MAX_CHANGES], "tempo_note": tempo_note,
            "chords": chords, "chords_note": chords_note, "notes": notes, "harmony": harmony,
            "harmony_note": f"멜로디 아래에 함께 적힌 음 {len(harmony)}개는 화음 파트로 넣어요." if harmony else None, "bpm": max(40, min(200, bpm)), "meter": METERS[sig], "bar_ticks": bar, "bars": int(bars),
            "key": _key_name(tonic, mode), "tonic": major, "snapped": int(snapped), "dropped": dropped,
            "range": [min(n["pitch"] for n in notes), max(n["pitch"] for n in notes)],
            "key_note": key_from}


MAX_CHANGES = 200
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _beats(ticks: float) -> str:
    return f"{ticks / TPQ:.3g}"


def _change(bar_ticks: int, start: float, pitch: int, kind: str, text: str) -> dict:
    """One difference from the file: where (bar and beat from 1), which note, what happened."""
    return {"bar": int(start // bar_ticks) + 1, "beat": round((start % bar_ticks) / TPQ + 1, 2),
            "note": f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}", "kind": kind, "text": text}


def require_consent(score: dict, accepted: bool) -> None:
    """Refuse to build anything from a converted score the user has not accepted (ScoreImportError says what to do)."""
    if score.get("exact", True) or accepted:
        return
    n = int(score.get("changes_total") or 0)
    parts = ([f"멜로디 {n}곳"] if n else []) + (["템포"] if score.get("tempo_note") else [])
    if score.get("omr"):                           # a PDF read from its picture: the whole result needs a look
        raise ScoreImportError("PDF에서 인식한 악보예요. 틀리게 읽힌 음이 있을 수 있으니 내용을 확인하고 "
                               "\"변환본으로 진행\"을 선택한 뒤 다시 눌러 주세요.")
    raise ScoreImportError(f"이 악보는 그대로 가져올 수 없어요: {', '.join(parts)}이(가) 달라져요. 달라지는 내용을 확인하고 "
                           f"\"변환본으로 진행\"을 선택한 뒤 다시 눌러 주세요.")


# music21 chord kinds -> the app's chord qualities (leadsheet.QUALITIES); anything else is reduced to its triad
_KINDS = {"major": "", "minor": "m", "dominant-seventh": "7", "major-seventh": "maj7", "minor-seventh": "m7",
          "diminished": "dim", "diminished-seventh": "dim7", "augmented": "aug", "suspended-second": "sus2",
          "suspended-fourth": "sus4", "power": "", "major-sixth": "6", "minor-sixth": "m6", "dominant-ninth": "9",
          "half-diminished": "m7b5", "suspended-fourth-seventh": "7sus4"}


def _symbol(cs) -> tuple[str, bool]:
    """(the app's chord symbol, whether it had to be simplified) for a music21 ChordSymbol."""
    root = cs.root().name.replace("-", "b")
    kind = getattr(cs, "chordKind", None) or "major"
    simplified = kind not in _KINDS or bool(cs.getChordStepModifications())
    quality = _KINDS.get(kind)
    if quality is None:                          # e.g. ninths, added tones, sus with a seventh: keep root and third
        pcs = {(p.pitchClass - cs.root().pitchClass) % 12 for p in cs.pitches}
        quality = "m" if 3 in pcs and 4 not in pcs else ""
    bass = cs.bass().name.replace("-", "b") if cs.bass() is not None else root
    name = root + quality + (f"/{bass}" if bass != root else "")
    if not _valid_chord(name):
        name, simplified = root + ("m" if quality.startswith("m") and not quality.startswith("maj") else ""), True
    return name, simplified


def xml_chords(path: Path, bars: int, sig: tuple[int, int]) -> tuple[list[str | None] | None, str]:
    """(one chord symbol per bar or None, what was done) from the chord symbols written in a MusicXML file.
    The lead sheet path takes one chord per bar: the chord at the start of the bar (or, failing that, the first one in
    it) is kept; a bar without a symbol keeps the chord before it (None in the list).  Nothing is guessed."""
    try:
        from music21 import converter, harmony
        score = converter.parse(str(path))
        found = list(score.recurse().getElementsByClass(harmony.ChordSymbol))
    except Exception:
        return None, "코드 기호를 읽지 못했어요. 코드는 앱이 붙여요."
    if not found:
        return None, "이 악보에는 코드 기호가 없어요. 코드는 앱이 붙여요."
    bar_q = 4.0 * sig[0] / sig[1]
    per_bar: dict[int, list[tuple[float, str, bool]]] = {}
    for cs in found:
        if isinstance(cs, harmony.NoChord) or not cs.pitches:
            continue
        at = float(cs.getOffsetInHierarchy(score))
        name, simplified = _symbol(cs)
        per_bar.setdefault(int(at // bar_q), []).append((at % bar_q, name, simplified))
    out: list[str | None] = [None] * bars
    kept = simplified_n = extra = 0
    for b, items in per_bar.items():
        if not 0 <= b < bars:
            continue
        items.sort()
        out[b] = items[0][1]
        kept += 1
        simplified_n += items[0][2]
        extra += len(items) - 1
    if kept == 0:
        return None, "이 악보의 코드 기호를 읽지 못했어요. 코드는 앱이 붙여요."
    note = f"악보에 적힌 코드 기호 {kept + extra}개 중 마디마다 하나씩 {kept}개를 가져왔어요."
    if extra:
        note += f" 마디 중간에 바뀌는 코드 {extra}개는 넣지 못했어요."
    if simplified_n:
        note += f" 앱이 모르는 종류 {simplified_n}개는 기본 3화음(또는 7화음)으로 줄였어요."
    if out[0] is None:
        note += " 첫 마디에 코드가 없어서 악보의 코드를 쓰지 않고 앱이 붙여요."
    return out, note


def _check(score: dict) -> None:
    try:
        ok = (isinstance(score["notes"], list) and score["notes"] and score["meter"] in METERS.values()
              and all(type(n[k]) is int for n in score["notes"] for k in ("start", "dur", "pitch"))
              and all(n["start"] >= 0 and n["dur"] > 0 and 0 <= n["pitch"] <= 127 for n in score["notes"])
              and type(score["bars"]) is int and type(score["bar_ticks"]) is int)
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise ScoreImportError("가져온 악보 정보가 온전하지 않아요. 파일을 다시 올려 주세요")


def to_tokens(score: dict, start_bar: int = 0, bars: int = SEED_MAX_BARS) -> dict:
    """{"tokens", "input_tonic", "bars", "bpm"} for bars [start_bar, start_bar + bars) (bar index from 0), as the
    melody input expects them: movable-do degrees relative to the major scale of `tonic`, rests for the gaps."""
    _check(score)
    if score["meter"] != "4/4":
        raise ScoreImportError("멜로디 입력은 4/4 박자만 받아요. '이 악보로 곡 만들기'를 써 주세요")
    if type(start_bar) is not int or type(bars) is not int or start_bar < 0 or not 1 <= bars <= SEED_MAX_BARS \
            or start_bar >= score["bars"]:
        raise ScoreImportError(f"마디 범위가 맞지 않아요 (곡은 {score['bars']}마디, 한 번에 {SEED_MAX_BARS}마디까지)")
    bar = score["bar_ticks"]
    lo, hi = start_bar * bar, min(start_bar + bars, score["bars"]) * bar
    tokens, now = [], lo
    for n in score["notes"]:
        if not lo <= n["start"] < hi:
            continue
        if n["start"] > now:
            tokens.append({"degree": 0, "sharp": False, "octave": 4, "dur": n["start"] - now})
        x = n["pitch"] - score["tonic"]
        degree, sharp = humtrans.STEP_TOKEN[x % 12]
        end = min(n["start"] + n["dur"], hi)
        tokens.append({"degree": degree, "sharp": sharp, "octave": x // 12 - 1, "dur": end - n["start"]})
        now = end
    if not any(t["degree"] for t in tokens):
        raise ScoreImportError(f"{start_bar + 1}번째 마디부터 {bars}마디 안에 음이 없어요")
    under = [{"at": e["at"] - lo, "offset": e["offset"], "dur": min(e["dur"], hi - e["at"])}
             for e in score.get("harmony") or [] if lo <= e["at"] < hi]
    return {"tokens": tokens, "input_tonic": score["tonic"], "bars": (hi - lo) // bar, "bpm": score["bpm"],
            "harmony": under, "seed_ticks": hi - lo}


def default_sections(bars: int) -> list[dict]:
    """A starting split the user can change: halves named verse and chorus (8-bar blocks), or one verse when short."""
    if bars < 8:
        return [{"name": "벌스", "start_bar": 0, "bars": bars}]
    half = max(4, (bars // 2) // 4 * 4)
    return [{"name": "벌스", "start_bar": 0, "bars": half}, {"name": "후렴", "start_bar": half, "bars": bars - half}]


def uncut_sections(score: dict, sections: list[dict]) -> list[dict]:
    """`sections` with every boundary that falls inside a held note moved to the first bar line after which no note
    is still sounding, so no note of the melody is shortened (a lead sheet cannot carry a note across sections).
    A section left without a bar joins the one before it.  Sections that cut nothing come back as they are."""
    bar, total = score["bar_ticks"], int(score["bars"])
    held = lambda b: any(n["start"] < b * bar < n["start"] + n["dur"] for n in score["notes"])
    out: list[dict] = []
    at = 0
    for i, sec in enumerate(sections):
        end = sec["start_bar"] + sec["bars"]
        if i == len(sections) - 1:
            end = total
        while end < total and held(end):
            end += 1
        if end > at:
            out.append({**sec, "start_bar": at, "bars": end - at})
            at = end
    return out


def to_leadsheet(score: dict, sections: list[dict] | None = None, genre: str = "발라드", key: str | None = None,
                 chords: list[str | None] | None = None) -> str:
    """Lead sheet text of the whole song.  `sections`: [{"name", "start_bar", "bars"}] covering bars 0..N in order
    without gaps or overlaps (default_sections when None).
    `chords`: one chord symbol per bar of the song (e.g. the comparison's AI or original chords).  It must cover every
    bar; None inside the list means "same chord as the bar before".  Without it no chords are written and the app
    harmonizes the melody."""
    _check(score)
    sections = [{**s, "name": SECTION_ALIAS.get(s.get("name"), s.get("name"))} if isinstance(s, dict) else s
                for s in (sections or default_sections(score["bars"]))]
    if chords is not None:
        if not isinstance(chords, list) or len(chords) != score["bars"]:
            raise ScoreImportError(f"코드는 마디마다 하나씩 {score['bars']}개가 필요해요 (받은 것: "
                                   f"{len(chords) if isinstance(chords, list) else '목록 아님'}개)")
        bad = [(i + 1, c) for i, c in enumerate(chords) if c is not None and not (isinstance(c, str) and _valid_chord(c))]
        if bad:
            raise ScoreImportError(f"{bad[0][0]}번째 마디의 코드를 읽지 못했어요: {bad[0][1]!r}")
        if chords[0] is None:
            raise ScoreImportError("첫 마디의 코드가 비어 있어요")
    bar, at = score["bar_ticks"], 0
    built = []
    for i, sec in enumerate(sections):
        try:
            name, start, n = sec["name"], sec["start_bar"], sec["bars"]
        except (KeyError, TypeError):
            raise ScoreImportError(f"{i + 1}번째 구간에 name, start_bar, bars가 필요해요") from None
        if name not in SECTION_NAMES:
            raise ScoreImportError(f"{i + 1}번째 구간 이름이 맞지 않아요: {name!r} (가능: {', '.join(SECTION_NAMES)})")
        if type(start) is not int or type(n) is not int or n < 1 or start != at:
            raise ScoreImportError(f"구간은 처음부터 빈틈없이 이어져야 해요 ({i + 1}번째 구간은 {at + 1}번째 마디에서 시작해야 해요)")
        at += n
    if at != score["bars"]:
        raise ScoreImportError(f"구간의 마디 수 합({at})이 곡의 마디 수({score['bars']})와 달라요")
    sections, at = uncut_sections(score, sections), 0     # the melody comes first: a boundary never shortens a note
    for i, sec in enumerate(sections):
        name, start, n = sec["name"], sec["start_bar"], sec["bars"]
        lo, hi = start * bar, (start + n) * bar
        notes = [Note(x["pitch"], x["start"] - lo, min(x["dur"], hi - x["start"]))
                 for x in score["notes"] if lo <= x["start"] < hi]
        built.append((name, notes, list(chords[start:start + n]) if chords is not None else [None] * n))
        at += n
    if at != score["bars"]:
        raise ScoreImportError(f"구간의 마디 수 합({at})이 곡의 마디 수({score['bars']})와 달라요")
    use = key or score.get("key") or "C"
    if use not in KEYS:
        raise ScoreImportError(f"알 수 없는 조예요: {use!r}")
    sheet = from_notes(genre, use, score["meter"], int(score["bpm"]), [(n, notes, ch) for n, notes, ch in built])
    return serialize_leadsheet(sheet)


__all__ = ["read", "to_tokens", "to_leadsheet", "default_sections", "require_consent", "ScoreImportError", "lakh"]
