"""Bring a band member's own part into the song: a MIDI file played or edited outside the app replaces one part.

    replace_part(result, part, midi)   checks that the file fits the song (tempo, meter, length), puts its notes on the
                                       song's own track for that part (same channel and sound), updates the playback
                                       notes, and records the import in result["member_parts"]
    keep_parts(new_result, previous)   after the app made a new version of the song: carries the members' parts over
                                       when the new version still has the same tempo, meter and length; otherwise it
                                       says which parts could not be kept and why (nothing is silently dropped)

What is kept from the member's file: every note (pitch, start, length, velocity) and the performance messages on that
track (controllers such as the sustain pedal, pitch bend, aftertouch), at their own times.  What is replaced: the MIDI
channel and the instrument sound (program), which stay those of the song's part.  Nothing is quantized or clipped.
A note that would ring past the end of the song is refused unless the caller asks for it to be cut there (and then the
cut is reported).  Range problems are reported as warnings.

Replacing the melody makes the song's notation (MusicXML, lead sheet) out of date: those are removed from the result
with the reason, instead of being handed on as if they still matched.
"""
from __future__ import annotations

import base64
import hashlib
import io
import time
from collections import Counter

import mido

from .arrange import RANGES

PARTS = ("melody", "lead", "piano", "keys", "bass", "guitar", "strings", "drums")
PART_KO = {"melody": "멜로디", "lead": "리드 기타", "piano": "피아노", "keys": "멜로디 피아노", "bass": "베이스",
           "guitar": "기타", "strings": "스트링", "drums": "드럼"}
TEMPO_TOLERANCE = 1.0          # BPM


class PartSwapError(ValueError):
    """The file cannot replace the part (Korean message with the reason)."""


def _name(track) -> str | None:
    return next((m.name for m in track if m.type == "track_name"), None)


def _notes(track, scale: float) -> list[tuple[int, int, int, int]]:
    """[(start, end, pitch, velocity)] in the song's ticks."""
    now, out, open_ = 0, [], {}
    for m in track:
        now += m.time
        if m.type == "note_on" and m.velocity > 0:
            open_.setdefault(m.note, []).append((now, m.velocity))
        elif m.type in ("note_off", "note_on") and open_.get(m.note):
            start, vel = open_[m.note].pop(0)
            out.append((round(start * scale), max(round(start * scale) + 1, round(now * scale)), m.note, vel))
    return sorted(out)


EXPRESSION = ("control_change", "pitchwheel", "aftertouch", "polytouch")


def _expression(track, scale: float) -> list[tuple[int, object]]:
    """[(tick in the song's resolution, message)] of the performance messages on the member's track."""
    now, out = 0, []
    for m in track:
        now += m.time
        if m.type in EXPRESSION:
            out.append((round(now * scale), m))
    return out


def layout(result: dict) -> list[tuple[str, int, int]]:
    return [(s["kind"], s["start_bar"], s["bars"]) for s in result["sections"]]


def _song_facts(mid: mido.MidiFile) -> tuple[float, tuple[int, int]]:
    tempo = next((m.tempo for m in mid.tracks[0] if m.type == "set_tempo"), 500000)
    sig = next(((m.numerator, m.denominator) for m in mid.tracks[0] if m.type == "time_signature"), (4, 4))
    return 60_000_000 / tempo, sig


def _pick_track(up: mido.MidiFile, part: str, allow_empty: bool = False):
    with_notes = [t for t in up.tracks if any(m.type == "note_on" and m.velocity > 0 for m in t)]
    named = [t for t in with_notes if _name(t) == part]
    if named:
        return named[0]
    if allow_empty:                              # a part the member emptied on purpose (a rest) is still their part
        silent = [t for t in up.tracks if _name(t) == part]
        if silent:
            return silent[0]
    if len(with_notes) == 1:
        return with_notes[0]
    if not with_notes:
        raise PartSwapError("올린 MIDI에 음표가 없어요")
    names = ", ".join(_name(t) or "(이름 없음)" for t in with_notes)
    raise PartSwapError(f"올린 MIDI에 음표가 있는 트랙이 {len(with_notes)}개예요 ({names}). {PART_KO[part]} 한 트랙만 내보내거나 "
                        f"트랙 이름을 '{part}'로 해 주세요")


DRUM_NOTES = range(35, 82)                   # General MIDI percussion keys
DRUM_CHANNEL = 9


def wrong_part_hint(part: str, notes: list[tuple], track) -> str | None:
    """A sentence for the user when the file does not look like the part it was brought in as (a bass line chosen as
    drums, a drum pattern chosen as bass...).  Only a hint: the file is used as it is, the app never decides for the
    member.  `notes`: (start, end, pitch, velocity)."""
    if not notes:
        return None
    pitches = [n[2] for n in notes]
    channels = {m.channel for m in track if m.type == "note_on" and m.velocity > 0}
    on_drum_channel = channels == {DRUM_CHANNEL}
    if part == "drums":
        if not on_drum_channel and len(set(pitches)) >= 4 and sum(p in (36, 38, 42, 46, 49, 51) for p in pitches) < len(pitches) * 0.3:
            return ("드럼: 올린 파일이 드럼 패턴처럼 보이지 않아요(드럼 채널이 아니고, 킥·스네어·하이햇에 해당하는 음이 거의 없어요). "
                    "다른 파트의 파일을 드럼으로 고른 것은 아닌지 확인해 주세요. 파일은 올린 그대로 넣었어요.")
        return None
    if on_drum_channel:
        return (f"{PART_KO[part]}: 올린 파일이 드럼 채널(10번)로 되어 있어요. 드럼 파일을 {PART_KO[part]}로 고른 것은 아닌지 "
                "확인해 주세요. 파일은 올린 그대로 넣었어요.")
    if part in RANGES:
        lo, hi = RANGES[part]
        outside = sum(not lo <= p <= hi for p in pitches)
        if outside > len(pitches) * 0.5:
            return (f"{PART_KO[part]}: 음의 절반 이상({outside}/{len(pitches)}개)이 {PART_KO[part]}의 음역 밖이에요. "
                    "다른 파트의 파일을 고른 것은 아닌지 확인해 주세요. 파일은 올린 그대로 넣었어요.")
    return None


def replace_part(result: dict, part: str, midi: bytes, author: str | None = None, trim_tail: bool = False,
                 allow_empty: bool = False) -> dict:
    """A copy of `result` whose `part` is the member's file.  PartSwapError when it does not fit the song.
    `trim_tail`: cut notes that ring past the song's end at the end (reported); default is to refuse such a file."""
    if part not in PARTS:
        raise PartSwapError(f"바꿀 수 없는 파트: {part!r}")
    try:
        song = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
        up = mido.MidiFile(file=io.BytesIO(midi))
    except Exception as exc:
        raise PartSwapError(f"MIDI를 읽지 못했어요: {exc}") from None
    target = next((t for t in song.tracks if _name(t) == part), None)
    if target is None:
        raise PartSwapError(f"이 곡에는 {PART_KO[part]} 파트가 없어요. 먼저 그 악기를 켜고 곡을 만든 뒤 바꿔 주세요")
    bpm, sig = _song_facts(song)
    up_bpm, up_sig = _song_facts(up)
    if abs(bpm - up_bpm) > TEMPO_TOLERANCE:
        raise PartSwapError(f"템포가 달라요: 곡 {bpm:.0f} BPM, 올린 파일 {up_bpm:.0f} BPM. 같은 템포로 내보내 주세요")
    if sig != up_sig:
        raise PartSwapError(f"박자가 달라요: 곡 {sig[0]}/{sig[1]}, 올린 파일 {up_sig[0]}/{up_sig[1]}")
    tpq = song.ticks_per_beat
    bar_len = tpq * 4 * sig[0] // sig[1]
    total = sum(s["bars"] for s in result["sections"]) * bar_len
    source = _pick_track(up, part, allow_empty)
    scale = tpq / up.ticks_per_beat
    notes = _notes(source, scale)
    late = [n for n in notes if n[0] >= total]
    if late:
        raise PartSwapError(f"곡은 {total // bar_len}마디인데 올린 파일은 {late[0][0] // bar_len + 1}마디째에도 음이 있어요. "
                            "곡의 맨 앞(0마디)에 맞춰 같은 길이로 내보냈는지 확인해 주세요")
    over = [n for n in notes if n[1] > total]
    if over and not trim_tail:
        worst = max(n[1] for n in over) - total
        raise PartSwapError(f"곡 끝({total // bar_len}마디 끝)을 넘어 울리는 음이 {len(over)}개 있어요 (가장 긴 것이 {worst / tpq:.2f}박 넘침). "
                            "음을 바꾸지 않으려고 가져오지 않았어요. 파일에서 길이를 맞추거나, '곡 끝에서 자르기'를 켜고 다시 올려 주세요")
    expression = [(t, m) for t, m in _expression(source, scale) if t <= total]
    channel = next((m.channel for m in target if hasattr(m, "channel")), 9 if part == "drums" else 0)
    head = [m for m in target if m.type in ("track_name", "program_change") or (m.is_meta and m.type not in ("end_of_track",))]
    msgs = [(0, 0, m.copy(time=0)) for m in head]
    for tick, m in expression:               # the member's pedal, bends...: same message, on the part's channel
        msgs.append((tick, 0, m.copy(channel=channel, time=0)))
    for start, end, pitch, vel in notes:
        msgs.append((start, 2, mido.Message("note_on", channel=channel, note=pitch, velocity=vel)))
        msgs.append((min(end, total), 1, mido.Message("note_off", channel=channel, note=pitch, velocity=0)))
    msgs.sort(key=lambda x: (x[0], x[1]))
    track, last = mido.MidiTrack(), 0
    for tick, _, m in msgs:
        track.append(m.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    song.tracks[song.tracks.index(target)] = track
    buf = io.BytesIO()
    song.save(file=buf)

    sec = 60.0 / bpm / tpq
    played = [{"track": part, "midi": p, "time": round(s * sec, 4), "duration": round((min(e, total) - s) * sec, 4),
               "velocity": round(v / 127, 3)} for s, e, p, v in notes]
    playback = dict(result.get("playback") or {})
    playback["notes"] = sorted([n for n in playback.get("notes", []) if n["track"] != part] + played, key=lambda n: n["time"])
    warnings = []
    if over:
        warnings.append(f"{PART_KO[part]}: 곡 끝을 넘는 음 {len(over)}개를 곡 끝에서 잘랐어요 (요청한 대로)")
    if part in RANGES and part != "drums":
        lo, hi = RANGES[part]
        out = [n for n in notes if not lo <= n[2] <= hi]
        if out:
            warnings.append(f"{PART_KO[part]}: 앱의 권장 음역(MIDI {lo}~{hi}) 밖의 음이 {len(out)}개 있어요 (그대로 둡니다)")
    mismatch = wrong_part_hint(part, notes, source)
    if mismatch:
        warnings.append(mismatch)
    record = {"part": part, "author": (author or "").strip()[:80] or None, "notes": len(notes),
              "expression_events": len(expression), "trimmed_notes": len(over),
              "sha256": hashlib.sha256(midi).hexdigest(), "imported": time.strftime("%Y-%m-%d %H:%M:%S")}
    members = [m for m in result.get("member_parts") or [] if m["part"] != part] + [record]
    out = {**result, "midi_b64": base64.b64encode(buf.getvalue()).decode(), "playback": playback, "member_parts": members,
           "member_part_warnings": warnings,
           "check_note": "가져온 파트는 앱의 충돌·음역 검사에 반영되지 않았어요 (검사 수치는 가져오기 전 편곡 기준)"}
    if part == "melody":                     # the notation described the old melody: do not pass it on as current
        reason = "멜로디를 멤버의 파일로 바꿔서, 이전 멜로디로 만든 악보(MusicXML)와 리드시트는 이 버전에 넣지 않았어요"
        out.update(musicxml=None, leadsheet=None, leadsheet_c=None, leadsheet_error=reason, stale_scores=reason)
        warnings.append(reason)
    return out


def keep_parts(new_result: dict, previous: dict) -> tuple[dict, list[str], list[dict]]:
    """(result, kept part names, [{"part", "reason"}] for parts that could not be carried over)."""
    members = previous.get("member_parts") or []
    if not members:
        return new_result, [], []
    old = mido.MidiFile(file=io.BytesIO(base64.b64decode(previous["midi_b64"])))
    result, kept, dropped = new_result, [], []
    same_layout = layout(new_result) == layout(previous)
    for rec in members:
        part = rec["part"]
        if not same_layout:                  # the bars no longer mean the same sections: never place it by guessing
            dropped.append({"part": part, "reason": f"곡의 길이나 구성이 달라졌어요 ({previous.get('bars')}마디 → "
                                                    f"{new_result.get('bars')}마디). 새 구성에 맞춰 파일을 다시 올려 주세요"})
            continue
        track = next((t for t in old.tracks if _name(t) == part), None)
        solo = mido.MidiFile(ticks_per_beat=old.ticks_per_beat, type=old.type)
        solo.tracks.append(old.tracks[0])
        solo.tracks.append(track)
        buf = io.BytesIO()
        solo.save(file=buf)
        try:
            result = replace_part(result, part, buf.getvalue(), rec.get("author"), allow_empty=True)
        except PartSwapError as exc:
            dropped.append({"part": part, "reason": str(exc)})
            continue
        result["member_parts"] = [m for m in result["member_parts"] if m["part"] != part] + [
            {**rec, "carried_over": time.strftime("%Y-%m-%d %H:%M:%S")}]
        kept.append(part)
    return result, kept, dropped


# ---------------------------------------------------------------- editing a section inside the app
MAX_EDIT_NOTES = 4000


def _abs(track) -> list[tuple[int, object]]:
    now, out = 0, []
    for m in track:
        now += m.time
        out.append((now, m))
    return out


def section_notes(result: dict, part: str, start_bar: int, bars: int) -> dict:
    """What the editor shows for bars [start_bar, start_bar + bars) of a part (bar index from 0; ticks, 480 per
    quarter note, counted from the start of the range):
        notes     the notes that START in the range: {"start", "dur", "pitch", "velocity"}; "over" marks a note that
                  rings past the end of the range (it is kept as it is unless the user changes it)
        ringing   notes from earlier bars still sounding in the range (shown read-only; they are never touched)
        bar_ticks, total_ticks, tpq"""
    song = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
    _, sig = _song_facts(song)
    bar_len = song.ticks_per_beat * 4 * sig[0] // sig[1]
    total_bars = sum(s["bars"] for s in result["sections"])
    if not (isinstance(start_bar, int) and isinstance(bars, int)) or start_bar < 0 or bars < 1 or start_bar + bars > total_bars:
        raise PartSwapError(f"마디 범위가 곡을 벗어나요 (곡은 {total_bars}마디, 요청은 시작 마디 번호 {start_bar!r}부터 {bars!r}마디)")
    track = next((t for t in song.tracks if _name(t) == part), None)
    lo, hi = start_bar * bar_len, (start_bar + bars) * bar_len
    notes, ringing = [], []
    for s, e, p, v in (_notes(track, 1.0) if track is not None else []):
        if lo <= s < hi:
            notes.append({"start": s - lo, "dur": e - s, "pitch": p, "velocity": v, "over": e > hi})
        elif s < lo < e:
            ringing.append({"start": s - lo, "dur": e - s, "pitch": p, "velocity": v})
    return {"part": part, "start_bar": start_bar, "bars": bars, "notes": notes, "ringing": ringing, "bar_ticks": bar_len,
            "total_ticks": hi - lo, "tpq": song.ticks_per_beat, "has_part": track is not None}


def edit_section(result: dict, part: str, start_bar: int, bars: int, notes: list[dict], author: str | None = None) -> dict:
    """A copy of `result` in which the notes of `part` that start in bars [start_bar, start_bar + bars) are exactly
    `notes` ({"start", "dur", "pitch", "velocity"}, ticks from the start of the range).

    Contract:
      - Only notes starting in the range are replaced.  Notes outside it, notes from earlier bars that ring into it,
        and every non-note message of the track (sound, pedal, pitch bend, controllers) stay exactly as they are.
      - A note may ring past the end of the range only if that very note (same start, pitch and length) was already
        there; a new or lengthened note that would cross the end is refused (nothing is cut silently).
      - An empty list is valid: the bars become a rest.
      - The same note twice at one tick is allowed and kept as two notes (existing parts contain such doubles).
      - Applying the notes that are already there changes nothing: the same result object is returned, no record.
      - The part must already exist in the song."""
    if part not in PARTS:
        raise PartSwapError(f"바꿀 수 없는 파트: {part!r}")
    try:
        song = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
    except Exception as exc:
        raise PartSwapError(f"MIDI를 읽지 못했어요: {exc}") from None
    target = next((t for t in song.tracks if _name(t) == part), None)
    if target is None:
        raise PartSwapError(f"이 곡에는 {PART_KO[part]} 파트가 없어요. 먼저 그 악기를 켜고 곡을 만든 뒤 편집해 주세요")
    view = section_notes(result, part, start_bar, bars)
    bar_len, span = view["bar_ticks"], view["total_ticks"]
    lo = start_bar * bar_len
    if not isinstance(notes, list) or len(notes) > MAX_EDIT_NOTES:
        raise PartSwapError(f"음이 너무 많아요 (한 번에 {MAX_EDIT_NOTES}개까지)")
    # multisets: a part may hold the same note twice at one tick (the app's own drums do, where a fill meets the
    # groove); such doubles are kept exactly, never merged
    before = Counter((n["start"], n["pitch"], n["dur"], n["velocity"]) for n in view["notes"])
    existing = {k[:3] for k in before}
    wanted: Counter = Counter()
    for i, n in enumerate(notes):
        try:
            s, d, p, v = n["start"], n["dur"], n["pitch"], n["velocity"]
        except (KeyError, TypeError):
            raise PartSwapError(f"{i + 1}번째 음에 start, dur, pitch, velocity가 모두 있어야 해요") from None
        if any(type(x) is not int for x in (s, d, p, v)):
            raise PartSwapError(f"{i + 1}번째 음: 값은 모두 정수여야 해요")
        if not (0 <= s < span and d >= 1 and 0 <= p <= 127 and 1 <= v <= 127):
            raise PartSwapError(f"{i + 1}번째 음이 범위를 벗어나요 (시작 0~{span - 1}틱, 길이 1 이상, 음높이 0~127, 세기 1~127)")
        if s + d > span and (s, p, d) not in existing:
            raise PartSwapError(f"{s // bar_len + start_bar + 1}번째 마디의 음(MIDI {p})이 편집 범위 끝을 "
                                f"{(s + d - span) / view['tpq']:.2f}박 넘어요. 길이를 줄이거나 범위를 넓혀 주세요")
        wanted[(s, p, d, v)] += 1
    if wanted == before:
        return result                            # nothing changed: no new record, notation stays valid
    channel = next((m.channel for m in target if hasattr(m, "channel")), 9 if part == "drums" else 0)
    events = _abs(target)
    open_, drop = {}, set()
    for i, (t, m) in enumerate(events):          # the note messages of the notes that start in the range
        if m.type == "note_on" and m.velocity > 0:
            open_.setdefault(m.note, []).append((i, t))
        elif m.type in ("note_off", "note_on") and open_.get(m.note):
            j, start = open_[m.note].pop(0)
            if lo <= start < lo + span:
                drop.update((i, j))
    kept = [(t, 0 if not m.type.startswith("note") else (1 if m.type == "note_off" or m.velocity == 0 else 2), m)
            for i, (t, m) in enumerate(events) if i not in drop and m.type != "end_of_track"]
    for s, p, d, v in wanted.elements():
        kept.append((lo + s, 2, mido.Message("note_on", channel=channel, note=p, velocity=v)))
        kept.append((lo + s + d, 1, mido.Message("note_off", channel=channel, note=p, velocity=0)))
    kept.sort(key=lambda x: (x[0], x[1]))        # stable: the track's own message order is kept at equal ticks
    track, last = mido.MidiTrack(), 0
    for t, _, m in kept:
        track.append(m.copy(time=t - last))
        last = t
    track.append(mido.MetaMessage("end_of_track", time=0))
    song.tracks[song.tracks.index(target)] = track
    buf = io.BytesIO()
    song.save(file=buf)

    bpm, _ = _song_facts(song)
    sec = 60.0 / bpm / song.ticks_per_beat
    a, b = lo * sec, (lo + span) * sec
    playback = dict(result.get("playback") or {})
    others = [n for n in playback.get("notes", []) if not (n["track"] == part and a - 1e-6 <= n["time"] < b - 1e-6)]
    fresh = [{"track": part, "midi": p, "time": round((lo + s) * sec, 4), "duration": round(d * sec, 4),
              "velocity": round(v / 127, 3)} for s, p, d, v in wanted.elements()]
    playback["notes"] = sorted(others + fresh, key=lambda n: n["time"])
    changed = sum(((wanted - before) + (before - wanted)).values())
    edit = {"start_bar": start_bar, "bars": bars, "author": (author or "").strip()[:80] or None,
            "changed_notes": changed, "edited": time.strftime("%Y-%m-%d %H:%M:%S")}
    previous = next((m for m in result.get("member_parts") or [] if m["part"] == part), None)
    record = {**(previous or {"part": part, "author": None, "source": "edited in the app"}),
              "last_editor": edit["author"], "notes": len(_notes(track, 1.0)),
              "edits": [*(previous or {}).get("edits", []), edit],
              "protects": "the whole part: the app will not regenerate any bar of it"}
    members = [m for m in result.get("member_parts") or [] if m["part"] != part] + [record]
    out = {**result, "midi_b64": base64.b64encode(buf.getvalue()).decode(), "playback": playback, "member_parts": members,
           "member_part_warnings": [],
           "check_note": "편집한 파트는 앱의 충돌·음역 검사에 반영되지 않았어요 (검사 수치는 편집 전 편곡 기준)"}
    if part == "melody":
        reason = "멜로디를 직접 고쳐서, 이전 멜로디로 만든 악보(MusicXML)와 리드시트는 이 버전에 넣지 않았어요"
        out.update(musicxml=None, leadsheet=None, leadsheet_c=None, leadsheet_error=reason, stale_scores=reason)
        out["member_part_warnings"].append(reason)
    return out
