"""The notes the user wrote under their melody, kept as a second voice (track "harmony").

    from_under(notes, under)                   score reader: which under-notes can follow the melody, as entries
    add(result, entries, period, whole)        a copy of a song result with the harmony track written

An entry is {"at": tick, "offset": semitones (negative: below the melody note), "dur": ticks}: "the melody note that
starts at `at` has a second note `offset` away".  Stored relative to the melody, the second voice follows wherever the
user's melody goes: another key, the singer's range, the key change of the last chorus, every repeat of the seed.
Only sections that hold the user's own melody get it (`origin == "input"`, or every section with `whole`); the app does
not invent a second voice for the sections it wrote.  Both the MIDI and the browser playback list are rewritten.
"""
from __future__ import annotations

import base64
import io

import mido

from .export import CHANNEL
from .introlead import _abs, _track

TRACK = "harmony"
LEVEL = 0.8             # the second voice sits a little under the melody
MAX_ENTRIES = 2000
LOWEST = -24            # a second voice lies within two octaves under the melody; a lower note is accompaniment


class HarmonyError(ValueError):
    """The entries are not usable; the message says why."""


def from_under(notes: list[dict], under: list[tuple[int, int, int]]) -> tuple[list[dict], list[tuple[int, int, int]]]:
    """(entries, left out).  `notes`: the melody on the grid [{"start", "dur", "pitch"}]; `under`: (start, end, pitch)
    on the same grid.  An under-note is kept when it starts with a melody note and lies below it; with several under one
    melody note the highest stays (one second voice).  Everything else is returned as left out: also a note more than
    two octaves under its melody note (a bass or left-hand note, not a second voice; check() refuses such an entry,
    and a file with one made the whole import fail) and whatever comes after MAX_ENTRIES entries."""
    by_start = {n["start"]: n for n in notes}
    best: dict[int, tuple[int, int, int]] = {}
    left = []
    for s, e, p in sorted(under, key=lambda x: (x[0], -x[2])):
        m = by_start.get(s)
        if m is None or p >= m["pitch"] or p - m["pitch"] < LOWEST or s in best or len(best) >= MAX_ENTRIES:
            left.append((s, e, p))
            continue
        best[s] = (s, e, p)
    entries = [{"at": s, "offset": p - by_start[s]["pitch"], "dur": max(1, min(e - s, by_start[s]["dur"]))}
               for s, e, p in (best[k] for k in sorted(best))]
    return entries, left


def check(entries, period: int | None = None) -> list[dict]:
    """Validated copy of the entries (they travel through the browser, so nothing is taken on trust)."""
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        raise HarmonyError(f"화음 파트는 음 {MAX_ENTRIES}개까지예요")
    out, seen = [], set()
    for e in entries:
        ok = isinstance(e, dict) and all(type(e.get(k)) is int for k in ("at", "offset", "dur"))
        if not ok or e["at"] < 0 or not LOWEST <= e["offset"] < 0 or e["dur"] < 1 or (period and e["at"] >= period):
            raise HarmonyError(f"화음 파트의 음이 맞지 않아요: {e!r}")
        if e["at"] in seen:
            raise HarmonyError(f"화음 파트에 같은 자리의 음이 두 개 있어요: {e['at']}")
        seen.add(e["at"])
        out.append({"at": e["at"], "offset": e["offset"], "dur": e["dur"]})
    return out


def add(result: dict, entries: list[dict], period: int | None = None, whole: bool = False) -> dict:
    """A copy of `result` with the second voice under the user's melody.  `period`: length of the seed in ticks (it
    repeats inside a longer section); None with `whole`: `at` counts from the start of the song.  No entry, no melody
    track or no matching note: the result is returned without a harmony track."""
    table = {e["at"]: e for e in check(entries, period)}
    mid = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
    names = [next((m.name for m in tr if m.type == "track_name"), None) for tr in mid.tracks]
    if names.count("melody") != 1 or names.count(TRACK) > 1:
        raise HarmonyError(f"expected exactly one melody track and at most one harmony track, got {names}")
    num, den = (int(x) for x in (result.get("meter") or "4/4").split("/"))
    bar = mid.ticks_per_beat * 4 * num // den
    spans = [(s["start_bar"] * bar, (s["start_bar"] + s["bars"]) * bar) for s in result.get("sections") or []
             if whole or s.get("origin") == "input"]
    ch = CHANNEL[TRACK]
    head, rows, open_notes, sung = [], [], {}, []
    for tick, m in _abs(mid.tracks[names.index("melody")]):
        if m.type == "note_on" and m.velocity > 0:
            open_notes.setdefault(m.note, []).append((tick, m.velocity))
        elif m.type in ("note_off", "note_on"):
            if open_notes.get(m.note):
                start, vel = open_notes[m.note].pop(0)
                sung.append((start, tick, m.note, vel))
        elif m.type == "program_change":
            head.append(m.copy(channel=ch, time=0))          # the second voice sounds like the melody does
    made = []
    for start, end, pitch, vel in sorted(sung):
        span = next(((a, b) for a, b in spans if a <= start < b), None)
        if span is None:
            continue
        rel = start if whole else (start - span[0]) % period if period else start - span[0]
        e = table.get(rel)
        if e is None or not 0 <= pitch + e["offset"] <= 127:
            continue
        stop = start + max(1, min(e["dur"], end - start))
        made.append((start, stop, pitch + e["offset"], max(1, int(vel * LEVEL))))
    tracks = [tr for tr, name in zip(mid.tracks, names) if name != TRACK]
    kept_names = [n for n in names if n != TRACK]
    play = dict(result.get("playback") or {})
    notes = [n for n in play.get("notes") or [] if n.get("track") != TRACK]
    out = {k: v for k, v in result.items() if k != "harmony"}
    if made:
        for start, stop, pitch, vel in made:
            rows.append((start, 1, mido.Message("note_on", note=pitch, velocity=vel, channel=ch)))
            rows.append((stop, 0, mido.Message("note_off", note=pitch, velocity=0, channel=ch)))
        tracks.insert(kept_names.index("melody") + 1, _track(TRACK, head, rows))
        if play.get("bpm"):
            sec = 60.0 / play["bpm"] / mid.ticks_per_beat
            notes += [{"track": TRACK, "midi": p, "time": round(a * sec, 4), "duration": round((b - a) * sec, 4),
                       "velocity": round(v / 127, 3)} for a, b, p, v in made]
        out["harmony"] = {"notes": len(made), "label": "화음"}
    out_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    out_mid.tracks.extend(tracks)
    buf = io.BytesIO()
    out_mid.save(file=buf)
    play["notes"] = notes
    return {**out, "midi_b64": base64.b64encode(buf.getvalue()).decode(), "playback": play}
