"""The user's intro melody played by the lead guitar instead of sung.

    to_lead(result) -> result      a copy: the melody notes of the opening intro moved to the "lead" track

Used when the input melody is the song's intro (seed_role "intro") and the user wants it played as a riff while the
rest of the song is sung.  Only the track changes: pitch, start, length and loudness of every note stay.  Both the
MIDI and the browser playback list are rewritten, so what is heard, downloaded and packed for the band agree.
"""
from __future__ import annotations

import base64
import io

import mido

from .arrange import PROGRAM
from .export import CHANNEL, lineup_programs


def _abs(track) -> list[tuple[int, mido.Message]]:
    now, out = 0, []
    for m in track:
        now += m.time
        out.append((now, m))
    return out


def _track(name: str, head: list, notes: list[tuple[int, int, mido.Message]]):
    """A track from (tick, order, message) rows; note_off (order 0) sorts before note_on at the same tick."""
    t = mido.MidiTrack()
    t.append(mido.MetaMessage("track_name", name=name, time=0))
    for m in head:
        t.append(m.copy(time=0))
    now = 0
    for tick, _, m in sorted(notes, key=lambda r: (r[0], r[1])):
        t.append(m.copy(time=tick - now))
        now = tick
    t.append(mido.MetaMessage("end_of_track", time=0))
    return t


def to_lead(result: dict) -> dict:
    secs = result.get("sections") or []
    if not secs or secs[0].get("kind") != "intro" or secs[0].get("origin") != "input":
        raise ValueError("the song does not open with the user's melody as its intro")
    mid = mido.MidiFile(file=io.BytesIO(base64.b64decode(result["midi_b64"])))
    names = [next((m.name for m in tr if m.type == "track_name"), None) for tr in mid.tracks]
    if names.count("melody") != 1 or names.count("lead") > 1:
        raise ValueError(f"expected exactly one melody track and at most one lead track, got {names}")
    mi = names.index("melody")
    end = secs[0]["bars"] * mid.ticks_per_beat * 4               # composed songs are 4/4
    genre = result.get("genre")
    lead_ch = CHANNEL["lead"]
    keep, move, open_notes, mel_head = [], [], {}, []
    for tick, m in _abs(mid.tracks[mi]):
        if m.type == "note_on" and m.velocity > 0:
            to = move if tick < end else keep
            open_notes.setdefault(m.note, []).append(to)
            to.append((tick, 1, m))
        elif m.type in ("note_off", "note_on"):
            to = open_notes[m.note].pop(0) if open_notes.get(m.note) else keep
            to.append((tick, 0, m))
        elif not m.is_meta:
            mel_head.append(m)                                  # program change etc.
    if not move:
        raise ValueError("the intro has no melody notes to move")
    lead_rows = [(tick, order, m.copy(channel=lead_ch)) for tick, order, m in move]
    if "lead" in names:
        li = names.index("lead")
        lead_head = [m for _, m in _abs(mid.tracks[li]) if not m.is_meta and m.type not in ("note_on", "note_off")]
        on = {}
        for tick, m in _abs(mid.tracks[li]):
            if m.type == "note_on" and m.velocity > 0:
                on.setdefault(m.note, []).append(tick)
                if tick >= end:
                    lead_rows.append((tick, 1, m))
            elif m.type in ("note_off", "note_on"):
                start = on[m.note].pop(0) if on.get(m.note) else tick
                if start >= end:                                # the app's own intro lead (the chorus hook) gives way
                    lead_rows.append((tick, 0, m))
    else:
        program = lineup_programs(genre).get("lead", PROGRAM["lead"].get(genre, 27)) if genre else 27
        lead_head = [mido.Message("program_change", program=program, channel=lead_ch)]
    tracks = list(mid.tracks)
    tracks[mi] = _track("melody", mel_head, keep)
    lead = _track("lead", lead_head, lead_rows)
    if "lead" in names:
        tracks[names.index("lead")] = lead
    else:
        tracks.insert(mi + 1, lead)
    out_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    out_mid.tracks.extend(tracks)
    buf = io.BytesIO()
    out_mid.save(file=buf)
    play = dict(result.get("playback") or {})
    # exact, not the section's rounded end_sec: a note starting right on the bar line belongs to the next section
    end_sec = secs[0]["bars"] * 240.0 / play["bpm"] if play.get("bpm") else float(secs[0]["end_sec"])
    notes = []
    for n in play.get("notes") or []:
        if n["track"] == "lead" and n["time"] < end_sec - 1e-6:
            continue                                            # the app's own intro lead, replaced by the user's line
        notes.append({**n, "track": "lead"} if n["track"] == "melody" and n["time"] < end_sec - 1e-6 else n)
    play["notes"] = notes
    sections = [dict(s) for s in secs]
    sections[0]["melody_source"] = f"{sections[0].get('melody_source') or '입력 그대로'} · 리드 기타가 연주"
    return {**result, "midi_b64": base64.b64encode(buf.getvalue()).decode(), "playback": play, "sections": sections,
            "intro_player": "guitar"}
