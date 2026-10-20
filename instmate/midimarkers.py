"""Section markers for a downloaded MIDI file (option, off by default).

    add(midi, sections) -> a copy with one marker event per section in the tempo track

The markers are written into a COPY at download time.  The song's own MIDI is never changed: the audio render seeds
its small timing and loudness offsets from the bytes of that MIDI, so a marker inside it would move every note of the
render.  Names are plain ASCII ("verse 1", "chorus 2": the kind and its count), because a MIDI text event has no agreed
encoding.  No note, tempo or program is added or moved.
"""
from __future__ import annotations

import io

import mido


class NotMidi(ValueError):
    """The bytes are not a MIDI file (Korean message)."""


def add(midi: bytes, sections: list[dict] | None) -> bytes:
    """`sections`: [{"kind", "start_bar", "bars"}] of the song (bars from 0).  With no sections the input is returned
    as it is.  Markers already in the tempo track are replaced, not doubled.  ValueError for a file that is not MIDI,
    a negative bar or a kind that is not plain ASCII."""
    if not sections:
        return midi
    try:
        mid = mido.MidiFile(file=io.BytesIO(midi))
    except Exception:
        raise NotMidi("MIDI를 읽지 못했어요") from None
    if not mid.tracks:
        raise NotMidi("MIDI에 트랙이 없어요")
    num, den = next(((m.numerator, m.denominator) for m in mid.tracks[0] if m.type == "time_signature"), (4, 4))
    bar = mid.ticks_per_beat * 4 * num // den
    seen: dict[str, int] = {}
    wanted = []
    for s in sections:
        kind, start = s.get("kind"), s.get("start_bar")
        if not isinstance(kind, str) or not kind or not kind.isascii() or type(start) is not int or start < 0:
            raise ValueError(f"구간 정보가 올바르지 않아요: {s!r}")
        seen[kind] = seen.get(kind, 0) + 1
        wanted.append((start * bar, f"{kind} {seen[kind]}"))
    rows, now, end = [], 0, None
    for i, m in enumerate(mid.tracks[0]):
        now += m.time
        if m.type == "end_of_track":
            end = now
        elif m.type != "marker":
            rows.append((now, 0, i, m))
    rows += [(tick, 1, i, mido.MetaMessage("marker", text=text)) for i, (tick, text) in enumerate(wanted)]
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    track, now = mido.MidiTrack(), 0
    for tick, _, _, m in rows:
        track.append(m.copy(time=tick - now))
        now = tick
    if end is not None:
        track.append(mido.MetaMessage("end_of_track", time=max(end, now) - now))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    out.tracks.append(track)
    out.tracks.extend(mid.tracks[1:])
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()
