"""Sound option "section_parts" (off unless asked for): the sections differ in how many instruments play, and the
level is left alone.

Works on the render copy only (the MIDI file and the score are not changed), after the performance is prepared:
  * verse        drums play kick and hi-hat only; of piano and guitar one stays (the piano when it plays in that
                 section, else the guitar); strings and keys rest.  Bass and melody stay.
  * pre-chorus   the whole drum part and the strings/keys come back; the second chord part still rests; the last bar
                 ends in a snare fill on its last beat unless the drummer already plays toms there.
  * chorus       everything as arranged, with a crash on its first beat when there is none.
  * bridge       the guitar and the lead rest when the piano or a pad plays; the drums go to half time (kick on one,
                 snare on three, hi-hat on the beats).
  * last chorus  (of two or more) the melody is doubled an octave up on the strings, else on the lead.
Intro, outro and other kinds are left as arranged.  A part a band member played (`exact`) and the melody are never
changed.  Which parts rest where is a design hypothesis to be judged by ear.
"""
from __future__ import annotations

import io

import mido

KICK, SNARE, CRASH = 36, 38, 49
HATS = frozenset({42, 44, 46, 51})
TOMS = frozenset({41, 43, 45, 47, 48, 50})
CHORDS = ("piano", "guitar")        # the first that plays in a verse or pre-chorus stays, the other rests
PADS = ("strings", "keys")
LAYERS = ("strings", "lead")        # the first present doubles the melody in the last chorus
LAYER_VELOCITY = 62
LAYER_TOP = 96
FILL = (72, 82, 92, 102)            # four sixteenths on the last beat
DRUMS = "drums"


def _name(track) -> str | None:
    return next((m.name for m in track if m.type == "track_name"), None)


def shape(midi: bytes, sections: list[dict] | None,
          exact: set[str] | frozenset[str] = frozenset()) -> bytes:
    """The performance with each section's parts thinned or filled as the module text says; the same bytes when
    there is nothing to do (no sections, a changing meter)."""
    from .breath import _notes
    if not sections:
        return midi
    mid = mido.MidiFile(file=io.BytesIO(midi))
    signatures = {(m.numerator, m.denominator) for t in mid.tracks for m in t if m.type == "time_signature"}
    if len(signatures) > 1 or mid.ticks_per_beat < 4:
        return midi                         # a variable meter needs its own bar map
    num, den = next(iter(signatures), (4, 4))
    tpb = mid.ticks_per_beat
    bar, grid = tpb * 4 * num // den, tpb // 4
    kind_of = {b: (i, s["kind"]) for i, s in enumerate(sections)
               for b in range(s["start_bar"], s["start_bar"] + s["bars"])}
    choruses = [i for i, s in enumerate(sections) if s["kind"] == "chorus"]
    last_chorus = choruses[-1] if len(choruses) > 1 else None
    at = lambda tick: round(tick / grid) * grid             # preparing moves starts a little: read the nearest 16th
    tracks = {_name(t): t for t in mid.tracks if _name(t)}
    noted = {name: _notes(t) for name, t in tracks.items()}
    plays = {name: {kind_of[at(n["start"]) // bar][0] for n in ns if at(n["start"]) // bar in kind_of}
             for name, ns in noted.items()}
    free = lambda name: name in tracks and name not in exact

    drop: dict[str, set[int]] = {name: set() for name in tracks}
    add: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in tracks}   # (start, end, note, velocity)

    def rest(name, i):
        """Every note of `name` that starts in section i."""
        for n in noted[name]:
            if kind_of.get(at(n["start"]) // bar, (None,))[0] == i:
                drop[name].update((n["on"], n["off"]))

    for i, s in enumerate(sections):
        k = s["kind"]
        if k in ("verse", "pre"):
            carriers = [c for c in CHORDS if i in plays.get(c, ())]
            for c in carriers[1:]:
                if free(c):
                    rest(c, i)
            if k == "verse":
                for p in PADS:
                    if free(p) and carriers:
                        rest(p, i)
        elif k == "bridge" and any(i in plays.get(c, ()) for c in ("piano",) + PADS):
            for c in ("guitar", "lead"):
                if free(c):
                    rest(c, i)

    if free(DRUMS):
        hits = noted[DRUMS]
        where = [(n, at(n["start"])) for n in hits]
        for i, s in enumerate(sections):
            first, end = s["start_bar"] * bar, (s["start_bar"] + s["bars"]) * bar
            mine = [(n, q) for n, q in where if first <= q < end]
            if s["kind"] == "verse":
                for n, q in mine:
                    if n["note"] != KICK and n["note"] not in HATS:
                        drop[DRUMS].update((n["on"], n["off"]))
            elif s["kind"] == "pre":
                beat = end - tpb
                if beat >= first and not any(n["note"] in TOMS and q >= end - bar for n, q in mine):
                    taken = {q for n, q in mine if n["note"] == SNARE}
                    add[DRUMS] += [(beat + j * grid, beat + (j + 1) * grid, SNARE, v)
                                   for j, v in enumerate(FILL) if beat + j * grid not in taken]
            elif s["kind"] == "chorus":
                if not any(n["note"] == CRASH and q == first for n, q in mine):
                    add[DRUMS].append((first, first + tpb, CRASH, 95))
            elif s["kind"] == "bridge":
                for n, q in mine:
                    pos = q % bar
                    keep = (n["note"] == KICK and pos == 0) or (n["note"] in HATS and pos % tpb == 0)
                    if n["note"] == SNARE and pos == tpb and (num, den) == (4, 4):
                        add[DRUMS].append((n["start"] + tpb, n["end"] + tpb, SNARE, 0))    # 0: the hit's own velocity
                    if not keep:
                        drop[DRUMS].update((n["on"], n["off"]))

    layer = next((name for name in LAYERS if free(name)), None)
    if last_chorus is not None and layer and "melody" in noted:
        add[layer] += [(n["start"], n["end"], n["note"] + 12, LAYER_VELOCITY) for n in noted["melody"]
                       if kind_of.get(at(n["start"]) // bar, (None,))[0] == last_chorus and n["note"] + 12 <= LAYER_TOP]

    if not any(drop.values()) and not any(add.values()):
        return midi
    out = mido.MidiFile(ticks_per_beat=tpb, type=mid.type)
    for track in mid.tracks:
        name = _name(track)
        if not name or not (drop[name] or add[name]):
            out.tracks.append(track)
            continue
        channel = next((m.channel for m in track if hasattr(m, "channel")), 9 if name == DRUMS else None)
        velocity = {n["on"]: track[n["on"]].velocity for n in noted[name]}
        moved = {(n["start"], n["note"]): velocity[n["on"]] for n in noted[name]}
        events, tick, end_tick = [], 0, 0
        for i, m in enumerate(track):
            tick += m.time
            if m.type == "end_of_track":
                end_tick = tick
            elif i not in drop[name]:
                is_on = m.type == "note_on" and m.velocity > 0
                events.append((tick, 2 if is_on else 1 if m.type in ("note_on", "note_off") else 0, i, m))
        if channel is not None:
            for j, (a, z, note, vel) in enumerate(add[name]):
                vel = vel or moved.get((a - tpb, note), 90)
                events.append((a, 2, len(track) + j, mido.Message("note_on", channel=channel, note=note, velocity=vel)))
                events.append((z, 1, len(track) + j, mido.Message("note_off", channel=channel, note=note, velocity=0)))
        events.sort(key=lambda e: e[:3])
        new, written = mido.MidiTrack(), 0
        for t, _, _, m in events:
            new.append(m.copy(time=t - written))
            written = t
        new.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - written)))
        out.tracks.append(new)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()
