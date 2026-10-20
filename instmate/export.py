"""Output: MIDI bytes, MusicXML lead sheet, playback events in seconds."""
from __future__ import annotations

import io

import mido
from music21 import clef, harmony, key as m21key, metadata, meter as m21meter, note, pitch, stream, tempo
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .arrange import PROGRAM, TEMPO, Event
from .chords import ChordDef, chord_symbol
from .theory import BAR, METER_TICKS, NOTE_NAMES, TPQ, Key, Note, num_bars

CHANNEL = {"melody": 0, "piano": 1, "bass": 2, "strings": 3, "guitar": 4, "keys": 5, "lead": 6, "harmony": 7, "drums": 9}


def lineup_programs(genre: str) -> dict[str, int]:
    """{track: General MIDI program} of the genre's instrument table (configs/instruments.yaml), so a downloaded MIDI
    opens with a voice-type sound on the melody and the genre's guitar, bass and drum kit.  A MIDI file can only name a
    General MIDI sound: an instrument the app plays through an effect chain is written as its `fallback` sound (crunch
    guitar -> overdriven guitar) and a plugin instrument as its SoundFont sound (synth pad -> string ensemble)."""
    from . import audio, instruments
    try:
        parts = instruments.lineup(genre)["parts"]
    except KeyError:
        return {}
    return {part: audio._program_of(part, inst.get("fallback", inst["sound"])) for part, inst in parts.items()}


def to_midi(events: list[Event], genre: str, bpm: int | None = None, meter: str = "4/4", lineup: bool = True,
            programs: dict[str, int] | None = None) -> bytes:
    """`lineup`: write the instrument table's sounds (melody = voice in the band genres, the genre's drum kit...).
    False keeps the old per-genre programs: only for reproducing the published listening-test clips.
    `programs`: {track: General MIDI program} that wins over the table (e.g. the melody played by a lead guitar)."""
    table = {**(lineup_programs(genre) if lineup else {}), **(programs or {})}
    mid = mido.MidiFile(ticks_per_beat=TPQ, type=1)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm or TEMPO[genre]), time=0))
    num, den = (int(x) for x in meter.split("/"))
    meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))
    mid.tracks.append(meta)
    for track in ["melody", "lead", "keys", "piano", "bass", "guitar", "strings", "drums"]:
        evs = [e for e in events if e.track == track]
        if not evs:
            continue
        ch = CHANNEL[track]
        t = mido.MidiTrack()
        t.append(mido.MetaMessage("track_name", name=track, time=0))
        if track != "drums":
            t.append(mido.Message("program_change", program=table.get(track, PROGRAM[track][genre]), channel=ch, time=0))
        elif "drums" in table:                   # the genre's drum kit (General MIDI / GS kit number)
            t.append(mido.Message("program_change", program=table["drums"], channel=ch, time=0))
        # (abs_tick, order, msg): note_off (0) sorts before note_on (1) at the same tick
        msgs = []
        for e in evs:
            msgs.append((e.start, 1, mido.Message("note_on", note=e.pitch, velocity=e.velocity, channel=ch)))
            msgs.append((e.start + e.dur, 0, mido.Message("note_off", note=e.pitch, velocity=0, channel=ch)))
        msgs.sort(key=lambda m: (m[0], m[1]))
        now = 0
        for tick, _, msg in msgs:
            t.append(msg.copy(time=tick - now))
            now = tick
        mid.tracks.append(t)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _m21_symbol(sym: str) -> str:
    # music21 chord symbols spell flats with '-', e.g. "E-" for Eb; a slash bass is spelled the same way
    head, slash, bass = sym.partition("/")
    if len(head) > 1 and head[1] == "b":
        head = head[0] + "-" + head[2:]
    if len(bass) > 1 and bass[1] == "b":
        bass = bass[0] + "-" + bass[2:]
    return head + slash + bass


def _melody_pitch(midi: int, k: m21key.Key) -> pitch.Pitch:
    """Spell a MIDI pitch within the key; hide accidentals the key signature already implies."""
    p = pitch.Pitch(midi=midi)
    for cand in (p, p.getEnharmonic()):
        if cand.name in {sp.name for sp in k.getScale().getPitches()}:
            p = cand
            break
    if p.accidental is not None and p.accidental.alter == 0:
        p.accidental = None
    return p


def to_musicxml(key: Key, genre: str, melody: list[Note], chords: list[ChordDef], bpm: int | None = None,
                symbols: list | None = None, meter: str = "4/4") -> str:
    """`symbols` (one per bar, see below) overrides the chord names, e.g. for sections in another key."""
    part = stream.Part()
    part.partName = ""
    sung = sorted(n.pitch for n in melody if n.pitch is not None)
    # a melody below middle C (a man's key) reads like a tenor part: treble clef sounding an octave lower
    part.append(clef.Treble8vbClef() if sung and sung[len(sung) // 2] < 57 else clef.TrebleClef())
    part.append(tempo.MetronomeMark(number=bpm or TEMPO[genre]))
    part.append(m21meter.TimeSignature(meter))
    bar_len = METER_TICKS[meter]
    tonic = NOTE_NAMES[key.tonic].replace("b", "-")
    k = m21key.Key(tonic, key.mode)
    part.append(k)
    for n in melody:
        el = note.Rest() if n.pitch is None else note.Note(_melody_pitch(n.pitch, k))
        el.quarterLength = n.dur / TPQ
        part.insert(n.start / TPQ, el)
    end = (len(chords) or num_bars(melody, bar_len)) * bar_len          # no chords: melody-only preview
    last = max(n.start + n.dur for n in melody)
    if last < end:                                    # pad the final bar with a rest
        r = note.Rest()
        r.quarterLength = (end - last) / TPQ
        part.insert(last / TPQ, r)
    for i, c in enumerate(chords):
        # an entry of `symbols` is one name for the bar, or [(tick in bar, name), ...] for chord changes
        marks = symbols[i] if symbols else chord_symbol(key, c)
        for tick, name in ([(0, marks)] if isinstance(marks, str) else marks):
            cs = harmony.ChordSymbol(_m21_symbol(name))
            cs.writeAsChord = False
            part.insert((i * bar_len + tick) / TPQ, cs)
    score = stream.Score([part.makeMeasures().makeTies()])
    score.metadata = metadata.Metadata(title="", composer="")
    return GeneralObjectExporter(score).parse().decode("utf-8")


def playback(events: list[Event], genre: str, bpm: int | None = None) -> dict:
    bpm = bpm or TEMPO[genre]
    sec_per_tick = 60.0 / bpm / TPQ
    return {
        "bpm": bpm,
        "notes": [{"track": e.track, "midi": e.pitch, "time": round(e.start * sec_per_tick, 4),
                   "duration": round(e.dur * sec_per_tick, 4), "velocity": round(e.velocity / 127, 3)}
                  for e in events],
    }
