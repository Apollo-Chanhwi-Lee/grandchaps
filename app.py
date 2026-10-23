"""InstMate web server. Run: .venv/bin/python app.py  ->  http://127.0.0.1:8000"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import json
from urllib.parse import quote

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from instmate import agent, bands as bands_mod, composer, refaudio, sectionfind, style as style_mod, upload
from instmate.arrange import BAND, INSTRUMENTS
from instmate.history import History, UnknownSong
from instmate.prefs import Prefs
from instmate.versions import NothingToUndo, StaleBase, UnknownHistory, VersionStore

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("INSTMATE_DATA", ROOT / "data"))   # separate folder for test/demo instances
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="InstMate")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.middleware("http")
async def revalidate_ui_files(request, call_next):
    """The page, script and stylesheet change during development: make browsers revalidate instead of using a
    stale cached copy (a stale style.css left the piano roll unstyled)."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@app.middleware("http")
async def local_app_guard(request, call_next):
    """The locally installed app (start_mac.command sets INSTMATE_LOCAL_APP=1) answers only to its own address: a request whose Host
    is not the loopback address is refused (DNS rebinding), and a browser request that changes something must come from this same
    page (Origin equal to Host, not Sec-Fetch-Site cross-site).  Tools such as curl send no Origin and are not affected.  Servers
    without the flag (tests of the page, the old public server) behave as before."""
    if os.environ.get("INSTMATE_LOCAL_APP") != "1":
        return await call_next(request)
    from urllib.parse import urlsplit
    host = request.headers.get("host", "")
    if (urlsplit("//" + host).hostname or "") not in LOOPBACK_HOSTS:
        return JSONResponse({"detail": "이 앱은 이 컴퓨터(127.0.0.1)에서만 열 수 있어요."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if (origin is not None and (urlsplit(origin).netloc != host or urlsplit(origin).scheme != "http")) \
                or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "다른 사이트에서 보낸 요청은 받지 않아요."}, status_code=403)
    return await call_next(request)


INVITE_FORM = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>초대 코드</title><body style="font:16px sans-serif;max-width:420px;margin:15vh auto;padding:0 16px">'
               '<h1>그랜드챕스 작곡메이트</h1><p>{msg}</p><form method="post" action="/invite">'
               '<input name="code" autocomplete="off" required aria-label="초대 코드" style="font:inherit;padding:8px;width:60%"> '
               '<button style="font:inherit;padding:8px 16px">들어가기</button></form></body>')


def invite_token(code: str) -> str:
    import hashlib
    return hashlib.sha256(("instmate-invite:" + code).encode()).hexdigest()


@app.middleware("http")
async def invite_gate(request, call_next):
    """With INSTMATE_INVITE_CODE set (the public Mac server behind Netlify), every request needs the invite cookie."""
    import hmac
    from urllib.parse import parse_qs
    code = os.environ.get("INSTMATE_INVITE_CODE")
    if not code:
        return await call_next(request)
    token = invite_token(code)
    if hmac.compare_digest(request.cookies.get("instmate_invite", ""), token):
        return await call_next(request)
    if request.url.path == "/invite" and request.method == "POST":
        given = parse_qs((await request.body()).decode("utf-8", "replace")).get("code", [""])[0].strip()
        if hmac.compare_digest(given.encode(), code.encode()):
            ok = RedirectResponse("/", status_code=303)
            ok.set_cookie("instmate_invite", token, max_age=30 * 86400, httponly=True, secure=True, samesite="lax")
            return ok
        return HTMLResponse(INVITE_FORM.format(msg="코드가 맞지 않아요. 다시 입력해 주세요."), status_code=401)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "초대 코드가 필요해요."}, status_code=401)
    return HTMLResponse(INVITE_FORM.format(msg="동아리 초대 코드를 입력해 주세요."), status_code=401)
prefs = Prefs(DATA / "prefs.db")
history: History | None = None            # composed songs and ratings (same SQLite file), created on first use


def _history() -> History:
    global history
    if history is None:
        history = History(DATA / "prefs.db")
    return history
versions = VersionStore()

# A trained BiLSTM (models/chord_bilstm.pt + chord_hmm.npz) becomes the agent's preferred chord tool.
from instmate import learned  # noqa: E402

if all(learned.available().values()):
    agent.register_chord_tool("dl", learned.bilstm_tool)


class KeyChoice(BaseModel):
    tonic: int = Field(ge=0, le=11)
    mode: str = Field(pattern="^(major|minor)$")


class GenerateReq(BaseModel):
    tokens: list[dict]
    input_tonic: int = Field(0, ge=0, le=11)
    genre: str = "ballad"
    instruments: list[str] = Field(default_factory=lambda: list(BAND))
    key_choice: KeyChoice | None = None
    energy: int = Field(0, ge=0, le=1)
    tempo: int | None = None                 # BPM; None = genre default (validated in instmate.mix)
    levels: dict[str, float] | None = None   # per-track level, 1.0 = unchanged


class ModifyReq(GenerateReq):
    key_choice: KeyChoice
    current: list[str]
    bar: int | None = None                   # one bar, or
    bars: list[int] | None = None            # a set of bars (e.g. a range) edited together
    mood: str
    history: str | None = None   # when set, the result is adopted as a new version of this history
    allow_clash: bool = False    # user accepted the least-clashing chord after a need_input question


class UndoReq(BaseModel):
    history: str


class AgentReq(BaseModel):
    history: str
    base_index: int
    request: str = Field(min_length=1, max_length=500)


class MixReq(BaseModel):
    history: str
    base_index: int                          # the version the user was looking at; must still be the head
    tempo: int | None = None
    levels: dict[str, float] | None = None


def _state(req: GenerateReq, result: dict) -> dict:
    """Everything a version is built from; a modify against a history must start from the current one."""
    return {"tokens": req.tokens, "input_tonic": req.input_tonic, "genre": req.genre,
            "instruments": req.instruments, "energy": req.energy,
            "key_choice": {"tonic": result["key"]["tonic"], "mode": result["key"]["mode"]},
            "chords": [c["roman"] for c in result["chords"]]}


def _adopt(history: str, base: dict, new_state: dict, result: dict) -> dict:
    """The one way a changed version is adopted (button edit, natural-language request, confirmed proposal).

    Adopts the new version (409 if the base is no longer the head) and only then records the user's preference, once
    per bar whose chord actually differs between the base version and the adopted one. Failed, asked, over-budget and
    rejected (409) requests never get here, and a mix-only change has no chord difference, so none of them teach a preference.
    """
    try:
        info = versions.adopt(history, base["index"], new_state, result)
    except StaleBase as e:
        raise HTTPException(409, str(e))
    for old, new in zip(base["state"]["chords"], new_state["chords"]):
        if old != new:
            prefs.record(new_state["genre"], result["key"]["mode"], new)
    return info


def _public(result: dict) -> dict:
    """Response body: the tick-level `events` stay server-side in the version store."""
    return {k: v for k, v in result.items() if k != "events"}


def _with_version(result: dict, info: dict) -> dict:
    return {**_public(result), "version": info}


@app.get("/")
def index():
    from instmate import yue
    reason = yue.unavailable_reason()
    status = json.dumps({"available": reason is None, "reason": reason}, ensure_ascii=False).replace("<", "\\u003c")
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    if os.environ.get("INSTMATE_PUBLIC") == "1":
        import html as html_escape
        marker = html_escape.escape(os.environ.get("INSTMATE_PUBLIC_ID", ""), quote=True)
        html = html.replace("</head>", f'<meta name="instmate-public-instance" content="{marker}"></head>', 1)
        banner = ('<aside id="public-demo-banner" role="note" style="position:sticky;top:0;z-index:1000;padding:12px;background:#33240e;color:#fff">'
                  '공개 시연 중: 녹음과 요청은 이 Mac에서 처리돼요. 한 번에 한 곡만 만들 수 있어요. '
                  + ('초대 코드가 있는 사람만 들어올 수 있어요.' if os.environ.get("INSTMATE_INVITE_CODE")
                     else '접근 제한이 없으니 주소는 수업 사람에게만 공유해 주세요.') + '</aside>')
        import re
        html = re.sub(r"(<body\b[^>]*>)", lambda m: m.group(1) + banner, html, count=1)
    return HTMLResponse(html.replace("</head>", f"<script>window.INSTMATE_YUE_STATUS = {status};</script></head>", 1))


@app.post("/api/generate")
def generate(req: GenerateReq):
    try:
        result = agent.run(req.tokens, req.input_tonic, req.genre, req.instruments,
                           req.key_choice.model_dump() if req.key_choice else None, req.energy, prefs,
                           tempo=req.tempo, levels=req.levels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result["status"] != "ok":
        return _public(result)
    return _with_version(result, versions.start(_state(req, result), result))


@app.post("/api/modify")
def modify(req: ModifyReq):
    base = None
    if req.history is not None:
        try:
            base = versions.head(req.history)
        except UnknownHistory:
            raise HTTPException(404, "unknown history")
        asked = {"tokens": req.tokens, "input_tonic": req.input_tonic, "genre": req.genre,
                 "instruments": req.instruments, "energy": req.energy,
                 "key_choice": req.key_choice.model_dump(), "chords": req.current}
        if asked != base["state"]:
            raise HTTPException(409, "request does not match the current version; reload it first")
    selection = req.bars if req.bars is not None else req.bar
    if selection is None:
        raise HTTPException(400, "bar or bars is required")
    try:
        # prefs=None: the preference is recorded by _adopt, only once the result is actually adopted
        result = agent.modify(req.tokens, req.input_tonic, req.genre, req.instruments, req.key_choice.model_dump(),
                              req.current, selection, req.mood, req.energy, None,
                              base_events=base["result"]["events"] if base else None, allow_clash=req.allow_clash,
                              # an edit keeps the adopted version's mix; without a history the request's mix is used
                              tempo=base["result"]["tempo"] if base else req.tempo,
                              levels=base["result"]["levels"] if base else req.levels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result["status"] not in agent.ADOPTABLE:
        return _public(result)          # need_input / budget_exceeded: the current version stays as it is
    if base is None:
        return _public(result)          # no history: a preview that is not stored, so it teaches no preference
    return _with_version(result, _adopt(req.history, base, _state(req, result), result))


# LLM planner factory; tests replace it with a scripted planner.
# Default: local Ollama (INSTMATE_LLM=ollama). INSTMATE_LLM=gemini uses the Gemini API key in .env instead.
# Returns (planner, None) or (None, reason).
def make_planner():
    from instmate import llm
    if os.environ.get("INSTMATE_LLM", "ollama") == "gemini":
        key = llm.load_api_key(ROOT)
        return (llm.GeminiPlanner(key), None) if key else (None, "Gemini API 키가 설정되지 않았어요 (.env의 GEMINI_API_KEY)")
    reason = llm.ollama_status()
    return (None, reason) if reason else (llm.OllamaPlanner(), None)


@app.post("/api/agent")
def agent_request(req: AgentReq):
    """Natural-language request: the LLM calls tools on a working copy; only a finished request is adopted."""
    from instmate.llm import run_request
    try:
        base = versions.head(req.history)
    except UnknownHistory:
        raise HTTPException(404, "unknown history")
    if base["index"] != req.base_index:
        raise HTTPException(409, "version changed; reload it first")
    planner, reason = make_planner()
    if planner is None:
        raise HTTPException(503, reason)
    out = run_request(planner, base["state"], base["result"], req.request)
    body = {"agent": {k: out[k] for k in ("status", "message", "log", "changed", "pending")}}
    # a failed check may be forced only by the user's explicit confirmation (POST /api/agent/confirm) of exactly these
    # proposals for exactly this version; the LLM's own arguments are never consent
    versions.set_pending(req.history, base["index"], out["pending"])
    if out["status"] == "ok" and out["changed"]:
        info = _adopt(req.history, base, out["state"], out["result"])
        return {**_with_version(out["result"], info), **body}
    return body                          # nothing adopted: the shown version stays current


class AgentConfirmReq(BaseModel):
    history: str
    base_index: int
    bars: list[int] = Field(min_length=1)    # 0-based, as in /api/modify
    mood: Literal["darker", "brighter"]


@app.post("/api/agent/confirm")
def agent_confirm(req: AgentConfirmReq):
    """The user's explicit "그래도 적용" for a clash proposal that a natural-language request ended on.

    Accepted only if the head is still `base_index`, a stored proposal for that version has the same mood and covers every
    requested bar (requested <= approved). Then the least-clashing chord is applied to exactly those bars and adopted as
    one version. Any other request is refused with 409 and nothing changes."""
    try:
        base = versions.head(req.history)
    except UnknownHistory:
        raise HTTPException(404, "unknown history")
    if base["index"] != req.base_index:
        raise HTTPException(409, "version changed; reload it first")
    asked = set(req.bars)
    if len(asked) != len(req.bars):
        raise HTTPException(400, "duplicate bar")
    if not any(p["mood"] == req.mood and asked <= set(p["bars"]) for p in versions.pending(req.history, req.base_index)):
        raise HTTPException(409, "no pending proposal for these bars and this mood on the current version; ask again")
    st = base["state"]
    try:
        result = agent.modify(st["tokens"], st["input_tonic"], st["genre"], st["instruments"], st["key_choice"],
                              st["chords"], sorted(asked), req.mood, st["energy"], None,
                              base_events=base["result"]["events"], allow_clash=True,
                              tempo=base["result"]["tempo"], levels=base["result"]["levels"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result["status"] not in agent.ADOPTABLE:
        return _public(result)
    new_state = {**st, "chords": [c["roman"] for c in result["chords"]]}
    return _with_version(result, _adopt(req.history, base, new_state, result))


class PreviewReq(BaseModel):
    tokens: list[dict]
    input_tonic: int = Field(0, ge=0, le=11)


@app.post("/api/preview")
def preview(req: PreviewReq):
    """Melody-only sheet for the input editor (no chords, nothing stored)."""
    from instmate.export import to_musicxml
    from instmate.theory import Key, analyze_key, parse_number_input
    try:
        melody = parse_number_input(req.tokens, req.input_tonic)
        top = analyze_key(melody, req.input_tonic)["candidates"][0]["key"] if any(n.pitch for n in melody) else \
            {"tonic": req.input_tonic, "mode": "major"}
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"musicxml": to_musicxml(Key(top["tonic"], top["mode"]), "ballad", melody, [], bpm=None)}


class SongReq(BaseModel):
    history: str
    base_index: int
    # the same choices as /api/compose (they used to be ignored here: always chorus, genre key change, sung intro)
    seed_role: Literal["chorus", "verse", "intro"] = "chorus"
    fill: Literal["own", "rest"] = "own"
    form: Literal["standard", "chorus_first"] | None = None      # None: the form stored with the version
    key_change: bool | None = None                               # None: the genre's habit; False: no key change
    intro_player: Literal["vocal", "guitar"] = "vocal"
    # the second voice written under the input melody (instmate/harmony.py): entries relative to the seed's first tick
    harmony: list[dict] | None = Field(None, max_length=2000)
    seed_ticks: int | None = Field(None, ge=1, le=8 * 1920)


def _second_voice(result: dict, entries, period: int | None, whole: bool = False) -> dict:
    """The song with the user's second voice under their melody; without entries the result as it is."""
    if not entries:
        return result
    from instmate import harmony
    try:
        return harmony.add(result, entries, period, whole)
    except harmony.HarmonyError as e:
        raise HTTPException(422, str(e))


@app.post("/api/song")
def make_song(req: SongReq):
    """Expand the current version (seed melody + adopted chords + mix) into a 2-4 minute song. Adopts nothing."""
    from instmate.song import build_song
    try:
        base = versions.head(req.history)
    except UnknownHistory:
        raise HTTPException(404, "unknown history")
    if base["index"] != req.base_index:
        raise HTTPException(409, "version changed; reload it first")
    state = {**base["state"], "seed_role": req.seed_role, "fill": req.fill, "key_change": req.key_change}
    if req.form:
        state["form"] = req.form
    try:
        result = build_song(state, base["result"])
        if req.seed_role == "intro" and req.intro_player == "guitar":
            from instmate import introlead
            result = introlead.to_lead(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["settings"] = {**(result.get("settings") or {}), "seed_role": req.seed_role, "fill": req.fill}
    result["melody_source"] = "own"          # this path never calls the trained model: rule variations only
    result = _second_voice(result, req.harmony, req.seed_ticks)
    return _with_leadsheet(result)


def _with_leadsheet(result: dict) -> dict:
    """Every song is also shown in the common lead sheet text format."""
    from instmate.leadsheet import LeadsheetError
    from instmate.render import song_to_leadsheet
    try:
        return {**result, "leadsheet": song_to_leadsheet(result)}
    except LeadsheetError as e:            # a song that cannot be written must not break the song itself
        return {**result, "leadsheet": None, "leadsheet_error": str(e)}


class LeadsheetReq(BaseModel):
    text: str = Field(max_length=200_000)
    target_key: str | None = None            # e.g. "G" or "Em" (same mode as the sheet's key)
    genre: str | None = None                 # 발라드/록/댄스/클래식 (or ballad/rock/dance/classical); default: the sheet's
    levels: dict[str, float] | None = None   # per-track level, 1.0 = unchanged
    key_change_last_chorus: bool = False     # with 2+ 후렴, the last one goes up a semitone


@app.post("/api/leadsheet/render")
def render_leadsheet(req: LeadsheetReq):
    """Lead sheet text -> rule-based arrangement -> song result (MIDI, sheet, playback, checks). Nothing is stored."""
    from instmate.leadsheet import validate_leadsheet
    from instmate.render import leadsheet_to_song
    check = validate_leadsheet(req.text)
    if not check["ok"]:
        raise HTTPException(400, {"errors": check["errors"]})
    try:
        result = leadsheet_to_song(req.text, req.target_key, req.genre, req.levels, req.key_change_last_chorus)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**_with_leadsheet(result), "warnings": check["warnings"]}


class AudioSection(BaseModel):
    kind: str = Field(max_length=20)
    start_bar: int = Field(ge=0, le=2000)
    bars: int = Field(ge=1, le=2000)


class AudioReq(BaseModel):
    midi_b64: str = Field(min_length=1, max_length=3_000_000)   # the result's own midi_b64 (mix already applied)
    format: Literal["auto", "wav", "mp3"] = "auto"           # auto: MP3 when ffmpeg exists, else WAV
    name: str = Field("instmate", max_length=80)             # file name without extension
    sounds: dict[str, str] | None = None                     # {track: sound key} (GET /api/sounds), e.g. melody: violin
    genre: str | None = None                                 # the result's genre: profile swing + master EQ
    swing: float | None = Field(None, ge=0, le=0.5)          # override: 0 straight .. 0.33 triplet shuffle
    vocal: bool = False                                      # sing the melody with the recorded hum (POST /api/voice)
    sections: list[AudioSection] | None = Field(None, max_length=64)   # the result's sections: per-section guitar tone
    exact_parts: list[str] | None = Field(None, max_length=8)   # members' own parts: played back without humanizing
    mode: Literal["demo", "release"] | None = None           # release: licence tier A instruments only
    exact_melody: bool = True                                # the melody is played as written: no swing, no timing drift
    vocal_fit: bool = False                                  # move the sung melody by octaves into the singer's range
    vocal_style: Literal["male", "female"] | None = None     # sing "na na na" (male) / "la la la" (female); with
                                                             # vocal=False the built-in voice of that style sings
    only_part: str | None = Field(None, max_length=20)       # practice: this part alone ...
    without_part: str | None = Field(None, max_length=20)    # ... or the song with this part left out
    options: list[str] | None = Field(None, max_length=16)   # opt-in sound features (GET /api/audio/options); none = as before


@app.get("/api/audio/status")
def audio_status():
    """Whether audio rendering can run here (FluidSynth + SoundFont; MP3 also needs ffmpeg)."""
    from instmate import audio
    return audio.status()


@app.get("/api/audio/options")
def audio_options():
    """Sound features the page shows as switches: id, label, help, default (on at first for the ones the user
    chose by ear, audio.DEFAULT_ON).  The API itself applies none unless `options` names it."""
    from instmate import audio
    return [{"id": k, "label": label, "help": about, "default": k in audio.DEFAULT_ON}
            for k, (label, about) in audio.OPTIONS.items()]


@app.get("/api/genres")
def genres_catalog(mode: Literal["demo", "release"] | None = None):
    """Genre profiles for the UI: label, country, tempo, length, loop, key change, swing, form and a description."""
    from instmate import genres, instruments
    out = []
    for g in genres.GENRES:
        s = genres.summary(g)
        s["lineup"] = instruments.api_lineup(g, mode)        # configs/instruments.yaml: part -> instrument, per section
        s["kit_labels"] = [{"part": p["part_label"], "sound": p["label"], "tier": p["tier"]} for p in s["lineup"]["parts"]
                           if p["part"] != "lead"]
        out.append(s)
    return out


@app.get("/api/sounds")
def sounds_catalog(mode: Literal["demo", "release"] | None = None):
    """Selectable sounds per part: {track: [{key, label, tier}]} (the first entry of each part is not the default; the
    default is the genre's own program, sent as no choice).  `tier` is the licence tier (A: may ship with the app,
    B: install it yourself); in release mode only tier A is listed."""
    from instmate import audio, instruments
    cfg = instruments.load()
    out = {t: [{"key": k, "label": lab, "tier": "A"} for k, (lab, _) in opts.items()] for t, opts in audio.SOUNDS.items()}
    for part, ids in (cfg.get("choices") or {}).items():     # gear-level choices (effect chains, plugins) for band parts
        rows = [{"key": i, "label": cfg["instruments"][i]["label"], "tier": cfg["instruments"][i].get("tier", "A")} for i in ids]
        out[part] = [r for r in rows if instruments.allowed(r["tier"], mode)]
    return out


@app.get("/api/lineup")
def lineup_for(genre: str, mode: Literal["demo", "release"] | None = None):
    """One genre's line-up as it plays in the given mode (release: tier-B instruments replaced)."""
    from instmate import genres, instruments
    if genre not in genres.GENRES:
        raise HTTPException(404, "unknown genre")
    return instruments.api_lineup(genre, mode)


@app.get("/api/credits")
def credits(mode: Literal["demo", "release"] | None = None):
    """Sound sources and plugin / sample instruments with licence tier, licence and attribution line."""
    from instmate import instruments
    return instruments.credits(mode)


_AUDIO_CACHE: dict[str, tuple[bytes, str, list[str]]] = {}   # same song + same settings rendered again: reuse (newest 4)


def _audio_key(req: "AudioReq", midi: bytes) -> str | None:
    """Cache key of a render, or None when it must not be cached (a recorded voice may have been replaced)."""
    import hashlib
    import json
    from instmate import audio, instruments, vst
    if req.vocal:
        return None
    h = hashlib.sha256(midi)
    sf, piano = audio.soundfont_path(), audio.piano_soundfont_path()   # another SoundFont or no ffmpeg: another render
    h.update(json.dumps([str(sf), str(piano), str(audio.ffmpeg_path())]).encode())
    h.update(req.model_dump_json(exclude={"midi_b64", "name"}).encode())
    h.update((instruments.default_mode() + os.environ.get("INSTMATE_VST", "")).encode())
    for f in [instruments.CONFIG, vst.CONFIG, *sorted(vst.VST_DIR.glob("*.state"))]:    # table or saved sounds changed
        if f.is_file():
            h.update(f"{f.name}:{f.stat().st_mtime_ns}:{f.stat().st_size}".encode())
    return h.hexdigest()


@app.post("/api/audio")
def render_audio(req: AudioReq):
    """M7: render a result's MIDI to WAV/MP3 offline (FluidSynth + General MIDI SoundFont). Nothing is stored.
    503 with the reason when FluidSynth, the SoundFont or (for MP3) ffmpeg is missing; 400 for an unusable MIDI."""
    import base64
    import binascii
    import re
    from instmate import audio
    try:
        midi = base64.b64decode(req.midi_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "midi_b64 is not valid base64")
    if req.only_part or req.without_part:                    # practice: one part alone, or everything but it
        from instmate import bandpack
        if req.only_part and req.without_part:
            raise HTTPException(422, "only_part와 without_part 중 하나만 보내 주세요")
        try:
            midi = bandpack.pick_tracks(midi, only=req.only_part, without=req.without_part)
        except bandpack.BandPackError as e:
            raise HTTPException(422, str(e))
        except Exception:
            raise HTTPException(400, "MIDI를 읽지 못했어요")
    try:
        voice = None
        if req.vocal:
            from instmate import vocal as vocal_mod
            if not VOICE_FILE.exists():
                raise HTTPException(409, "먼저 '내 목소리 허밍'을 녹음해 주세요")
            voice = vocal_mod.load_voice(VOICE_FILE.read_bytes())
        sections = [s.model_dump() for s in req.sections] if req.sections else None
        key = _audio_key(req, midi)
        if key in _AUDIO_CACHE:
            data, fmt, notes = _AUDIO_CACHE[key]
        else:
            notes = []
            exact = set(req.exact_parts or []) | ({"melody"} if req.exact_melody else set())
            data, fmt = audio.render(midi, req.format, req.sounds, req.genre, req.swing, voice, sections, notes=notes,
                                     vocal_style=req.vocal_style, exact_parts=exact, mode=req.mode,
                                     vocal_fit=req.vocal_fit, options=frozenset(req.options or ()))
            notes = list(dict.fromkeys(notes))
            if key is not None:
                _AUDIO_CACHE[key] = (data, fmt, notes)
                while len(_AUDIO_CACHE) > 4:
                    _AUDIO_CACHE.pop(next(iter(_AUDIO_CACHE)))
    except ValueError as e:                                  # unknown part or sound key
        raise HTTPException(422, str(e))
    except audio.AudioUnavailable as e:
        raise HTTPException(503, str(e))
    except audio.BadMidi as e:
        raise HTTPException(400, str(e))
    except audio.AudioRenderError as e:
        raise HTTPException(500, str(e))
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", req.name).strip("._") or "instmate"
    import json
    from urllib.parse import quote
    return Response(data, media_type=audio.MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"',
                             # which instruments were replaced and why (percent-encoded JSON list, shown on screen)
                             "X-Instmate-Notes": quote(json.dumps(notes[:12], ensure_ascii=False))})


class MidiReq(BaseModel):
    midi_b64: str = Field(min_length=1, max_length=3_000_000)
    sounds: dict[str, str] | None = None                     # {track: sound key}: what the user chose on screen


@app.post("/api/midi")
def midi_with_sounds(req: MidiReq):
    """The result's MIDI with the sounds the user chose on screen written into it (program numbers only: no note,
    tempo or controller is changed). Without choices the same bytes come back."""
    import base64
    import binascii
    from instmate import audio
    try:
        midi = base64.b64decode(req.midi_b64, validate=True)
        out = audio.set_sounds(midi, req.sounds)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(400, "MIDI를 읽지 못했어요")
    return {"midi_b64": base64.b64encode(out).decode(), "changed": out != midi}


class MidiMarkersReq(BaseModel):
    midi_b64: str = Field(min_length=1, max_length=3_000_000)
    sections: list[AudioSection] = Field(max_length=64)      # the result's sections


@app.post("/api/midi/markers")
def midi_with_markers(req: MidiMarkersReq):
    """Option at download time: a copy of the result's MIDI with one marker per section in the tempo track
    (instmate/midimarkers.py).  The song itself is not changed; no note, tempo or program is added."""
    import base64
    import binascii
    from instmate import midimarkers
    try:
        midi = base64.b64decode(req.midi_b64, validate=True)
        out = midimarkers.add(midi, [s.model_dump() for s in req.sections])
    except midimarkers.NotMidi as e:
        raise HTTPException(400, str(e))
    except (binascii.Error, ValueError) as e:
        raise HTTPException(422, str(e))
    return {"midi_b64": base64.b64encode(out).decode(), "markers": len(req.sections) if out != midi else 0}


class BandPackReq(BaseModel):
    result: dict | None = None                               # the finished song as the page holds it
    sounds: dict[str, str] | None = None                     # sounds chosen on screen, written into the exported MIDI
    band_version_id: int | None = None                       # instead of `result`: export exactly this saved version
    melody_source: Literal["ai", "own"] | None = None
    band_id: int | None = None                               # the band's line-up for the guide's checks


@app.post("/api/bandpack")
def band_pack(req: BandPackReq):
    """A zip for the band: printable chord chart, per-part playing guide, per-part MIDI, score and lead sheet."""
    from instmate import bandpack
    if req.band_version_id is not None:      # a saved band version: its stored content is exported, nothing from the page
        try:
            ver = _bands().get_version(req.band_version_id)
        except bands_mod.UnknownVersion:
            raise HTTPException(404, "unknown band version")
        band = {k: v for k, v in _bands().get_band(ver["band_id"]).items() if k != "created"}
        provenance = {"source": "band_version", "band_id": ver["band_id"], "version_id": ver["id"],
                      "parent_id": ver["parent_id"], "result_sha256": ver["sha256"], "author": ver["author"],
                      "note": ver["note"], "credits": ver["credits"], "saved": ver["created_text"],
                      "meaning": "the pack was built from this stored version; hashes identify content, not rights"}
        try:
            data = bandpack.build(ver["result"], ver["result"].get("melody_source"), band, provenance)
        except bandpack.BandPackError as e:
            raise HTTPException(422, str(e))
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="jakgokmate_bandpack_v{ver["id"]}.zip"'})
    if req.result is None:
        raise HTTPException(422, "result 또는 band_version_id가 필요해요")
    source = req.melody_source or req.result.get("melody_source")
    song_id = req.result.get("song_id")
    provenance = None
    if isinstance(song_id, int):             # a stored song: how it was made comes from the record, not the request
        try:
            rec = _history().get_song(song_id)
            source = rec["compose_stats"].get("melody_source", source)
            provenance = {"source": "record", "song_id": song_id,
                          "matches_record": req.result.get("leadsheet_c") == rec["leadsheet"],
                          "note": "matches_record compares the melody and chords with the stored song; member parts "
                                  "and later edits are listed separately"}
        except UnknownSong:
            provenance = {"source": "request", "song_id": song_id, "note": "this song id is not in the local history"}
    band = None
    if req.band_id is not None:
        try:
            band = {k: v for k, v in _bands().get_band(req.band_id).items() if k != "created"}
        except bands_mod.UnknownBand:
            raise HTTPException(404, "unknown band")
    result = req.result
    if req.sounds:                           # the user's own choice wins over the genre table in the exported files too
        import base64
        from instmate import audio
        try:
            midi = audio.set_sounds(base64.b64decode(result["midi_b64"]), req.sounds)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e))
        result = {**result, "midi_b64": base64.b64encode(midi).decode()}
    try:
        data = bandpack.build(result, source, band, provenance)
    except bandpack.BandPackError as e:
        raise HTTPException(422, str(e))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="jakgokmate_bandpack.zip"'})


# ---------------------------------------------------------------- band workspace: versions and members' own parts
def _bands() -> bands_mod.Bands:
    return bands_mod.Bands(DATA / "bands.db")


class BandReq(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    members: str | None = Field(None, max_length=300)
    guitarists: int | None = Field(None, ge=0, le=4)
    vocal_low: int | None = Field(None, ge=24, le=96)        # MIDI note numbers
    vocal_high: int | None = Field(None, ge=24, le=96)
    source_note: str | None = Field(None, max_length=1000)   # where the melody came from and on what terms (free text)


class BandVersionReq(BaseModel):
    result: dict
    author: str | None = Field(None, max_length=80)
    note: str | None = Field(None, max_length=1000)
    parent_id: int | None = None
    credits: dict[str, str] | None = None


class PartImportReq(BaseModel):
    result: dict
    part: str = Field(max_length=20)
    midi_b64: str = Field(min_length=1, max_length=3_000_000)
    author: str | None = Field(None, max_length=80)
    trim_tail: bool = False                                  # cut notes that ring past the song's end (default: refuse)


class PartSectionReq(BaseModel):
    result: dict
    part: str = Field(max_length=20)
    start_bar: int = Field(ge=0, le=2000)                    # bar index from 0
    bars: int = Field(ge=1, le=64)


class PartEditReq(PartSectionReq):
    notes: list[dict] = Field(max_length=4000)               # {start, dur, pitch, velocity}: ticks from the range start
    author: str | None = Field(None, max_length=80)


class PartKeepReq(BaseModel):
    result: dict                                             # the new version
    previous: dict                                           # the version that held the members' parts


@app.get("/api/bands")
def bands_list():
    return _bands().list_bands()


@app.post("/api/bands")
def bands_create(req: BandReq):
    try:
        band_id = _bands().create_band(**req.model_dump())
    except bands_mod.BandError as e:
        raise HTTPException(422, str(e))
    return _bands().get_band(band_id)


class BandProfileReq(BaseModel):
    profile: dict                            # bands.PROFILE_KEYS: mood, genre, tempo, lead, instruments, energy, ...


@app.put("/api/bands/{band_id}/profile")
def band_profile_set(band_id: int, req: BandProfileReq):
    """This band's own direction (feeling in its own words, genre, tempo, lead voice, line-up, gear, key change):
    what its next song starts from. It belongs to this band only and is never applied to another band."""
    try:
        return _bands().set_profile(band_id, req.profile)
    except bands_mod.UnknownBand:
        raise HTTPException(404, "unknown band")
    except bands_mod.BandError as e:
        raise HTTPException(422, str(e))


@app.get("/api/bands/{band_id}/versions")
def band_versions(band_id: int):
    try:
        return _bands().list_versions(band_id)
    except bands_mod.UnknownBand:
        raise HTTPException(404, "unknown band")


@app.post("/api/bands/{band_id}/versions")
def band_version_save(band_id: int, req: BandVersionReq):
    """Save the song as this band has it now. Nothing is overwritten: every save is a new version."""
    try:
        return _bands().save_version(band_id, req.result, req.author, req.note, req.parent_id, req.credits)
    except bands_mod.UnknownBand:
        raise HTTPException(404, "unknown band")
    except bands_mod.UnknownVersion:
        raise HTTPException(404, "unknown parent version")
    except bands_mod.BandError as e:
        raise HTTPException(422, str(e))


@app.get("/api/bands/versions/{version_id}")
def band_version_get(version_id: int):
    try:
        return _bands().get_version(version_id)
    except bands_mod.UnknownVersion:
        raise HTTPException(404, "unknown version")


@app.get("/api/bands/compare")
def band_compare(a: int, b: int):
    try:
        return _bands().compare(a, b)
    except bands_mod.UnknownVersion:
        raise HTTPException(404, "unknown version")
    except bands_mod.BandError as e:
        raise HTTPException(422, str(e))


@app.post("/api/part/import")
def part_import(req: PartImportReq):
    """Replace one part with a member's own MIDI (played or edited outside the app). The notes are kept as they are."""
    import base64
    import binascii
    from instmate import partswap
    try:
        midi = base64.b64decode(req.midi_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "midi_b64 is not valid base64")
    try:
        return partswap.replace_part(req.result, req.part, midi, req.author, req.trim_tail)
    except partswap.PartSwapError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")


class PartScoreReq(BaseModel):
    result: dict
    part: str = Field(max_length=20)
    start_bar: int = Field(0, ge=0, le=2000)
    bars: int | None = Field(None, ge=1, le=2000)            # None: to the end of the song


@app.post("/api/part/score")
def part_score(req: PartScoreReq):
    """One instrument's part as sheet music (MusicXML) for practising: read from the song's own MIDI, nothing changed."""
    from instmate import partscore
    try:
        xml = partscore.musicxml(req.result, req.part, req.start_bar, req.bars)
    except partscore.PartScoreError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")
    return {"part": req.part, "title": partscore.TITLE.get(req.part, req.part), "musicxml": xml}


class PartPrintReq(BaseModel):
    result: dict
    parts: list[str] = Field(min_length=1, max_length=8)
    start_bar: int = Field(0, ge=0, le=2000)
    bars: int | None = Field(None, ge=1, le=2000)
    title: str = Field("", max_length=80)
    tabs: bool = True                                        # tablature / drum chart pages after each part's sheet


@app.post("/api/part/engrave")
def part_engrave(req: PartScoreReq):
    """One part laid out on A4 pages by the engraving engine (one SVG per page), with its tablature / drum chart.
    501 when the engine is not installed: the page then draws its own simple sheet."""
    from instmate import partscore
    try:
        pages = partscore.engrave(req.result, req.part, req.start_bar, req.bars)
        tab = partscore.tab(req.result, req.part, req.start_bar, req.bars) \
            if req.part in partscore.TAB_PARTS or req.part == "drums" else None
    except partscore.EngraverMissing as e:
        raise HTTPException(501, str(e))
    except partscore.PartScoreError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")
    return {"part": req.part, "title": partscore.TITLE.get(req.part, req.part), "pages": pages,
            "tab_svg": partscore.tab_svg(tab) if tab else None, "tab_kind": tab["kind"] if tab else None,
            "tab_note": tab["note"] if tab else None}


@app.post("/api/part/print")
def part_print(req: PartPrintReq):
    """A complete A4 document for the browser's print (or "save as PDF"): title, part names, page numbers, the
    engraved sheets and the tablature pages of the wanted parts."""
    from instmate import partscore
    try:
        return partscore.print_html(req.result, req.parts, req.start_bar, req.bars, req.title, req.tabs)
    except partscore.EngraverMissing as e:
        raise HTTPException(501, str(e))
    except partscore.PartScoreError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")


@app.post("/api/part/tab")
def part_tab(req: PartScoreReq):
    """Tablature (guitar, lead guitar, bass) or a drum chart of a stretch, as data the page draws.  Fingerings are
    suggestions computed from the pitches."""
    from instmate import partscore
    try:
        return partscore.tab(req.result, req.part, req.start_bar, req.bars)
    except partscore.PartScoreError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")


@app.post("/api/part/section")
def part_section(req: PartSectionReq):
    """The notes of one part in a bar range, for the in-app editor (ticks: 480 per quarter note, from the range start)."""
    from instmate import partswap
    try:
        return partswap.section_notes(req.result, req.part, req.start_bar, req.bars)
    except partswap.PartSwapError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")


@app.post("/api/part/edit")
def part_edit(req: PartEditReq):
    """Replace the notes that start in a bar range of one part with the edited ones. Everything else in the part is
    kept as it is; applying unchanged notes returns {"changed": false} and the same song."""
    from instmate import partswap
    try:
        out = partswap.edit_section(req.result, req.part, req.start_bar, req.bars, req.notes, req.author)
    except partswap.PartSwapError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")
    return {"changed": out is not req.result, "result": out}


@app.post("/api/part/keep")
def part_keep(req: PartKeepReq):
    """After a new version of the song was made: carry the members' parts over when they still fit."""
    from instmate import partswap
    try:
        result, kept, dropped = partswap.keep_parts(req.result, req.previous)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"곡 결과가 온전하지 않아요: {e}")
    return {"result": result, "kept": kept, "dropped": dropped}


class YueReq(BaseModel):
    leadsheet: str = Field(min_length=1, max_length=200_000)   # the result's own lead sheet text (result.leadsheet)
    style: Literal["male", "female", "female_high"] = "female_high"   # E: the take chosen by ear on 2026-10-06
    genre: str | None = Field(None, max_length=40)
    full: bool = False                                         # False: the highlight (lead-in + first chorus)
    mode: Literal["demo", "release"] | None = None             # release: refused (the model's weights are CC BY-NC)


yue_jobs: dict[str, dict] = {}


@app.get("/api/yue/status")
def yue_status():
    """Whether the sung demo (YuE2, MLX port in models/yue2_mlx) can run here, and the voice presets."""
    from instmate import yue
    reason = yue.unavailable_reason()
    free = _free_memory_gb()
    return {"available": reason is None, "reason": reason,
            "styles": [{"id": k, "label": v["label"]} for k, v in yue.STYLE.items()],
            # the model peaks around 18 GB: the page warns before starting it when less than that can be freed
            "memory_free_gb": free, "memory_need_gb": YUE_NEED_GB, "memory_ok": free is None or free >= YUE_NEED_GB,
            # "a whole song from my bars" runs with a capped cache and a tiled decoder (instmate/yuesong.py)
            "song_need_gb": YUE_SONG_NEED_GB, "song_memory_ok": free is None or free >= YUE_SONG_NEED_GB,
            "song_need_gb_full": YUE_NEED_GB}


class ModelDownloadReq(BaseModel):
    accept_terms: bool = False


@app.get("/api/models/status")
def models_status():
    """First-run model download state (instmate/modelsetup.py); `enabled` is false on any server that is not a locally installed app."""
    from instmate import modelsetup, omr, yueblend
    # what this install can do beyond the basics, so the page can say so before a click fails (review F07)
    features = {"pdf": omr.unavailable_reason() is None, "pdf_reason": omr.unavailable_reason(),
                "practice_parts": yueblend.separator() is not None,
                "practice_reason": None if yueblend.separator() is not None else "보컬 분리 도구(Demucs)가 없어요. 보컬 연습만 돼요."}
    return {**modelsetup.status(), "features": features}


@app.post("/api/models/yue2/download")
def models_download(req: ModelDownloadReq):
    from instmate import modelsetup
    try:
        return modelsetup.start(req.accept_terms)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


YUE_NEED_GB = 18.0
YUE_SONG_NEED_GB = 8.0          # measured 6.5 GB at the peak for a 36 s take on 2026-10-12; longer takes were not measured


def _free_memory_gb() -> float | None:
    """Memory the system can hand out without swapping (macOS `vm_stat`: free + inactive + speculative + purgeable
    pages), in GB; None where it cannot be read (the page then does not warn)."""
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = sum(int(re.search(rf"{name}:\s+(\d+)", out).group(1))
                    for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
        return round(pages * page / 1e9, 1)
    except Exception:
        return None


@app.post("/api/yue")
def yue_start(req: YueReq):
    """Start a sung demo of the song's melody and chords (YuE2). One at a time, and not while a song is being
    composed: the model needs most of this machine's memory. Poll GET /api/yue/{job}."""
    from instmate import audio, instruments, yue
    from instmate.leadsheet import validate_leadsheet
    if not instruments.allowed("B", req.mode):
        raise HTTPException(409, "배포 모드에서는 AI 보컬 데모(YuE2, 비영리 조건)를 쓰지 않아요. 배포 모드를 끄면 쓸 수 있어요")
    reason = yue.unavailable_reason()
    if reason:
        raise HTTPException(503, reason)
    check = validate_leadsheet(req.leadsheet)
    if not check["ok"]:
        raise HTTPException(422, "; ".join(check["errors"][:3]))
    with _jobs_lock:
        if _busy() or any(j["status"] == "running" for j in yue_jobs.values()):
            raise HTTPException(409, "다른 생성 작업이 진행 중이에요. 끝난 뒤 다시 시도하세요")
        job_id = uuid.uuid4().hex[:12]
        job = {"status": "running", "progress": [], "info": None, "error": None, "audio": None, "format": None}
        yue_jobs[job_id] = job
        for old in [k for k, j in yue_jobs.items() if j["status"] != "running"][:-2]:
            del yue_jobs[old]

    def target():
        try:
            _llm_unload()                                  # qwen out of memory before YuE2 loads (user, 2026-10-19)
            wav, info = yue.generate(req.leadsheet, req.style, req.genre, req.full, job["progress"].append)
            if audio.ffmpeg_path():          # the demo was noticeably quieter than the app's own audio: same level now
                try:
                    wav = audio.match_loudness(wav)
                    info["loudness"] = f"{audio.MASTER_LUFS:g} LUFS"
                except (audio.AudioRenderError, audio.AudioUnavailable):
                    pass                     # the take itself is fine; it just keeps its own level
            job["audio"], job["format"] = (audio.to_mp3(wav), "mp3") if audio.ffmpeg_path() else (wav, "wav")
            job["info"] = info
            job["status"] = "done"
        except Exception as exc:
            job["error"] = str(exc)
            job["status"] = "error"

    threading.Thread(target=target, daemon=True).start()
    return {"job": job_id}


class YueSongReq(BaseModel):
    leadsheet: str | None = Field(None, min_length=1, max_length=200_000)   # the song's lead sheet (not for stage "full")
    seed_section: int | None = Field(None, ge=0, le=63)        # the one given section (needed with 2+ sections) ...
    seed_sections: list[int] | Literal["all"] | None = None    # ... or several, in the song's order, or the whole song
    ins_sections: list[int] = Field(default_factory=list)      # of the given ones (0 = the first given): an instrument plays
    continue_song: bool = True                                 # False: sing and play the given part as it is
    genre: str | None = Field(None, max_length=40)
    voice: Literal["female", "mezzo", "female_low", "male_low"] = "female"   # yuesong.VOICES
    key_shift: int = Field(0, ge=-4, le=0)                     # the finished take lowered by semitones
    low_memory: bool = True                                    # about 6 GB at the peak; off: about 18 GB
    lyrics: str | None = Field(None, max_length=4000)          # none: every line is sung on "아"
    lyrics_mode: Literal["ah", "model"] = "ah"                 # model: no words given, YuE2 writes its own
    extra_sections: list[Literal["Chorus", "Bridge", "Outro"]] = Field(default_factory=list, max_length=4)   # parts added to the plan
    seed: int = Field(1, ge=0, le=2_000_000_000)
    max_seconds: float | None = Field(None, ge=20.0, le=900.0)  # none: 90 s when continued, the song's length as given
    stage: Literal["all", "first", "full"] = "all"             # first: the best of `plans` scores, sung to the end of the
    plans: int = Field(3, ge=1, le=8)                          # first chorus; full: the whole song from that `score`
    score: str | None = Field(None, min_length=1, max_length=200_000)
    target_seconds: float = Field(210.0, ge=30.0, le=900.0)
    check: bool = True                                         # stage "first": compare the take with its score, sing again if off
    title: str = Field("", max_length=80)
    parent: str | None = Field(None, max_length=40)            # the library id of the first part a whole song grew from
    mode: Literal["demo", "release"] | None = None


@app.post("/api/yue/song")
def yue_song(req: YueSongReq):
    """Start "a whole song from my bars" (instmate/yuesong.py): YuE2 continues a score that starts with sections of the
    user's song, sings and plays it, and the score it wrote comes back as the app's lead sheet with the app's own
    arrangement of it (`info.result`).  stage "first" sings only the first part and asks whether to go on; stage
    "full" goes on from that part's `score` to the whole song.  Every finished song is kept in the library.
    Same job list and the same polling / audio paths as POST /api/yue."""
    from instmate import audio, instruments, yue, yueimport, yuelib, yuesong
    from instmate.leadsheet import validate_leadsheet
    from instmate.render import leadsheet_to_song
    if not instruments.allowed("B", req.mode):
        raise HTTPException(409, "배포 모드에서는 YuE2(비영리 조건)를 쓰지 않아요. 배포 모드를 끄면 쓸 수 있어요")
    reason = yue.unavailable_reason()
    if reason:
        raise HTTPException(503, reason)
    seed_sheet, sheet_genre = None, "팝"
    if req.stage == "full":
        if not req.score:
            raise HTTPException(422, "전곡을 만들려면 1절의 악보(score)가 필요해요")
        try:
            yueimport.read(req.score)
        except yueimport.AbcError as exc:
            raise HTTPException(422, f"1절 악보를 읽을 수 없어요: {exc}") from None
    else:
        if not req.leadsheet:
            raise HTTPException(422, "리드시트가 필요해요")
        if req.stage == "first" and not req.continue_song:
            raise HTTPException(422, "1절 만들기는 이어 쓰기에서만 돼요")
        check = validate_leadsheet(req.leadsheet)
        if not check["ok"]:
            raise HTTPException(422, "; ".join(check["errors"][:3]))
        try:
            if req.seed_sections is not None and req.seed_section is not None:
                raise yuesong.YueSongError("seed_section과 seed_sections 중 하나만 보내 주세요")
            seed_sheet = yuesong.pick_seed(req.leadsheet, req.seed_sections if req.seed_sections is not None else req.seed_section)
            yuesong.seed_prefix(seed_sheet, req.ins_sections, open_end=req.continue_song)
        except yuesong.YueSongError as exc:
            raise HTTPException(422, str(exc)) from None
        sheet_genre = seed_sheet.genre
    max_seconds = req.max_seconds if req.max_seconds is not None or not req.continue_song else yuesong.MAX_SECONDS
    with _jobs_lock:
        if _busy() or any(j["status"] == "running" for j in yue_jobs.values()):
            raise HTTPException(409, "다른 생성 작업이 진행 중이에요. 끝난 뒤 다시 시도하세요")
        job_id = uuid.uuid4().hex[:12]
        job = {"status": "running", "progress": [], "info": None, "error": None, "audio": None, "format": None}
        yue_jobs[job_id] = job
        for old in [k for k, j in yue_jobs.items() if j["status"] != "running"][:-2]:
            del yue_jobs[old]
    yuesong._cancel.clear()
    more = {**({"lyrics_mode": req.lyrics_mode} if req.lyrics_mode != "ah" else {}),
            **({"extra": req.extra_sections} if req.extra_sections else {})}     # only what was asked: the defaults are the old call

    def target():
        try:
            _llm_unload()                                  # qwen out of memory before YuE2 loads (user, 2026-10-19)
            say = job["progress"].append
            if req.stage == "full":
                wav, info = yuesong.make_full(req.score, genre=req.genre, voice=req.voice, lyrics=req.lyrics, seed=req.seed,
                                              target_seconds=req.target_seconds, progress=say, sheet_genre=sheet_genre, plans=req.plans,
                                              key_shift=req.key_shift, low_memory=req.low_memory, **more)
            elif req.stage == "first":
                wav, info = yuesong.make_first(seed_sheet, genre=req.genre, voice=req.voice, lyrics=req.lyrics, seed=req.seed,
                                               plans=req.plans, ins_sections=req.ins_sections, progress=say, check=req.check,
                                               key_shift=req.key_shift, low_memory=req.low_memory, **more)
            else:
                wav, info = yuesong.make(seed_sheet, genre=req.genre, voice=req.voice, lyrics=req.lyrics, seed=req.seed,
                                         max_seconds=max_seconds, progress=say,
                                         ins_sections=req.ins_sections, continue_song=req.continue_song,
                                         key_shift=req.key_shift, low_memory=req.low_memory, **more)
            result = leadsheet_to_song(info["leadsheet"])      # chords, parts and edits now work on the song YuE2 wrote
            result.setdefault("leadsheet", info["leadsheet"])
            for i, sec in enumerate(result["sections"]):       # the given bars open the first section; the rest is YuE2's
                sec["origin"] = "input" if i == 0 else "model"
            if req.stage == "full":                            # the given bars were checked in the first part: carry that verdict
                parent = {}
                try:
                    parent = yuelib.get(DATA, req.parent) if req.parent else {}
                except KeyError:
                    pass
                info["given_bars"] = parent.get("given_bars")
                info["given_kept"] = bool(parent.get("given_kept")) and info.get("first_kept", False) if parent else None
            given_at = (req.seed_sections[0] if isinstance(req.seed_sections, list) and req.seed_sections else req.seed_section or 0)
            result["yue_song"] = {"given_section": given_at, "given_bars": info.get("given_bars"),
                                  **{k: info[k] for k in ("target_seconds", "target_bars", "length_met") if k in info},
                                  **({"continued": False} if not req.continue_song else {}),
                                  **({"cut": True, "wanted_seconds": info.get("wanted_seconds")} if info.get("cut") else {}),
                                  "given_kept": info.get("given_kept"), "lyrics": info["lyrics"],
                                  "seconds": info["seconds"], "stage": req.stage}
            info["result"] = result
            if info.get("given_kept") is not True or info.get("first_kept") is False:
                info["warning"] = (info.get("warning", "") + " 주어진 마디가 악보에서 다르게 돌아왔어요. 악보를 확인해 주세요.").strip()
            if req.stage == "first":
                info["ask"] = "이 곡으로 전곡을 만들까요?"
            job["audio"], job["format"] = (audio.to_mp3(wav), "mp3") if audio.ffmpeg_path() else (wav, "wav")
            info["library_id"] = yuelib.save(DATA, job["audio"], job["format"], info["abc"], {
                "title": req.title, "stage": req.stage, "parent": req.parent, "seconds": info["seconds"], "bars": info["bars"],
                "sections": info["sections"], "key": result["key"]["label"], "tempo": result.get("tempo"), "voice": req.voice,
                "genre": req.genre, "sheet_genre": sheet_genre, "lyrics": info["lyrics"], "lyrics_text": req.lyrics or "",
                "sounding_key": info.get("sounding_key"), "male_shift": info.get("male_shift", 0),
                "follow": (info.get("follow") or {}).get("match"), "given_bars": info.get("given_bars"),
                "given_kept": info.get("given_kept"),
                **{k: info[k] for k in ("target_seconds", "target_bars", "length_met", "warning") if k in info}})
            job["info"] = info
            job["status"] = "done"
        except yuesong.YueCancelled as exc:
            job["error"] = str(exc)
            job["status"] = "cancelled"
        except Exception as exc:
            job["error"] = str(exc)
            job["status"] = "error"

    threading.Thread(target=target, daemon=True).start()
    return {"job": job_id}


@app.get("/api/yue/library")
def yue_library():
    """The songs YuE2 made here, newest first."""
    from instmate import yuelib
    return {"songs": yuelib.entries(DATA)}


@app.get("/api/yue/library/{song_id}")
def yue_library_song(song_id: str):
    """One library song: what was saved, the score YuE2 wrote and the app's arrangement of it (made on request)."""
    from instmate import yueimport, yuelib
    from instmate.leadsheet import serialize_leadsheet
    from instmate.render import leadsheet_to_song
    try:
        row = yuelib.get(DATA, song_id)
    except KeyError:
        raise HTTPException(404, "unknown song") from None
    try:
        text = serialize_leadsheet(yueimport.read(row["abc"], row.get("sheet_genre") or "팝")[0])
    except yueimport.AbcError as exc:                  # a damaged saved score: say so instead of a server error
        raise HTTPException(422, f"저장된 악보를 읽을 수 없어요: {exc}") from None
    result = leadsheet_to_song(text)
    result.setdefault("leadsheet", text)
    result["yue_song"] = {"library_id": song_id, "stage": row.get("stage"), "seconds": row.get("seconds"),
                          **{k: row[k] for k in ("target_seconds", "target_bars", "length_met") if k in row}}
    return {**row, "result": result}


@app.get("/api/yue/library/{song_id}/minus")
def yue_library_minus(song_id: str, part: str):
    """Practice track: the song's score played by the app's band without my `part`, with YuE2's own voice on top
    (no voice when my part is the vocal).  The voice is split from the take once and kept beside the song."""
    from instmate import audio, yuelib, yuepractice
    if part not in yuepractice.PARTS:
        raise HTTPException(422, f"내 파트는 {', '.join(yuepractice.PARTS)} 중 하나여야 해요")
    try:
        yuelib.get(DATA, song_id)
    except KeyError:
        raise HTTPException(404, "unknown song") from None
    if part != "vocal":
        reason = yuepractice.available()
        if reason:
            raise HTTPException(503, reason)
    wav, info = yuepractice.minus(DATA, song_id, part)
    data, fmt = (audio.to_mp3(wav), "mp3") if audio.ffmpeg_path() else (wav, "wav")
    return Response(data, media_type=audio.MEDIA[fmt], headers={"X-Instmate-Removed": part,
                    "X-Instmate-Practice": quote(json.dumps(info, ensure_ascii=False))})


@app.get("/api/yue/library/{song_id}/audio")
def yue_library_audio(song_id: str):
    from instmate import audio, yuelib
    try:
        data, fmt = yuelib.audio(DATA, song_id)
    except KeyError:
        raise HTTPException(404, "unknown song") from None
    return Response(audio.yue_attributed(data, fmt), media_type=audio.MEDIA[fmt],
                    headers={"Content-Disposition": f'inline; filename="{audio.YUE_FILENAME.format(ext=fmt)}"'})


class YueOverlayReq(BaseModel):
    leadsheet: str = Field(min_length=1, max_length=200_000)   # the result's own lead sheet text
    genre: str | None = Field(None, max_length=40)
    instruments: list[str] | None = None
    mode: Literal["demo", "release"] | None = None
    full: bool = False                                         # the whole song (up to 300 s) instead of the highlight
    lyrics: str | None = Field(None, max_length=4000)          # typed words (any language); none = "la" syllables


@app.post("/api/yue/overlay")
def yue_overlay(req: YueOverlayReq):
    """Start "AI vocal on the app's band" (instmate/yueblend.py): YuE2 sings the highlight five semitones lower, the
    voice alone is separated and laid over the app's own accompaniment in that key.  Same job list and the same
    polling / audio paths as POST /api/yue; one job at a time, never while a song is being composed."""
    from instmate import audio, instruments, yueblend
    from instmate.leadsheet import validate_leadsheet
    if not instruments.allowed("B", req.mode):
        raise HTTPException(409, "배포 모드에서는 AI 보컬(YuE2, 비영리 조건)을 쓰지 않아요. 배포 모드를 끄면 쓸 수 있어요")
    reason = yueblend.unavailable_reason()
    if reason:
        raise HTTPException(503, reason)
    check = validate_leadsheet(req.leadsheet)
    if not check["ok"]:
        raise HTTPException(422, "; ".join(check["errors"][:3]))
    with _jobs_lock:
        if _busy() or any(j["status"] == "running" for j in yue_jobs.values()):
            raise HTTPException(409, "다른 생성 작업이 진행 중이에요. 끝난 뒤 다시 시도하세요")
        job_id = uuid.uuid4().hex[:12]
        job = {"status": "running", "progress": [], "info": None, "error": None, "audio": None, "format": None}
        yue_jobs[job_id] = job
        for old in [k for k, j in yue_jobs.items() if j["status"] != "running"][:-2]:
            del yue_jobs[old]

    def target():
        try:
            _llm_unload()                                  # qwen out of memory before YuE2 loads (user, 2026-10-19)
            wav, info = yueblend.overlay(req.leadsheet, req.genre, req.instruments, req.mode, job["progress"].append,
                                         full=req.full, lyrics=req.lyrics)
            job["audio"], job["format"] = (audio.to_mp3(wav), "mp3") if audio.ffmpeg_path() else (wav, "wav")
            job["info"] = info
            job["status"] = "done"
        except Exception as exc:
            job["error"] = str(exc)
            job["status"] = "error"

    threading.Thread(target=target, daemon=True).start()
    return {"job": job_id}


@app.get("/api/yue/{job_id}")
def yue_poll(job_id: str):
    job = yue_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {k: job[k] for k in ("status", "progress", "info", "error", "format")}


@app.post("/api/yue/{job_id}/cancel")
def yue_cancel(job_id: str):
    """Stop a running take (the runner is killed within a second); the job ends as "cancelled"."""
    from instmate import yuesong
    job = yue_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "알 수 없는 작업이에요")
    if job["status"] != "running":
        raise HTTPException(409, "이미 끝난 작업이라 취소할 수 없어요")
    yuesong.request_cancel()
    return {"ok": True}


@app.get("/api/yue/{job_id}/audio")
def yue_audio(job_id: str):
    job = yue_jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "no audio for this job")
    from instmate import audio
    return Response(audio.yue_attributed(job["audio"], job["format"]), media_type=audio.MEDIA[job["format"]],
                    headers={"Content-Disposition": f'attachment; filename="{audio.YUE_FILENAME.format(ext=job["format"])}"'})


@app.post("/api/mix")
def mix(req: MixReq):
    """Change tempo and/or track levels of the current version (adopted as a new version, undoable)."""
    try:
        base = versions.head(req.history)
    except UnknownHistory:
        raise HTTPException(404, "unknown history")
    if base["index"] != req.base_index:
        raise HTTPException(409, "version changed; reload it first")
    try:
        result = agent.remix(base["state"], base["result"], req.tempo, req.levels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        return _with_version(result, versions.adopt(req.history, base["index"], base["state"], result))
    except StaleBase as e:
        raise HTTPException(409, str(e))


@app.post("/api/undo")
def undo(req: UndoReq):
    try:
        v = versions.undo(req.history)
    except UnknownHistory:
        raise HTTPException(404, "unknown history")
    except NothingToUndo as e:
        raise HTTPException(409, str(e))
    return _with_version(v["result"], v["info"])


@app.get("/api/versions/{history}")
def list_versions(history: str):
    try:
        return versions.listing(history)
    except UnknownHistory:
        raise HTTPException(404, "unknown history")


@app.post("/api/prefs/clear")
def clear_prefs():
    prefs.clear()
    return {"status": "ok"}


# ---------------------------------------------------------------- M4: compose (background job, one at a time)
compose_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_workspace_model_busy = False
MAX_KEPT_JOBS = 20


def make_generator():
    """The lead sheet model, or None when it cannot run (no adapter / no mlx / base model not cached).
    Tests replace this function; compose then falls back to the rule draft for every section."""
    g = composer.default_generator()
    return g if g.available() else None


def _busy() -> bool:
    return _workspace_model_busy or any(j["status"] == "running" for j in compose_jobs.values())


def _start_job(work) -> str:
    """Run `work(progress_callback) -> result` in a thread. Only one job runs at a time (409)."""
    with _jobs_lock:
        if _busy():
            raise HTTPException(409, "다른 곡 생성이 진행 중이에요. 끝난 뒤 다시 시도하세요")
        job_id = uuid.uuid4().hex[:12]
        job = {"status": "running", "progress": [], "result": None, "error": None, "created": time.time()}
        compose_jobs[job_id] = job
        for old in [k for k, j in compose_jobs.items() if j["status"] != "running"][:-MAX_KEPT_JOBS]:
            del compose_jobs[old]

    def target():
        try:
            job["result"] = work(job["progress"].append)
            job["status"] = "done"
        except Exception as exc:          # the job must always end in done or error
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["status"] = "error"

    threading.Thread(target=target, daemon=True).start()
    return job_id


def _section_sources(leadsheet_c: str, stats: dict, previous: list[str] | None = None,
                     seed_role: str = "chorus") -> list[str]:
    """One label per section of the song, in order: "seed" (the section that carries the user's melody: the 후렴,
    or the 벌스 / first 인트로 when the input was placed there), "model" or "rule".
    A repeated section shows the source of its first occurrence. After a one-section rewrite `previous` (the parent
    song's labels) supplies the sections that were not part of the rewrite."""
    from instmate.leadsheet import parse_leadsheet
    sheet = parse_leadsheet(leadsheet_c)
    by_uid = {s["index"]: s["source"] for s in stats["sections"]}
    out = []
    seed_name = {"chorus": "후렴", "verse": "벌스", "intro": "인트로"}[seed_role or "chorus"]
    for i, uid in enumerate(composer.groups_of(sheet)):
        if sheet.sections[i].name == seed_name and (seed_role != "intro" or i == 0):
            out.append("seed")
        elif uid in by_uid:
            out.append(by_uid[uid])
        elif previous is not None and i < len(previous):
            out.append(previous[i])
        else:
            out.append("rule")
    return out


def _label_similar_parts(res: dict) -> None:
    """The similar-song report lists the parts with the final per-section source labels (model / rule)."""
    rep = res.get("similar_check")
    if not rep or not rep.get("available"):
        return
    from instmate.leadsheet import parse_leadsheet
    first = {}
    for sec, src in zip(parse_leadsheet(res["leadsheet_c"]).sections, res["section_sources"]):
        first.setdefault(sec.name, src)
    for part in rep["ai"]["parts"]:
        part["source"] = first.get(part["name"], part["source"])
    res["compose_stats"]["similar_check"] = rep


from instmate.genres import PROFILES as _GENRE_PROFILES  # noqa: E402
PROFILES_KO = {g: p["label"] for g, p in _GENRE_PROFILES.items()}
OWN_FROM = "모델을 쓸 수 없어 모든 구간을 규칙 기반으로 만들었어요."
OWN_TO = "'내 멜로디 변형만' 방식이라 학습 모델을 쓰지 않고, 후렴 멜로디를 규칙으로 변형해 모든 구간을 만들었어요."


class ComposeReq(BaseModel):
    tokens: list[dict]
    input_tonic: int = Field(0, ge=0, le=11)
    key_choice: KeyChoice | None = None
    genre: str | None = None                 # None: the confirmed style's genre, else ballad
    instruments: list[str] | None = None
    energy: int = Field(0, ge=0, le=1)
    tempo: int | None = None
    levels: dict[str, float] | None = None
    style: dict | None = None                # an already confirmed StyleAttrs dict (confirmation happens in the UI)
    candidates: int = Field(composer.CANDIDATES_DEFAULT, ge=1, le=10)
    max_rounds: int = Field(composer.MAX_ROUNDS_DEFAULT, ge=1, le=3)
    form: Literal["standard", "chorus_first"] = "standard"   # chorus_first: the hook right after the intro (C-A-B)
    # last chorus a semitone up.  False (default): the input melody stays as written in every chorus.  "genre": the
    # genre's own habit, the behaviour before 2026-10-06, kept only for reproducing older results.
    key_change: bool | Literal["genre"] = False
    melody_role: Literal["vocal", "lead_guitar", "acoustic_guitar", "keys"] = "vocal"   # who plays the melody (render.MELODY_ROLES)
    seed_role: Literal["chorus", "verse", "intro"] = "chorus"   # which section the input melody is
    # an intro melody played as a lead-guitar riff instead of sung (instmate/introlead.py); only with seed_role intro
    intro_player: Literal["vocal", "guitar"] = "vocal"
    fill: Literal["own", "rest"] = "own"                     # without the model: rule variations, or no melody elsewhere
    # who writes the melody outside the chorus: "ai" = the lead sheet model (practice / class use: its training data is
    # non-commercial), "own" = only rule-made variations of the user's own seed melody (no trained model involved)
    melody_source: Literal["ai", "own"] = "ai"
    seed_chords: list[str | None] | None = Field(None, max_length=8)   # chords chosen for the seed's bars (kept in the chorus)
    # move the whole melody (and key, and chosen chords) into this singer's range (instmate/voicefit.py); the request
    # itself stays in the key the user typed.  None: the key is not touched.
    fit_voice: Literal["male", "female"] | None = None
    melody_shape: bool = False               # option: open pre-chorus ending, a copied bridge written anew (melodyshape.py)
    # the second voice written under the input melody (instmate/harmony.py): entries relative to the seed's first tick
    harmony: list[dict] | None = Field(None, max_length=2000)
    seed_ticks: int | None = Field(None, ge=1, le=8 * 1920)


def _harmony_report(result: dict) -> dict | None:
    """The harmony diagnosis of a finished song (instmate/harmonyreport.py); a report problem never costs the user
    the song."""
    from instmate import harmonyreport
    try:
        return harmonyreport.report(result)
    except Exception as e:                               # a diagnosis must not break song making
        return {"notes": [], "summary": {"total": 0}, "about": f"화성 점검을 만들지 못했어요 ({type(e).__name__})."}


@app.post("/api/compose")
def compose_start(req: ComposeReq):
    """Seed melody -> 2+ minute song with the lead sheet model (candidates, scoring, regeneration; rule fallback).
    Returns {job_id}; poll GET /api/compose/{job_id}."""
    from instmate.mix import resolve as resolve_mix
    from instmate.theory import parse_number_input
    if _busy():
        raise HTTPException(409, "다른 곡 생성이 진행 중이에요. 끝난 뒤 다시 시도하세요")
    if req.melody_source == "ai" and (req.seed_role != "chorus" or req.fill == "rest"):
        raise HTTPException(422, "입력 멜로디를 인트로·벌스에 놓거나 나머지를 비워 둘 때는 AI로 채우기를 쓸 수 없어요. "
                                 "멜로디 출처를 \"내 멜로디 변형만\"으로 골라 주세요.")
    voice_fit = None
    try:                                     # reject bad requests now instead of inside the job
        parse_number_input(req.tokens, req.input_tonic)
        if req.fit_voice and req.melody_role == "vocal":
            from instmate import voicefit
            from instmate.theory import analyze_key
            # an unsettled key is asked about first, in the key the user typed; the move happens once it is known
            if req.key_choice or not analyze_key(parse_number_input(req.tokens, req.input_tonic), req.input_tonic)["ambiguous"]:
                tokens, tonic, choice, seed_chords, voice_fit = voicefit.fit_tokens(
                    req.tokens, req.input_tonic, req.fit_voice,
                    req.key_choice.model_dump() if req.key_choice else None, req.seed_chords)
                req = req.model_copy(update={"tokens": tokens, "input_tonic": tonic, "seed_chords": seed_chords,
                                             "key_choice": KeyChoice(**choice) if choice else None})
        applied = style_mod.apply_style(req.style) if req.style else None     # clamps; never raises for odd values
        genre = req.genre or (applied["genre"] if applied else "ballad")
        agent._validate(genre, req.instruments or (applied["instrument_list"] if applied else list(BAND)))
        resolve_mix(genre, req.tempo if req.tempo is not None else (applied["bpm"] if applied else None), req.levels)
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(400, str(e))

    voicefit_range = None
    if voice_fit:                            # the sections the app writes follow the singer too, not only the input
        from instmate import voicefit as _vf
        voicefit_range = _vf.RANGES[req.fit_voice]

    def work(progress):
        # a confirmed style's dark/bright hint steers the rule-draft chords (model-written sections keep their own chords)
        chord_bias = None
        if applied and applied["chord_bias"]["direction"]:
            from instmate.theory import Key, analyze_key
            if req.key_choice:
                key = Key(int(req.key_choice.tonic), req.key_choice.mode)
            else:
                top = analyze_key(parse_number_input(req.tokens, req.input_tonic), req.input_tonic)["candidates"][0]["key"]
                key = Key(top["tonic"], top["mode"])
            chord_bias = style_mod.chord_bias_for_key(key, applied["chord_bias"]) or None
        res = composer.compose(req.tokens, req.input_tonic, req.key_choice.model_dump() if req.key_choice else None,
                               req.genre, req.tempo, req.levels, req.style, req.candidates, req.max_rounds,
                               generator=make_generator() if req.melody_source == "ai" else None, progress=progress,
                               instruments=req.instruments, energy=req.energy, prefs=prefs, bias=chord_bias,
                               ref_index="auto", form=req.form, seed_chords=req.seed_chords,
                               key_change=None if req.key_change == "genre" else req.key_change,
                               melody_role=req.melody_role, seed_role=req.seed_role, fill=req.fill,
                               voice_range=voicefit_range, shape=req.melody_shape)
        if res.get("status") == "need_key":
            return res
        res["melody_source"] = res["compose_stats"]["melody_source"] = req.melody_source
        if voice_fit:                        # the key was moved for the singer: part of the song's record
            res["voice_fit"] = res["settings"]["voice_fit"] = voice_fit
        if req.melody_source == "own":       # not a fallback: the user chose to keep the trained model out
            for key, value in list(res.items()):
                if isinstance(value, str) and OWN_FROM in value:
                    res[key] = value.replace(OWN_FROM, OWN_TO)
        res["section_sources"] = _section_sources(res["leadsheet_c"], res["compose_stats"], seed_role=req.seed_role)
        res["compose_stats"]["section_sources"] = res["section_sources"]
        res["harmony_report"] = _harmony_report(res)
        _label_similar_parts(res)
        res["song_id"] = _history().add_song(req.tokens, res["settings"], res["leadsheet_c"], res["compose_stats"])
        if req.seed_role == "intro" and req.intro_player == "guitar" and req.melody_role == "vocal":
            from instmate import introlead
            res = introlead.to_lead(res)             # after every check: the notes are the same, only the track differs
            res["settings"]["intro_player"] = "guitar"
        if req.harmony:                              # kept with the song so that a rewritten section keeps the voice
            res = _second_voice(res, req.harmony, req.seed_ticks)
            res["settings"]["harmony"] = {"entries": req.harmony, "period": req.seed_ticks}
        return res

    return {"job_id": _start_job(work)}


@app.get("/api/compose/{job_id}")
def compose_status(job_id: str):
    job = compose_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    body = {"status": job["status"], "progress": list(job["progress"])}
    if job["status"] == "done":
        body["result"] = job["result"]
    if job["status"] == "error":
        body["error"] = job["error"]
    return body


class SectionReq(BaseModel):
    section_index: int = Field(ge=0)
    candidates: int | None = Field(None, ge=1, le=10)
    max_rounds: int | None = Field(None, ge=1, le=3)


@app.post("/api/compose/{song_id}/section")
def compose_section(song_id: int, req: SectionReq):
    """Rewrite one section (same generate/score/regenerate loop with a single section) of a stored song.
    A repeated section is rewritten as a group; the chorus (the seed) cannot be rewritten (400).
    Returns {job_id}; the new song is stored with parent_id = song_id."""
    from instmate.leadsheet import parse_leadsheet
    if _busy():
        raise HTTPException(409, "다른 곡 생성이 진행 중이에요. 끝난 뒤 다시 시도하세요")
    try:
        rec = _history().get_song(song_id)
    except UnknownSong:
        raise HTTPException(404, "unknown song")
    sections = parse_leadsheet(rec["leadsheet"]).sections
    if req.section_index >= len(sections):
        raise HTTPException(400, f"section_index must be 0-{len(sections) - 1}")
    if sections[req.section_index].name == "후렴":
        raise HTTPException(400, "후렴은 씨앗 멜로디라 다시 만들지 않아요")
    if rec["compose_stats"].get("melody_source") == "own":
        raise HTTPException(400, "'내 멜로디 변형만' 방식의 곡은 AI로 다시 쓰지 않아요. 악보를 직접 고치거나 'AI 멜로디 제안'으로 새로 만들어 주세요")

    def work(progress):
        res = composer.regenerate_section(rec["settings"], rec["leadsheet"], req.section_index, make_generator(),
                                          progress, req.candidates, req.max_rounds, ref_index="auto")
        res["section_sources"] = _section_sources(res["leadsheet_c"], res["compose_stats"],
                                                  rec["compose_stats"].get("section_sources"))
        res["compose_stats"]["section_sources"] = res["section_sources"]
        # the rewrite's own statistics name one section only: every section's origin comes from the labels above,
        # which carry the parent song's for the rest (the page draws the cover from the section marked "input")
        composer.label_origins(res, (rec["settings"] or {}).get("seed_role") or "chorus", list(res["section_sources"]))
        res["harmony_report"] = _harmony_report(res)
        _label_similar_parts(res)
        res["song_id"] = _history().add_song(rec["seed"], res["settings"], res["leadsheet_c"], res["compose_stats"],
                                          parent_id=song_id)
        kept = (rec["settings"] or {}).get("harmony")
        if kept:
            res = _second_voice(res, kept.get("entries"), kept.get("period"))
        return res

    return {"job_id": _start_job(work)}


class RateReq(BaseModel):
    song_id: int
    section: int | None = Field(None, ge=0)        # index into the song's sections; None = the whole song
    rating: Literal[-1, 1]


@app.post("/api/rate")
def rate(req: RateReq):
    from instmate.leadsheet import parse_leadsheet
    try:
        rec = _history().get_song(req.song_id)
    except UnknownSong:
        raise HTTPException(404, "unknown song")
    if req.section is not None and req.section >= len(parse_leadsheet(rec["leadsheet"]).sections):
        raise HTTPException(400, "section out of range")
    return {"status": "ok", "rating_id": _history().rate(req.song_id, req.rating, req.section)}


# ---------------------------------------------------------------- M6: style request and reference audio
def make_style_planner():
    """Planner for POST /api/style: None = local Ollama (style.OllamaStylePlanner), False = rules only.
    INSTMATE_STYLE_LLM=off forces rules (no network at all). Without a reachable Ollama that has the model, or with
    INSTMATE_LLM=gemini (no style planner for it), the rules answer at once instead of waiting for a timeout.
    Tests replace this function."""
    if os.environ.get("INSTMATE_STYLE_LLM", "auto") == "off" or os.environ.get("INSTMATE_LLM", "ollama") == "gemini":
        return False
    from instmate import llm
    return False if llm.ollama_status() else None


class StyleReq(BaseModel):
    text: str = Field(min_length=1, max_length=300)


class StyleApplyReq(BaseModel):
    attrs: dict


@app.post("/api/style")
def style_request(req: StyleReq):
    """Free text ("성시경 희재 같은 느낌") -> style attributes + a confirmation question. Nothing is applied here."""
    try:
        return style_mod.style_from_request(req.text, planner=make_style_planner())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/style/apply")
def style_apply(req: StyleApplyReq):
    """The user confirmed: attrs -> concrete settings (genre, bpm, instruments, mood, chord_bias). Validated by code."""
    attrs, notes = style_mod.validate_attrs(req.attrs, str(req.attrs.get("source", "")))
    return {"attrs": attrs.to_dict(), "settings": style_mod.apply_style(attrs), "notes": notes}


UPLOADS = DATA / "uploads"


@app.post("/api/reference")
async def reference_upload(request: Request):
    """multipart/form-data with one audio file (<= 30 MB). Streamed to DATA/uploads, analysed, ALWAYS deleted.
    Returns {analysis (no pitch arrays), suggestion (style proposal, needs confirmation), deleted: true}."""
    length = request.headers.get("content-length")
    try:
        path = await upload.save_first_file(
            request.stream(), request.headers.get("content-type"), UPLOADS, refaudio.MAX_BYTES, refaudio.ALLOWED_EXT,
            int(length) if length and length.isdigit() else None)
    except upload.UploadError as e:
        raise HTTPException(e.status, str(e))
    try:
        analysis = await run_in_threadpool(refaudio.analyze_reference, path, True)
    except refaudio.RefAudioError as e:
        raise HTTPException(400, str(e))
    finally:
        path.unlink(missing_ok=True)             # analyze_reference deletes already; this covers any other failure
    if not analysis.get("deleted"):
        raise HTTPException(500, "업로드한 파일을 삭제하지 못했어요. 파일이 서버에 남았을 수 있으니 data/uploads를 확인해 주세요.")
    return {"analysis": analysis, "suggestion": refaudio.reference_to_style(analysis), "deleted": True}


@app.post("/api/hum")
async def hum_upload(request: Request, bpm: int = 90):
    """multipart/form-data with one recording (<= 10 MB) of the user humming or singing a melody -> number-notation
    tokens for the melody input (instmate.humtrans). The recording is converted, analysed and deleted at once."""
    from instmate import audio, humtrans
    import subprocess
    length = request.headers.get("content-length")
    try:
        path = await upload.save_first_file(request.stream(), request.headers.get("content-type"), UPLOADS,
                                            10 * 1024 * 1024, VOICE_EXT,
                                            int(length) if length and length.isdigit() else None)
    except upload.UploadError as e:
        raise HTTPException(e.status, str(e))
    wav = path.with_suffix(".hum.wav")
    try:
        ff = audio.ffmpeg_path()
        if ff is None:
            raise HTTPException(503, "녹음을 변환하려면 ffmpeg가 필요해요")
        proc = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(path),
                               "-ac", "1", "-ar", "16000", str(wav)], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not wav.exists():
            raise HTTPException(400, "녹음 파일을 읽지 못했어요")
        try:
            return await run_in_threadpool(humtrans.transcribe, wav.read_bytes(), bpm)
        except humtrans.HumError as e:
            raise HTTPException(400, str(e))
    finally:
        path.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


VOICE_DIR = DATA / "voice"
VOICE_FILE = VOICE_DIR / "hum.wav"
VOICE_EXT = {".wav", ".webm", ".ogg", ".m4a", ".mp3", ".aiff", ".aif", ".mp4"}


@app.get("/api/voice")
def voice_status():
    """Whether a hum recording is stored (the user's own voice, kept only in DATA/voice/)."""
    if not VOICE_FILE.exists():
        return {"ready": False}
    import soundfile as sf
    info = sf.info(str(VOICE_FILE))
    return {"ready": True, "seconds": round(info.frames / info.samplerate, 1)}


@app.post("/api/voice")
async def voice_upload(request: Request):
    """multipart/form-data with one recording (<= 10 MB) of the user humming: converted to WAV (ffmpeg), checked
    (instmate.vocal.load_voice) and stored as DATA/voice/hum.wav, replacing an earlier one."""
    from instmate import audio, vocal as vocal_mod
    import subprocess
    length = request.headers.get("content-length")
    try:
        path = await upload.save_first_file(request.stream(), request.headers.get("content-type"), UPLOADS,
                                            10 * 1024 * 1024, VOICE_EXT,
                                            int(length) if length and length.isdigit() else None)
    except upload.UploadError as e:
        raise HTTPException(e.status, str(e))
    try:
        ff = audio.ffmpeg_path()
        if ff is None:
            raise HTTPException(503, "녹음을 변환하려면 ffmpeg가 필요해요")
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = VOICE_DIR / "hum.new.wav"
        proc = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(path),
                               "-ac", "1", "-ar", "24000", str(tmp)], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not tmp.exists():
            raise HTTPException(400, "녹음 파일을 읽지 못했어요")
        try:
            v = vocal_mod.load_voice(tmp.read_bytes())
        except vocal_mod.VoiceError as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, str(e))
        tmp.replace(VOICE_FILE)
        return {"ready": True, "median_hz": round(v["f0"], 1), "frames": v["frames"]}
    finally:
        path.unlink(missing_ok=True)


@app.delete("/api/voice")
def voice_delete():
    VOICE_FILE.unlink(missing_ok=True)
    return {"ready": False}


@app.post("/api/score")
async def score_upload(request: Request, genre: str = "ballad", instruments: str | None = None):
    """multipart/form-data with one MIDI or MusicXML file (<= 2 MB): the user's score as uploaded plus the AI's chords for
    the same melody, bar by bar (instmate.scorecheck). The file is always deleted; nothing is stored or trained on."""
    from instmate import scorecheck
    length = request.headers.get("content-length")
    try:
        path = await upload.save_first_file(
            request.stream(), request.headers.get("content-type"), UPLOADS, scorecheck.MAX_BYTES, scorecheck.ALLOWED_EXT,
            int(length) if length and length.isdigit() else None)
    except upload.UploadError as e:
        raise HTTPException(e.status, str(e))
    try:
        inst = [x for x in (instruments or "").split(",") if x] or None
        return await run_in_threadpool(scorecheck.analyze, path, genre, inst)
    except scorecheck.ScoreError as e:
        raise HTTPException(400, str(e))
    finally:
        path.unlink(missing_ok=True)
        if path.exists():
            raise HTTPException(500, "업로드한 악보를 삭제하지 못했어요. data/uploads를 확인해 주세요.")


class WorkspaceContext(BaseModel):
    tempo: int = Field(90, ge=40, le=200, strict=True)
    genre: Literal["ballad", "classical", "rock", "dance", "kpop", "jpop", "newage", "hiphop", "piano_rock", "pop_punk", "hard_rock", "emo_rock", "modern_rock"] = "ballad"
    voice: Literal["female", "mezzo", "female_low", "male_low"] = "female"
    lyrics_mode: Literal["ah", "model"] = "ah"
    key_shift: int = Field(0, ge=-4, le=0, strict=True)
    melody_role: Literal["vocal", "lead_guitar", "acoustic_guitar", "keys"] = "vocal"
    form: Literal["standard", "chorus_first"] = "standard"
    seed_role: Literal["chorus", "verse", "intro"] = "chorus"
    melody_source: Literal["ai", "own", "rest"] = "ai"
    intro_player: Literal["guitar", "vocal"] = "guitar"
    fit_voice: bool = Field(True, strict=True)
    key_change: bool = Field(False, strict=True)
    melody_shape: bool = Field(True, strict=True)


class WorkspaceMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=1000)


class WorkspaceReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    context: WorkspaceContext = Field(default_factory=WorkspaceContext)
    history: list[WorkspaceMessage] = Field(default_factory=list, max_length=10)


def _llm_unload():
    from instmate import llm
    llm.unload()


def make_workspace_planner():
    """Lazy local LLM: precise commands need no model and make no network call."""
    if os.environ.get("INSTMATE_WORKSPACE_LLM", "auto").lower() == "off":
        return None
    from instmate import workspace_assistant
    if not workspace_assistant.connection_status().get("available"):   # Ollama down: say so instead of a failed call's error
        return None
    return lambda text, context, history: workspace_assistant.ollama_plan(text, context, history)


@app.get("/api/workspace/assistant/status")
def workspace_status():
    from instmate import workspace_assistant
    return workspace_assistant.connection_status()


@app.post("/api/workspace/assistant")
async def workspace_input(request: Request):
    """Bound bytes before JSON parsing; audio/score uploads use their existing limits."""
    parts, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 65536:
            raise HTTPException(413, "도우미 요청은 64KB까지 보낼 수 있습니다.")
        parts.append(chunk)
    try:
        req = WorkspaceReq.model_validate_json(b"".join(parts))
    except ValidationError:
        raise HTTPException(422, "요청 형식 또는 설정 값의 범위를 확인해 주세요.") from None
    key = request.headers.get("x-gemini-key")
    return await run_in_threadpool(workspace_turn, req, key)


def workspace_turn(req: WorkspaceReq, key: str | None = None):
    """Return safe settings, not a score rewrite or a generation job."""
    from instmate import workspace_assistant
    if not req.text.strip():
        raise HTTPException(400, "수정하고 싶은 내용을 말씀해 주세요.")
    context = req.context.model_dump()
    history = [m.model_dump() for m in req.history]
    global _workspace_model_busy
    fast = workspace_assistant.respond(req.text, context, planner=None, history=history)
    if not fast.pop("requires_llm", False):
        return fast
    from instmate import usergemini
    events: list = []
    if usergemini.valid(key):              # the visitor's own Gemini key first, the local model when it fails
        planner = lambda text, context, history: workspace_assistant.ollama_plan(text, context, history, key=key, events=events)
    else:
        planner = make_workspace_planner()
    if planner is None:
        return {"reply": "언어 모델이 꺼져 있어 추천은 처리할 수 없습니다. BPM처럼 명확한 설정 수정은 가능합니다.",
                "patch": {}, "confirmation_required": False, "source": "unavailable"}
    # Reserve shared model capacity atomically, but keep the job mutex available to cancel/status/start requests.
    with _jobs_lock:
        if _busy() or any(j.get("status") == "running" for j in yue_jobs.values()):
            return {"reply": "다른 작업을 처리 중입니다. 메모리를 보호하기 위해 추천은 작업이 끝난 뒤 처리합니다. 빠른 설정 수정은 가능합니다.",
                    "patch": {}, "confirmation_required": False, "source": "busy"}
        _workspace_model_busy = True
    try:
        out = workspace_assistant.respond(req.text, context, planner=planner, history=history)
        if usergemini.valid(key) and not out.get("patch") and out.get("source") == "gemini":
            # nothing to change but a question was asked: Gemini answers it from the app's own facts
            said = workspace_assistant.answer_question(req.text, context, key, events)
            if said:
                out = {**out, "reply": said}
        return {**out, "llm_note": events[0]} if events else out
    finally:
        with _jobs_lock:
            _workspace_model_busy = False


class ChatReq(BaseModel):
    text: str = Field("", max_length=1000)
    session: dict | None = None                              # what the last /api/chat answer returned (the page keeps it)
    score: dict | None = None                                # POST /api/score/read's score for a file attached to this message
    event: dict | None = None                                # {"type": "composed", "song": {...}} when a job finished
    attached: bool = False                                   # a file came with this message and was not read yet
    find_sections: bool = False                              # a whole score: sections found from its repeats (off by default)


def make_chat_llm():
    """The dialogue model as a reader of free text for the conversation, or None (tests replace this function).
    INSTMATE_CHAT_LLM=off keeps the conversation on its plain rules."""
    from instmate import chat
    if os.environ.get("INSTMATE_CHAT_LLM", "").lower() == "off":
        return None
    return chat.ollama_reader()


def make_chat_tools():
    """The local model as the tool picker for requests on a finished song (chat.TOOLS), or None."""
    from instmate import chat
    if os.environ.get("INSTMATE_CHAT_LLM", "").lower() == "off":
        return None
    return chat.ollama_tools()


def make_chat_explainer(key=None, events=None):
    """question, song -> text: instmate/explain.py's lookup, worded by the local model when it runs."""
    from instmate import chat, explain
    from instmate import usergemini, workspace_assistant
    off = os.environ.get("INSTMATE_CHAT_LLM", "").lower() == "off"
    local = None if off else chat.ollama_text()

    def run(question, song):
        if usergemini.valid(key):          # Gemini answers from the app's own facts (and this song) first
            said = workspace_assistant.answer_question(question, {"현재 곡": explain.facts(song)} if song else {}, key, events)
            if said:
                return "[Gemini] " + said
        return explain.say(explain.answer(question, song, local))   # no key, or Gemini failed: the local model or the passage
    return run


@app.post("/api/chat")
def chat_turn(req: ChatReq, x_gemini_key: str | None = Header(None)):
    """One turn of the conversation that leads from a score file to a song (instmate/chat.py).  The order of the work
    is fixed in code; the answer says what to tell the user and which existing API the page calls next.  Nothing is
    stored: the session travels with the page."""
    from instmate import chat, scoreimport
    try:
        from instmate import audio
        options = [{"id": k, "label": label, "help": about} for k, (label, about) in audio.OPTIONS.items()]
        from instmate import usergemini
        key = x_gemini_key if usergemini.valid(x_gemini_key) else None     # the visitor's own key: this request only
        events: list = []
        out = chat.step(req.session, req.text, req.score, req.event,
                        llm=chat.ollama_reader(key=key, events=events) if key else make_chat_llm(), attached=req.attached,
                        # made only when a turn needs them: neither asks for the model otherwise
                        tools=lambda *a: ((chat.ollama_tools(key=key, events=events) if key else make_chat_tools())
                                          or (lambda *_: None))(*a), sound_options=options,
                        explainer=lambda question, song: (make_chat_explainer(key, events) if key
                                                          else make_chat_explainer())(question, song),
                        find_sections=req.find_sections, answer_any=bool(key))
        return {**out, "choices": chat.choices(out["session"], out.get("ask")),   # the answers shown as buttons
                "llm_note": events[0] if events else None}
    except scoreimport.ScoreImportError as e:
        raise HTTPException(422, str(e))
    except (KeyError, TypeError, ValueError, AttributeError):
        raise HTTPException(422, "대화 상태를 읽지 못했어요. 새로 시작해 주세요.")


class ScoreTokensReq(BaseModel):
    score: dict                                              # what POST /api/score/read returned
    accept_changes: bool = False                             # the user saw what differs from the file and goes on
    start_bar: int = Field(0, ge=0, le=400)                  # bar index from 0
    bars: int = Field(8, ge=1, le=8)


class ScoreSongReq(BaseModel):
    score: dict
    sections: list[dict] | None = Field(None, max_length=40)   # [{name, start_bar, bars}] in order, no gaps
    genre: str = Field("ballad", max_length=40)
    key: str | None = Field(None, max_length=4)              # None: the key read from the score
    instruments: list[str] | None = None
    chords: list[str | None] | None = Field(None, max_length=400)   # one per bar, e.g. the comparison's AI chords
    chords_source: Literal["ai", "original"] | None = None   # which comparison column the chords came from
    accept_changes: bool = False                             # the user saw what differs from the file and goes on
    melody_role: Literal["vocal", "lead_guitar", "acoustic_guitar", "keys"] = "vocal"
    fit_voice: Literal["male", "female"] | None = None    # move the whole song into this singer's range (voicefit)


@app.post("/api/score/read")
async def score_read(request: Request):
    """multipart/form-data with one MIDI or MusicXML file (<= 2 MB) or a score PDF (<= 30 MB, read by optical music
    recognition, instmate/omr.py): the user's own song read as a melody on the app's grid (tempo, meter, key, bars)
    plus a first split into sections. The file and its page images are always deleted; nothing is stored."""
    import shutil
    import tempfile
    from instmate import omr, scorecheck, scoreimport
    length = request.headers.get("content-length")
    try:
        path = await upload.save_first_file(
            request.stream(), request.headers.get("content-type"), UPLOADS, omr.MAX_BYTES, scorecheck.ALLOWED_EXT | {".pdf"},
            int(length) if length and length.isdigit() else None)
    except upload.UploadError as e:
        raise HTTPException(e.status, str(e))
    work = None
    try:
        if path.suffix.lower() == ".pdf":
            work = Path(tempfile.mkdtemp(prefix="omr_", dir=UPLOADS))
            score = await run_in_threadpool(omr.read_pdf, path, work)
        elif path.stat().st_size > scorecheck.MAX_BYTES:
            raise HTTPException(413, "파일이 너무 커요(최대 2MB).")
        else:
            score = await run_in_threadpool(scoreimport.read, path)
    except scoreimport.ScoreImportError as e:
        raise HTTPException(400, str(e))
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        path.unlink(missing_ok=True)
        if path.exists() or (work is not None and work.exists()):
            raise HTTPException(500, "업로드한 악보를 삭제하지 못했어요. data/uploads를 확인해 주세요.")
    # the split is learned from what repeats in the melody (user, 2026-10-19); the halves when nothing repeats
    try:
        found = sectionfind.find(score)
    except Exception:
        found = None
    return {"score": score, "sections": found or scoreimport.default_sections(score["bars"]), "musicxml": _score_sheet(score)}


def _score_sheet(score: dict) -> str | None:
    """The melody the app read, as a sheet for the page (melody only).  A drawing problem never costs the import."""
    from instmate.export import to_musicxml
    from instmate.render import key_of
    from instmate.theory import Note
    try:
        melody, at = [], 0
        for n in score["notes"]:
            if n["start"] > at:
                melody.append(Note(None, at, n["start"] - at))
            melody.append(Note(n["pitch"], n["start"], n["dur"]))
            at = n["start"] + n["dur"]
        end = score["bars"] * score["bar_ticks"]
        if at < end:
            melody.append(Note(None, at, end - at))
        return to_musicxml(key_of(score["key"]), "ballad", melody, [], bpm=score["bpm"], meter=score["meter"])
    except Exception:
        return None


@app.post("/api/score/tokens")
def score_tokens(req: ScoreTokensReq):
    """Up to 8 bars of the imported melody as tokens for the melody input (the seed)."""
    from instmate import scoreimport
    try:
        scoreimport.require_consent(req.score, req.accept_changes)
        return scoreimport.to_tokens(req.score, req.start_bar, req.bars)
    except scoreimport.ScoreImportError as e:
        raise HTTPException(422, str(e))


@app.post("/api/score/song")
def score_song(req: ScoreSongReq):
    """The whole imported song with the band arrangement: the user's melody as written, sections as they named them,
    chords from the app's chord tool. The result is a normal song (band pack, band workspace, part editing)."""
    from instmate import scoreimport
    from instmate.leadsheet import LeadsheetError
    from instmate.render import leadsheet_to_song
    try:
        scoreimport.require_consent(req.score, req.accept_changes)
        genre_ko = PROFILES_KO.get(req.genre, req.genre)
        written = req.score.get("chords") if isinstance(req.score, dict) else None
        from_file = (req.chords is None and isinstance(written, list) and len(written) == req.score.get("bars")
                     and written[0] is not None)              # the file's own chord symbols: kept as written
        score, key, chords, voice_fit = req.score, req.key, req.chords, None
        if req.fit_voice and req.melody_role == "vocal":
            from instmate import voicefit
            scoreimport._check(score)
            score, key, chords, voice_fit = voicefit.fit_score(score, req.fit_voice, key, chords)
            written = score.get("chords")
        text = scoreimport.to_leadsheet(score, req.sections, genre_ko, key, written if from_file else chords)
        result = leadsheet_to_song(text, instruments=req.instruments, melody_role=req.melody_role)
    except (scoreimport.ScoreImportError, LeadsheetError) as e:
        raise HTTPException(422, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    result["melody_source"] = "own"          # every melody note is the user's; no trained model wrote a section
    result = _second_voice(result, req.score.get("harmony") if isinstance(req.score, dict) else None, None, whole=True)
    if voice_fit:
        result["voice_fit"] = voice_fit
    # the chords are NOT the song's own: the file carried none that we read, so the app's chord tool wrote them
    tool_ko = {"dl": "학습된 코드 모델", "rule": "규칙 기반 코드 도구", "cycle": "고정 코드 순환"}
    result["chords_from"] = {"source": "app", "tool": result.get("chord_tool"),
                             "label": tool_ko.get(result.get("chord_tool"), str(result.get("chord_tool")))}
    if from_file:                            # the user's own chords: not replaced, not mixed with estimated ones
        result["chords_from"] = {"source": "file", "tool": None, "label": "악보에 적힌 코드 (그대로)",
                                 "note": req.score.get("chords_note")}
    if req.chords is not None:               # chords chosen in the comparison card were written into the song as given
        label = {"ai": "비교에서 고른 AI 추천 코드", "original": "올린 악보의 반주에서 추정한 원래 코드"}
        result["chords_from"] = {"source": f"compare-{req.chords_source or 'ai'}", "tool": None,
                                 "label": label.get(req.chords_source or "ai")}
    result["duration_ok"], result["duration_note"] = None, None   # the song is as long as the user wrote it
    for sec in result["sections"]:                                # every melody note of an imported song is the user's
        sec["origin"] = "input"
    result["harmony_report"] = _harmony_report(result)
    result["imported_score"] = {"bars": req.score.get("bars"), "snapped": req.score.get("snapped"),
                                "dropped": req.score.get("dropped"), "key_note": req.score.get("key_note"),
                                "chords_note": req.score.get("chords_note"),
                                # False: a converted version the user accepted, not the file note for note
                                "exact": bool(req.score.get("exact", True)),
                                "changes_total": req.score.get("changes_total", 0), "tempo_note": req.score.get("tempo_note")}
    return _with_leadsheet(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("INSTMATE_PORT", "8000")))
