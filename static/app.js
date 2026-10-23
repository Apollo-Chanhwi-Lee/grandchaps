"use strict";

const TPQ = 480, BAR = TPQ * 4, MAX_TICKS = BAR * 8;
const DURS = [240, 480, 960, 1920];
const DUR_KO = { 240: "8분음표", 480: "4분음표", 960: "2분음표", 1920: "온음표" };
const NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11];
const GENRES = { ballad: "발라드", classical: "클래식", rock: "록", dance: "댄스", kpop: "케이팝", jpop: "제이팝", newage: "뉴에이지", hiphop: "힙합",
  piano_rock: "피아노 록", pop_punk: "팝펑크", hard_rock: "하드록", emo_rock: "감성록", modern_rock: "모던록" };
const INSTR = { piano: "피아노", bass: "베이스", drums: "드럼", guitar: "기타", strings: "스트링", keys: "멜로디 피아노" };
const BAND = ["piano", "bass", "drums", "guitar"];
// the app is scoped to band songs: other genres stay in the backend but get no button
const BAND_GENRES = ["ballad", "rock", "modern_rock", "emo_rock", "piano_rock", "pop_punk", "hard_rock", "jpop", "kpop"];   // default line-up (strings: optional)
const MIX_TRACKS = { melody: "멜로디", ...INSTR };
const LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];
const SAMPLE = "3 3 4 5 5 4 3 2 1 1 2 3 3 2 2h 3 3 4 5 5 4 3 2 1 1 2 3 2 1 1h"; // Ode to Joy

const state = {
  tokens: [], octave: 4, dur: 480, sharpNext: false, inputTonic: 0,
  genre: "ballad", instruments: new Set(BAND),
  keyChoice: null, result: null, changedBar: null,
  inputVersion: 0,                  // bumped on every input change; older in-flight responses are dropped
  tempo: null,                      // BPM chosen in the settings; null = the genre default
  src: { genre: null, instr: null, tempo: null },   // where each setting came from: null (default) | "rec" | "manual"
  style: null,                      // confirmed StyleAttrs (only after the user said "맞아요"); cleared by manual changes
  pending: null,                    // a style proposal waiting for confirmation: { attrs, question }
  composing: false, composed: null, selected: null,    // AI song: job running / shown song meta / selected section
  rated: new Map(),                 // "songId:section|song" -> "saving" | "saved" (one rating per song or section)
};
const $ = (id) => document.getElementById(id);

// ---------- melody input ----------
const totalTicks = () => state.tokens.reduce((s, t) => s + t.dur, 0);
// Notes picked in the number list (indices into state.tokens): the octave buttons move these instead of setting the
// octave of the next note.
const melodySel = new Set();
function pickNote(i) {
  const only = melodySel.size === 1 && melodySel.has(i);
  melodySel.clear();
  if (!only) melodySel.add(i);
  renderMelody();
  if (!only && state.tokens[i]?.degree > 0) previewNote(state.tokens[i]);
}
function pickAllNotes() {
  const all = state.tokens.length > 0 && melodySel.size === state.tokens.length;
  melodySel.clear();
  if (!all) state.tokens.forEach((_, i) => melodySel.add(i));
  renderMelody();
}

function addNote(degree) {
  if (totalTicks() + state.dur > MAX_TICKS) return showError("최대 8마디까지 입력할 수 있어요.");
  const tok = { degree, sharp: degree > 0 && state.sharpNext, octave: state.octave, dur: state.dur };
  state.tokens.push(tok);
  state.sharpNext = false;
  melodyChanged();
  if (degree > 0) previewNote(tok);
}

function melodyChanged(keepSelection = false) {
  if (!keepSelection) melodySel.clear();   // another melody: the old picks point at other notes
  state.keyChoice = null;           // melody changed: analyze the key again
  state.seedChords = null;          // chords chosen for another melody no longer apply
  invalidateResult();
  hideError();
  renderMelody();
  schedulePreview();
}

function renderMelody() {
  renderPianoRoll();
  for (const i of [...melodySel]) if (i >= state.tokens.length) melodySel.delete(i);
  $("octave-view").textContent = melodySel.size
    ? (melodySel.size === state.tokens.length ? "전체 선택됨" : `음 ${melodySel.size}개 선택됨`) + " (▲▼로 옮겨요)"
    : `다음에 넣을 음: ${state.octave} (도 = ${NOTE_NAMES[state.inputTonic]}${state.octave})`;
  if ($("btn-pick-all")) $("btn-pick-all").textContent = state.tokens.length && melodySel.size === state.tokens.length ? "선택 풀기" : "전체 선택";
  $("dur-view").textContent = DUR_KO[state.dur];
  $("sharp-view").textContent = state.sharpNext ? "다음 음 ♯" : "꺼짐";
  const view = $("melody-view");
  view.innerHTML = "";
  if (!state.tokens.length) {
    view.innerHTML = '<span class="placeholder">숫자 1~7을 눌러 멜로디를 입력하세요 (0은 쉼표)</span>';
    return;
  }
  let t = 0, barEl = null, barIdx = -1;
  for (const [i, tok] of state.tokens.entries()) {
    const b = Math.floor(t / BAR);
    if (b !== barIdx) { barIdx = b; barEl = document.createElement("div"); barEl.className = "bar"; view.append(barEl); }
    const el = document.createElement("span");
    el.className = "note" + (melodySel.has(i) ? " sel" : "");
    el.onclick = () => pickNote(i);
    const oct = tok.octave > 4 ? "'".repeat(tok.octave - 4) : ",".repeat(Math.max(0, 4 - tok.octave));
    el.innerHTML = (tok.degree === 0 ? "0" : tok.degree + (tok.sharp ? "♯" : "") + oct) + `<sub>${tok.dur / TPQ}</sub>`;
    barEl.append(el);
    t += tok.dur;
  }
}

function tokenMidi(tok) {
  return 12 * (tok.octave + 1) + state.inputTonic + MAJOR_STEPS[tok.degree - 1] + (tok.sharp ? 1 : 0);
}

function loadSample() {
  state.tokens = SAMPLE.split(" ").map((s) => ({
    degree: +s[0], sharp: false, octave: 4, dur: s.endsWith("h") ? 960 : 480,
  }));
  state.inputTonic = 0;
  $("input-tonic").value = "0";
  melodyChanged();
}


// ---------- piano roll (tokens stay the single source of truth; number keys keep working) ----------
const PR = { col: 240, cols: MAX_TICKS / 240, cellW: 18, rowH: 12, labelW: 34, lo: 48, hi: 84 };
const BLACK_PC = new Set([1, 3, 6, 8, 10]);

function notesFromTokens(tokens) {
  const out = [];
  let t = 0;
  for (const tok of tokens) {
    if (tok.degree > 0) out.push({ start: t, dur: tok.dur, pitch: tokenMidi(tok) });
    t += tok.dur;
  }
  return out;
}

// absolute MIDI pitch -> number token in the current input key (non-scale notes become a sharp of the step below)
function tokenFromPitch(pitch, dur) {
  const rel = pitch - state.inputTonic;
  const octave = Math.floor(rel / 12) - 1;
  const pc = ((rel % 12) + 12) % 12;
  let i = MAJOR_STEPS.indexOf(pc), sharp = false;
  if (i < 0) { i = MAJOR_STEPS.indexOf(pc - 1); sharp = true; }
  return { degree: i + 1, sharp, octave, dur };
}

function tokensFromNotes(notes) {
  const tokens = [];
  let t = 0;
  for (const n of [...notes].sort((a, b) => a.start - b.start)) {
    if (n.start > t) tokens.push({ degree: 0, sharp: false, octave: state.octave, dur: n.start - t });
    tokens.push(tokenFromPitch(n.pitch, n.dur));
    t = n.start + n.dur;
  }
  return tokens;
}

// monophonic insert: notes that start inside the new one are removed, a note sounding into it is cut short
function placeNote(notes, start, dur, pitch) {
  const end = start + dur, out = [];
  for (const n of notes) {
    if (n.start + n.dur <= start || n.start >= end) out.push(n);
    else if (n.start < start) out.push({ ...n, dur: start - n.start });
  }
  out.push({ start, dur, pitch });
  return out.sort((a, b) => a.start - b.start);
}

let prDrag = null;      // { notes (without the dragged one), start, pitch, col0, dur, moved, hit }
function prRange(notes) {
  const ps = notes.map((n) => n.pitch);
  return [Math.min(PR.lo, ...ps), Math.max(PR.hi, ...ps)];
}

function renderPianoRoll() {
  const box = $("piano-roll");
  const notes = prDrag ? placeNote(prDrag.notes, prDrag.start, prDrag.dur, prDrag.pitch) : notesFromTokens(state.tokens);
  const [lo, hi] = prRange(notes);
  const W = PR.labelW + PR.cols * PR.cellW, H = (hi - lo + 1) * PR.rowH;
  const scale = new Set(MAJOR_STEPS.map((s) => (s + state.inputTonic) % 12));
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" data-lo="${lo}" data-hi="${hi}">`;
  for (let p = hi; p >= lo; p--) {
    const y = (hi - p) * PR.rowH, pc = p % 12;
    const cls = BLACK_PC.has(pc) ? "pr-row-black" : scale.has(pc) ? "pr-row-key" : "pr-row-white";
    svg += `<rect class="${cls}" x="${PR.labelW}" y="${y}" width="${W - PR.labelW}" height="${PR.rowH}"/>`;
    if (pc === state.inputTonic % 12 || pc === 0) svg += `<text class="pr-label" x="2" y="${y + PR.rowH - 2}">${NOTE_NAMES[pc]}${Math.floor(p / 12) - 1}</text>`;
  }
  for (let c = 0; c <= PR.cols; c++) {
    const x = PR.labelW + c * PR.cellW;
    svg += `<line class="${c % 8 === 0 ? "pr-bar" : "pr-grid"}" x1="${x}" y1="0" x2="${x}" y2="${H}"/>`;
  }
  for (const n of notes) {
    const x = PR.labelW + (n.start / PR.col) * PR.cellW, w = (n.dur / PR.col) * PR.cellW;
    svg += `<rect class="pr-note" x="${x + 1}" y="${(hi - n.pitch) * PR.rowH + 1}" width="${w - 2}" height="${PR.rowH - 2}"/>`;
  }
  box.innerHTML = svg + "</svg>";
}

function prCell(e) {
  const svg = $("piano-roll").querySelector("svg");
  const r = svg.getBoundingClientRect();
  const col = Math.floor((e.clientX - r.left - PR.labelW) / PR.cellW);
  const pitch = Number(svg.dataset.hi) - Math.floor((e.clientY - r.top) / PR.rowH);
  return { col, pitch };
}

function prDown(e) {
  const { col, pitch } = prCell(e);
  if (col < 0 || col >= PR.cols) return;
  const notes = notesFromTokens(state.tokens), tick = col * PR.col;
  const hit = notes.find((n) => n.pitch === pitch && n.start <= tick && tick < n.start + n.dur);
  if (hit) { prDrag = { hit, moved: false }; return; }            // click on a note: delete on release
  prDrag = { notes, start: tick, pitch, col0: col, dur: PR.col, moved: false };
  $("piano-roll").querySelector("svg").setPointerCapture?.(e.pointerId);
  previewNote(tokenFromPitch(pitch, PR.col));
  renderPianoRoll();
}

function prMove(e) {
  if (!prDrag || prDrag.hit) return;
  const { col } = prCell(e);
  const dur = Math.max(1, Math.min(col, PR.cols - 1) - prDrag.col0 + 1) * PR.col;
  if (dur !== prDrag.dur) { prDrag.dur = dur; prDrag.moved = true; renderPianoRoll(); }
}

function prUp() {
  if (!prDrag) return;
  const d = prDrag;
  prDrag = null;
  const notes = d.hit ? notesFromTokens(state.tokens).filter((n) => n !== d.hit && !(n.start === d.hit.start && n.pitch === d.hit.pitch))
    : placeNote(d.notes, d.start, d.dur, d.pitch);
  state.tokens = tokensFromNotes(notes);
  melodyChanged();
}

// live sheet under the editor: re-rendered 0.3 s after the last edit
let previewTimer = null, inputOsmd = null, previewSeq = 0;
function schedulePreview() {
  if (typeof setTimeout !== "function") return;         // (test harness without timers)
  clearTimeout(previewTimer);
  previewTimer = setTimeout(requestPreview, 300);
}

async function requestPreview() {
  const seq = ++previewSeq;
  if (!state.tokens.some((t) => t.degree > 0)) { $("input-sheet").innerHTML = ""; inputOsmd = null; return; }
  try {
    const data = await api("/api/preview", { tokens: state.tokens, input_tonic: state.inputTonic });
    if (seq !== previewSeq || !window.opensheetmusicdisplay) return;
    if (!inputOsmd) inputOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("input-sheet", {
      autoResize: true, drawTitle: false, drawComposer: false, drawCredits: false, drawPartNames: false });
    await inputOsmd.load(data.musicxml);
    if (seq === previewSeq) inputOsmd.render();
  } catch (e) { /* preview is best effort; generation reports real errors */ }
}

// ---------- audio ----------
let synths = null;
function ensureSynths() {
  if (synths) return synths;
  const out = new Tone.Volume(-6).toDestination();
  synths = {
    // humming guide vocal: a soft sine with a slow attack and a light vibrato (when Tone provides it)
    melody: new Tone.Synth({ oscillator: { type: "sine" }, envelope: { attack: 0.06, decay: 0.1, sustain: 0.8, release: 0.5 } })
      .connect(Tone.Vibrato ? new Tone.Vibrato(5.5, 0.08).connect(out) : out),
    piano: new Tone.PolySynth(Tone.Synth, { oscillator: { type: "sine" }, envelope: { attack: 0.005, decay: 0.3, sustain: 0.3, release: 0.8 } }).connect(out),
    bass: new Tone.MonoSynth({ oscillator: { type: "sawtooth" }, filter: { Q: 1, type: "lowpass" },
      filterEnvelope: { baseFrequency: 120, octaves: 2 }, envelope: { attack: 0.01, release: 0.2 } }).connect(out),
    strings: new Tone.PolySynth(Tone.AMSynth, { envelope: { attack: 0.4, release: 1.2 } }).connect(new Tone.Volume(-10).connect(out)),
    guitar: new Tone.PolySynth(Tone.Synth, { oscillator: { type: "fatsawtooth", count: 2, spread: 12 },
      envelope: { attack: 0.003, decay: 0.25, sustain: 0.15, release: 0.3 } }).connect(new Tone.Volume(-14).connect(out)),
    kick: new Tone.MembraneSynth().connect(out),
    snare: new Tone.NoiseSynth({ envelope: { attack: 0.001, decay: 0.15, sustain: 0 } }).connect(new Tone.Volume(-8).connect(out)),
    hat: new Tone.MetalSynth({ envelope: { decay: 0.05, release: 0.01 }, harmonicity: 5.1, resonance: 4000 }).connect(new Tone.Volume(-22).connect(out)),
    tom: new Tone.MembraneSynth({ pitchDecay: 0.03, octaves: 3 }).connect(new Tone.Volume(-4).connect(out)),
    cymbal: new Tone.MetalSynth({ envelope: { decay: 0.9, release: 0.3 }, harmonicity: 3.1, resonance: 6000 }).connect(new Tone.Volume(-24).connect(out)),
  };
  return synths;
}

// Drum notes that get their own preview voice: toms by pitch (low to high), crash and ride as cymbals.
const TOM_NOTE = { 41: "F1", 43: "G1", 45: "A1", 47: "C2", 48: "D2", 50: "F2" };
const CYMBALS = new Set([49, 51, 52, 55, 57]);

// Monophonic Tone voices reject two starts at the same instant (e.g. crash + hat on beat 1): nudge by 1 ms.
let lastStart = {};
function monoTime(voice, time) {
  const t = Math.max(time, (lastStart[voice] ?? -1) + 0.001);
  lastStart[voice] = t;
  return t;
}

function playNote(n, time) {
  const s = ensureSynths();
  if (n.track === "drums") {
    if (n.midi === 36) s.kick.triggerAttackRelease("C1", 0.2, monoTime("kick", time), n.velocity);
    else if ([37, 38, 39].includes(n.midi)) s.snare.triggerAttackRelease(0.12, monoTime("snare", time), n.velocity);
    else if (n.midi in TOM_NOTE) s.tom.triggerAttackRelease(TOM_NOTE[n.midi], 0.25, monoTime("tom", time), n.velocity);
    else if (CYMBALS.has(n.midi)) s.cymbal.triggerAttackRelease(n.midi === 51 ? "C5" : "C4", n.midi === 51 ? 0.3 : 0.9, monoTime("cymbal", time), n.velocity);
    else if (n.midi === 54) s.cymbal.triggerAttackRelease("G5", 0.08, monoTime("cymbal", time), n.velocity * 0.7);   // tambourine: short and bright, not the hi-hat
    else s.hat.triggerAttackRelease("C6", n.midi === 46 ? 0.15 : 0.04, monoTime("hat", time), n.velocity);
    return;
  }
  const t = ["melody", "bass"].includes(n.track) ? monoTime(n.track, time) : time;
  // shared preview voices (the melody synth plays one note at a time, so the second voice borrows the piano)
  const voice = n.track === "keys" || n.track === "harmony" ? "piano" : n.track === "lead" ? "guitar" : n.track;
  s[voice].triggerAttackRelease(Tone.Frequency(n.midi, "midi").toFrequency(), n.duration, t, n.velocity);
}

// Key / piano-roll preview. Several presses in the same millisecond must not hit the monophonic synth with equal start
// times (Tone throws "Start time must be strictly greater than previous start time"): every start goes through monoTime.
async function previewNote(tok) {
  await Tone.start();
  const t = monoTime("melody", Tone.now());
  ensureSynths().melody.triggerAttackRelease(Tone.Frequency(tokenMidi(tok), "midi").toFrequency(), 0.3, t);
}

// ---------- player bar: what is playing, how far, and an end.  Playback stops by itself when the music is over.
let pbTimer = null;
// The song's route map follows a playback only when that playback was started for it (a station, or the whole song):
// `routeNext` is the song time the next player bar starts at, taken by showPlayerBar; stopPlayback ends the following.
let routeNext = null, routeFollow = null;
const fmtClock = (s) => `${Math.floor(Math.max(0, s) / 60)}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, "0")}`;
function showPlayerBar(title, total, position, loops = false) {
  $("pb-title").textContent = title;
  $("playerbar").classList.remove("hidden");
  if (pbTimer != null) clearInterval(pbTimer);
  routeFollow = routeNext != null ? { offset: routeNext } : null;
  routeNext = null;
  const tick = () => {
    const at = position(), len = total();
    if (routeFollow) moveRouteDot(routeFollow.offset + at);
    if ($("pb-time")) $("pb-time").textContent = len > 0 ? `${fmtClock(loops ? at % len : at)} / ${fmtClock(len)}` : "";
    if ($("pb-fill")?.style) $("pb-fill").style.width = len > 0 ? `${Math.min(100, 100 * (loops ? (at % len) / len : at / len))}%` : "0%";
    if (!loops && len > 0 && at >= len) stopPlayback();        // the music is over: no bar left saying "playing"
  };
  tick();
  pbTimer = typeof setInterval === "function" ? setInterval(tick, 200) : null;
}

async function playEvents(notes, title = "재생 중") {
  await Tone.start();
  stopPlayback();
  ensureSynths();
  lastStart = {};
  for (const n of notes) Tone.Transport.schedule((time) => playNote(n, time), n.time);
  const end = notes.reduce((m, n) => Math.max(m, n.time + (n.duration || 0)), 0) + 1.2;   // let the last notes ring out
  Tone.Transport.start("+0.1");
  showPlayerBar(title, () => end, () => Tone.Transport.seconds ?? 0);
}

function stopPlayback() {
  if (pbTimer != null) { clearInterval(pbTimer); pbTimer = null; }
  routeFollow = null;
  if (editDraft) editDraft.playing = false;
  $("playerbar").classList.add("hidden");
  if (hqAudio) { hqAudio.pause(); hqAudio = null; }
  if (hqUrl) { URL.revokeObjectURL(hqUrl); hqUrl = null; }
  $("yue-audio")?.pause?.();                  // the AI vocal take stops with everything else (it stays ready to replay)
  Tone.Transport.stop();
  Tone.Transport.cancel();
  Tone.Transport.loop = false;               // the part editor's preview loops; ordinary playback never does
  if (synths) { synths.piano.releaseAll(); synths.strings.releaseAll(); synths.guitar.releaseAll(); }
}

function playMelodyOnly() {
  if (!state.tokens.length) return showError("먼저 멜로디를 입력하세요.");
  const spb = 60 / 90 / TPQ;           // preview at 90 bpm
  let t = 0;
  const notes = [];
  for (const tok of state.tokens) {
    if (tok.degree > 0) notes.push({ track: "melody", midi: tokenMidi(tok), time: t * spb, duration: tok.dur * spb * 0.95, velocity: 0.8 });
    t += tok.dur;
  }
  playEvents(notes, "입력한 멜로디");
}

// ---------- server ----------
// The visitor's own Gemini key: kept in this browser only, sent as a header with the conversation requests only.
window.instmateGeminiKey = () => { try { return localStorage.getItem("instmate-gemini-key") || ""; } catch (e) { return ""; } };
const geminiHeaders = (path) => (/^\/api\/(chat|workspace)/.test(path) && window.instmateGeminiKey() ? { "X-Gemini-Key": window.instmateGeminiKey() } : {});
async function api(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", ...geminiHeaders(path) }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({ detail: "서버 응답이 비었어요(시간 초과이거나 연결이 끊겼어요). 잠시 뒤 다시 보내 주세요." }));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail
    : Array.isArray(data.detail?.errors) ? data.detail.errors.join("\n") : JSON.stringify(data.detail));
  return data;
}

function baseRequest() {
  return {
    tokens: state.tokens, input_tonic: state.inputTonic, genre: state.genre,
    instruments: [...state.instruments], energy: $("energy").checked ? 1 : 0,
    ...(state.tempo ? { tempo: state.tempo } : {}),
  };
}

let resultRequestSeq = 0;   // only the newest generate or modify request may update the UI
async function generate() {
  if (state.editing) return;
  if (!state.tokens.length) return showError("먼저 멜로디를 입력하세요.");
  hideError();
  $("ask").classList.add("hidden");
  $("btn-generate").disabled = true;
  setMaking(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const data = await api("/api/generate", { ...baseRequest(), key_choice: state.keyChoice });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    if (data.status === "need_key") return askKey(data);
    state.changedBar = null;
    showResult(data);
  } catch (e) { if (seq === resultRequestSeq) showError(e.message); }
  finally { setMaking(false); if (seq === resultRequestSeq) $("btn-generate").disabled = false; }
}
// The ring on the make step: on while any making request is running (they may overlap, so it counts).
let makingCount = 0;
function setMaking(on) {
  makingCount = Math.max(0, makingCount + (on ? 1 : -1));
  $("make-busy")?.classList?.toggle("hidden", makingCount === 0);
}

function askKey(data, next = generate) {
  const box = $("ask");
  box.innerHTML = `<p><b>에이전트 질문:</b> ${data.question}</p>`;
  const version = state.inputVersion;
  for (const c of data.key_analysis.candidates) {
    const b = document.createElement("button");
    b.textContent = c.key.label;
    b.onclick = () => {
      if (version !== state.inputVersion) return;   // question was for an older melody/setting
      state.keyChoice = { tonic: c.key.tonic, mode: c.key.mode }; next();
    };
    box.append(b, " ");
  }
  box.classList.remove("hidden");
  if (next === generate) renderTrace(data.trace);
}

// `bar`: one bar index or a list of bar indices (edited together as one version)
async function modifyBar(bar, mood, allowClash = false) {
  if (state.editing) return;
  const bars = Array.isArray(bar) ? bar : [bar];           // one version-changing request at a time (server adopts what it receives)
  hideError();
  hideAgentAsk();
  setEditing(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const data = await api("/api/modify", {
      ...baseRequest(), key_choice: state.keyChoice, allow_clash: allowClash,
      current: state.result.chords.map((c) => c.roman), bars, mood, history: state.result.version?.history,
    });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    if (data.status === "need_input" || data.status === "budget_exceeded") return askAgent(data, bars, mood);
    state.changedBar = bars;
    showResult(data);
  } catch (e) { if (seq === resultRequestSeq && version === state.inputVersion) showError(e.message); }
  finally { setEditing(false); }
}

function setEditing(on) {
  state.editing = on;
  for (const b of state.editButtons || []) b.disabled = on;
  $("btn-generate").disabled = on;
  $("btn-undo").disabled = on || !state.result?.version?.can_undo;
  $("btn-mix").disabled = on;
  $("btn-agent").disabled = on;
  $("btn-song").disabled = on;
  $("btn-range-dark").disabled = on;
  $("btn-range-bright").disabled = on;
}

// ---------- 작곡메이트: the current version expanded into a 2-4 minute song (nothing is adopted) ----------
async function makeSong() {
  const v = state.result?.version;
  if (!v || state.editing) return;
  hideError();
  setEditing(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const previous = state.song;
    const role = $("seed-role")?.value || "chorus";
    const data = await api("/api/song", { history: v.history, base_index: v.index, seed_role: role,
      fill: $("melody-source")?.value === "rest" ? "rest" : "own", form: $("form-choice")?.value || null,
      key_change: !!$("key-change")?.checked,
      intro_player: role === "intro" && melodyRole() === "vocal" ? $("intro-player")?.value || "guitar" : "vocal",
      ...seedHarmony() });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    const carried = await carryMemberParts(data, previous);
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    showSong(carried.result);
    autoAiVocal();
    if (carried.note && $("band-status")) $("band-status").textContent = carried.note;
  } catch (e) { if (seq === resultRequestSeq && version === state.inputVersion) showError(e.message); }
  finally { setEditing(false); }
}

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
let songOsmd = null;
function showSong(data) {
  state.song = data;
  bandParent = null;                         // another song: not a continuation of the band version that was open
  if (editDraft) { editSeq++; editDraft = null; $("edit-panel")?.classList.add("hidden"); }   // its draft belonged to the old song
  if ($("edit-box")?.open) editSections();
  goStep(3);                                 // a finished song: on to "listen and edit"
  $("song").classList.remove("hidden");
  for (const id of ["yue2-card", "practice-card"]) $(id)?.classList.remove("hidden");   // the flow goes on: YuE2 audio, practice
  if ($("yue-box") && sungMelody(data)) $("yue-box").open = true;   // the sung demo is offered right with the song
  fillPractice(data);
  renderCover(data);
  fillYueSections(data);
  renderRoute(data);
  if ($("song-seed-warning")) {
    const warn = seedWarning(data);
    $("song-seed-warning").textContent = warn || "";
    $("song-seed-warning").classList.toggle("hidden", !warn);
  }
  $("song-view").textContent = `${fmtTime(data.duration_sec)} · ${data.bars}마디 · ${data.tempo} BPM`
    + (data.duration_ok === false ? ` · 2~4분 범위 밖${data.duration_note ? " (" + data.duration_note + ")" : ""}` : "");
  renderHarmony(data.harmony_report);
  if ($("song-about")) $("song-about").textContent = songAbout(data);
  if ($("btn-song-hq")) $("btn-song-hq").textContent = playLabel(data);
  $("song-sections").innerHTML = data.sections.map((s) =>
    `<li><b>${escapeHtml(s.label)}</b>${fmtTime(s.start_sec)}~${fmtTime(s.end_sec)} · ${s.bars}마디 · 멜로디: ${escapeHtml(s.melody_source)}</li>`).join("");
  const m = data.metrics;
  $("song-metrics").textContent = (m.chorus_seed_match == null ? "" : `후렴=씨앗 일치 ${m.chorus_seed_match} · `)
    + `파트 경계 평균 충돌률 ${m.boundary_mean_clash}`
    + ` · 전체 평균 충돌률 ${m.mean_clash} · 기준 초과 마디 ${m.bars_over_threshold}`
    + ` (새로 만든 파트 ${m.new_part_bars_over_threshold ?? "-"})`
    + (m.seed_bars_over_threshold?.length ? ` · 후렴의 초과 마디는 원본 버전의 ${m.seed_bars_over_threshold.join(", ")}마디에서 왔어요. 원본에서 수정하면 모든 후렴에 반영돼요.` : "");
  $("song-leadsheet").textContent = data.leadsheet || data.leadsheet_error || "";
  $("song-leadsheet-box").open = false;
  $("song-sheet-box").open = false;
  songOsmd = null;
  $("song-sheet").innerHTML = "";
}

// ---------- the song's route map (#song-route, user order 2026-10-13, after the "가닥" app's subway map): one line,
// a station per section spaced by its length; my melody's stretch in lime (--seed), the model's in teal (--ai), the
// rest grey.  A station plays the song from that section; the dot follows that playback (and the library take whose
// score is shown).  With reduced motion the dot stays where it is.
let routeTotal = 0, routeLib = null;
const reduceMotion = () => !!window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
const routeInk = (origin) => (origin === "input" ? "var(--seed)" : origin === "model" ? "var(--ai)" : "var(--line-strong)");
function routeHtml(song) {
  const secs = song?.sections || [];
  const total = Number(song?.duration_sec) || Number(secs[secs.length - 1]?.end_sec) || 0;
  if (!secs.length || !(total > 0)) return "";
  const at = (t, scale) => +(scale * Math.min(1, Math.max(0, Number(t) / total))).toFixed(3);
  // a YuE2 song carries no origin per section: the sections inside its first given_bars are mine, the rest YuE2's
  const given = Number(song.yue_song?.given_bars) || 0;
  let bar = 0;
  const origin = secs.map((s) => {
    bar += Number(s.bars) || 0;
    return s.origin ?? (given ? (bar <= given ? "input" : "model") : undefined);
  });
  const lines = secs.map((s, i) => `<line x1="${at(s.start_sec, 1000)}" x2="${at(s.end_sec, 1000)}" stroke="${routeInk(origin[i])}" y1="6" y2="6"></line>`).join("");
  const stops = secs.map((s, i) => {
    const name = escapeHtml(s.label || s.name || "구간"), what = `${name}부터 재생 (${fmtTime(Number(s.start_sec) || 0)})`;
    return `<button type="button" class="route-stop${i % 2 ? " up" : ""}${origin[i] === "input" ? " mine" : ""}" data-sec="${i}" style="left: ${at(s.start_sec, 100)}%"`
      + ` aria-label="${what}" title="${what}">${name}</button>`;
  }).join("");
  return `<div class="route-track"><svg class="route-line" viewBox="0 0 1000 12" preserveAspectRatio="none" aria-hidden="true" focusable="false">${lines}</svg>`
    + `${stops}<i id="route-dot" class="route-dot" aria-hidden="true"></i></div>`
    + '<p class="route-key"><span class="k-seed">내 멜로디</span><span class="k-ai">모델이 쓴 구간</span><span class="k-rule">규칙으로 만든 구간</span></p>';
}
function renderRoute(song) {
  const box = $("song-route");
  if (!box) return;
  const secs = song?.sections || [];
  routeTotal = Number(song?.duration_sec) || Number(secs[secs.length - 1]?.end_sec) || 0;
  routeLib = null;                                          // another song: the library take no longer belongs to it
  box.innerHTML = routeHtml(song);
  box.onclick = (e) => {
    const stop = e.target?.closest?.("[data-sec]");
    return stop ? playRouteFrom(Number(stop.dataset.sec)) : undefined;
  };
}
function playRouteFrom(i) {
  const song = state.song, s = song?.sections?.[i];
  if (!s || !song.playback?.notes) return;
  const start = Number(s.start_sec) || 0;
  routeNext = start;
  return playEvents(notesInRange(song.playback.notes, { start, end: routeTotal || Infinity }), `${s.label || s.name || "구간"}부터 재생`);
}
function moveRouteDot(t) {
  const dot = $("route-dot");
  if (!dot?.style || !(routeTotal > 0) || reduceMotion()) return;
  dot.style.left = `${+(100 * Math.min(1, Math.max(0, t / routeTotal))).toFixed(3)}%`;
}

function renderSongSheet() {
  if (!state.song || !$("song-sheet-box").open || songOsmd || !window.opensheetmusicdisplay) return;
  songOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("song-sheet", {
    autoResize: true, drawTitle: false, drawComposer: false, drawCredits: false, drawPartNames: false });
  songOsmd.load(state.song.musicxml).then(() => songOsmd.render()).catch((e) => showError("곡 악보 표시 실패: " + e.message));
}

// Lead sheet text -> rule-based arrangement -> shown in the song card (nothing is adopted or stored)
let leadsheetSeq = 0;
async function renderLeadsheet() {
  const text = $("leadsheet-input").value;
  const err = $("leadsheet-error");
  err.classList.add("hidden");
  if (!text.trim()) { err.textContent = "리드시트 텍스트를 붙여넣으세요."; err.classList.remove("hidden"); return; }
  const body = { text }, key = $("leadsheet-key").value.trim();
  if (key) body.target_key = key;
  $("btn-leadsheet").disabled = true;
  const seq = ++leadsheetSeq;
  try {
    const previous = state.song;
    const data = await api("/api/leadsheet/render", body);
    if (seq !== leadsheetSeq) return;
    const carried = await carryMemberParts(data, previous);
    if (seq !== leadsheetSeq) return;
    showSong({ ...carried.result, from_leadsheet: true });
    if (carried.note && $("band-status")) $("band-status").textContent = carried.note;
  } catch (e) {
    if (seq === leadsheetSeq) { err.textContent = e.message; err.classList.remove("hidden"); }
  } finally { if (seq === leadsheetSeq) $("btn-leadsheet").disabled = false; }
}

function hideSong() {
  state.song = null; state.composed = null; state.selected = null;
  $("song").classList.add("hidden");
  $("compose-extras").classList.add("hidden");
}

// Natural-language request -> LLM tool calls on the server. Adopted as one version only if it finished with a change.
const AGENT_STATUS_KO = { ok: "완료", need_input: "질문", budget_exceeded: "예산 초과로 중단", llm_error: "LLM 오류" };
async function agentRequest() {
  const v = state.result?.version;
  const text = $("agent-text").value.trim();
  if (!v || state.editing) return;
  if (!text) return showError("요청 내용을 입력하세요.");
  hideError(); hideAgentAsk();
  setEditing(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const data = await api("/api/agent", { history: v.history, base_index: v.index, request: text });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    if (data.version) { state.changedBar = null; showResult(data); $("agent-text").value = ""; }
    showAgentReply(data.agent, !!data.version);
  } catch (e) { if (seq === resultRequestSeq && version === state.inputVersion) showError(e.message); }
  finally { setEditing(false); }
}

function showAgentReply(ag, adopted) {
  const box = $("agent-reply");
  const steps = (ag.log || []).map((e) => `${e.tool}(${JSON.stringify(e.args)}) → ${e.result?.status ?? ""}`).join("<br>");
  box.innerHTML = `<p><b>에이전트 (${AGENT_STATUS_KO[ag.status] || ag.status}):</b> ${escapeHtml(ag.message || "")}</p>`
    + `<p>${adopted ? "새 버전으로 적용했어요. ↶ 이전 버전으로 되돌릴 수 있어요." : "곡은 바뀌지 않았어요."}</p>`
    + (steps ? `<details><summary>도구 호출 ${ag.log.length}회</summary><p>${escapeHtml(steps).replace(/&lt;br&gt;/g, "<br>")}</p></details>` : "");
  // A check failed: the server kept the proposal for this version. Only this explicit click can force it (the model can not).
  for (const p of ag.pending || []) {
    if (p.can_force !== true || !Array.isArray(p.bars)) continue;
    const btn = textEl("button", `그래도 적용 (${p.bars.map((b) => b + 1).join(", ")}마디 ${p.mood === "darker" ? "어둡게" : "밝게"})`);
    btn.onclick = () => confirmAgent(p);
    box.append(btn, " ");
  }
  box.classList.remove("hidden");
}

// "그래도 적용" for a clash proposal of a natural-language request (bound by the server to this version, these bars, this mood)
async function confirmAgent(p) {
  const v = state.result?.version;
  if (!v || state.editing) return;
  hideError();
  setEditing(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const data = await api("/api/agent/confirm", { history: v.history, base_index: v.index, bars: p.bars, mood: p.mood });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    state.changedBar = p.bars;
    showResult(data);
  } catch (e) { if (seq === resultRequestSeq && version === state.inputVersion) showError(e.message); }
  finally { setEditing(false); }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

// "마디 a ~ b" (1-based, inclusive) -> one edit of those bars
function rangeEdit(mood) {
  if (!state.result || state.editing) return;
  const n = state.result.chords.length;
  const from = Number($("range-from").value), to = Number($("range-to").value);
  if (!Number.isInteger(from) || !Number.isInteger(to) || from < 1 || to > n || from > to) {
    return showError(`마디 범위는 1~${n} 사이에서 시작 ≤ 끝으로 입력하세요.`);
  }
  const bars = [];
  for (let b = from - 1; b <= to - 1; b++) bars.push(b);
  modifyBar(bars, mood);
}

// Tempo / track levels of the shown version -> a new version (server re-renders; chords and events unchanged).
async function applyMix() {
  const v = state.result?.version;
  if (!v || state.editing) return;
  const tempo = Number($("mix-tempo").value);
  if (!Number.isInteger(tempo) || tempo < 40 || tempo > 200) return showError("템포는 40~200 사이 정수로 입력하세요.");
  const levels = {};
  for (const [k, sel] of Object.entries(state.levelSelects || {})) levels[k] = Number(sel.value);
  hideError();
  setEditing(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const data = await api("/api/mix", { history: v.history, base_index: v.index, tempo, levels });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    state.changedBar = null;
    showResult(data);
  } catch (e) { if (seq === resultRequestSeq && version === state.inputVersion) showError(e.message); }
  finally { setEditing(false); }
}

// The agent could not find a chord that passes the check; the current version is unchanged.
// Server contract (agent.modify): tool_result.tried is an OBJECT keyed by the 0-based bar index (a JSON key, so a string),
// each value a list of {roman, symbol, clash}. Bars shown to the user are 1-based. Text is inserted with textContent.
function triedLines(tried) {
  if (!tried || typeof tried !== "object" || Array.isArray(tried)) return [];
  return Object.entries(tried)
    .map(([bar, list]) => [Number(bar), Array.isArray(list) ? list : []])
    .filter(([bar, list]) => Number.isInteger(bar) && list.length)
    .sort((a, b) => a[0] - b[0])
    .map(([bar, list]) => `${bar + 1}마디: ` + list.map((t) => `${t.symbol}(${t.clash})`).join(", "));
}

function textEl(tag, text) { const e = document.createElement(tag); e.textContent = text; return e; }

function askAgent(data, bar, mood) {
  const box = $("agent-ask");
  box.innerHTML = "";
  const q = document.createElement("p");
  q.append(textEl("b", "에이전트 질문:"), textEl("span", " " + String(data.question ?? "")));
  const lines = triedLines(data.tool_result?.tried);
  const tried = document.createElement("p");
  tried.append(textEl("span", "시도한 코드(충돌률): " + (lines.length ? lines.join(" / ") : "없음")));
  const apply = textEl("button", "그래도 적용");
  const keep = textEl("button", "그대로 두기");
  apply.onclick = () => modifyBar(bar, mood, true);
  keep.onclick = hideAgentAsk;
  box.append(q, tried);
  if (data.can_force === true) box.append(apply, " ");
  box.append(keep);
  box.classList.remove("hidden");
  renderTrace(data.trace || []);
}

function hideAgentAsk() { const b = $("agent-ask"); b.classList.add("hidden"); b.innerHTML = ""; }

// Undo returns the stored earlier version as-is (no regeneration).
async function undoVersion() {
  const v = state.result?.version;
  if (!v?.can_undo || state.editing) return;
  hideError();
  setEditing(true);
  const version = state.inputVersion, seq = ++resultRequestSeq;
  try {
    const data = await api("/api/undo", { history: v.history });
    if (seq !== resultRequestSeq || version !== state.inputVersion) return;
    state.changedBar = null;
    showResult(data);
  } catch (e) { if (seq === resultRequestSeq && version === state.inputVersion) showError(e.message); }
  finally { setEditing(false); }
}

// ---------- result ----------
// Any input change makes the shown result stale: drop it so it can't be played, modified or downloaded.
function invalidateResult() {
  if (state.result || state.composed) stopPlayback();
  state.inputVersion++;
  cancelCompose();
  state.result = null;
  state.changedBar = null;
  $("result").classList.add("hidden");
  $("ask").classList.add("hidden");   // a shown key question is stale too
  $("ask").innerHTML = "";
  hideAgentAsk();
  $("agent-reply").classList.add("hidden");
  if (state.song) hideSong();
}

let osmd = null;
function showResult(data) {
  hideAgentAsk();
  $("agent-reply").classList.add("hidden");   // an agent reply (and its confirm buttons) belongs to the previous version
  if (state.song) hideSong();          // a song was rendered from the previous version
  state.result = data;
  state.keyChoice = { tonic: data.key.tonic, mode: data.key.mode };
  $("result").classList.remove("hidden");
  if (currentStep() < 3) goStep(3);        // an accompaniment was made: on to "listen and edit"
  $("btn-undo").disabled = !data.version?.can_undo;
  if ($("btn-hq")) $("btn-hq").textContent = playLabel(data);
  $("key-view").textContent = `${data.key.label} · ${GENRES[data.genre]} · ${data.playback.bpm} BPM`
    + (data.status === "partial" ? " · 충돌 허용해 반영" : "");

  const box = $("chords");
  box.innerHTML = "";
  state.editButtons = [];
  for (const c of data.chords) {
    const el = document.createElement("div");
    el.className = "chord" + ((state.changedBar || []).includes(c.bar) ? " changed" : "");
    el.innerHTML = `<div><b>${c.bar + 1}마디</b> <span class="sym">${c.symbol}</span><span class="rn">${c.roman}</span></div>
      <p>${c.reason}</p>`;
    const dark = document.createElement("button"); dark.textContent = "어둡게";
    const bright = document.createElement("button"); bright.textContent = "밝게";
    dark.onclick = () => modifyBar(c.bar, "darker");
    bright.onclick = () => modifyBar(c.bar, "brighter");
    state.editButtons.push(dark, bright);
    el.append(dark, " ", bright);
    box.append(el);
  }
  renderTrace(data.trace);
  $("mix-tempo").value = String(data.tempo ?? data.playback.bpm);
  const nBars = data.chords.length;
  for (const id of ["range-from", "range-to"]) $(id).max = String(nBars);
  if (!(Number($("range-to").value) >= 1 && Number($("range-to").value) <= nBars)) {
    $("range-from").value = "1"; $("range-to").value = String(nBars);
  }
  for (const [k, sel] of Object.entries(state.levelSelects || {})) sel.value = String(data.levels?.[k] ?? 1);

  const chk = data.check;
  $("check").innerHTML = `<p>마디별 충돌률 (기준 ${chk.threshold} 이하):
    ${chk.clash_rates.map((r, i) => `<span class="${r > chk.threshold ? "bad" : "good"}">${i + 1}마디 ${r}</span>`).join(", ")}</p>
    <p>악기 음역 위반 ${chk.range_violations.length}건 · 마디 길이 위반 ${chk.length_violations.length}건</p>`;

  if (window.opensheetmusicdisplay) {
    if (!osmd) osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("sheet", {
      autoResize: true, drawTitle: false, drawComposer: false, drawCredits: false, drawPartNames: false,
    });
    osmd.load(data.musicxml).then(() => osmd.render()).catch((e) => showError("악보 표시 실패: " + e.message));
  }
}

function renderTrace(trace) {
  $("trace").innerHTML = trace.map((t) => `<li><b>${t.step}</b>${t.detail}</li>`).join("");
}

// Option (off by default): section markers are written into the downloaded copy only, never into the song's own MIDI.
function saveMidiMarked(b64, data, name) {
  const sections = $("midi-markers")?.checked ? audioSections(data) : null;
  if (!sections) return saveMidi(b64, name);
  api("/api/midi/markers", { midi_b64: b64, sections })
    .then((r) => saveMidi(r.midi_b64, name)).catch((e) => showError("MIDI 다운로드: " + e.message));
}

function downloadMidi(data = state.result, name = `jakgokmate_accompaniment_${state.result?.genre}.mid`) {
  if (!data) return;
  const sounds = chosenSounds();
  if (sounds) {                              // sounds changed on screen go into the file too (program numbers only)
    api("/api/midi", { midi_b64: data.midi_b64, sounds })
      .then((r) => saveMidiMarked(r.midi_b64, data, name)).catch((e) => showError("MIDI 다운로드: " + e.message));
    return;
  }
  saveMidiMarked(data.midi_b64, data, name);
}

function saveMidi(b64, name) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// M7: the server renders the same MIDI (mix applied) to MP3/WAV with FluidSynth; 503 explains what is missing.
// ---------- selectable sounds for the audio download (GET /api/sounds); "" = the genre's own sound
// The melody is the vocal line: it is chosen as a male / female voice (#vocal-style), not as an instrument.
const SOUND_PART = { guitar: "기타", bass: "베이스", drums: "드럼", piano: "피아노(건반)", keys: "멜로디 피아노", strings: "스트링" };
const EXTRA_INSTR = ["strings", "keys"];     // not band members' parts: kept in a folded box below the band line-up
const soundChoice = {};
// Licence mode of the server render: "release" uses tier-A instruments only (configs/instruments.yaml modes).
function currentMode() { return $("release-mode")?.checked ? "release" : "demo"; }
function showAudioNotes(res) {
  const box = $("audio-notes");
  if (!box) return;
  let notes = [];
  try { notes = JSON.parse(decodeURIComponent(res.headers.get("X-Instmate-Notes") || "[]")); } catch (e) { notes = []; }
  box.textContent = notes.length ? "음원 안내: " + notes.join(" / ") : "";
  box.classList.toggle("hidden", !notes.length);
}
async function loadCredits() {
  const box = $("credits");
  if (!box) return;
  try {
    const res = await fetch(`/api/credits?mode=${currentMode()}`);
    if (!res.ok) return;
    const data = await res.json();
    box.textContent = "";
    for (const r of data.rows) {
      const li = document.createElement("li");
      li.textContent = `[${r.tier}] ${r.name} — ${r.license}` + (r.credit ? ` · ${r.credit}` : "") + (r.allowed ? "" : " (배포 모드에서는 쓰지 않음)");
      box.append(li);
    }
  } catch (e) { /* optional panel */ }
}
// Who plays the melody (render.MELODY_ROLES on the server).  A song remembers the role it was made with; the audio
// then plays the melody with that instrument instead of singing it.
const ROLE_SOUND = { lead_guitar: "electric_guitar", acoustic_guitar: "acoustic_guitar", keys: "piano" };
const ROLE_KO = { vocal: "보컬", lead_guitar: "일렉 기타", acoustic_guitar: "어쿠스틱 기타", keys: "피아노" };
// The main play button says what will be heard: a sung melody, or the instrument that plays it.
function playLabel(data) { return sungMelody(data) ? "▶ 가이드 보컬로 미리듣기" : "▶ 실제 악기 소리로 듣기"; }
function melodyRole() { return $("melody-role")?.value || "vocal"; }
// Why an uploaded melody differs, said from the changes themselves: notes sounding together (one voice sings one note)
// and notes off the 16th-note grid are two different reasons, and a score may have only one of them.
function changeReasons(changes) {
  const list = changes || [];
  const together = list.some((c) => (c.text || "").includes("같은 순간"));
  const grid = list.some((c) => c.kind === "moved" || c.kind === "length" || (c.text || "").includes("같은 칸"));
  const why = [];
  if (together) why.push("멜로디는 한 번에 한 음이라, 함께 울리는 음은 맨 위 음만 남겨요");
  if (grid || !together) why.push("앱은 16분음표 칸 단위로 음을 놓아요");
  return why.join(" · ");
}
// The singer whose range the song's key is fitted to; null = keep the key as given (box off, or nobody sings).
function fitVoice() {
  return melodyRole() === "vocal" && $("fit-voice")?.checked ? $("vocal-style")?.value || null : null;
}
function sungMelody(data) { return (data?.melody_role || "vocal") === "vocal"; }
function audioSounds(data) {
  const chosen = chosenSounds() || {};
  const role = data?.melody_role || "vocal";
  if (ROLE_SOUND[role]) chosen.melody = ROLE_SOUND[role];
  return Object.keys(chosen).length ? chosen : null;
}
function chosenSounds() {
  const out = {};
  for (const [part, key] of Object.entries(soundChoice)) if (key) out[part] = key;
  return Object.keys(out).length ? out : null;
}
async function loadGenreNotes() {
  const box = $("genre-notes");
  if (!box) return;
  try {
    const res = await fetch(`/api/genres?mode=${currentMode()}`);
    if (!res.ok) return;
    box.textContent = "";
    for (const g of await res.json()) {
      const d = document.createElement("div");
      d.className = "genre-note";
      const h = document.createElement("b");
      h.textContent = `${g.label} · ${g.country} · 템포 ${g.tempo_range[0]}~${g.tempo_range[1]} · ${Math.round(g.length_s[0] / 60 * 10) / 10}~${Math.round(g.length_s[1] / 60 * 10) / 10}분`
        + (g.loop_bars ? ` · ${g.loop_bars}마디 루프` : "") + (g.last_chorus_shift ? " · 마지막 후렴 전조" : "")
        + (g.inherits.length ? ` · ${g.inherits.map((x) => GENRES[x] || x).join("·")}에서 물려받음` : "");
      const p = document.createElement("p");
      p.textContent = g.notes;
      const kit = document.createElement("p");
      kit.className = "kit";
      kit.textContent = "추천 악기: " + (g.kit_labels || []).map((k) => `${k.part} ${k.sound}${k.tier ? ` [${k.tier}]` : ""}`).join(" · ");
      d.append(h, p, kit);
      const lu = g.lineup || { parts: [], sections: [] };
      const changes = lu.sections.map((x) => [x.guitar ? `기타 → ${x.guitar.label}` : "",
        x.add.length ? `${x.add.join("·")} 추가 (켜 둔 경우)` : ""].filter(Boolean).join(", ")).map((t, i) => t && `${lu.sections[i].kind_label}: ${t}`).filter(Boolean);
      const subs = lu.parts.filter((x) => x.note).map((x) => `${x.part_label} ${x.label}: ${x.note}`);
      for (const [cls, text] of [["kit", changes.length ? "파트별 변화: " + changes.join(" / ") : ""], ["kit", subs.join(" / ")]]) {
        if (!text) continue;
        const e = document.createElement("p");
        e.className = cls;
        e.textContent = text;
        d.append(e);
      }
      box.append(d);
    }
  } catch (e) { /* optional panel */ }
}

// Opt-in sound features (GET /api/audio/options): one set of switches, shown under both result cards, all off at first.
const soundOptions = new Set();
function audioOptions() { return [...soundOptions]; }
async function loadSoundOptions() {
  const lists = document.querySelectorAll(".sound-option-list");
  if (!lists.length) return;
  try {
    const res = await fetch("/api/audio/options");
    if (!res.ok) return;
    const options = await res.json();
    for (const list of lists) {
      list.textContent = "";
      for (const o of options) {
        const lab = document.createElement("label");
        lab.className = "row";
        lab.title = o.help;
        const box = document.createElement("input");
        box.type = "checkbox";
        box.dataset.soundOption = o.id;
        if (o.default) { box.checked = true; soundOptions.add(o.id); }     // chosen by ear: on at first, can be switched off
        box.onchange = () => {
          if (box.checked) soundOptions.add(o.id); else soundOptions.delete(o.id);
          document.querySelectorAll(`input[data-sound-option="${o.id}"]`).forEach((b) => { b.checked = box.checked; });
        };
        lab.append(box, ` ${o.label} — ${o.help}`);
        list.append(lab);
      }
    }
  } catch (e) { /* optional panel: with no switches every render is the plain one */ }
}

async function loadSounds() {
  const box = $("sounds");
  if (!box) return;
  try {
    const res = await fetch(`/api/sounds?mode=${currentMode()}`);
    if (!res.ok) return;
    const cat = await res.json();
    box.textContent = "";
    for (const [part, label] of Object.entries(SOUND_PART)) {
      if (!cat[part]) continue;
      const lab = document.createElement("label");
      lab.textContent = label + " ";
      const sel = document.createElement("select");
      sel.dataset.part = part;
      sel.append(new Option("장르 추천 악기", ""));
      for (const o of cat[part]) sel.append(new Option(`${o.label} [${o.tier || "A"}]`, o.key));
      if (cat[part].some((o) => o.key === soundChoice[part])) sel.value = soundChoice[part];
      else delete soundChoice[part];              // a choice the mode no longer offers falls back to the genre's own
      sel.onchange = () => { soundChoice[part] = sel.value; };
      if (EXTRA_INSTR.includes(part)) lab.classList.add("extra");
      lab.append(sel);
      box.append(lab);
    }
  } catch (e) { /* optional panel: the download still works with the genre sounds */ }
}

// The song's sections for the server render: the guitar tone follows the genre's per-section instrument table.
function audioSections(data) {
  if (!Array.isArray(data?.sections) || !data.sections.length) return null;
  return data.sections.map((x) => ({ kind: x.kind, start_bar: x.start_bar, bars: x.bars }));
}

// High-quality preview: the same server render as the download (genre kit, humanize, mastering), played in the page.
let hqAudio = null, hqUrl = null;
async function renderAudio(data, practice = null) {
  const res = await fetch("/api/audio", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...(practice || {}),
      midi_b64: data.midi_b64, name: "preview", sounds: audioSounds(data), genre: data.genre || null,
      swing: $("swing-choice")?.value ? Number($("swing-choice").value) : null, vocal: false,                     // the built-in guide vocal only (no recorded voice on the page)
      vocal_style: sungMelody(data) ? $("vocal-style")?.value || null : null, sections: audioSections(data),
      exact_parts: (data.member_parts || []).map((m) => m.part), mode: currentMode(),
      vocal_fit: !!$("vocal-fit")?.checked, options: audioOptions() }) });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(typeof err.detail === "string" ? err.detail : `음원 만들기 실패 (${res.status})`);
  }
  showAudioNotes(res);
  return res.blob();
}

async function playHq(data, button, title, practice = null) {
  if (!data?.midi_b64) return;
  hideError();
  stopPlayback();
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "음원 만드는 중…";
  try {
    const blob = await renderAudio(data, practice);
    hqUrl = URL.createObjectURL(blob);
    hqAudio = new Audio(hqUrl);
    hqAudio.onended = () => stopPlayback();
    const audio = hqAudio;
    showPlayerBar(title, () => (Number.isFinite(audio.duration) ? audio.duration : 0), () => audio.currentTime || 0);
    await hqAudio.play();
  } catch (e) {
    // "■ 정지" pressed while the audio was still starting: the browser reports the stopped start as an error
    if (e?.name !== "AbortError") showError("실제 소리로 듣기: " + e.message);
    routeNext = null;                                        // no player bar came: the route does not wait for one
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function downloadAudio(data, name, button) {
  if (!data?.midi_b64) return;
  hideError();
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "음원 만드는 중…";
  try {
    const res = await fetch("/api/audio", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ midi_b64: data.midi_b64, name, sounds: audioSounds(data), genre: data.genre || null,
        swing: $("swing-choice")?.value ? Number($("swing-choice").value) : null, vocal: false,                     // the built-in guide vocal only (no recorded voice on the page)
      vocal_style: sungMelody(data) ? $("vocal-style")?.value || null : null, sections: audioSections(data),
      exact_parts: (data.member_parts || []).map((m) => m.part), mode: currentMode(),
      vocal_fit: !!$("vocal-fit")?.checked, options: audioOptions() }) });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(typeof err.detail === "string" ? err.detail : `음원 만들기 실패 (${res.status})`);
    }
    showAudioNotes(res);
    const ext = res.headers.get("Content-Type")?.includes("mpeg") ? "mp3" : "wav";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(await res.blob());
    a.download = `${name}.${ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    showError("음원 다운로드: " + e.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

// ---------- band pack: chord chart, part guide and per-part MIDI for the rehearsal room ----------
async function downloadBandPack() {
  const song = state.song;
  if (!song) return;
  hideError();
  const button = $("btn-bandpack");
  button.disabled = true;
  try {
    const res = await fetch("/api/bandpack", { method: "POST", headers: { "Content-Type": "application/json" },
      // an unchanged saved band version is exported from the server's stored copy, so the pack names that version
      body: JSON.stringify(bandParent?.clean ? { band_version_id: bandParent.version }
        : { result: song, melody_source: song.melody_source || null, sounds: chosenSounds(),
            band_id: $("band-select")?.value ? Number($("band-select").value) : null }) });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(typeof err.detail === "string" ? err.detail : `밴드 팩 만들기 실패 (${res.status})`);
    }
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url; a.download = `jakgokmate_bandpack_${song.genre || "song"}.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { showError(e.message); } finally { button.disabled = false; }
}

// ---------- own score import: the read score and the section split typed by the user ----------
let importedScore = null, importSeq = 0;

// "벌스 1-8, 후렴 9-16" -> [{name, start_bar, bars}] (bars from 0); null when a piece cannot be read
const SECTION_ALIAS = { "도입부": "인트로", "도입": "인트로", "전주": "인트로", "후크": "프리", "훅": "프리", "프리코러스": "프리",
  "코러스": "후렴", "사비": "후렴", "하이라이트": "후렴", "브리지": "브릿지", "엔딩": "아웃트로", "후주": "아웃트로" };   // = scoreimport.SECTION_ALIAS
function parseSections(text) {
  // "이름 시작-끝" pieces; a stray comma between the name and its bars ("벌스, 5-9") is forgiven
  const out = [], re = /([^\s,\d~-][^\s,]*)[\s,]*(\d+)\s*[-~]\s*(\d+)/g;
  const rest = (text || "").replace(re, "").replace(/[\s,]/g, "");
  for (const m of (text || "").matchAll(re)) {
    if (Number(m[3]) < Number(m[2]) || Number(m[2]) < 1) return null;
    out.push({ name: SECTION_ALIAS[m[1]] || m[1], start_bar: Number(m[2]) - 1, bars: Number(m[3]) - Number(m[2]) + 1 });
  }
  return out.length && !rest ? out : null;
}
// Sections must follow one another from bar 1 to the last bar; the first problem in plain words, or "".
function sectionGap(list, total) {
  let at = 0;
  for (const x of list) {
    if (x.start_bar < at) return `${x.name}(${x.start_bar + 1}마디부터)가 앞 구간과 겹쳐요. ${at + 1}마디부터 시작해 주세요.`;
    if (x.start_bar > at) return `${at + 1}~${x.start_bar}마디가 어느 구간에도 없어요.`;
    at += x.bars;
  }
  return at === total ? "" : at < total ? `${at + 1}~${total}마디가 어느 구간에도 없어요.` : `구간이 악보(${total}마디)보다 길어요.`;
}
const sectionsText = (list) => list.map((x) => `${x.name} ${x.start_bar + 1}-${x.start_bar + x.bars}`).join(", ");

// What differs between the uploaded file and the melody the app can keep (16th-note grid, one tempo).  Nothing is
// made from such a converted version until the user ticks the box; `prefix` is "import" or "score" (the two cards).
function renderScoreChanges(prefix, score) {
  const box = $(`${prefix}-changes`);
  if (!box) return;
  const exact = !score || score.exact !== false;
  box.classList.toggle("hidden", exact);
  if ($(`${prefix}-accept`)) $(`${prefix}-accept`).checked = false;
  if (exact) return;
  const n = score.changes_total || 0;
  $(`${prefix}-changes-text`).textContent = (score.omr ? score.omr.note
    + (score.omr.pages_empty?.length ? ` ${score.omr.pages_empty.join(", ")}쪽은 악보를 읽지 못해 뺐어요.` : "") + " "
    : "이 악보는 그대로 가져올 수 없어요. ")
    + (n ? `멜로디 ${n}곳이 달라져요(${changeReasons(score.changes)}). ` : "") + (score.tempo_note || "");
  const ul = $(`${prefix}-changes-list`);
  ul.textContent = "";
  for (const c of score.changes || []) {
    const li = document.createElement("li");
    li.textContent = `${c.bar}마디 ${c.beat}박 ${c.note}: ${c.text}`;
    ul.append(li);
  }
  if (n > (score.changes || []).length) {
    const li = document.createElement("li");
    li.textContent = `... 그 밖에 ${n - score.changes.length}곳`;
    ul.append(li);
  }
  if (score.omr) {                        // a recognized PDF: what was read, bar by bar, to compare with the paper
    for (const line of barNoteLines(score)) {
      const li = document.createElement("li");
      li.textContent = line;
      ul.append(li);
    }
  }
}
const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
function barNoteLines(score, maxBars = 64) {
  const bars = new Map();
  for (const x of score.notes || []) {
    const b = Math.floor(x.start / score.bar_ticks);
    if (!bars.has(b)) bars.set(b, []);
    bars.get(b).push(`${PITCH_NAMES[x.pitch % 12]}${Math.floor(x.pitch / 12) - 1}`);
  }
  const out = [];
  for (let b = 0; b < Math.min(score.bars || 0, maxBars); b++) out.push(`읽은 음 ${b + 1}마디: ${(bars.get(b) || ["(쉼)"]).join(" ")}`);
  if ((score.bars || 0) > maxBars) out.push(`... ${score.bars - maxBars}마디 더 있어요`);
  return out;
}
// true when the score may be used: it is exact, or the user accepted the converted version (else says what to do)
function scoreAccepted(prefix, score) {
  if (!score || score.exact !== false || $(`${prefix}-accept`)?.checked) return true;
  if (prefix === "import") setStart("score");
  if (currentStep() !== 1) goStep(1);
  showError("이 악보는 그대로 가져올 수 없어 달라지는 곳이 있어요. 목록을 확인하고 \"이 변환본으로 진행할게요\"에 표시해 주세요.");
  $(`${prefix}-changes`)?.scrollIntoView?.({ block: "center", behavior: "instant" });
  $(`${prefix}-accept`)?.focus?.();
  return false;
}

// The melody the app read from the file, drawn in the score card (so step 1 shows the score that was uploaded).
let importOsmd = null;
async function showImportSheet(xml, seq) {
  const box = $("import-sheet");
  if (!box) return;
  box.innerHTML = "";
  importOsmd = null;
  if (!xml || !window.opensheetmusicdisplay) return;
  try {
    importOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("import-sheet", {
      autoResize: true, drawTitle: false, drawComposer: false, drawCredits: false, drawPartNames: false });
    await importOsmd.load(xml);
    if (seq === importSeq) importOsmd.render();
  } catch (e) { /* the summary and the list of changes still say what was read */ }
}

async function readScoreFile() {
  const input = $("import-file");
  const file = input.files && input.files[0];
  if (!file) return;
  hideError();
  const pdf = /\.pdf$/i.test(file.name || "");
  if (file.size > (pdf ? PDF_MAX_BYTES : SCORE_MAX_BYTES)) {
    input.value = "";
    return showError(pdf ? "PDF가 너무 커요(최대 30MB)." : "악보 파일이 너무 커요(최대 2MB).");
  }
  const form = new FormData();
  form.append("file", file);
  const seq = ++importSeq;
  // the score already read stays (and stays usable) until the new file has been read successfully (review F06)
  const kept = importedScore ? " 이전에 올린 악보는 그대로 남아 있어요." : "";
  $("import-status").textContent = pdf ? "PDF 악보를 인식하고 있어요... 쪽마다 수십 초가 걸려요." : "악보를 읽고 있어요...";
  try {
    const res = await fetch("/api/score/read", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (seq !== importSeq) return;
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `악보 읽기 실패 (${res.status})`);
    if ($("import-file-name")) $("import-file-name").textContent = `올린 파일: ${file.name}`;   // the file field itself is emptied after reading
    displayImportedScore(data, seq);
  } catch (e) {
    if (seq === importSeq) { $("import-status").textContent = "새 파일을 읽지 못했어요: " + e.message + kept; showError("내 악보 가져오기: " + e.message + kept); }
  } finally { input.value = ""; }
}

function displayImportedScore(data, seq) {
  importedScore = data.score;
  const sc = data.score;
  $("import-summary").textContent = `${sc.bars}마디 · ${sc.meter} · ${sc.bpm} BPM · 음 ${sc.notes.length}개 · 조 ${sc.key} (${sc.key_note})`
    + (sc.snapped ? ` · 격자에 맞추며 ${sc.snapped}개 음의 위치가 눈에 띄게 옮겨졌어요` : "")
    + (sc.dropped ? ` · 겹치는 음 ${sc.dropped}개는 뺐어요(멜로디는 한 번에 한 음)` : "");
  renderScoreChanges("import", sc);
  showImportSheet(data.musicxml, seq);
  $("import-key").value = sc.key;
  $("import-sections").value = sectionsText(data.sections);
  $("import-box").classList.remove("hidden");
  $("import-status").textContent = "읽었어요. 구간을 확인하고 아래에서 골라 주세요.";
}

async function importSeed() {
  if (!importedScore || !scoreAccepted("import", importedScore)) return;
  hideError();
  const score = importedScore;
  try {
    const d = await api("/api/score/tokens", { score, start_bar: Math.max(0, Number($("import-start").value || 1) - 1), bars: 8,
      accept_changes: true });
    if (importedScore !== score) return;
    state.tokens = d.tokens;
    // the second voice written under these bars; it belongs to exactly this melody (seedHarmony)
    state.seedHarmony = d.harmony?.length ? { entries: d.harmony, ticks: d.seed_ticks, of: JSON.stringify(d.tokens) } : null;
    state.inputTonic = d.input_tonic;
    $("input-tonic").value = String(d.input_tonic);
    melodyChanged();
    setStart("type");
    goStep(1, `악보의 ${d.bars}마디를 멜로디 입력에 채웠어요. 장르·악기를 확인하고 다음 단계로 넘어가세요.`);
    $("import-status").textContent = `멜로디 입력에 ${d.bars}마디를 채웠어요.`
      + (d.harmony?.length ? ` 멜로디 아래에 적힌 음 ${d.harmony.length}개는 화음 파트로 따라가요(멜로디를 고치면 화음은 빠져요).` : "");
  } catch (e) { showError("내 악보 가져오기: " + e.message); }
}

// The second voice of the imported seed, sent with the song requests while the melody is still the imported one.
function seedHarmony() {
  const h = state.seedHarmony;
  return h && h.of === JSON.stringify(state.tokens) ? { harmony: h.entries, seed_ticks: h.ticks } : {};
}
// The make step's two ways to use an uploaded score.
async function makeScoreSeedSong() {
  if (!importedScore || needStyle() || !scoreAccepted("import", importedScore)) return;
  await importSeed();                          // fills the melody input with the chosen 8 bars (and shows that input)
  if (!state.tokens.length) return;
  goStep(2);
  await startCompose();
}

async function importSong() {
  if (!importedScore || needStyle() || !scoreAccepted("import", importedScore)) return;
  hideError();
  const sections = parseSections($("import-sections").value);
  if (!sections) {
    if (currentStep() !== 1) goStep(1);        // the section split is edited on the score card
    return showError('구간은 "벌스 1-8, 후렴 9-16"처럼 적어 주세요.');
  }
  const gap = sectionGap(sections, Number(importedScore.bars));
  if (gap) {
    if (currentStep() !== 1) goStep(1);
    return showError("구간 나누기: " + gap + ' 예: "인트로 1-4, 벌스 5-9, 후렴 10-13"');
  }
  if ($("compose-status")) $("compose-status").textContent = "악보 그대로 편곡하고 있어요...";
  const score = importedScore, seq = ++importSeq, previous = state.song;
  $("btn-import-song").disabled = true;
  setMaking(true);
  $("import-status").textContent = "편곡하고 있어요...";
  try {
    const data = await api("/api/score/song", requestWithSettings({ score, sections, genre: state.genre, key: $("import-key").value.trim() || null,
      accept_changes: true, melody_role: melodyRole(), fit_voice: fitVoice(),
      instruments: [...state.instruments] }));
    if (seq !== importSeq) return;
    const carried = await carryMemberParts(data, previous);
    if (seq !== importSeq) return;
    stopPlayback();
    showSong(carried.result);
    if (window.workspaceController) autoYueSong(); else autoAiVocal();
    const fromFile = data.chords_from?.source === "file";     // the file's own chord symbols were kept as written
    $("import-status").textContent = `곡으로 만들었어요 (${data.bars}마디). 멜로디는 악보 그대로이고, `
      .replace("멜로디는 악보 그대로이고", data.imported_score?.exact === false ? "멜로디는 확인하신 변환본이고(악보 그대로가 아니에요)" : "멜로디는 악보 그대로이고")
      + (fromFile ? `코드는 악보에 적힌 것을 그대로 썼어요. ${data.chords_from.note || ""}`
        : `코드는 악보에서 읽지 못해 앱이 붙였어요(${data.chords_from?.label || "앱의 코드 도구"}). 원곡의 코드가 아니에요.`)
      + (data.harmony?.notes ? ` 멜로디 아래에 적힌 음 ${data.harmony.notes}개는 화음 파트로 넣었어요.` : "")
      + ` 코드를 바꾸려면 "3. 곡 완성"의 리드시트 텍스트에서 코드를 고쳐 "리드시트에서 곡 만들기"에 붙여 넣으세요.`;
  } catch (e) {
    if (seq === importSeq) { $("import-status").textContent = ""; showError("내 악보 가져오기: " + e.message); }
  } finally {
    setMaking(false);
    if (seq === importSeq) {
      $("btn-import-song").disabled = false;
      if ($("compose-status")?.textContent === "악보 그대로 편곡하고 있어요...") $("compose-status").textContent = "";
    }
  }
}

// ---------- one step per page: 1 genre -> 2 melody -> 3 make -> 4 listen and edit ----------
// The <main> element carries the current step and the chosen way to enter the melody; CSS shows only the matching cards.
const STEP_HINT = { 1: "멜로디를 넣고 장르·악기와 보컬을 고른 뒤 다음으로 넘어가세요.",
  2: "\"반주 만들기\"나 \"곡 완성\"을 누르면 결과 단계로 넘어가요.", 3: "" };
function currentStep() { return Number($("steps").dataset.step || 1); }
function goStep(n, hint = "") {
  const step = Math.min(3, Math.max(1, n));
  $("steps").dataset.step = String(step);
  for (const b of document.querySelectorAll("[data-go]")) {
    const k = Number(b.dataset.go);
    b.classList?.toggle("on", k === step);
    b.classList?.toggle("done", k < step);
  }
  // the make step offers what fits the melody at hand: an uploaded score, or the typed / filled-in melody
  const fromScore = $("steps").dataset.start === "score" && !!importedScore;
  $("make-score")?.classList?.toggle("hidden", !fromScore);
  $("make-typed")?.classList?.toggle("hidden", fromScore && !state.tokens.length);
  if ($("make-style")) $("make-style").textContent = makeStyleText();
  $("btn-step-prev").disabled = step === 1;
  if (step === 1 && importOsmd) { try { importOsmd.render(); } catch (e) { /* drawn again on the next read */ } }
  $("btn-step-next").disabled = step === 3;
  // the make step moves on by itself when something is made; the button shows there only to go back to an earlier result
  $("btn-step-next").classList?.toggle("hidden", step === 2 && !state.result && !state.song);
  const empty = step === 3 && !state.result && !state.song;
  $("step-hint").textContent = hint || (empty ? "아직 만든 곡이 없어요. 2단계에서 만들어 주세요." : STEP_HINT[step]);
  $("stepper").scrollIntoView?.({ block: "start", behavior: "instant" });   // each step starts at its top
}
// The genre and line-up are the user's to choose, also when the melody comes from an uploaded score: nothing is made
// with the silent default.  Returns true (and shows the genre card) while no genre was picked.
function needStyle() {
  if (state.src.genre) return needVoice();
  if (currentStep() !== 1) goStep(1);
  $("step-hint").textContent = "먼저 아래 \"장르·악기\"에서 장르와 편성을 골라 주세요.";
  showError("장르를 아직 고르지 않았어요. \"장르·악기\"에서 장르와 편성을 고른 뒤 다시 눌러 주세요.");
  $("style-card")?.scrollIntoView?.({ block: "start", behavior: "instant" });
  return true;
}
// The melody is sung: the singer (female / male) is the user's to choose as well, so the page does not start with one.
function needVoice() {
  if (melodyRole() !== "vocal" || $("vocal-style")?.value !== "") return false;
  if (currentStep() !== 1) goStep(1);
  $("step-hint").textContent = "먼저 \"멜로디를 누가 맡나요\"에서 여성·남성 보컬을 골라 주세요.";
  showError("보컬을 아직 고르지 않았어요. 여성·남성 중 하나를 고른 뒤 다시 눌러 주세요.");
  $("vocal-style")?.scrollIntoView?.({ block: "center", behavior: "instant" });
  return true;
}
function makeStyleText() {
  const parts = [...state.instruments].map((k) => INSTR[k] || k).join("·");
  const voice = melodyRole() === "vocal" ? ` (${$("vocal-style")?.value === "male" ? "남성" : "여성"}${fitVoice() ? ", 음역 맞춤 켜짐" : ""})` : "";
  const seed = SEED_KO[$("seed-role")?.value || "chorus"]
    + ($("seed-role")?.value === "intro" && melodyRole() === "vocal" ? ($("intro-player")?.value === "vocal" ? ", 보컬" : ", 리드 기타가 연주") : "");
  return state.src.genre ? `장르: ${GENRES[state.genre] || state.genre} · 편성: ${parts} · 멜로디: ${ROLE_KO[melodyRole()]}${voice}`
    + ` · "곡 완성"에서 입력 멜로디의 자리: ${seed} (바꾸려면 1단계의 "장르·악기")` : "";
}
const SEED_KO = { chorus: "후렴(하이라이트)", verse: "벌스", intro: "인트로" };
// "who plays the intro" is asked only when the input is the intro and the song is sung
function syncIntroPlayer() {
  $("intro-player-row")?.classList?.toggle("hidden", !($("seed-role")?.value === "intro" && melodyRole() === "vocal"));
  $("vocal-style-row")?.classList?.toggle("hidden", melodyRole() !== "vocal");       // nobody sings: no singer to choose
}
// One line under a finished song that says what it really is (an imported score is not "a seed in the chorus").
function songAbout(data) { return songAboutBase(data) + (data.voice_fit?.text ? " " + data.voice_fit.text : ""); }
function songAboutBase(data) {
  const role = data.melody_role || "vocal";
  const heard = role === "vocal" ? "가사는 없고, 멜로디는 가이드 보컬로 들을 수 있어요."
    : `멜로디는 ${ROLE_KO[role] || role}가 맡아요(노래하지 않아요).`;
  if (data.from_leadsheet) return "붙여 넣은 리드시트의 멜로디와 코드를 그대로 두고 밴드 편곡을 붙였어요. " + heard;
  if (data.imported_score) {
    const moved = data.voice_fit?.shift ? "올린 악보의 멜로디(조만 옮김)" : "올린 악보의 멜로디 그대로";
    return (data.imported_score.exact === false ? "올린 악보의 멜로디(확인하신 변환본)" : moved)
      + "에 코드와 밴드 편곡을 붙였어요. " + heard;
  }
  const where = { verse: "벌스", intro: "인트로" }[data.settings?.seed_role] || "후렴";
  const rest = data.settings?.fill === "rest" ? "나머지 구간은 멜로디 없이 반주만 있어요."
    : data.melody_source === "own" ? "나머지 구간의 멜로디는 입력을 규칙으로 변형한 것이에요."
      : "나머지 구간의 멜로디와 코드는 앱이 만들어 붙였어요.";
  const kept = data.voice_fit?.shift ? "조만 옮겨" : "그대로";
  const riff = data.intro_player === "guitar" ? " 인트로의 멜로디는 리드 기타가 연주하고 노래는 그 뒤부터예요." : "";
  return `입력한 멜로디를 ${where}에 ${kept} 두고, ${rest} ${heard}${riff}`;
}

// The harmony diagnosis of a finished song: listed for the band to look at, never applied by the app.
function renderHarmony(rep) {
  const box = $("harmony-box");
  if (!box) return;
  box.classList?.toggle("hidden", !rep);
  if (!rep) return;
  $("harmony-count").textContent = rep.summary?.total ? `${rep.summary.total}건` : "없음";
  $("harmony-about").textContent = rep.about || "";
  const ul = $("harmony-list");
  ul.textContent = "";
  for (const n of rep.notes || []) {
    const li = document.createElement("li");
    li.textContent = `[${n.level}] ${n.bar}마디 ${n.beat}박 · ${n.text}` + (n.on_input ? " (입력하신 멜로디: 앱이 바꾸지 않아요)" : "");
    ul.append(li);
  }
}
// Humming or a recording as the melody (POST /api/hum, user decision 2026-10-13: back on the page).  What was written
// down is shown first; the melody input changes only when the user presses "이 멜로디 쓰기".
let humPending = null, humRec = null;
async function humSend(blob, name) {
  const status = $("hum-status");
  status.textContent = "받아 적는 중…";
  try {
    const form = new FormData();
    form.append("file", blob, name);
    const bpm = Number($("style-tempo")?.value) || DEFAULT_BPM[state.genre] || 90;
    const res = await fetch(`/api/hum?bpm=${bpm}`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));            // a server fault answers in plain text, not JSON
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `받아 적지 못했어요 (서버 ${res.status ?? "오류"})`);
    humPending = data;
    const shown = data.tokens.map((t) => `${t.sharp ? "#" : ""}${t.degree}`).join(" ");
    const cut = data.cut ? ` 녹음에서 ${data.bars_heard}마디(${data.notes_heard}음)를 들었고, 작곡 씨앗에는 앞 ${data.bars}마디(첫 음부터 약 ${data.kept_seconds}초)만 들어가요. 뒷부분은 쓰지 않아요.` : "";
    $("hum-result-text").textContent = `받아 적었어요: ${data.notes}음, ${data.bars}마디, ${data.key} 추정 (템포 ${data.bpm}).${cut} 숫자 악보: ${shown}`;
    $("hum-result").classList.remove("hidden");
    status.textContent = "";
  } catch (e) {
    status.textContent = "";
    showError("허밍·녹음 입력: " + e.message);
  }
}
function humFromFile() {
  const file = $("hum-file")?.files?.[0];
  if (file) return humSend(file, file.name);
}
async function humRecord() {
  const btn = $("btn-hum");
  if (humRec) { humRec.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    return showError("이 브라우저는 녹음을 지원하지 않아요. 녹음 파일을 올려 주세요.");
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream), chunks = [];
    humRec = rec;
    rec.ondataavailable = (e) => chunks.push(e.data);
    const limit = setTimeout(() => rec.state === "recording" && rec.stop(), 60000);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      humRec = null;
      clearTimeout(limit);
      btn.textContent = "🎤 녹음해서 넣기";
      await humSend(new Blob(chunks, { type: rec.mimeType }), "melody.webm");
    };
    rec.start();
    btn.textContent = "■ 멈추기 (녹음 중)";
  } catch (e) { humRec = null; showError("마이크를 쓸 수 없어요: " + e.message); }
}
function humUse() {
  if (!humPending) return;
  state.tokens = humPending.tokens;
  state.inputTonic = humPending.input_tonic;
  if ($("input-tonic")) $("input-tonic").value = String(humPending.input_tonic);
  humPending = null;
  $("hum-result").classList.add("hidden");
  melodyChanged();
  setStart("type");                                         // shown in the melody input, to check and edit
}
function humDrop() {
  humPending = null;
  $("hum-result").classList.add("hidden");
}
function setStart(kind) {
  $("steps").dataset.start = kind;
  for (const b of document.querySelectorAll("[data-start-go]")) b.classList?.toggle("on", b.dataset.startGo === kind);
}
// "다음 단계": a step is left only when it has what the next one needs.
function nextStep() {
  const step = currentStep();
  if (step === 1 && !state.tokens.length && $("steps").dataset.start === "score" && importedScore) {
    if (needStyle()) return;
    return goStep(2, "올린 악보로 어떻게 만들지 골라 주세요.");      // an uploaded score is enough to move on
  }
  if (step === 1 && state.song && !state.tokens.length) {
    return goStep(3, "이미 만든 곡이 있어요. 여기서 듣고 고쳐요.");
  }
  if (step === 1 && !state.tokens.length) {
    const how = $("steps").dataset.start;
    return goStep(1, how === "type" ? "먼저 멜로디를 입력해 주세요." : how === "score"
      ? "먼저 악보 파일을 올려 주세요."
      : "리드시트를 붙여 넣고 \"곡 만들기\"를 눌러 주세요.");
  }
  if (step === 2 && !state.result && !state.song) return goStep(2, "먼저 \"반주 만들기\"나 \"곡 완성\"을 눌러 주세요.");
  if (step === 1 && needStyle()) return;
  goStep(step + 1);
}

// ---------- part editor: change the notes of one part in one section, inside the app ----------
// The server contract (POST /api/part/edit): only notes starting in the range are replaced; everything else in the part
// stays.  Here: notes the user does not touch keep their exact tick and velocity; the grid and the three levels apply
// only to notes that are added or explicitly changed.
const EDIT_STEP = 120;                                   // one cell = a 16th note (480 ticks per quarter note)
const EDIT_LEVELS = { soft: 55, mid: 85, hard: 112 };    // MIDI velocity of 약 / 중 / 강 (new or explicitly changed notes)
const EDIT_RANGE = { bass: [28, 60], guitar: [40, 76], lead: [52, 88], piano: [48, 84] };
const DRUM_ROWS = [[49, "크래시"], [51, "라이드"], [46, "하이햇(열림)"], [42, "하이햇"], [50, "탐 높음"], [47, "탐 중간"],
  [43, "탐 낮음"], [38, "스네어"], [36, "킥"], [54, "탬버린"]];
const NOTE_LABEL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const EG = { cell: 11, row: 12, label: 76 };
let editDraft = null, editSeq = 0;

function editSections() {
  const sel = $("edit-section"), song = state.song;
  if (!sel || !song) return;
  const keep = sel.value;
  sel.textContent = "";
  song.sections.forEach((x, i) => sel.append(new Option(`${x.label} (${x.start_bar + 1}~${x.start_bar + x.bars}마디)`, String(i))));
  if (keep && Number(keep) < song.sections.length) sel.value = keep;
}

function editRows(d) {
  if (d.part === "drums") {
    const extra = [...new Set(d.notes.map((n) => n.pitch))].filter((p) => !DRUM_ROWS.some(([q]) => q === p)).sort((a, b) => b - a);
    return [...DRUM_ROWS.map(([p, name]) => ({ pitch: p, name })), ...extra.map((p) => ({ pitch: p, name: `타악기 ${p}` }))];
  }
  const [lo0, hi0] = EDIT_RANGE[d.part];
  const ps = [...d.notes, ...d.ringing].map((n) => n.pitch);
  const lo = Math.min(lo0, ...ps), hi = Math.max(hi0, ...ps), rows = [];
  for (let p = hi; p >= lo; p--) rows.push({ pitch: p, name: `${NOTE_LABEL[p % 12]}${Math.floor(p / 12) - 1}` });
  return rows;
}

async function openPartEditor() {
  const song = state.song;
  if (!song) return;
  hideError();
  stopEditPreview();
  const sec = song.sections[Number($("edit-section").value || 0)], part = $("edit-part").value;
  const seq = ++editSeq;
  editDraft = null;
  $("edit-panel").classList.add("hidden");
  $("edit-status").textContent = "여는 중…";
  try {
    const v = await api("/api/part/section", { result: song, part, start_bar: sec.start_bar, bars: sec.bars });
    if (seq !== editSeq || state.song !== song) return;
    if (!v.has_part) { $("edit-status").textContent = "이 곡에는 그 파트가 없어요. 악기를 켜고 곡을 만든 뒤 편집해 주세요."; return; }
    const notes = v.notes.map((n, i) => ({ ...n, id: i }));
    editDraft = { song, part, start_bar: sec.start_bar, bars: sec.bars, label: sec.label, total: v.total_ticks, bar: v.bar_ticks,
      notes, ringing: v.ringing, original: JSON.stringify(notes.map(editKey)), nextId: notes.length, selected: null,
      undo: [], redo: [], playing: false };
    $("edit-panel").classList.remove("hidden");
    $("edit-status").textContent = `${sec.label} · ${PART_KO[part] || part}`;
    renderEditGrid();
  } catch (e) { if (seq === editSeq) { $("edit-status").textContent = ""; showError("파트 편집: " + e.message); } }
}

const editKey = (n) => [n.start, n.dur, n.pitch, n.velocity];
const editChanged = (d) => JSON.stringify(d.notes.map(editKey)) !== d.original;

function renderEditGrid() {
  const d = editDraft, box = $("edit-grid");
  if (!d || !box) return;
  const rows = editRows(d), cols = d.total / EDIT_STEP;
  const W = EG.label + cols * EG.cell, H = rows.length * EG.row;
  const y = new Map(rows.map((r, i) => [r.pitch, i * EG.row]));
  const x = (t) => EG.label + (t / EDIT_STEP) * EG.cell;
  let h = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
  rows.forEach((r, i) => {
    const black = d.part !== "drums" && [1, 3, 6, 8, 10].includes(r.pitch % 12);
    h += `<rect x="0" y="${i * EG.row}" width="${W}" height="${EG.row}" class="${black ? "eg-row-black" : "eg-row-white"}"/>`
      + `<text class="eg-label" x="4" y="${i * EG.row + 9}">${r.name}</text>`;
  });
  for (let c = 0; c <= cols; c++) {
    const t = c * EDIT_STEP, strong = t % d.bar === 0, beat = t % 480 === 0;
    h += `<line class="${strong ? "eg-line-bar" : beat ? "eg-line-beat" : "eg-line-sub"}" x1="${x(t)}" y1="0" x2="${x(t)}" y2="${H}"/>`;
  }
  for (const n of d.ringing) if (y.has(n.pitch))
    h += `<rect class="eg-ring" x="${EG.label}" y="${y.get(n.pitch) + 2}" width="${Math.max(3, x(n.start + n.dur) - EG.label)}" height="${EG.row - 4}"/>`;
  for (const n of d.notes) if (y.has(n.pitch)) {
    const w = Math.max(4, (Math.min(n.start + n.dur, d.total) - n.start) / EDIT_STEP * EG.cell - 1);
    h += `<rect class="eg-note${n.id === d.selected ? " sel" : ""}${n.start + n.dur > d.total ? " over" : ""}" x="${x(n.start)}" y="${y.get(n.pitch) + 1}"`
      + ` width="${w}" height="${EG.row - 2}" rx="2" opacity="${(0.45 + 0.55 * n.velocity / 127).toFixed(2)}"/>`;
  }
  box.innerHTML = h + "</svg>";
  const sel = d.notes.find((n) => n.id === d.selected);
  $("edit-info").textContent = (sel ? `선택한 음: ${editRows(d).find((r) => r.pitch === sel.pitch)?.name || sel.pitch} · 시작 ${sel.start}틱 · 길이 ${sel.dur}틱 · 세기 ${sel.velocity}` : "음을 누르면 선택돼요.")
    + ` · 음 ${d.notes.length}개` + (editChanged(d) ? " · 바뀐 내용이 있어요(아직 적용 전)" : " · 바뀐 내용 없음")
    + (d.ringing.length ? ` · 회색 = 앞 구간에서 이어지는 음 ${d.ringing.length}개(고칠 수 없음)` : "");
  $("btn-edit-undo").disabled = !d.undo.length;
  $("btn-edit-redo").disabled = !d.redo.length;
}

function editSnapshot() {
  const d = editDraft;
  d.undo.push(JSON.stringify(d.notes));
  if (d.undo.length > 200) d.undo.shift();
  d.redo = [];
}
function editAfterChange() { renderEditGrid(); if (editDraft.playing) scheduleEditPreview(); }

// A click in the grid: on a note = select it; on an empty cell = add a note there (new note: grid, chosen length and level).
function editCell(col, rowIndex) {
  const d = editDraft;
  if (!d) return;
  const row = editRows(d)[rowIndex];
  if (!row || col < 0 || col >= d.total / EDIT_STEP) return;
  const t = col * EDIT_STEP;
  const hit = d.notes.find((n) => n.pitch === row.pitch && n.start <= t && t < n.start + Math.max(n.dur, EDIT_STEP));
  if (hit) { d.selected = hit.id; renderEditGrid(); return; }
  editSnapshot();
  const want = d.part === "drums" ? EDIT_STEP : Number($("edit-len").value || 240);
  const note = { id: d.nextId++, start: t, dur: Math.min(want, d.total - t), pitch: row.pitch, velocity: EDIT_LEVELS[$("edit-level").value] || EDIT_LEVELS.mid };
  d.notes.push(note);
  d.selected = note.id;
  editAfterChange();
}

function editSelected(change) {
  const d = editDraft, n = d?.notes.find((x) => x.id === d.selected);
  if (!n) return;
  const rows = editRows(d).map((r) => r.pitch);
  const next = { ...n };
  change(next, rows);
  if (next.start < 0 || next.start >= d.total || next.dur < 1 || !rows.includes(next.pitch)) return;
  if (d.notes.some((x) => x.id !== n.id && x.pitch === next.pitch && x.start === next.start)) return;   // same slot taken
  if (JSON.stringify(editKey(next)) === JSON.stringify(editKey(n))) return;
  editSnapshot();
  Object.assign(n, next);
  editAfterChange();
}
function editDelete() {
  const d = editDraft;
  if (!d || d.selected == null || !d.notes.some((n) => n.id === d.selected)) return;
  editSnapshot();
  d.notes = d.notes.filter((n) => n.id !== d.selected);
  d.selected = null;
  editAfterChange();
}
const editMove = (dt) => editSelected((n) => { n.start += dt; });
const editPitch = (dir) => editSelected((n, rows) => { const i = rows.indexOf(n.pitch) - dir; if (rows[i] != null) n.pitch = rows[i]; });
const editLength = (dd) => editSelected((n) => { n.dur = Math.max(EDIT_STEP, Math.round((n.dur + dd) / EDIT_STEP) * EDIT_STEP); });
const editLevel = () => editSelected((n) => { n.velocity = EDIT_LEVELS[$("edit-level").value] || n.velocity; });
function editUndo(redo) {
  const d = editDraft, from = redo ? d?.redo : d?.undo, to = redo ? d?.undo : d?.redo;
  if (!d || !from.length) return;
  to.push(JSON.stringify(d.notes));
  d.notes = JSON.parse(from.pop());
  if (!d.notes.some((n) => n.id === d.selected)) d.selected = null;
  editAfterChange();
}

// Preview of the draft (not yet applied), looped over the section; other parts come from the shown song.
function editPreviewNotes() {
  const d = editDraft, spt = 60 / (d.song.playback?.bpm ?? d.song.tempo) / 480, len = d.total * spt;
  const mine = d.notes.map((n) => ({ track: d.part, midi: n.pitch, time: n.start * spt,
    duration: Math.min(n.dur, d.total - n.start) * spt, velocity: n.velocity / 127 }));
  if (!$("edit-with-band").checked) return { notes: mine, len };
  const start = d.start_bar * d.bar * spt;
  const others = notesInRange(d.song.playback.notes.filter((n) => n.track !== d.part), { start, end: start + len });
  return { notes: [...others, ...mine], len };
}
function scheduleEditPreview() {
  const { notes, len } = editPreviewNotes();
  Tone.Transport.cancel();
  lastStart = {};
  for (const n of notes) Tone.Transport.schedule((time) => playNote(n, time), Math.min(n.time, len - 0.001));
  Tone.Transport.loop = true;
  Tone.Transport.loopStart = 0;
  Tone.Transport.loopEnd = len;
}
async function playEditPreview() {
  if (!editDraft) return;
  await Tone.start();
  stopPlayback();
  ensureSynths();
  editDraft.playing = true;
  scheduleEditPreview();
  Tone.Transport.start("+0.1");
  const draft = editDraft;                                 // the looped preview shows in the bar too, with its stop button
  showPlayerBar(`${draft.label} · ${PART_KO[draft.part] || draft.part} 반복 듣기`, () => Tone.Transport.loopEnd || 0,
                () => Tone.Transport.seconds ?? 0, true);
}
function stopEditPreview() {
  if (editDraft) editDraft.playing = false;
  Tone.Transport.loop = false;
  stopPlayback();
}

function cancelPartEdit() {
  stopEditPreview();
  editSeq++;                                              // a reply still on its way is dropped
  editDraft = null;
  $("edit-panel").classList.add("hidden");
  $("edit-status").textContent = "편집을 취소했어요. 곡은 그대로예요.";
}

async function applyPartEdit() {
  const d = editDraft;
  if (!d) return;
  hideError();
  if (!editChanged(d)) { $("edit-status").textContent = "바뀐 것이 없어 적용하지 않았어요."; return; }
  const seq = ++editSeq;
  $("btn-edit-apply").disabled = true;
  try {
    const r = await api("/api/part/edit", { result: d.song, part: d.part, start_bar: d.start_bar, bars: d.bars,
      notes: d.notes.map((n) => ({ start: n.start, dur: n.dur, pitch: n.pitch, velocity: n.velocity })), author: $("edit-author").value || null });
    if (seq !== editSeq || state.song !== d.song || editDraft !== d) return;   // cancelled, or another song / section meanwhile
    if (!r.changed) { $("edit-status").textContent = "바뀐 것이 없어 적용하지 않았어요."; return; }
    stopEditPreview();
    state.song = r.result;                                // same song view; playback, downloads, band pack use the edit
    if (bandParent) bandParent = { ...bandParent, clean: false };
    $("song-leadsheet").textContent = r.result.leadsheet || r.result.leadsheet_error || "";
    editDraft = null;
    $("edit-panel").classList.add("hidden");
    $("edit-status").textContent = `${d.label}의 ${withObjectParticle(PART_KO[d.part] || d.part)} 적용했어요. 이제 이 파트 전체는 앱이 다시 만들지 않아요.`
      + (r.result.member_part_warnings || []).map((w) => " " + w).join("");
    // a sheet of this part on screen is now out of date: drawn again from the edited song
    if (($("practice-sheet")?.innerHTML || "").trim() && $("practice-part")?.value === d.part) showPracticeScore();
  } catch (e) {
    if (seq === editSeq) showError("파트 편집: " + e.message);
  } finally { $("btn-edit-apply").disabled = false; }
}

// ---------- band workspace: per-band versions and members' own parts ----------
const PART_KO = { melody: "멜로디", lead: "리드 기타", piano: "피아노", keys: "멜로디 피아노", bass: "베이스", guitar: "기타",
  strings: "스트링", drums: "드럼", harmony: "화음" };
const NOTE_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

// "A3" / "C#4" / "Bb3" -> MIDI number, null for empty, NaN for text that is not a note name
function noteToMidi(text) {
  const t = (text || "").trim();
  if (!t) return null;
  const m = /^([A-Ga-g])([#b]?)(-?\d)$/.exec(t);
  if (!m) return NaN;
  return (Number(m[3]) + 1) * 12 + NOTE_PC[m[1].toUpperCase()] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
}

// A new version of the song replaces the shown one: the members' own parts go with it when they still fit.
async function carryMemberParts(result, previous) {
  if (!previous?.member_parts?.length) return { result, note: "" };
  const kept = await api("/api/part/keep", { result, previous });
  return { result: kept.result, note: memberKeepNote(kept) };
}

function memberKeepNote(kept) {
  const names = (list) => list.map((p) => PART_KO[p] || p).join(", ");
  return [kept.kept.length ? `멤버 파트 유지: ${names(kept.kept)}` : "",
    ...kept.dropped.map((d) => `${PART_KO[d.part] || d.part} 파트는 새 버전에 넣지 못했어요: ${d.reason}`)].filter(Boolean).join(" · ");
}

let bandsLoaded = false;
async function loadBands(selectId) {
  const res = await fetch("/api/bands");
  if (!res.ok) return;
  const list = await res.json();
  bandList = list;
  const sel = $("band-select");
  sel.textContent = "";
  if (!list.length) sel.append(new Option("(밴드를 만들어 주세요)", ""));
  for (const b of list) sel.append(new Option(`${b.name} · 버전 ${b.versions}개`, String(b.id)));
  if (selectId != null) sel.value = String(selectId);
  await loadBandVersions();
}

// ---------- a band's own direction (PUT /api/bands/{id}/profile): saved per band, applied only when asked
let bandList = [];
function bandProfileText(p) {
  const bits = [];
  if (p.mood) bits.push(`"${p.mood}"`);
  if (p.genre) bits.push(GENRES[p.genre] || p.genre);
  if (p.tempo) bits.push(`${p.tempo}BPM`);
  if (p.melody_role && p.melody_role !== "vocal") bits.push(`멜로디: ${ROLE_KO[p.melody_role] || p.melody_role}`);
  else if (p.lead) bits.push(p.lead === "male" ? "남성 보컬" : "여성 보컬");
  if (p.instruments) bits.push(p.instruments.map((k) => INSTR[k] || k).join("·"));
  if (p.sounds) bits.push(`악기 지정 ${Object.keys(p.sounds).length}개`);
  if (p.key_change != null) bits.push(p.key_change ? "마지막 후렴 전조" : "전조 없음");
  return bits.length ? "이 밴드의 방향: " + bits.join(" · ") : "이 밴드는 아직 저장한 방향이 없어요.";
}
function showBandProfile() {
  const band = bandList.find((b) => String(b.id) === $("band-select").value);
  if ($("band-profile-view")) $("band-profile-view").textContent = band ? bandProfileText(band.profile || {}) : "";
  if ($("band-mood") && band) $("band-mood").value = band.profile?.mood || "";
}
async function saveBandProfile() {
  const id = $("band-select").value;
  if (!id) { showError("먼저 밴드를 고르거나 만들어 주세요"); return; }
  hideError();
  const profile = { mood: $("band-mood").value.trim() || null, genre: state.genre, tempo: state.tempo || null,
    lead: $("vocal-style")?.value || null, melody_role: melodyRole(), instruments: [...state.instruments], energy: !!$("energy").checked,
    key_change: !!$("key-change")?.checked, sounds: chosenSounds() };
  const res = await fetch(`/api/bands/${id}/profile`, { method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile }) });
  if (!res.ok) { const err = await res.json().catch(() => ({})); showError(typeof err.detail === "string" ? err.detail : `저장 실패 (${res.status})`); return; }
  await loadBands(id);
  $("band-status").textContent = "이 밴드의 방향을 저장했어요. 다른 밴드에는 영향이 없어요.";
}
function applyBandProfile() {
  const band = bandList.find((b) => String(b.id) === $("band-select").value);
  const p = band?.profile || {};
  if (!band || !Object.keys(p).length) { showError("이 밴드가 저장해 둔 방향이 없어요"); return; }
  hideError();
  if (p.genre) { state.genre = p.genre; state.src.genre = "manual"; }
  if (p.tempo) state.tempo = p.tempo;
  if (p.instruments) { state.instruments = new Set(p.instruments); state.src.instr = "manual"; }
  if (p.lead && $("vocal-style")) $("vocal-style").value = p.lead;
  if (p.melody_role && $("melody-role")) $("melody-role").value = p.melody_role;
  if (p.energy != null) $("energy").checked = p.energy;
  if (p.key_change != null && $("key-change")) $("key-change").checked = p.key_change;
  for (const k of Object.keys(soundChoice)) delete soundChoice[k];
  Object.assign(soundChoice, p.sounds || {});
  for (const sel of document.querySelectorAll?.("#sounds select") || []) sel.value = soundChoice[sel.dataset.part] || "";
  state.style = null;
  syncStyleButtons();
  $("band-status").textContent = `${band.name}의 방향을 1단계 설정에 적용했어요. 지금 곡은 그대로이고, 다음 "곡 만들기"부터 쓰여요.`;
}

async function loadBandVersions() {
  showBandProfile();
  const id = $("band-select").value, ul = $("band-versions");
  ul.textContent = "";
  if (!id) return;
  const res = await fetch(`/api/bands/${id}/versions`);
  if (!res.ok) return;
  const list = await res.json();
  for (const v of list.slice().reverse()) {
    const li = document.createElement("li");
    const text = document.createElement("span");
    text.textContent = `#${v.id} · ${v.created_text} · ${v.author || "이름 없음"} · ${v.note || "메모 없음"}`
      + ` · ${v.summary.bars}마디 ${v.summary.tempo}BPM`
      + (v.summary.member_parts.length ? ` · 멤버 파트: ${v.summary.member_parts.map((p) => PART_KO[p] || p).join(", ")}` : "");
    const open = document.createElement("button");
    open.textContent = "불러오기";
    open.onclick = () => openBandVersion(v.id);
    li.append(text, " ", open);
    if (v.parent_id != null) {
      const cmp = document.createElement("button");
      cmp.textContent = `#${v.parent_id}과 비교`;
      cmp.onclick = () => compareBandVersions(v.parent_id, v.id);
      li.append(" ", cmp);
    }
    ul.append(li);
  }
}

async function createBand() {
  hideError();
  const low = noteToMidi($("band-vocal-low").value), high = noteToMidi($("band-vocal-high").value);
  if (Number.isNaN(low) || Number.isNaN(high)) return showError("보컬 음역은 A3, D5처럼 음이름과 옥타브로 적어 주세요.");
  const g = $("band-guitarists").value;
  try {
    const band = await api("/api/bands", { name: $("band-name").value, members: $("band-members").value || null,
      guitarists: g === "" ? null : Number(g), vocal_low: low, vocal_high: high, source_note: $("band-source").value || null });
    $("band-status").textContent = `밴드 "${band.name}"를 만들었어요.`;
    await loadBands(band.id);
  } catch (e) { showError(e.message); }
}

// The band version the shown song came from (parent of the next save); clean = the shown song is exactly that version.
let bandParent = null, bandOpenSeq = 0;

// "작곡: 지수; 편곡: 민호" -> {작곡: "지수", 편곡: "민호"}; null when nothing usable was typed
function parseCredits(text) {
  const out = {};
  for (const piece of (text || "").split(/[;\n]/)) {
    const i = piece.indexOf(":");
    if (i > 0 && piece.slice(i + 1).trim()) out[piece.slice(0, i).trim()] = piece.slice(i + 1).trim();
  }
  return Object.keys(out).length ? out : null;
}
async function saveBandVersion() {
  const song = state.song, id = $("band-select").value;
  if (!song) return;
  if (!id) return showError("먼저 밴드를 만들거나 골라 주세요.");
  hideError();
  try {
    const parent = bandParent && String(bandParent.band) === id ? bandParent.version : null;
    const body = { result: song, author: $("band-author").value || null, note: $("band-note").value || null, parent_id: parent };
    const credits = parseCredits($("band-credits").value);
    if (credits) body.credits = credits;     // left empty: the server keeps the parent version's credits
    const saved = await api(`/api/bands/${id}/versions`, body);
    if (state.song !== song) return;         // another song was opened meanwhile: the save stands, the view is not touched
    bandParent = { band: Number(id), version: saved.id, clean: true };
    $("band-status").textContent = `버전 #${saved.id}로 저장했어요.`;
    $("band-note").value = "";
    await loadBands(id);
  } catch (e) { showError(e.message); }
}

async function openBandVersion(versionId) {
  hideError();
  const seq = ++bandOpenSeq;
  try {
    const res = await fetch(`/api/bands/versions/${versionId}`);
    if (!res.ok) throw new Error(`버전을 불러오지 못했어요 (${res.status})`);
    const v = await res.json();
    if (seq !== bandOpenSeq) return;         // a newer "불러오기" click wins
    stopPlayback();
    showSong(v.result);
    bandParent = { band: v.band_id, version: v.id, clean: true };
    $("band-credits").value = Object.entries(v.credits || {}).map(([k, x]) => `${k}: ${x}`).join("; ");
    $("band-status").textContent = `버전 #${v.id}를 불러왔어요. 여기서 고쳐 저장하면 #${v.id}의 다음 버전이 돼요.`;
  } catch (e) { showError(e.message); }
}

async function compareBandVersions(a, b) {
  hideError();
  try {
    const res = await fetch(`/api/bands/compare?a=${a}&b=${b}`);
    if (!res.ok) throw new Error(`비교하지 못했어요 (${res.status})`);
    const d = await res.json();
    const parts = Object.entries(d.parts).map(([p, x]) => `${PART_KO[p] || p} ${x.change === "edited" ? x.bars_text + "번째 마디가 달라요" : x.change === "added" ? "추가" : "삭제"}`);
    $("band-status").textContent = d.same ? `#${a}과 #${b}는 같아요.`
      : `#${a} → #${b}: ` + [...d.facts, d.chord_bars.length ? `코드 ${d.chord_bars_text}번째 마디가 달라요` : "", ...parts].filter(Boolean).join(" · ");
  } catch (e) { showError(e.message); }
}

async function importMemberPart(file) {
  const song = state.song;
  if (!song || !file) return;
  hideError();
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let bin = "";
    for (const b of bytes) bin += String.fromCharCode(b);
    const part = $("member-part").value;
    const result = await api("/api/part/import", { result: song, part, midi_b64: btoa(bin), author: $("member-author").value || null,
      trim_tail: !!$("member-trim")?.checked });
    if (state.song !== song) {               // another song or version was opened while the file was being read
      $("band-status").textContent = "가져오는 동안 다른 곡이 열려서 적용하지 않았어요. 다시 올려 주세요.";
      return;
    }
    stopPlayback();
    state.song = result;                     // same song view; playback, downloads and the band pack use the new part
    if (bandParent) bandParent = { ...bandParent, clean: false };
    $("song-leadsheet").textContent = result.leadsheet || result.leadsheet_error || "";
    $("band-status").textContent = `${PART_KO[part] || part} 파트를 가져왔어요 (${result.member_parts.find((m) => m.part === part).notes}음). `
      + (result.member_part_warnings || []).join(" ") + " 재생·음원·밴드 팩에 바로 반영돼요.";
  } catch (e) { showError("멤버 파트 가져오기: " + e.message); }
  finally { $("member-file").value = ""; }
}

// ---------- practice: one instrument's sheet, and the song with that part alone or left out ----------
// "드럼을" / "베이스를": the object particle follows whether the word ends in a consonant
function withObject(word) {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  return word + (code >= 0 && code <= 11171 && code % 28 !== 0 ? "을" : "를");
}
const PRACTICE_ORDER = ["guitar", "bass", "drums", "piano", "keys", "lead", "strings", "melody", "harmony"];
let practiceOsmd = null, practiceSeq = 0, practiceXml = null;
function fillPractice(song) {
  const partSel = $("practice-part"), secSel = $("practice-section");
  if (!partSel || !secSel) return;
  const have = new Set((song.playback?.notes || []).map((n) => n.track));
  const keepPart = partSel.value, parts = PRACTICE_ORDER.filter((p) => have.has(p));
  partSel.textContent = "";
  for (const p of parts) {
    const o = document.createElement("option");
    o.value = p; o.textContent = PART_KO[p] || p;
    partSel.append(o);
  }
  if (parts.includes(keepPart)) partSel.value = keepPart;       // the same member keeps practising after an edit
  secSel.textContent = "";
  const all = document.createElement("option");
  all.value = "all"; all.textContent = `곡 전체 (${song.bars}마디)`;
  secSel.append(all);
  (song.sections || []).forEach((s, i) => {
    const o = document.createElement("option");
    o.value = String(i); o.textContent = `${s.label} · ${s.start_bar + 1}~${s.start_bar + s.bars}마디`;
    secSel.append(o);
  });
  const firstChorus = (song.sections || []).findIndex((s) => s.kind === "chorus");
  secSel.value = firstChorus >= 0 ? String(firstChorus) : "all";    // a section draws fast and is what one practises
  practiceSeq++; practiceXml = null; practiceOsmd = null;
  if ($("practice-sheet")) $("practice-sheet").innerHTML = "";
  if ($("practice-tab")) $("practice-tab").innerHTML = "";
  if ($("practice-tab-note")) $("practice-tab-note").textContent = "";
  if ($("practice-status")) $("practice-status").textContent = parts.length
    ? "내 악기를 고르고 \"악보 보기\"를 누른 뒤, \"이 파트 빼고 듣기\"에 맞춰 연주해 보세요."
    : "이 곡에는 연습할 파트가 없어요.";
}
// {start_bar, bars, t0, t1 (seconds at the song's own tempo), label} of the chosen stretch
function practiceRange(song) {
  const v = $("practice-section")?.value || "all", bar = 240 / song.playback.bpm;       // 4/4: seconds per bar
  if (v === "all") return { start_bar: 0, bars: song.bars, t0: 0, t1: song.bars * bar, label: "곡 전체" };
  const s = song.sections[Number(v)];
  return { start_bar: s.start_bar, bars: s.bars, t0: s.start_bar * bar, t1: (s.start_bar + s.bars) * bar, label: s.label };
}
// The notes to play: "solo" = only the part, "minus" = everything but the part, "all" = everything.  Times start at
// the stretch's beginning (after an optional one-bar count) and are stretched by 1 / speed.
function practiceNotes(song, mode, part, range, speed, countIn) {
  const eps = 1e-6, lead = countIn ? (240 / song.playback.bpm) / speed : 0;
  const notes = song.playback.notes
    .filter((n) => n.time >= range.t0 - eps && n.time < range.t1 - eps)
    .filter((n) => (mode === "solo" ? n.track === part : mode === "minus" ? n.track !== part : true))
    .map((n) => ({ ...n, time: (n.time - range.t0) / speed + lead, duration: Math.min(n.duration, range.t1 - n.time) / speed }));
  const clicks = [];
  if (countIn) for (let b = 0; b < 4; b++) clicks.push({ track: "drums", midi: 37, time: (b * 60 / song.playback.bpm) / speed, duration: 0.05, velocity: b === 0 ? 0.9 : 0.6 });
  return { notes: [...clicks, ...notes], len: (range.t1 - range.t0) / speed + lead, lead };
}
async function playPractice(mode) {
  const song = state.song, part = $("practice-part")?.value;
  if (!song || !part) return;
  hideError();
  const range = practiceRange(song), speed = Number($("practice-speed")?.value) || 1, loop = !!$("practice-loop")?.checked;
  const { notes, len } = practiceNotes(song, mode, part, range, speed, !!$("practice-count")?.checked);
  if (!notes.some((n) => n.track !== "drums" || n.midi !== 37) && mode === "solo") {
    $("practice-status").textContent = `${range.label}에서는 ${PART_KO[part] || part} 파트가 쉬어요. 다른 구간을 골라 주세요.`;
    return;
  }
  await Tone.start();
  stopPlayback();
  ensureSynths();
  lastStart = {};
  for (const n of notes) Tone.Transport.schedule((time) => playNote(n, time), Math.min(n.time, len - 0.001));
  const what = { solo: `${PART_KO[part] || part}만`, minus: `${PART_KO[part] || part} 빼고`, all: "전체" }[mode];
  const title = `연습 · ${range.label} · ${what}${speed !== 1 ? ` · ${Math.round(speed * 100)}%` : ""}`;
  if (loop) {
    Tone.Transport.loop = true; Tone.Transport.loopStart = 0; Tone.Transport.loopEnd = len;
    Tone.Transport.start("+0.1");
    showPlayerBar(title + " · 반복", () => len, () => Tone.Transport.seconds ?? 0, true);
  } else {
    Tone.Transport.start("+0.1");
    showPlayerBar(title, () => len + 1.2, () => Tone.Transport.seconds ?? 0);
  }
  $("practice-status").textContent = mode === "minus"
    ? `${withObject(PART_KO[part] || part)} 빼고 틀고 있어요. 악보를 보면서 그 자리를 직접 연주해 보세요.` : `${title} 재생 중`;
}
async function loadPracticeScore() {
  const song = state.song, part = $("practice-part")?.value;
  if (!song || !part) return null;
  const range = practiceRange(song), seq = ++practiceSeq;
  $("practice-status").textContent = "악보를 만들고 있어요…";
  const data = await api("/api/part/score", { result: song, part, start_bar: range.start_bar, bars: range.bars });
  if (seq !== practiceSeq) return null;
  practiceXml = { xml: data.musicxml, name: `jakgokmate_${part}_${range.start_bar + 1}-${range.start_bar + range.bars}.musicxml`, title: data.title, range };
  return practiceXml;
}
// The sheet as the engraving engine lays it out on A4 pages (what the print gives).  False when the engine is not
// installed on the server: the caller then draws the simple sheet in the browser.
async function showEngravedScore(song, part, range, seq) {
  const res = await fetch("/api/part/engrave", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ result: song, part, start_bar: range.start_bar, bars: range.bars }) });
  if (res.status === 501) return false;
  const d = await res.json();
  if (!res.ok) throw new Error(typeof d.detail === "string" ? d.detail : `악보 조판 실패 (${res.status})`);
  if (seq !== practiceSeq) return true;
  const box = $("practice-sheet"), tab = $("practice-tab"), note = $("practice-tab-note");
  box.classList.add("engraved");
  box.innerHTML = d.pages.map((svg) => `<div class="sheet-page">${svg}</div>`).join("");
  tab.innerHTML = d.tab_svg || "";
  note.textContent = d.tab_svg ? (d.tab_kind === "drums" ? "드럼 차트: " : "타브: ") + d.tab_note : "";
  practiceXml = { ...(practiceXml || {}), title: d.title, range, pages: d.pages.length, engraved: true };
  $("practice-status").textContent = `${d.title} · ${range.label} (${range.start_bar + 1}~${range.start_bar + range.bars}마디) · A4 ${d.pages.length}쪽으로 조판했어요. `
    + "인쇄하면 이 모습 그대로 나와요. \"이 파트만 듣기\"로 확인하고 \"이 파트 빼고 듣기\"에 맞춰 연주해 보세요.";
  return true;
}
async function showPracticeScore() {
  hideError();
  try {
    const song = state.song, part = $("practice-part")?.value;
    if (song && part) {
      const seq = ++practiceSeq, range = practiceRange(song);
      $("practice-status").textContent = "악보를 조판하고 있어요…";
      if (await showEngravedScore(song, part, range, seq)) return;
    }
    const got = await loadPracticeScore();
    if (!got) return;
    const box = $("practice-sheet");
    box.classList.remove("engraved");
    box.innerHTML = "";
    if (window.opensheetmusicdisplay) {
      practiceOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("practice-sheet", {
        autoResize: true, drawTitle: false, drawComposer: false, drawCredits: false, drawPartNames: false });
      await practiceOsmd.load(got.xml);
      practiceOsmd.render();
    }
    await showPracticeTab(got);
    $("practice-status").textContent = `${got.title} · ${got.range.label} (${got.range.start_bar + 1}~${got.range.start_bar + got.range.bars}마디) 악보예요. `
      + "\"이 파트만 듣기\"로 확인하고 \"이 파트 빼고 듣기\"에 맞춰 연주해 보세요.";
  } catch (e) { $("practice-status").textContent = ""; showError("파트 악보: " + e.message); }
}
// Tablature under the sheet: guitar / lead / bass as fret numbers on string lines, drums as a chart (x / o per drum).
const TAB_PARTS = new Set(["guitar", "lead", "bass", "drums"]);
function tabSvg(d, perRow = 4) {
  const cell = 15, gap = 16, left = 34, pad = 12, head = 30, lines = d.lines.length;
  const barW = d.steps * cell + 10, rowH = head + (lines - 1) * gap + 30, rows = Math.ceil(d.bars.length / perRow);
  const W = left + perRow * barW + pad, H = rows * rowH + pad;
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" font-family="ui-monospace, Menlo, Consolas, monospace" font-size="12" fill="#111">`
    + `<rect width="${W}" height="${H}" fill="#fff"/>`;
  d.bars.forEach((bar, i) => {
    const row = Math.floor(i / perRow), col = i % perRow, x0 = left + col * barW, y0 = row * rowH + head + pad;
    if (col === 0) d.lines.forEach((name, l) => { s += `<text x="${left - 8}" y="${y0 + l * gap + 4}" text-anchor="end" fill="#555">${esc(name)}</text>`; });
    for (let l = 0; l < lines; l++) s += `<line x1="${x0}" y1="${y0 + l * gap}" x2="${x0 + barW}" y2="${y0 + l * gap}" stroke="#999" stroke-width="1"/>`;
    s += `<line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y0 + (lines - 1) * gap}" stroke="#333" stroke-width="${col === 0 ? 2 : 1}"/>`;
    if (col === perRow - 1 || i === d.bars.length - 1) s += `<line x1="${x0 + barW}" y1="${y0}" x2="${x0 + barW}" y2="${y0 + (lines - 1) * gap}" stroke="#333" stroke-width="1"/>`;
    for (let b = 1; b < d.steps / 4; b++) s += `<line x1="${x0 + 5 + b * 4 * cell}" y1="${y0 + (lines - 1) * gap + 4}" x2="${x0 + 5 + b * 4 * cell}" y2="${y0 + (lines - 1) * gap + 9}" stroke="#bbb"/>`;
    s += `<text x="${x0 + 3}" y="${y0 - 18}" font-size="10" fill="#777">${bar.number}</text>`;
    if (bar.label) s += `<text x="${x0 + 22}" y="${y0 - 18}" font-size="11" font-weight="700" fill="#2f6b1f">${esc(bar.label)}</text>`;
    let lastX = -99;
    for (const [at, name] of bar.chords || []) {
      const x = Math.max(x0 + 5 + at * cell, lastX + 26);
      s += `<text x="${x}" y="${y0 - 5}" font-weight="700" font-family="system-ui, sans-serif">${esc(name)}</text>`;
      lastX = x;
    }
    for (const [at, l, text] of bar.cells) {
      const x = x0 + 5 + at * cell + cell / 2, y = y0 + l * gap;
      s += `<rect x="${x - 4 * String(text).length - 1}" y="${y - 6}" width="${8 * String(text).length + 2}" height="12" fill="#fff"/>`
        + `<text x="${x}" y="${y + 4}" text-anchor="middle"${text === "?" ? ' fill="#b00"' : ""}>${esc(text)}</text>`;
    }
  });
  return s + "</svg>";
}
async function showPracticeTab(got) {
  const box = $("practice-tab"), note = $("practice-tab-note"), part = $("practice-part")?.value;
  if (!box) return;
  box.innerHTML = "";
  if (note) note.textContent = "";
  if (!TAB_PARTS.has(part)) return;
  const seq = practiceSeq;
  const d = await api("/api/part/tab", { result: state.song, part, start_bar: got.range.start_bar, bars: got.range.bars });
  if (seq !== practiceSeq) return;
  box.innerHTML = tabSvg(d, (box.clientWidth || 1000) < 820 ? 2 : 4);      // a narrow screen: two bars per row, readable
  if (note) note.textContent = (d.kind === "drums" ? "드럼 차트: " : "타브: ") + d.note;
}
// The parts as an A4 document made by the server (engraved sheets, title, page numbers, tab pages), opened in a window
// of its own with the browser's print sheet.  The window is opened at once, inside the click, so that pop-up blockers
// let it through; it is filled when the document arrives.
async function printEngraved(all) {
  const song = state.song, part = $("practice-part")?.value;
  if (!song || !part) return;
  hideError();
  const parts = all ? [...$("practice-part").options].map((o) => o.value) : [part];
  const range = all ? { start_bar: 0, bars: song.bars, label: "곡 전체" } : practiceRange(song);
  const w = window.open("", "_blank");
  if (!w) return showError("새 창이 막혀 있어요. 브라우저의 팝업 차단을 이 페이지에서 풀어 주세요.");
  w.document.write('<!doctype html><meta charset="utf-8"><title>악보 만드는 중…</title><p style="font-family:sans-serif;margin:40px">악보를 조판하고 있어요…</p>');
  $("practice-status").textContent = all ? "모든 파트의 악보를 조판하고 있어요… (곡 전체는 10초쯤)" : "악보를 조판하고 있어요…";
  try {
    const res = await fetch("/api/part/print", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: song, parts, start_bar: range.start_bar, bars: range.bars, title: $("practice-title")?.value || "" }) });
    if (res.status === 501) { w.close?.(); return printPracticeScore(); }       // no engine: the simple drawing instead
    const d = await res.json();
    if (!res.ok) throw new Error(typeof d.detail === "string" ? d.detail : `악보 만들기 실패 (${res.status})`);
    w.document.open(); w.document.write(d.html); w.document.close();
    w.focus?.();
    if (typeof setTimeout === "function") setTimeout(() => { try { w.print(); } catch (e) { /* print by hand */ } }, 600);
    $("practice-status").textContent = `${d.parts.join(", ")} 악보 ${d.pages}쪽을 열었어요. 종이로 뽑거나, 대상에서 \"PDF로 저장\"을 고르면 PDF 파일이 돼요.`;
  } catch (e) { w.close?.(); $("practice-status").textContent = ""; showError("악보 인쇄: " + e.message); }
}
// From the sheet to the part editor: the same part and stretch, opened for changing notes.
const EDITABLE_PARTS = new Set(["bass", "guitar", "lead", "piano", "drums"]);
async function editPracticeScore() {
  const song = state.song, part = $("practice-part")?.value, v = $("practice-section")?.value;
  if (!song || !part) return;
  hideError();
  if (!EDITABLE_PARTS.has(part)) return showError(`${PART_KO[part] || part} 파트는 여기서 음을 고칠 수 없어요. 멜로디는 1단계의 입력에서 고쳐 주세요.`);
  if (v === "all") return showError("고칠 구간을 하나 골라 주세요(\"구간\"에서 곡 전체 말고 인트로·벌스·후렴 중 하나).");
  $("edit-box").open = true;
  editSections();
  $("edit-section").value = v;
  $("edit-part").value = part;
  await openPartEditor();
  $("edit-box").scrollIntoView?.({ block: "start", behavior: "instant" });
  $("practice-status").textContent = "아래 \"파트 편집\"에서 음을 고치고 \"이 파트 적용\"을 누르면, 이 악보가 고친 내용으로 다시 그려져요.";
}
// The simple drawing (browser-made sheet and tab) in a window of its own: used when the engraving engine is missing.
function printPracticeScore() {
  const sheet = $("practice-sheet")?.innerHTML || "", tab = $("practice-tab")?.innerHTML || "";
  if (!sheet.trim() && !tab.trim()) return showError("먼저 \"악보 보기\"를 눌러 악보를 그려 주세요.");
  const w = window.open("", "_blank");
  if (!w) return showError("새 창이 막혀 있어요. 브라우저의 팝업 차단을 이 페이지에서 풀어 주세요.");
  const title = `${practiceXml?.title || "파트"} · ${practiceXml?.range?.label || ""}`;
  w.document.write(`<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>${title}</title>`
    + "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#111;background:#fff}svg{max-width:100%;height:auto}h1{font-size:18px}p{color:#555;font-size:12px}</style></head>"
    + `<body><h1>${title}</h1>${sheet}${tab ? `<p>${$("practice-tab-note")?.textContent || ""}</p>${tab}` : ""}</body></html>`);
  w.document.close();
  w.focus?.();
  // straight to the browser's print sheet: "PDF로 저장" there gives the file, with no other program needed
  if (typeof setTimeout === "function") setTimeout(() => { try { w.print(); } catch (e) { /* the window stays open to print by hand */ } }, 400);
  if ($("practice-status")) $("practice-status").textContent = "인쇄 창이 열렸어요. 종이로 뽑거나, 대상(프린터)에서 \"PDF로 저장\"을 고르면 PDF 파일이 돼요.";
}
function playPracticeHq(mode, button) {
  const song = state.song, part = $("practice-part")?.value;
  if (!song || !part) return;
  const name = PART_KO[part] || part;
  $("practice-status").textContent = mode === "solo" ? `실제 악기 소리로 ${name}만 만들고 있어요…` : `실제 악기 소리로 ${withObject(name)} 뺀 반주를 만들고 있어요…`;
  return playHq(song, button, `실제 악기 소리 · ${name}${mode === "solo" ? "만" : " 빼고"}`, mode === "solo" ? { only_part: part } : { without_part: part });
}

async function downloadPracticeScore() {
  hideError();
  try {
    const got = await loadPracticeScore();
    if (!got) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([got.xml], { type: "application/vnd.recordare.musicxml+xml" }));
    a.download = got.name;
    a.click();
    $("practice-status").textContent = `${got.name}을 받았어요. MuseScore 같은 악보 프로그램에서 열어 인쇄할 수 있어요.`;
  } catch (e) { showError("파트 악보: " + e.message); }
}

// ---------- sung demo (YuE2, runs on this Mac; optional) ----------
let yueAvailable = null, yueReason = "";
function applyYueAvailability(st) {
  if (!st || typeof st.available !== "boolean") return;
  yueAvailable = st.available;
  yueReason = st.available ? "" : st.reason || "YuE2를 쓸 수 없어요";
  for (const id of ["yuesong-on", "btn-yuesong", "btn-yuesong-yes", "btn-yue", "btn-song-overlay"]) {
    const el = $(id);
    if (el) { el.disabled = !st.available || yueBusy; el.title = st.available ? "" : st.reason || "YuE2를 쓸 수 없어요"; }
  }
  if (!yueBusy && $("yuesong-status")) $("yuesong-status").textContent = st.available ? "" : st.reason || "YuE2를 쓸 수 없어요";
}
function showPracticeTiming(timing) {
  const el = $("lib-tempo-note");
  if (!el) return;
  const measured = timing && Number.isFinite(timing.min_s) && Number.isFinite(timing.max_s) && timing.min_s > 0 && timing.max_s >= timing.min_s;
  el.textContent = "⚠ 연습 반주는 YuE2 보컬의 박을 따라가요. 그래서 곡이 진행되는 동안 템포가 조금씩 달라질 수 있어요"
    + (measured ? ` (이번 곡은 박마다 ${timing.min_s.toFixed(3)}~${timing.max_s.toFixed(3)}초)` : "")
    + ". 메트로놈과 함께 쓰면 어긋날 수 있어요.";
}
async function loadYueStatus() {
  try {
    const res = await fetch("/api/yue/status");
    if (!res.ok) return;
    const st = await res.json();
    applyYueAvailability(st);
    const sel = $("yue-style");
    sel.textContent = "";
    for (const x of st.styles) {
      const o = document.createElement("option");
      o.value = x.id; o.textContent = x.label;
      sel.append(o);
    }
    syncYueVoice();
    if (yueBusy) return;                     // a take is being made: its progress line and the busy button stay
    $("btn-yue").disabled = !st.available;
    $("yue-status").textContent = st.available ? "" : st.reason;
  } catch (e) { /* optional feature */ }
}

// The AI vocal demo sings with the voice chosen for the melody in step 1: male -> the male demo, female -> female E.
function syncYueVoice() {
  const sel = $("yue-style"), want = $("vocal-style")?.value === "male" ? "male" : "female_high";
  if (sel && [...(sel.options || [])].some((o) => o.value === want)) sel.value = want;
}
let yueSeq = 0, yueUrl = null, yueLoaded = false;
let yueFor = null;                          // which song + voice the finished take belongs to (yueKey)
let yueBusy = false;                        // a take is being made (only one at a time, here and on the server)
const yueVoice = () => ($("vocal-style")?.value === "male" ? "male" : "female_high");
const yueKey = (song, style, full) => `${style}|${full ? 1 : 0}|${song.leadsheet}`;
// "AI 보컬로 듣기" in the play row: the take for this song and the voice chosen in step 1.  Made once (minutes),
// then replayed at once.  A new song, another voice or a changed melody makes a new take.
async function playAiVocal() {
  const song = state.song, note = $("song-ai-status");
  if (!song) return;
  const style = yueVoice(), full = !!$("yue-full")?.checked;
  if (yueFor === yueKey(song, style, full) && yueUrl) {
    stopPlayback();
    const a = $("yue-audio");
    a.currentTime = 0;
    try { await a.play(); if (note) note.textContent = "AI 보컬 데모를 재생하고 있어요 (반주도 AI가 연주한 별도 음원이에요)."; }
    catch (e) { if (note) note.textContent = "AI 보컬 데모가 준비됐어요. 아래 재생기에서 ▶를 눌러 주세요."; }
    return;
  }
  if (yueBusy) { if (note) note.textContent = "AI 보컬 데모를 이미 만들고 있어요. 끝나면 여기에 알려 드려요."; return; }
  if ($("btn-yue")?.disabled && $("yue-status")?.textContent) {
    return showError("AI 보컬 데모: " + $("yue-status").textContent);        // not installed / not available on this computer
  }
  if (note) note.textContent = "AI 보컬 데모를 만들고 있어요. 곡 길이만큼(몇 분) 걸려요. 그동안 다른 버튼은 그대로 쓸 수 있어요.";
  const ok = await makeYueDemo(style);
  if (!ok) { if (note) note.textContent = ""; return; }
  try { stopPlayback(); await $("yue-audio").play(); if (note) note.textContent = "AI 보컬 데모를 재생하고 있어요. 다시 누르면 처음부터 들어요."; }
  catch (e) { if (note) note.textContent = "AI 보컬 데모가 완성됐어요. \"🎤 AI 보컬로 듣기\"를 한 번 더 누르면 재생돼요."; }
}
// A newly made song gets its AI vocal take without a click (owner decision 2026-10-08): the voice chosen in step 1,
// started in the background and never played by itself.  Skipped when the box is off, nobody sings, the feature is
// not available here, a take is already being made, or this song and voice already have one.
async function autoAiVocal() {
  const song = state.song, note = $("song-ai-status");
  if (!song || !$("yue-auto")?.checked || !song.leadsheet || !sungMelody(song) || yueBusy || state.composing) return;
  if ($("btn-yue")?.disabled) return;
  const style = yueVoice();
  if (yueFor === yueKey(song, style, !!$("yue-full")?.checked) && yueUrl) return;
  if (note) note.textContent = "AI 보컬을 자동으로 만들고 있어요 (몇 분). 끝나면 \"🎤 AI 보컬로 듣기\"로 바로 들을 수 있어요.";
  const ok = await makeYueDemo(style, true);
  if (state.song !== song) return;                       // another song came meanwhile: its own take says its own state
  if (note) note.textContent = ok ? "AI 보컬이 준비됐어요. \"🎤 AI 보컬로 듣기\"를 누르면 재생돼요."
    : "AI 보컬을 자동으로 만들지 못했어요. \"🎤 AI 보컬로 듣기\"를 누르면 다시 시도해요.";
}
// "Finish the song with YuE2" (POST /api/yue/song, off until its box is ticked): the section that carries the user's
// melody opens a score that YuE2 continues, sings and plays.  The finished job holds the audio and the score as the
// app's own song (info.result); "불러오기" shows that song like one imported from a lead sheet.
let yueSongSeq = 0, yueSongUrl = null, yueSongResult = null, yueSongFirst = null;
let yueSongSource = null;
let yueRunning = null;                       // { job, first } while a take is made: what "취소" stops and "다시" repeats
// Boxes that change what is asked of YuE2: the model's own words, and parts added to the plan.  Only what was asked is sent.
function yueExtras() {
  const random = !!$("yuesong-random")?.checked && !$("yuesong-lyrics")?.value.trim(); // typed words win over random
  const extra = [["yue-add-chorus", "Chorus"], ["yue-add-bridge", "Bridge"], ["yue-add-outro", "Outro"]].filter(([id]) => $(id)?.checked).map(([, name]) => name);
  return { random, body: { ...(random ? { lyrics_mode: "model" } : {}), ...(extra.length ? { extra_sections: extra } : {}) } };
}
function syncRandomLyrics() { if ($("yuesong-lyrics")) $("yuesong-lyrics").disabled = false; }
function showYueCancel(on) { for (const id of ["btn-yuesong-cancel", "btn-yuesong-redo"]) $(id)?.classList.toggle("hidden", !on); }
async function cancelYue() {
  if (!yueRunning) return false;
  try { await api(`/api/yue/${yueRunning.job}/cancel`, {}); return true; } catch (e) { showError("취소하지 못했어요: " + e.message); return false; }
}
async function redoYue() {
  const first = yueRunning?.first || null;
  if (!(await cancelYue())) return;
  const t0 = Date.now();
  while (yueBusy && Date.now() - t0 < 20000) await new Promise((r) => setTimeout(r, 300));    // the runner is stopped within a second
  makeYueSong(first);
}
// Per-section choice (#yuesong-sections): one row per section of the shown song, "주기" (given to YuE2) and "악기가 연주"
// (an instrument plays it).  Until the user changes a row the rows only mirror the range select and the instrument box,
// and the request is the one of before; after that the rows are what is sent.
const YUE_MAX_BARS = 200;                                   // yuesong.MAX_BARS: the most bars a request may give
let yueSecs = { touched: false, given: 0, rows: [] };       // rows: { give, ins, bars, boxes: { give, ins } }
function fillYueSections(data) {
  const list = $("yuesong-sections");
  if (!list) return;
  const secs = data?.sections || [];
  yueSecs = { touched: false, given: Math.max(0, (data?.section_sources || []).indexOf("seed")), rows: [] };
  list.innerHTML = secs.length ? "" : '<li class="help">곡이 만들어지면 구간이 여기에 나와요.</li>';
  secs.forEach((s, i) => {
    const row = { give: false, ins: false, bars: Array.isArray(s.bars) ? s.bars.length : Number(s.bars) || 0, boxes: {} };
    const li = document.createElement("li"), name = document.createElement("span");
    name.textContent = `${i + 1}. ${s.label || s.name || "구간"}`;
    li.append(name);
    for (const [kind, text] of [["give", "주기"], ["ins", "악기가 연주"]]) {
      const label = document.createElement("label"), box = document.createElement("input");
      box.type = "checkbox";
      box.dataset.sec = String(i);
      box.dataset.kind = kind;
      box.onchange = () => { yueSecs.touched = true; row[kind] = box.checked; syncYueSections(); };
      row.boxes[kind] = box;
      label.append(box, " " + text);
      li.append(label);
    }
    yueSecs.rows.push(row);
    list.append(li);
  });
  syncYueSections();
}
// Untouched rows follow the range select and the instrument box; the boxes are set in place (a redraw would drop the focus).
function syncYueSections() {
  const whole = $("yuesong-scope")?.value || "seed", played = !!$("yuesong-ins")?.checked;
  yueSecs.rows.forEach((row, i) => {
    if (!yueSecs.touched) {
      row.give = whole !== "seed" || i === yueSecs.given;
      row.ins = whole === "seed" && played && i === yueSecs.given;
    }
    row.ins = row.give && row.ins;                           // a section that is not given has nothing to play
    row.boxes.give.checked = row.give;
    row.boxes.ins.checked = row.ins;
    row.boxes.ins.disabled = !row.give;
  });
  $("btn-yuesong-sections-reset")?.classList.toggle("hidden", !yueSecs.touched);
}
// The changed rows as the request's fields (YueSongReq: seed_sections in the song's order, ins_sections as positions in
// that list), or the reason the server would refuse them, told here in Korean before anything is sent.
function yueSectionRequest(whole) {
  const picked = yueSecs.rows.map((row, i) => (row.give ? i : -1)).filter((i) => i >= 0);
  if (!picked.length) return { error: "YuE2에 줄 구간을 하나 이상 고르세요 (구간별 선택의 \"주기\")." };
  const bars = picked.reduce((n, i) => n + yueSecs.rows[i].bars, 0);
  if (bars > YUE_MAX_BARS) return { error: `고른 구간이 모두 ${bars}마디예요. YuE2에는 최대 ${YUE_MAX_BARS}마디까지 줄 수 있어요.` };
  const ins = picked.map((i, at) => (yueSecs.rows[i].ins ? at : -1)).filter((at) => at >= 0);
  if (whole === "asis" && ins.length === picked.length) return { error: "곡을 그대로 부르려면 보컬이 부르는 구간이 하나는 있어야 해요. \"악기가 연주\"를 한 줄은 꺼 주세요." };
  return { body: { seed_sections: picked, ins_sections: ins, continue_song: whole !== "asis" } };
}
// The queue view: the job being made (title · step · voice, then its last progress line) and the last finished job's
// one-line result, which stays until the next job starts.
let yueQueue = { now: "", line: "", last: "" };
function showQueue() {
  if ($("queue-now")) $("queue-now").textContent = yueQueue.now ? `만드는 중: ${yueQueue.now}${yueQueue.line ? " · " + yueQueue.line : ""}` : "진행 중인 작업이 없어요.";
  const last = $("queue-last");
  if (!last) return;
  last.textContent = yueQueue.last ? `마지막 결과: ${yueQueue.last}` : "";
  last.classList.toggle("hidden", !yueQueue.last);
}
// `first`: the finished first part ({ abc, library_id, genre, title }) when the user said yes to the whole song; its genre
// and title are the ones it was made with, so the whole song needs no song on screen.
async function makeYueSong(first = null, play = true) {
  if (yueAvailable === false) return showError(yueReason || "YuE2를 쓸 수 없어요."), false;
  const song = state.song;
  if (!first && !song) return;
  if (!first && !song.leadsheet) return showError("이 곡은 리드시트로 내보낼 수 없어 YuE2에 넘길 수 없어요."), false;
  if (yueBusy) return showError("AI 보컬을 이미 만들고 있어요. 끝난 뒤 다시 눌러 주세요."), false;
  hideError();
  const rows = !first && yueSecs.touched ? yueSectionRequest($("yuesong-scope")?.value || "seed") : null;   // only once a row was changed
  if (rows?.error) return showError(rows.error), false;
  const seq = ++yueSongSeq, button = $("btn-yuesong"), status = $("yuesong-status");
  const given = Math.max(0, (song?.section_sources || []).indexOf("seed"));
  const title = first && "title" in first ? first.title : $("yuesong-title")?.value?.trim() || "";
  const genre = first && "genre" in first ? first.genre : song?.genre || null;
  button.disabled = true;
  yueBusy = true;
  status.textContent = "시작하는 중…";
  const what = `${title || "제목 없음"} · ${first ? "전곡" : $("yuesong-scope")?.value === "asis" ? "곡 그대로" : "1절"}`
    + ` · ${voiceLabel($("yuesong-voice")?.value)}`;
  yueQueue = { now: what, line: status.textContent, last: "" };   // a new job: the result of the one before leaves the queue view
  showQueue();
  try {
    let whole = $("yuesong-scope")?.value || "seed";         // seed: my section, then YuE2 writes on · all: every section,
    if (whole === "seed" && song?.imported_score) whole = "all";   // an uploaded score is all the user's melody: give every bar
    let body = whole === "seed" ? { seed_section: given }       // then YuE2 writes on · asis: the whole song as it is
      : { seed_sections: "all", continue_song: whole !== "asis" };
    if ($("yuesong-ins")?.checked && whole === "seed") body.ins_sections = [0];
    if (rows) body = rows.body;                                // the user's rows replace the range and the instrument box
    body = first ? { stage: "full", score: first.abc, parent: first.library_id }       // step two: the whole song
      : whole === "asis" ? { ...body, leadsheet: song.leadsheet }                      // sung as given: one step
      : { ...body, leadsheet: song.leadsheet, stage: "first" };                        // step one: the first part
    $("yuesong-ask").classList.add("hidden");
    const ex = yueExtras();
    const { job } = await api("/api/yue/song", { ...body, title, genre,
      voice: $("yuesong-voice").value, key_shift: Number($("yuesong-key").value), low_memory: $("yuesong-lowmem").checked,
      lyrics: ex.random ? null : $("yuesong-lyrics").value.trim() || null, ...ex.body,
      mode: currentMode() });
    yueRunning = { job, first };
    showYueCancel(true);
    const t0 = Date.now();
    for (;;) {
      await new Promise((r) => setTimeout(r, 2000));
      const st = await getJson(`/api/yue/${job}`);
      if (seq !== yueSongSeq) return;
      if (st.status === "cancelled") {                       // the user's own stop: no error box, the next take is free
        status.textContent = "취소했어요";
        yueQueue = { now: "", line: "", last: `${what} · 취소했어요` };
        showQueue();
        if (first) $("yuesong-ask").classList.remove("hidden");
        return false;
      }
      if (st.status === "error") throw new Error(st.error || "생성 실패");
      if (st.status === "done") {
        const blob = await (await fetch(`/api/yue/${job}/audio`)).blob();
        if (yueSongUrl) URL.revokeObjectURL(yueSongUrl);
        yueSongUrl = URL.createObjectURL(blob);
        $("yuesong-audio").src = yueSongUrl;
        $("yuesong-audio").classList.remove("hidden");
        yueSongResult = st.info.result;
        yueSongSource = song;
        const repaired = st.info.plans?.find(p => p.seed === st.info.plan_seed)?.repaired_bars || 0;
        const parts = st.info.sections.map(([name, bars]) => `${name} ${bars}마디`).join(" → ");
        $("yuesong-note").textContent = `YuE2가 쓴 악보: ${parts}.`      // a job without given bars (the whole song) skips that sentence
          + (st.info.given_bars != null ? ` 처음 ${st.info.given_bars}마디가 내 마디예요` + (st.info.given_kept ? " (그대로 들어갔어요)." : ".") : "")
          + (st.info.lyrics === "ah" ? " 가사 없이 '아'로 불렀어요." : st.info.lyrics === "model" ? " 랜덤 가사 모드로 요청했어요. 실제 가사는 음원에서 확인해 주세요." : "")
          + (repaired ? ` AI가 새로 쓴 ${repaired}마디의 쉼표 길이를 박자에 맞췄어요. 입력 악보는 보존했어요.` : "")
          + (st.info.warning ? " " + st.info.warning : "");
        $("yuesong-result").classList.remove("hidden");
        $("yuesong-result").scrollIntoView?.({ block: "nearest" });
        if (play) Promise.resolve($("yuesong-audio").play?.()).catch(() => {});
        const sung = $("yuesong-lyrics-view");                  // the words the take was sung with
        if (sung) {
          sung.textContent = st.info.lyrics === "ah" ? "가사 없이 '아'로 불렀어요."
            : st.info.lyrics === "model" ? "랜덤 가사 모드로 요청했어요. 실제 가사는 음원에서 확인해 주세요." : ($("yuesong-lyrics")?.value || "").trim();
          sung.classList.toggle("hidden", !sung.textContent);
        }
        riseIn($("yuesong-result"));
        libRise = st.info.library_id || null;                   // its library card rises once when it is listed
        const ask = st.info.ask || (st.info.stage === "first" ? "이 곡으로 전곡을 만들까요?" : "");   // a first part always asks
        yueSongFirst = ask ? { abc: st.info.abc, library_id: st.info.library_id, genre, title } : null;
        $("yuesong-ask-text").textContent = ask;
        $("yuesong-ask").classList.toggle("hidden", !ask || !play);
        if (ask) $("yuesong-ask").scrollIntoView?.({ block: "nearest" });
        loadLibrary();
        status.textContent = `${st.info.length_met === false ? "목표 길이 미달 · 짧은 결과 저장됨" : st.info.stage === "first" ? "1절 완성" : "완성"} (${Math.round(st.info.seconds)}초 분량, ${Math.round(st.info.elapsed_sec)}초 걸림)`;
        yueQueue = { now: "", line: "", last: `${what} · ${status.textContent}` };
        showQueue();
        return true;
      }
      status.textContent = `${st.progress[st.progress.length - 1] || "만드는 중"} · ${Math.round((Date.now() - t0) / 1000)}초 지남`;
      yueQueue.line = status.textContent;
      showQueue();
    }
  } catch (e) {
    if (seq === yueSongSeq) {
      status.textContent = "";
      yueQueue = { now: "", line: "", last: `${what} · 실패: ${String(e.message).split("\n")[0]}` };
      showQueue();
      showError("YuE2 곡 완성: " + e.message);
      if (first) $("yuesong-ask").classList.remove("hidden");   // refused or failed: the user can say yes again
    }
    return false;                                            // true when the take was made (the conversation says when it was not)
  } finally {
    if (seq === yueSongSeq) {
      for (const id of ["yuesong-on", "btn-yuesong", "btn-yuesong-yes", "btn-yue", "btn-song-overlay"]) {
        if ($(id)) $(id).disabled = yueAvailable === false;
      }
      yueBusy = false; yueRunning = null; showYueCancel(false);
    }
  }
}
// A short rise into view (CSS .rise, only without reduced motion); taken off and put back so it plays again.
function riseIn(el) {
  if (!el?.classList) return;
  el.classList.remove("rise");
  void el.offsetWidth;                                       // a reflow between the two restarts the animation
  el.classList.add("rise");
}
function adoptYueSong() {
  if (!yueSongResult) return;
  stopPlayback();
  showSong({ ...yueSongResult, from_leadsheet: true, yue_song: yueSongResult.yue_song });
}

// Layout A: the side menu switches what the content column shows (the create column stays).
function setView(v) {
  document.body.dataset.view = v;
  for (const a of $("views")?.querySelectorAll?.("a") || []) {
    a.classList.toggle("on", a.dataset.view === v);
    if (a.dataset.view === v) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");   // read out as the current page
  }
  for (const id of ["library", "queue", "settings", "practice"]) $("view-" + id)?.classList.toggle("hidden", v !== id);
  if (v === "library" || v === "practice") loadLibrary();
  if (v === "queue") showQueue();
}

// The library of YuE2 songs (GET /api/yue/library): the list sits in the side menu; a song plays in its own bar with a
// volume slider and its words, and its score can be loaded into the song card.
let libOpen = null, libRise = null;
// A voice id as the user reads it: the option texts of #yuesong-voice are the one list of labels (an unknown id stays as it is).
function voiceLabel(id) {
  const opt = [...($("yuesong-voice")?.options || [])].find((o) => o.value === id);
  return (opt?.textContent || "").trim() || id || "";
}
async function loadLibrary() {
  const list = $("lib-list");
  if (!list) return;
  let songs = [];
  try { songs = (await getJson("/api/yue/library")).songs; } catch (e) { return; }
  list.innerHTML = songs.length ? "" : '<li class="help">아직 만든 곡이 없어요</li>';
  for (const s of songs) {
    const li = document.createElement("li"), b = document.createElement("button");
    const len = `${Math.floor(s.seconds / 60)}:${String(Math.round(s.seconds % 60)).padStart(2, "0")}`;
    b.textContent = s.title;
    const small = document.createElement("small");
    small.textContent = `${s.length_met === false ? "목표 길이 미달" : s.stage === "first" ? "1절" : s.stage === "full" ? "전곡" : "곡"} · ${len} · ${s.created}`;
    b.append(small);
    b.classList.toggle("on", libOpen === s.id);
    b.onclick = () => openLibrarySong(s);
    li.append(b);
    list.append(li);
  }
  const plist = $("practice-songs");                           // and in the practice menu: pick a take to practise on
  if (plist) {
    plist.innerHTML = songs.length ? "" : '<li class="help">아직 만든 곡이 없어요. "만들기"에서 1절을 만들면 여기에 나와요.</li>';
    for (const s of songs) {
      const li = document.createElement("li"), b = document.createElement("button"), small = document.createElement("small");
      b.textContent = s.title;
      small.textContent = [s.length_met === false ? "목표 길이 미달" : s.stage === "first" ? "1절" : s.stage === "full" ? "전곡" : "곡",
        `${Math.floor(s.seconds / 60)}:${String(Math.round(s.seconds % 60)).padStart(2, "0")}`, s.created].filter(Boolean).join(" · ");
      b.append(small);
      b.classList.toggle("on", libOpen === s.id);
      b.onclick = () => openLibrarySong(s);
      li.append(b);
      plist.append(li);
    }
  }
  const view = $("lib-view-list");                             // the same songs as cards in the library view
  if (!view) return;
  view.innerHTML = songs.length ? "" : '<li class="help">아직 만든 곡이 없어요. "곡 만들기"에서 1절을 만들어 보세요.</li>';
  for (const s of songs) {
    const li = document.createElement("li"), b = document.createElement("button"), small = document.createElement("small");
    b.textContent = s.title;
    small.textContent = [s.length_met === false ? "목표 길이 미달" : s.stage === "first" ? "1절" : s.stage === "full" ? "전곡" : "곡",
      `${Math.floor(s.seconds / 60)}:${String(Math.round(s.seconds % 60)).padStart(2, "0")}`,
      s.key, s.tempo ? `${s.tempo} BPM` : "", s.created].filter(Boolean).join(" · ");   // a missing key or tempo is left out
    b.append(small);
    b.onclick = () => {
      openLibrarySong(s);
      $("lib-detail").textContent = `${s.title}: ${(s.sections || []).map(([n, k]) => `${n} ${k}마디`).join(" → ")} · 목소리 ${voiceLabel(s.voice)}`
        + (s.male_shift ? ` · ${-s.male_shift}키 낮춤(들리는 조 ${s.sounding_key})` : "")
        + (s.follow != null ? ` · 보컬이 악보를 따른 비율 ${s.follow}` : "") + (s.lyrics === "ah" ? " · 가사 없이 '아'" : "");
    };
    li.append(b);
    if (libRise && s.id === libRise) riseIn(li);
    view.append(li);
  }
  libRise = null;
}
// The part lines over the library bar (#lib-lines): the part picked as mine (#lib-part) is dimmed and dashed and says
// it is my turn; the vocal line otherwise says the voice is YuE2's.
const LINE_PARTS = ["vocal", "guitar", "bass", "keys", "drums"];
function markMyPart() {
  const mine = $("lib-part")?.value;
  for (const p of LINE_PARTS) {
    const li = $(`lib-line-${p}`), tag = $(`lib-line-${p}-tag`);
    if (!li || !tag) continue;
    li.classList.toggle("mine", p === mine);
    if (p === mine) li.setAttribute("aria-current", "true"); else li.removeAttribute("aria-current");
    tag.textContent = p === mine ? "내 차례" : p === "vocal" ? "YuE2" : "";
  }
}
let libSong = null;
function openLibrarySong(s) {
  showPracticeTiming(null);
  libOpen = s.id;
  libSong = s;
  $("btn-lib-full")?.classList.toggle("hidden", s.stage !== "first");     // a first part can be taken on to the whole song
  libFullFirst = null;                                       // a question asked for another song is gone
  $("lib-full-ask")?.classList.add("hidden");
  if ($("lib-full-note")) $("lib-full-note").textContent = "";
  if ($("practice-song-now")) $("practice-song-now").textContent = `연습할 곡: ${s.title} (${s.length_met === false ? "목표 길이 미달" : s.stage === "first" ? "1절" : s.stage === "full" ? "전곡" : "곡"}). 아래에서 내 파트를 고르세요.`;
  stopPlayback();
  for (const id of ["yuesong-audio", "yue-audio"]) $(id)?.pause?.();   // one sound at a time: the other takes stop
  $("lib-title").textContent = s.title;
  $("lib-audio").src = `/api/yue/library/${s.id}/audio`;
  $("lib-audio").volume = Number($("lib-volume").value) / 100;
  $("lib-lyrics").textContent = s.lyrics_text || "가사 없이 '아'로 부른 곡이에요.";
  $("lib-lyrics").classList.add("hidden");
  $("libbar").classList.remove("hidden");
  markMyPart();
  for (const b of $("lib-list").querySelectorAll("button")) b.classList.toggle("on", b.firstChild.textContent === s.title && libOpen === s.id);
  loadLibrary();
}
// The library's first part -> the whole song.  It says why when nothing can start, asks next to the button, and on
// "예" sends one request and shows the YuE2 card (the old version moved to a view without that card: a click seemed dead).
let libFullFirst = null;
function libFullSay(text) {
  if ($("lib-full-note")) $("lib-full-note").textContent = text;
  $("lib-full-ask")?.classList.add("hidden");
}
async function libraryFull() {
  libFullFirst = null;
  if (!libSong || libSong.stage !== "first") return libFullSay("1절로 저장된 곡에서만 전곡을 만들 수 있어요.");
  if (yueAvailable === false) return libFullSay(`전곡을 만들 수 없어요: ${yueReason}`);
  if (yueBusy) return libFullSay("다른 곡을 만드는 중이에요. 끝난 뒤 다시 눌러 주세요.");
  const song = libSong;
  libFullSay("저장된 1절을 읽는 중…");
  try {
    const row = await getJson(`/api/yue/library/${song.id}`);
    if (libSong !== song) return;                            // another song was opened meanwhile
    if (!row.abc) return libFullSay("이 곡에는 저장된 악보가 없어 전곡을 만들 수 없어요.");
    libFullFirst = { abc: row.abc, library_id: song.id, genre: row.genre || row.sheet_genre || null, title: row.title || song.title };
    libFullSay("");
    $("lib-full-ask-text").textContent = `"${libFullFirst.title}" 1절로 전곡을 만들까요?`;
    $("lib-full-ask").classList.remove("hidden");
    $("btn-lib-full-yes").disabled = false;
    $("btn-lib-full-yes").focus?.();                         // the keyboard lands on the answer
  } catch (e) { libFullSay(e.message); }
}
function libraryFullYes() {
  const first = libFullFirst;
  if (!first) return;
  if (yueBusy) return libFullSay("다른 곡을 만드는 중이에요. 끝난 뒤 다시 눌러 주세요.");
  libFullFirst = null;                                       // one request per question
  yueSongFirst = first;                                      // a cancelled or failed take asks again about this first part
  if ($("yuesong-ask-text")) $("yuesong-ask-text").textContent = `"${first.title}" 1절로 전곡을 만들까요?`;
  $("btn-lib-full-yes").disabled = true;
  $("lib-full-ask").classList.add("hidden");
  $("yue2-card")?.classList.remove("hidden");
  setView("studio");                                         // the workspace view that shows the YuE2 card
  $("yue2-card")?.scrollIntoView?.({ block: "start" });
  makeYueSong(first);
}
async function loadLibraryScore() {
  if (!libOpen) return;
  try {
    const row = await getJson(`/api/yue/library/${libOpen}`);
    showSong({ ...row.result, from_leadsheet: true, yue_song: row.result.yue_song });
    routeLib = libOpen;                                     // the take now playing in the library bar is this song
  } catch (e) { showError("라이브러리: " + e.message); }
}

// `quiet`: started by the page itself, so a failure is told next to the button and not in the error box.
async function makeYueDemo(styleOverride, quiet = false) {
  const song = state.song;
  if (!song) return false;
  const useStyle = typeof styleOverride === "string" ? styleOverride : $("yue-style").value;
  const useFull = !!$("yue-full")?.checked;
  if (!song.leadsheet) { showError("이 곡은 리드시트로 내보낼 수 없어 AI 보컬 데모를 만들 수 없어요."); return; }
  if (!sungMelody(song)) { showError(`이 곡의 멜로디는 ${ROLE_KO[song.melody_role] || song.melody_role}가 맡아서 AI 보컬 데모를 만들지 않아요.`); return; }
  hideError();
  const seq = ++yueSeq, button = $("btn-yue"), status = $("yue-status");
  button.disabled = true;
  yueBusy = true;
  status.textContent = "시작하는 중…";
  try {
    const { job } = await api("/api/yue", { leadsheet: song.leadsheet, style: useStyle, genre: song.genre || null,
      full: useFull, mode: currentMode() });
    const t0 = Date.now();
    for (;;) {
      await new Promise((r) => setTimeout(r, 2000));
      const res = await fetch(`/api/yue/${job}`);
      if (!res.ok) throw new Error(`상태 확인 실패 (${res.status})`);
      const st = await res.json();
      if (seq !== yueSeq) return;
      if (st.status === "error") throw new Error(st.error || "생성 실패");
      if (st.status === "done") {
        const blob = await (await fetch(`/api/yue/${job}/audio`)).blob();
        if (yueUrl) URL.revokeObjectURL(yueUrl);
        yueUrl = URL.createObjectURL(blob);
        const a = $("yue-audio");
        a.src = yueUrl;
        a.classList.remove("hidden");
        yueFor = yueKey(song, useStyle, useFull);
        status.textContent = `완성 (${st.info.label}, ${st.info.sections.join(" → ")}, ${Math.round(st.info.elapsed_sec)}초 걸림)`
          + (st.info.melody === "simplified" ? " · 멜로디를 단순하게 줄여 부른 데모예요" : "")
          + (st.info.pitch_shift ? ` · 전체를 ${Math.abs(st.info.pitch_shift)}키 ${st.info.pitch_shift < 0 ? "낮췄어요" : "올렸어요"}` : "");
        return true;
      }
      status.textContent = `${st.progress[st.progress.length - 1] || "만드는 중"} · ${Math.round((Date.now() - t0) / 1000)}초 지남`;
    }
  } catch (e) {
    if (seq === yueSeq) { status.textContent = quiet ? "자동 생성 실패: " + e.message : ""; if (!quiet) showError("AI 보컬 데모: " + e.message); }
  } finally {
    if (seq === yueSeq) { button.disabled = false; yueBusy = false; }
  }
  return false;
}

function showError(msg) { const e = $("error"); e.textContent = msg; e.classList.remove("hidden"); }
function hideError() { $("error").classList.add("hidden"); }

// ---------- controls ----------
// Octave change: play "do" (degree 1) in the new octave so the register can be heard.
function changeOctave(delta) {
  if (melodySel.size) return shiftPicked(delta);
  const next = Math.min(6, Math.max(2, state.octave + delta));
  if (next === state.octave) showError("옥타브는 2~6까지예요."); else hideError();
  state.octave = next;
  renderMelody();
  previewNote({ degree: 1, sharp: false, octave: state.octave });
}
// The picked notes (one, or all of them) an octave up or down; nothing moves if any would leave octaves 2~6.
function shiftPicked(delta) {
  const picked = [...melodySel].map((i) => state.tokens[i]).filter((t) => t && t.degree > 0);
  if (!picked.length) return showError("선택한 것이 쉼표뿐이에요. 음을 골라 주세요.");
  if (picked.some((t) => t.octave + delta < 2 || t.octave + delta > 6)) return showError("옥타브는 2~6까지예요. 더 옮길 수 없는 음이 있어요.");
  for (const t of picked) t.octave += delta;
  melodyChanged(true);
  previewNote(picked[0]);
}
const octUp = () => changeOctave(1);
const octDown = () => changeOctave(-1);
const durHalf = () => { state.dur = DURS[Math.max(0, DURS.indexOf(state.dur) - 1)]; renderMelody(); };
const durDouble = () => { state.dur = DURS[Math.min(DURS.length - 1, DURS.indexOf(state.dur) + 1)]; renderMelody(); };
const toggleSharp = () => { state.sharpNext = !state.sharpNext; renderMelody(); };
const delLast = () => { state.tokens.pop(); melodyChanged(); };

// ---------- M6: style request / reference audio -> attributes -> the user confirms -> applied ----------
const genreButtons = {}, instrButtons = {};
// Genre default tempo = arrange.TEMPO in the backend (tests/test_api.py checks that this table matches).
// ---------- song cover: a square picture drawn from data the page already has (no request, no Math.random) ----------
// Genre-colour gradient + two soft lights, with the seed melody's notes as rounded bars (x = time, y = pitch).
// The song has no stored id on this path (/api/song returns none), so a hash of stable song data (genre, bars, tempo, key,
// section labels, seed melody notes) seeds the generator: the same song always gives the same picture.
const COVER_FALLBACK = ["#9BE15D", "#4FD1C5"];       // lime + teal: the page's default pair
function hash32(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return h >>> 0;
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// The cover (user, 2026-10-13: "the cover is poor"): a dark field of the genre's two colours with a faint starfield, the
// seed melody as a radial piano roll (time runs round the circle from the top, pitch is the distance from the centre,
// like the grooves of a record), a strip of the sections under it (my melody lime, the model's teal) and the genre, key,
// tempo and bars as words.  The picture is a pure function of the song, so the same song is always the same picture.
const COVER_FONT = '"Pretendard", "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif';
// the melody notes of the first section that carries the user's seed (origin "input"), the first notes when none does
function coverNotes(song) {
  const mel = (song?.playback?.notes || []).filter((n) => n.track === "melody");
  const sec = (song?.sections || []).find((s) => s.origin === "input");
  const t0 = sec ? sec.start_sec : 0;
  const list = sec ? mel.filter((n) => n.time >= sec.start_sec - 1e-6 && n.time < sec.end_sec - 1e-6) : mel.slice(0, 64);
  return list.map((n) => ({ t: n.time - t0, d: n.duration, p: n.midi, seed: !!sec })).slice(0, 160);
}
function coverKey(song) {
  return [song?.genre, song?.bars, song?.tempo, song?.key?.name, (song?.sections || []).map((s) => s.label).join(","),
    coverNotes(song).map((n) => `${n.t.toFixed(3)}:${n.p}:${n.d.toFixed(3)}`).join(",")].join("|");
}
function coverModel(song, colors = COVER_FALLBACK, size = 1024) {
  const rng = mulberry32(hash32(coverKey(song)));
  const r2 = (v) => Math.round(v * 100) / 100;
  const model = { size, colors: [colors[0], colors[1]], angle: r2(20 + rng() * 140), cx: size / 2, cy: r2(size * .40),
    r0: r2(size * .095), r1: r2(size * .285), arcs: [], stars: [], strip: [], rings: [.15, .215, .285].map((k) => r2(size * k)),
    text: { brand: "그랜드챕스 작곡메이트", genre: GENRES[song?.genre] || song?.genre || "",
      meta: [song?.key?.name, song?.tempo ? `${song.tempo} BPM` : "", song?.bars ? `${song.bars}마디` : ""].filter(Boolean).join(" · ") } };
  for (let i = 0; i < 70; i++) model.stars.push({ x: r2(rng() * size), y: r2(rng() * size * .98), r: r2(.6 + rng() * 1.9), a: r2(.15 + rng() * .5) });
  const notes = coverNotes(song);
  if (notes.length) {
    const span = Math.max(1e-6, ...notes.map((n) => n.t + n.d));      // one turn is the whole seed
    const lo = Math.min(...notes.map((n) => n.p)), hi = Math.max(...notes.map((n) => n.p)), range = Math.max(12, hi - lo);   // a flat tune stays a ring
    const mid = (lo + hi) / 2, rm = (model.r0 + model.r1) / 2;
    for (const n of notes) {
      const a0 = -Math.PI / 2 + (n.t / span) * Math.PI * 2 * .985;
      const a1 = Math.max(a0 + .045, a0 + (n.d / span) * Math.PI * 2 * .985 - .012);
      model.arcs.push({ a0: +a0.toFixed(5), a1: +a1.toFixed(5), r: r2(rm + ((n.p - mid) / range) * (model.r1 - model.r0)), seed: n.seed });
    }
  }
  const secs = song?.sections || [];
  const total = secs.reduce((n, x) => n + Math.max(0, (x.end_sec - x.start_sec) || 0), 0);
  if (total > 0) {
    const pad = size * .1, gap = size * .006, width = size - 2 * pad;
    let x = pad;
    for (const x0 of secs) {
      const w = Math.max(0, ((x0.end_sec - x0.start_sec) / total) * width);
      model.strip.push({ x: r2(x), w: r2(Math.max(0, w - gap)), origin: x0.origin || "rule", label: x0.label || "" });
      x += w;
    }
  }
  return model;
}
function drawCover(canvas, song, colors = COVER_FALLBACK, size = 1024) {
  const ctx = canvas?.getContext?.("2d");
  if (!ctx) return null;
  const m = coverModel(song, colors, size), S = m.size, rad = m.angle * Math.PI / 180, dx = Math.cos(rad) * S / 2, dy = Math.sin(rad) * S / 2;
  const g = ctx.createLinearGradient(S / 2 - dx, S / 2 - dy, S / 2 + dx, S / 2 + dy);      // the genre's two colours, deepened below
  g.addColorStop(0, m.colors[0]); g.addColorStop(1, m.colors[1]);
  ctx.fillStyle = g; ctx.fillRect(0, 0, S, S);
  const v = ctx.createRadialGradient(m.cx, m.cy, S * .08, m.cx, m.cy, S * .78);           // a night vignette: the melody is the light
  v.addColorStop(0, "rgba(10,14,34,.35)"); v.addColorStop(1, "rgba(8,10,26,.9)");
  ctx.fillStyle = v; ctx.fillRect(0, 0, S, S);
  const glow = ctx.createRadialGradient(m.cx, m.cy, 0, m.cx, m.cy, S * .42);
  glow.addColorStop(0, "rgba(255,255,255,.16)"); glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = glow; ctx.fillRect(0, 0, S, S);
  for (const t of m.stars) { ctx.globalAlpha = t.a; ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(t.x, t.y, t.r, 0, Math.PI * 2); ctx.fill(); }
  ctx.globalAlpha = 1;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(255,255,255,.14)"; ctx.lineWidth = S * .0016;                   // the record's guide rings
  for (const r of m.rings) { ctx.beginPath(); ctx.arc(m.cx, m.cy, r, 0, Math.PI * 2); ctx.stroke(); }
  ctx.lineWidth = S * .016;
  for (const a of m.arcs) {                                                                // one arc per note
    ctx.beginPath();
    ctx.arc(m.cx, m.cy, a.r, a.a0, a.a1);
    ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.shadowColor = "rgba(255,255,255,.7)"; ctx.shadowBlur = S * .02; ctx.stroke();
  }
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255,255,255,.92)"; ctx.beginPath(); ctx.arc(m.cx, m.cy, S * .014, 0, Math.PI * 2); ctx.fill();   // the spindle
  const ink = { input: "#9BE15D", model: "#4FD1C5" };
  for (const p of m.strip) {                                                               // the sections, side by side
    ctx.fillStyle = ink[p.origin] || "rgba(255,255,255,.55)";
    const h = S * .018, r = h / 2;
    ctx.beginPath();
    ctx.moveTo(p.x + r, S * .835); ctx.arcTo(p.x + p.w, S * .835, p.x + p.w, S * .835 + h, r); ctx.arcTo(p.x + p.w, S * .835 + h, p.x, S * .835 + h, r);
    ctx.arcTo(p.x, S * .835 + h, p.x, S * .835, r); ctx.arcTo(p.x, S * .835, p.x + p.w, S * .835, r); ctx.closePath(); ctx.fill();
  }
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.font = `600 ${S * .026}px ${COVER_FONT}`; ctx.textAlign = "center";
  ctx.fillText(m.text.brand, S / 2, S * .075);
  ctx.fillStyle = "#fff"; ctx.font = `900 ${S * .085}px ${COVER_FONT}`; ctx.textAlign = "left";
  ctx.fillText(m.text.genre, S * .1, S * .795);
  ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.font = `600 ${S * .03}px ${COVER_FONT}`;
  ctx.fillText(m.text.meta, S * .1, S * .915);
  return m;
}
// the genre's own two colours are read from its tile's --g1 / --g2 (style.css is the one place they are defined)
function genreColors(g) {
  const cs = genreButtons[g] && globalThis.getComputedStyle?.(genreButtons[g]);
  const a = cs?.getPropertyValue?.("--g1")?.trim(), b = cs?.getPropertyValue?.("--g2")?.trim();
  return a && b ? [a, b] : COVER_FALLBACK;
}
function renderCover(song) {
  const canvas = $("song-cover");
  if (!canvas) return;
  const m = drawCover(canvas, song, genreColors(song.genre));
  canvas.setAttribute?.("aria-label", `곡 표지 그림: ${GENRES[song.genre] || song.genre} ${song.bars}마디 ${song.tempo} BPM, 씨앗 멜로디 음표 ${m ? m.arcs.length : 0}개를 원형 악보로 그렸어요`);
}
function saveCoverPng() {
  const canvas = $("song-cover");
  if (!state.song || !canvas?.toBlob) return;
  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `jakgokmate_${state.song?.genre}_cover.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/png");
}
// the page's two lights: lime + teal until a genre is chosen, then that genre's pair (style.css eases the change over 0.6 s)
function syncGlow() {
  const st = document.documentElement?.style;
  if (!st?.setProperty) return;
  const [a, b] = state.src.genre ? genreColors(state.genre) : COVER_FALLBACK;
  st.setProperty("--glow-a", a); st.setProperty("--glow-b", b);
}

const DEFAULT_BPM = { ballad: 72, classical: 96, rock: 120, dance: 124, kpop: 112, jpop: 148, newage: 72, hiphop: 90, piano_rock: 128, pop_punk: 172, hard_rock: 118, emo_rock: 92, modern_rock: 122 };
const SRC_TEXT = {
  genre: { rec: "추천", manual: "직접 선택" }, instr: { rec: "추천", manual: "직접 선택" }, tempo: { rec: "추천", manual: "직접 입력" },
};
function renderSourceTags() {
  const set = (id, key, fallback) => {
    const el = $(id), v = state.src[key];
    el.textContent = v ? SRC_TEXT[key][v] : fallback;
    el.classList.toggle("rec", v === "rec");
  };
  set("genre-tag", "genre", `기본 (${GENRES[state.genre]})`);
  set("instr-tag", "instr", "기본 (밴드: 피아노·베이스·드럼·기타)");
  set("tempo-tag", "tempo", "");
  $("style-tempo").placeholder = `장르 기본 (${GENRES[state.genre]} ${DEFAULT_BPM[state.genre]})`;
}
function syncStyleButtons() {
  for (const [g, b] of Object.entries(genreButtons)) b.classList.toggle("on", g === state.genre);
  for (const [k, b] of Object.entries(instrButtons)) b.classList.toggle("on", state.instruments.has(k));
  syncGlow();
  const xb = $("extras-box");                // an extra part that is on must stay visible
  if (xb && EXTRA_INSTR.some((k) => state.instruments.has(k))) xb.open = true;
  $("style-tempo").value = state.tempo ? String(state.tempo) : "";
  renderSourceTags();
}


// Korean object particle: 을 after a final consonant, 를 otherwise.
function withObjectParticle(word) {
  const c = word.charCodeAt(word.length - 1);
  return word + (c >= 0xac00 && c <= 0xd7a3 && (c - 0xac00) % 28 !== 0 ? "을" : "를");
}


// Proposal -> recommendation card + two buttons. Nothing changes in the settings until "추천대로 적용".




// "추천대로 적용": the server validates the attributes again and returns the concrete settings; only then the UI changes.


function setTempoFromInput() {
  const raw = $("style-tempo").value.trim();
  if (raw === "") { state.tempo = null; state.src.tempo = null; }
  else {
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 40 || n > 200) { $("style-tempo").value = state.tempo ? String(state.tempo) : ""; return showError("템포는 40~200 사이 정수로 입력하세요."); }
    state.tempo = n;
    state.src.tempo = "manual";
  }
  state.style = null;
  renderSourceTags();
  invalidateResult();
}

// Reference audio: analysed on the server (tempo, key, structure, energy only) and deleted at once.

// Upload limits of the score import (the comparison card was removed 2026-10-07; POST /api/score stays on the server).
const SCORE_MAX_BYTES = 2 * 1024 * 1024, PDF_MAX_BYTES = 30 * 1024 * 1024;

// ---------- M5: AI song (background job), progress list, structure timeline, section playback, ratings ----------
const TOOL_KO = {
  analyze_melody: "멜로디 분석", style_from_request: "스타일 적용", analyze_reference_audio: "레퍼런스 분석",
  plan_structure: "구조 설계", generate_section: "구간 생성", arrange_section: "구간 반주", score_section: "점수 검사",
  render: "악보·음원 만들기", similar_check: "유사곡 검사", explain: "요약",
};
const KIND_KO = { intro: "인트로", verse: "벌스", pre: "프리", chorus: "후렴", bridge: "브릿지", interlude: "간주", outro: "아웃트로" };
const SOURCE_BADGE = { model: "AI", rule: "규칙", seed: "씨앗" };
let POLL_MS = 1000;
let composeSeq = 0;                      // only the newest job may touch the UI (older polls end silently)

// Block widths in percent, proportional to the number of bars (they sum to 100).
function timelineWidths(sections) {
  const total = sections.reduce((s, x) => s + x.bars, 0);
  return sections.map((x) => (total ? (100 * x.bars) / total : 0));
}

// Seconds of a section from its first bar and length: bar length = beats per bar (quarter notes) * 60 / BPM.
function beatsPerBar(meter) {
  const m = /^(\d+)\/(\d+)$/.exec(meter || "4/4");
  return m ? (Number(m[1]) * 4) / Number(m[2]) : 4;
}
function sectionRange(section, bpm, meter = "4/4") {
  const bar = (beatsPerBar(meter) * 60) / bpm;
  return { start: section.start_bar * bar, end: (section.start_bar + section.bars) * bar };
}

// Notes that start inside [start, end), moved to start at 0 and cut at the section end.
function notesInRange(notes, range) {
  return notes.filter((n) => n.time >= range.start - 1e-6 && n.time < range.end - 1e-6)
    .map((n) => ({ ...n, time: n.time - range.start, duration: Math.min(n.duration, range.end - n.time) }));
}

// "AI n개 · 규칙 m개": a repeated section counts once (it reuses its first occurrence); 후렴 is the seed.
function sourceCounts(sections, sources) {
  const seen = new Set(), out = { model: 0, rule: 0, seed: 0 };
  sections.forEach((s, i) => {
    const src = sources[i] || "rule";
    if (src === "seed") { out.seed++; return; }
    if (seen.has(s.name)) return;
    seen.add(s.name);
    out[src] = (out[src] || 0) + 1;
  });
  return out;
}

function statsSummary(data) {
  const c = sourceCounts(data.sections, data.section_sources || []);
  const g = data.compose_stats?.grammar?.round1;
  const gram = g && g.candidates ? `${g.grammar_valid}/${g.candidates} (${Math.round((100 * g.grammar_valid) / g.candidates)}%)`
    : "모델을 쓰지 않아 해당 없음";
  return `AI가 쓴 구간 ${c.model}개 · 규칙 기반 구간 ${c.rule}개 · 후렴(씨앗) ${c.seed}개 · 이번 생성의 문법 통과율(1회차) ${gram}`;
}

const ratingKey = (songId, section) => `${songId}:${section == null ? "song" : section}`;
const isRated = (songId, section) => state.rated.has(ratingKey(songId, section));
const RATE_MSG = { saving: "저장 중...", saved: "저장됨" };

function shortText(s, n = 100) { s = String(s ?? ""); return s.length > n ? s.slice(0, n) + "…" : s; }

function renderProgress(progress) {
  const box = $("compose-progress");
  box.innerHTML = progress.map((t) => `<li><b>${escapeHtml(TOOL_KO[t.tool] || t.tool)}</b>${escapeHtml(shortText(t.output))}</li>`).join("");
  box.classList.toggle("hidden", !progress.length);
}

function setComposeBusy(on) {
  if (on !== !!state.composing) setMaking(on);
  state.composing = on;
  $("btn-compose").disabled = on;
  const rewrite = $("btn-rewrite");
  if (rewrite) rewrite.disabled = on || state.selected == null || state.composed?.sources[state.selected] === "seed";
  if (!on && state.selected != null && state.composed) renderSectionPanel();     // the full rule (incl. songs without the model)
}

// The input changed (or a new job started): the old job's responses must not touch the UI any more.
function cancelCompose() {
  composeSeq++;
  if (state.composing) { setComposeBusy(false); renderProgress([]); }
}

let sleep = (ms) => new Promise((r) => setTimeout(r, ms));     // (tests replace it: their fake window has no timers)
async function getJson(path) {
  const res = await fetch(path, { method: "GET" });
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

// Start a job and poll it every POLL_MS. Returns the finished result, or null when this job is no longer current.
async function runJob(startPath, body, seq) {
  const { job_id } = await api(startPath, body);
  for (;;) {
    await sleep(POLL_MS);
    if (seq !== composeSeq) return null;
    const st = await getJson(`/api/compose/${job_id}`);
    if (seq !== composeSeq) return null;
    renderProgress(st.progress || []);
    if (st.status === "done") return st.result;
    if (st.status === "error") throw new Error(st.error || "곡 만들기에 실패했어요");
  }
}

function composeRequest() {
  const body = { ...baseRequest(), key_choice: state.keyChoice, ...seedHarmony() };
  if (state.style) body.style = state.style;
  const form = $("form-choice")?.value;
  if (form && form !== "standard") body.form = form;     // chorus_first: the hook right after the intro
  const src = $("melody-source")?.value || "ai";
  if (src !== "ai") body.melody_source = "own";             // "own" and "rest" never call the trained model
  if (src === "rest") body.fill = "rest";
  body.seed_role = $("seed-role")?.value || "chorus";
  if (body.seed_role === "intro" && melodyRole() === "vocal") body.intro_player = $("intro-player")?.value || "guitar";
  if (state.seedChords) body.seed_chords = state.seedChords;             // chords the user chose for the seed's bars
  body.fit_voice = fitVoice();
  body.key_change = !!$("key-change")?.checked;           // off: the input melody stays as written in every chorus
  body.melody_shape = !!$("melody-shape")?.checked;       // option, off by default (instmate/melodyshape.py)
  body.melody_role = melodyRole();
  return body;
}

function requestWithSettings(body) {
  const out = { ...body };
  if (state.tempo != null && state.src.tempo) {
    if (out.score) out.score = { ...out.score, bpm: state.tempo };
    else out.tempo = state.tempo;
  }
  if (state.src.genre) out.genre = state.genre;
  if (state.src.instr) out.instruments = [...state.instruments];
  const selected = composeRequest();
  const controls = out.score ? [['melody-role', 'melody_role'], ['vocal-style', 'fit_voice'], ['fit-voice', 'fit_voice']]
    : [['form-choice', 'form'], ['seed-role', 'seed_role'], ['intro-player', 'intro_player'],
      ['melody-source', 'melody_source', 'fill'], ['key-change', 'key_change'], ['melody-shape', 'melody_shape'],
      ['melody-role', 'melody_role'], ['vocal-style', 'fit_voice'], ['fit-voice', 'fit_voice']];
  for (const [id, ...keys] of controls) {
    if ($(id)?.dataset.userSet !== '1') continue;
    for (const key of keys) {
      if (key in selected) out[key] = selected[key];
      else delete out[key];
    }
  }
  if (out.score && $('import-key')?.dataset.userSet === '1') out.key = $('import-key').value.trim() || null;
  if (!out.score && state.style) out.style = state.style;
  return out;
}

async function startCompose() {
  if (state.composing) return;
  if (!state.tokens.length) return showError("먼저 멜로디를 입력하세요.");
  if (($("seed-role")?.value || "chorus") !== "chorus" && ($("melody-source")?.value || "ai") === "ai") {
    return showError("입력을 인트로나 벌스에 놓을 때는 AI 멜로디 제안을 쓸 수 없어요(모델은 후렴이 주어진 곡을 채우도록 학습됐어요). "
      + "\"나머지 구간의 멜로디\"를 \"내 멜로디 변형만\"이나 \"비워 두기\"로 바꿔 주세요.");
  }
  await runCompose("/api/compose", composeRequest(), "곡을 만들고 있어요...");
}

async function rewriteSection() {
  const c = state.composed, i = state.selected;
  if (state.composing || !c || i == null || c.sources[i] === "seed") return;
  await runCompose(`/api/compose/${c.song_id}/section`, { section_index: i }, `${c.sections[i].label}만 다시 만들고 있어요...`);
}

async function runCompose(path, body, message) {
  hideError();
  $("ask").classList.add("hidden");
  const seq = ++composeSeq, version = state.inputVersion;
  const previous = state.song;               // members' own parts in the shown song are carried into the new version
  setComposeBusy(true);
  renderProgress([]);
  $("compose-status").textContent = message;
  let made = false;
  try {
    let result = await runJob(path, body, seq);
    if (result === null || seq !== composeSeq) return;
    if (version !== state.inputVersion) return;
    if (result.status === "need_key") { askKey(result, startCompose); $("compose-status").textContent = ""; return; }
    const carried = await carryMemberParts(result, previous);
    if (seq !== composeSeq || version !== state.inputVersion) return;
    const parent = bandParent;               // rewriting one section stays in the band version's line of history
    showComposed(carried.result);
    made = true;
    if (path.includes("/section") && parent) bandParent = { ...parent, clean: false };
    if (carried.note && $("band-status")) $("band-status").textContent = carried.note;
    // status is "duration_out_of_range" (not "ok") when the song is not 2-4 minutes: say so instead of a plain success
    $("compose-status").textContent = result.status === "duration_out_of_range"
      ? `만들었지만 길이가 2~4분 범위 밖이에요. ${result.duration_note || ""}` : "완성했어요.";
  } catch (e) {
    if (seq === composeSeq) { showError(e.message); $("compose-status").textContent = ""; }
  } finally {
    if (seq === composeSeq) setComposeBusy(false);
  }
  // after the model is done: the two never run together.  In the conversation the AI vocal starts only from its button.
  if (made && !document.body?.classList?.contains("chat-mode") && !autoYueSong()) autoAiVocal();
}

// YuE2 makes the song (user, 2026-10-13): with "이 기능 켜기" on (the default) a song the app has just made goes straight on
// to YuE2's first part and the page moves to its card, so the user hears the song YuE2 writes from the melody.  One song
// starts one take; nothing starts while another is being made, with the feature off, or for a song that came from YuE2.
let yueAutoSong = null;
async function completeYueSong(song) {
  const version = state.inputVersion;
  const settings = () => JSON.stringify([$("yuesong-voice").value, $("yuesong-key").value,
    $("yuesong-lyrics").value, $("yuesong-lowmem").checked, yueExtras().body, window.workspaceController.readSettings?.()]);
  const initial = settings();
  const current = () => state.song === song && state.inputVersion === version && settings() === initial && $("yuesong-on").checked;
  $("btn-yuesong-adopt").disabled = true;
  try {
    window.workspaceController.openYue();
    if (!await makeYueSong(null, false)) return;
    if (!current()) return chatSay("app", "입력이나 설정이 바뀌어 전곡 자동 진행을 멈췄어요. 완성된 1절은 보존됩니다.");
    const first = yueSongFirst;
    if (first) {
      $("yuesong-ask").classList.add("hidden");
      $("yuesong-status").textContent = "1절을 저장했어요. 이어서 전곡과 악보를 만들어요…";
      if (!await makeYueSong(first)) return;
    }
    if (!current() || !yueSongResult?.leadsheet) return;
    adoptYueSong();
    window.workspaceController.openYue();
    $("yuesong-status").textContent += yueSongResult.yue_song?.length_met === false ? " · 목표 길이에 미달한 음원과 악보예요" : " · 음원과 악보 준비됨";
  } finally { $("btn-yuesong-adopt").disabled = false; }
}
function autoYueSong(inChat = false) {
  if (yueAvailable === false) return false;
  const song = state.song;
  if (!song?.leadsheet || song === yueAutoSong || yueBusy || song.yue_song || !$("yuesong-on")?.checked) return false;
  yueAutoSong = song;
  if (window.workspaceController?.openYue) {
    completeYueSong(song).catch(e => showError("YuE2 자동 완성: " + e.message));
    return true;
  }
  if (inChat) chatSay("app", "YuE2가 1절을 만들기 시작했어요. 1~7분 걸리고, 끝나면 바로 들려 드려요. 그동안 다른 생성은 할 수 없어요.");
  $("yue2-card")?.scrollIntoView?.({ block: "start", behavior: "instant" });      // after the line: the conversation's own scroll must not undo it
  makeYueSong().then((ok) => { if (inChat && ok === false) chatSay("app", "YuE2로 곡을 만들지 못했어요. " + ($("error")?.textContent || "")); });
  return true;
}

function showComposed(data) {
  stopPlayback();
  showSong(data);
  state.composed = { song_id: data.song_id, sections: data.sections, sources: data.section_sources || [],
    bpm: data.playback?.bpm ?? data.tempo, meter: data.meter || "4/4" };
  state.selected = null;
  $("compose-extras").classList.remove("hidden");
  $("compose-explain").textContent = data.explain || "";
  $("compose-stats").textContent = statsSummary(data);
  renderSimilar(data.similar_check || data.compose_stats?.similar_check || null);
  renderTimeline();
  renderSectionPanel();
  renderSongRating();
  $("song").scrollIntoView?.({ block: "nearest" });
}

// ---------- similar-song check box (server field `similar_check`, see instmate/plagcheck.py) ----------
function similarItemHtml(m) {
  const who = [m.artist, m.dataset_label, m.year].filter((x) => x).join(", ");
  return `<li>'${escapeHtml(m.title)}'${who ? ` (${escapeHtml(who)})` : ""} · ${m.notes}음 연속 · `
    + `${escapeHtml(m.section)} ${m.our_bars[0]}~${m.our_bars[1]}마디</li>`;
}

function similarListHtml(label, rep) {
  const block = (title, items) => `<p><b>${title}</b></p>`
    + (items.length ? `<ol>${items.map(similarItemHtml).join("")}</ol>` : `<p class="help">없음</p>`);
  return `<h4>${label}</h4>` + block("음높이+리듬이 같은 구간", rep.top_rhythm)
    + block("음높이만 같은 구간 (참고용, 위험도에는 쓰지 않아요)", rep.top_pitch);
}

function renderSimilar(report) {
  const box = $("similar-box");
  if (!report) { box.classList.add("hidden"); return; }
  $("similar-summary").textContent = report.summary || "";
  $("similar-note").textContent = report.note || "";
  $("similar-lists").innerHTML = report.available
    ? similarListHtml("AI가 쓴 파트", report.ai) + similarListHtml("씨앗 멜로디", report.seed) : "";
  $("similar-details").classList.toggle("hidden", !report.available);
  box.classList.remove("hidden");
}

function renderTimeline() {
  const box = $("timeline"), c = state.composed;
  box.innerHTML = "";
  const widths = timelineWidths(c.sections);
  c.sections.forEach((s, i) => {
    const b = document.createElement("button");
    const src = c.sources[i] || "rule";
    b.className = `tl tl-${s.kind}` + (state.selected === i ? " selected" : "");
    b.style.flexGrow = String(widths[i]);      // proportional to bars; the row fits the card (see .timeline in style.css)
    b.dataset.index = String(i);
    b.title = `${s.label} · ${s.bars}마디 · ${SOURCE_BADGE[src] || src}`;
    b.setAttribute("aria-pressed", state.selected === i ? "true" : "false");
    b.innerHTML = `<span class="tl-name">${escapeHtml(s.label)}</span><span class="tl-bars">${s.bars}마디</span>`
      + `<span class="badge badge-${src}">${SOURCE_BADGE[src] || src}</span>`;
    b.onclick = () => selectSection(i);
    box.append(b);
  });
}

function selectSection(i) {
  const c = state.composed;
  if (!c || !state.song) return;
  state.selected = i;
  renderTimeline();
  renderSectionPanel();
  playSection(i);
}

// Plays only this section: the song's notes inside its time range, moved to time 0 (same scheduling as the full song).
function playSection(i) {
  const c = state.composed;
  if (!c || !state.song) return;
  const range = sectionRange(c.sections[i], c.bpm, c.meter);
  const notes = notesInRange(state.song.playback.notes, range);
  if (notes.length) playEvents(notes, `${c.sections[i].label} 미리듣기`);
}

function renderSectionPanel() {
  const c = state.composed, i = state.selected, box = $("section-panel");
  if (!c || i == null) { box.classList.add("hidden"); return; }
  const s = c.sections[i], src = c.sources[i] || "rule", isSeed = src === "seed";
  const r = sectionRange(s, c.bpm, c.meter);
  $("section-title").textContent = `${s.label} · ${s.bars}마디 · ${fmtTime(r.start)}~${fmtTime(r.end)} · ${SOURCE_BADGE[src] || src}`;
  const rw = $("btn-rewrite");
  // the model rewrites a section only in a song built around a chorus input with model-written sections
  const st = state.song?.settings || {};
  const noModel = (st.seed_role || "chorus") !== "chorus" || st.fill === "rest" || state.song?.melody_source === "own";
  const why = isSeed ? "입력한 멜로디가 들어간 구간이라 그대로 둬요"
    : noModel ? "이 곡은 AI 멜로디 없이 만들어서 구간 다시 만들기를 쓸 수 없어요. 음을 고치려면 위의 \"파트 편집\"을 쓰세요." : "";
  rw.disabled = state.composing || isSeed || noModel;
  rw.title = why || "이 파트만 새로 만들어요";
  $("rewrite-note").textContent = why;
  renderRatingButtons("section");
  box.classList.remove("hidden");
}

// ---------- ratings: one per song / section, disabled after the click ----------
function renderSongRating() { renderRatingButtons("song"); }

function renderRatingButtons(scope) {
  const c = state.composed;
  if (!c) return;
  const section = scope === "song" ? null : state.selected;
  if (scope === "section" && section == null) return;
  const done = isRated(c.song_id, section);
  for (const [id, r] of [[`rate-${scope}-up`, 1], [`rate-${scope}-down`, -1]]) {
    const b = $(id);
    b.disabled = done;
    b.onclick = () => rate(section, r);
  }
  $(`rate-${scope}-msg`).textContent = done ? RATE_MSG[state.rated.get(ratingKey(c.song_id, section))] : "";
}

async function rate(section, rating) {
  const c = state.composed;
  if (!c || isRated(c.song_id, section)) return;
  const scope = section == null ? "song" : "section", songId = c.song_id;
  const key = ratingKey(songId, section);
  state.rated.set(key, "saving");                   // disable at once so a double click sends one rating
  renderRatingButtons(scope);
  try {
    await api("/api/rate", { song_id: songId, section, rating });
    state.rated.set(key, "saved");
    if (state.composed?.song_id === songId && (section == null || state.selected === section)) renderRatingButtons(scope);
  } catch (e) {
    state.rated.delete(key);                         // not saved: allow another try
    if (state.composed?.song_id === songId) { renderRatingButtons(scope); showError(e.message); }
  }
}

// ---------- conversation (POST /api/chat): the server fixes the order of the work and says which existing API comes
// next; this page calls it the same way the form does.  The session lives here, the server keeps nothing.
let chatSession = null, chatBusy = false;
let chatMainScore = null;
function chatSay(who, text) {
  const log = $("chat-log");
  if (!log || !text) return;
  const li = document.createElement("li");
  li.className = "chat-" + who;             // not "app" / "me" alone: ".app" is the page's own layout class (full height)
  li.textContent = text;
  log.append(li);
  log.scrollTop = log.scrollHeight;
}

function setMode(mode) {
  document.body?.classList?.toggle("chat-mode", mode === "chat");
  $("mode-chat")?.classList?.toggle("on", mode === "chat");
  $("mode-form")?.classList?.toggle("on", mode !== "chat");
}

// The text shown when a finished song's chorus is not the melody the user gave (the server's own words when it sent
// them; an older result that only carries the failed check gets the plain sentence).  null when all is well.
function seedWarning(song) {
  if (song?.seed_warning?.text) return song.seed_warning.text;
  return song?.checks?.chorus_equals_seed === false ? "입력 멜로디와 다르게 나왔어요. 후렴이 입력한 멜로디와 같지 않으니 악보를 확인해 주세요." : null;
}

function chatSongSummary(song) {
  const sources = song.section_sources || [];
  return { song_id: song.song_id, melody_source: song.melody_source || null,
    key: song.key?.label || null, tempo: song.tempo || null, genre: song.genre || null,   // facts for explanations
    seed_warning: seedWarning(song),            // the chorus is not the typed melody: the conversation says it too
    sections: (song.sections || []).map((s, i) => ({ label: s.label, kind: s.kind, source: sources[i] || null })) };
}

// The AI vocal (YuE2) needs about 18 GB: in the conversation it never starts by itself.  The button asks the server how
// much memory is free first; when that is short it warns, and only a second click starts it.  It makes "AI vocal on the
// app's band" (POST /api/yue/overlay): the YuE2 voice alone over this song's own accompaniment, five semitones lower.
// One run of "AI vocal on the app's band": memory check (a warning the first time when memory is short), the job, its
// progress lines through `say`, and the finished audio as a blob.  Returns null when it only warned.
const overlayWarned = new WeakSet();
async function makeOverlay(song, full, say) {
  const st = await getJson("/api/yue/status");
  if (!st.available) throw new Error(st.reason || "AI 보컬을 쓸 수 없어요");
  if (!st.memory_ok && !overlayWarned.has(song)) {
    overlayWarned.add(song);
    say(`남은 메모리가 약 ${st.memory_free_gb}GB로, 필요한 ${st.memory_need_gb}GB보다 적어요. 다른 프로그램을 닫은 뒤 다시 누르거나, 그래도 시작하려면 한 번 더 눌러 주세요.`);
    return null;
  }
  say(`AI 보컬을 만들어 이 곡의 반주에 얹고 있어요 (${full ? "곡 전체" : "하이라이트"}, 몇 분 걸려요). AI 보컬: YuE2 (m-a-p)`);
  const { job } = await api("/api/yue/overlay", { leadsheet: song.leadsheet, genre: song.genre || null, mode: currentMode(), full, lyrics: $("yuesong-lyrics")?.value.trim() || null });
  let seen = 0;
  for (;;) {
    await sleep(3000);
    const j = await getJson(`/api/yue/${job}`);
    for (const line of (j.progress || []).slice(seen)) say(line);
    seen = (j.progress || []).length;
    if (j.status === "error") throw new Error(j.error || "실패");
    if (j.status === "done") {
      const blob = await (await fetch(`/api/yue/${job}/audio`)).blob();
      return { blob, info: j.info || {} };
    }
  }
}
function overlayNote(info) {
  return `AI 보컬을 얹은 ${info.full ? "완성곡" : "하이라이트"}예요. 원래 조보다 ${info.lowered ?? 5}키 낮은 ${info.key || ""} 조이고, 멜로디는 줄여서 불러요.`;
}

// The song card's main button: the finished song with the AI voice.
async function playSongOverlay() {
  const song = state.song, b = $("btn-song-overlay"), note = $("song-overlay-status"), player = $("song-overlay-audio");
  if (!song?.leadsheet || !b) return;
  b.disabled = true;
  try {
    // the highlight, not the whole song: in the one whole-song run (2026-10-10) the model stopped after 178 of 230
    // seconds and its timing drifted by more than a second between sections
    const made = await makeOverlay(song, false, (line) => { if (note) note.textContent = line; });
    if (!made || state.song !== song) return;
    stopPlayback();
    player.src = URL.createObjectURL(made.blob);
    player.classList.remove("hidden");
    if (note) note.textContent = overlayNote(made.info) + " 아래 재생 버튼으로 들어요.";
  } catch (e) {
    if (note) note.textContent = "AI 보컬: " + e.message;
  } finally { b.disabled = false; }
}

function chatOfferAiVocal() {
  const log = $("chat-log");
  const song = state.song;
  if (!log || !song?.leadsheet) return;
  const li = document.createElement("li");
  li.className = "chat-app";
  if (yueAvailable === false) {                              // no button that could only fail: say why instead
    li.textContent = `AI 보컬 얹기는 이 컴퓨터에서 쓸 수 없어요: ${yueReason}`;
    log.append(li);
    return;
  }
  const b = document.createElement("button");
  b.textContent = "AI 보컬 얹기: 하이라이트 듣기 (약 18GB, 2~3분 소요)";
  b.onclick = async () => {
    b.disabled = true;
    try {
      const made = await makeOverlay(song, false, (line) => chatSay("app", line));
      if (!made) { b.disabled = false; return; }
      const out = document.createElement("li");
      out.className = "chat-app";
      const player = document.createElement("audio");
      player.controls = true;
      player.src = URL.createObjectURL(made.blob);
      out.append(overlayNote(made.info) + " ", player);
      log.append(out);
      out.scrollIntoView?.({ block: "nearest" });
    } catch (e) { chatSay("app", "AI 보컬: " + e.message); b.disabled = false; }
  };
  li.append(b);
  log.append(li);
}

let chatFile = null;                          // the file attached to the message being handled

async function chatReadScore(file) {
  chatSay("app", /\.pdf$/i.test(file.name) ? "PDF 악보를 인식하고 있어요. 쪽마다 수십 초가 걸려요..." : "악보를 읽고 있어요...");
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/score/read", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `악보 읽기 실패 (${res.status})`);
  displayImportedScore(data, ++importSeq);
  setStart("score");
  return data.score;
}

async function chatAct(action, text = "") {
  if (!action) return;
  if (window.workspaceController && ['compose', 'score_song', 'regenerate'].includes(action.type)) {
    // an uploaded score always goes in whole and on to YuE2 (user, 2026-10-19: no 8-bar choice to understand)
    if (action.type !== 'regenerate' && importedScore && $("steps").dataset.start === "score") {
      if (action.body?.key && $("import-key")) $("import-key").value = action.body.key;   // the key the feeling recommended
      return importSong();
    }
    chatSay("app", "입력과 설정을 확인했습니다. 다음 만들기 화면에서 직접 시작해 주세요.");
    return;
  }
  const before = state.song;
  if (action.type === "sound_option") {       // a tool call the server checked: the same switch as the option list
    if (action.on) soundOptions.add(action.id); else soundOptions.delete(action.id);
    document.querySelectorAll(`input[data-sound-option="${action.id}"]`).forEach((b) => { b.checked = !!action.on; });
    return;
  }
  if (action.type === "read_score") {          // the attached file is a melody: read it, then go on with the same words
    if (!chatFile) return;
    const score = await chatReadScore(chatFile);
    chatFile = null;
    return chatTurn(text, score, null);
  }
  if (action.type === "part_import") {         // the attached file is a band part of the finished song
    const song = state.song, file = chatFile;
    chatFile = null;
    if (!song || !file) return chatSay("app", "먼저 곡을 만들어 주세요.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    let bin = "";
    for (const x of bytes) bin += String.fromCharCode(x);
    const result = await api("/api/part/import", { result: song, part: action.part, midi_b64: btoa(bin), author: null, trim_tail: false });
    if (state.song !== song) return chatSay("app", "가져오는 동안 다른 곡이 열려서 적용하지 않았어요.");
    stopPlayback();
    state.song = result;
    const got = (result.member_parts || []).find((m) => m.part === action.part);
    chatSay("app", `파트를 넣었어요${got ? ` (${got.notes}음)` : ""}. 아래 재생기로 들어 보세요.`
      + ((result.member_part_warnings || []).length ? " 확인할 점: " + result.member_part_warnings.join(" ") : ""));
    return;
  }
  if (action.type === "compose") {
    const body = requestWithSettings(action.body);
    if ($("vocal-style") && body.fit_voice) $("vocal-style").value = body.fit_voice;
    await runCompose("/api/compose", body, "곡을 만들고 있어요...");
  } else if (action.type === "score_song") {   // the score is the whole song: arranged as written
    const body = requestWithSettings(action.body);
    if ($("vocal-style") && body.fit_voice) $("vocal-style").value = body.fit_voice;
    const data = await api("/api/score/song", body);
    stopPlayback();
    showSong(data);
  } else if (action.type === "regenerate") {
    await runCompose(`/api/compose/${action.song_id}/section`, { section_index: action.section_index }, "그 구간만 다시 만들고 있어요...");
  } else return;
  if (!state.song || state.song === before) {
    return chatSay("app", "곡을 만들지 못했어요. " + ($("error")?.textContent || $("ask")?.textContent || ""));
  }
  $("song")?.scrollIntoView?.({ block: "start", behavior: "instant" });   // the conversation is done: on to the result
  await chatTurn("", null, { type: "composed", song: chatSongSummary(state.song) });
  // with the YuE2 feature on, its first part starts now and plays when it is done; with it off the song is heard once the
  // AI vocal take is ready (user decision 2026-10-11: only the finished version is presented)
  if (!autoYueSong(true)) chatOfferAiVocal();
}

let chatUnderstood = "";
async function chatTurn(text, score, event, attached = false) {
  const r = await api("/api/chat", { session: chatSession, text, score, event, attached,
    find_sections: !!$("chat-find-sections")?.checked });
  chatSession = r.session;
  if (r.llm_note) chatSay("app", r.llm_note);          // Gemini was not used this turn: say what answered instead
  if (score && score === importedScore) chatMainScore = score;
  if (importedScore && chatMainScore === importedScore && chatSession?.consent) $("import-accept").checked = true;
  const said = (r.understood || []).length ? "이해한 것 · " + r.understood.join(" · ") : "";
  if (said && said !== chatUnderstood) chatSay("understood", said);     // what the rules took; once per message, not per request
  if (said) chatUnderstood = said;
  for (const line of r.say || []) chatSay("app", line);
  showChatChoices(r.choices || []);
  await chatAct(r.action, text);
}

// The answers of the open question as buttons (#chat-choices); a press sends that answer as if it were typed.
function showChatChoices(list) {
  const box = $("chat-choices");
  if (!box) return;
  box.innerHTML = "";
  for (const c of list) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = c;
    b.onclick = () => { showChatChoices([]); $("chat-text").value = c; return chatSend(); };
    box.append(b);
  }
  box.classList.toggle("hidden", !list.length);
}

async function chatSend(ev) {
  ev?.preventDefault?.();
  if (chatBusy) return;
  const text = ($("chat-text")?.value || "").trim(), file = $("chat-file")?.files?.[0] || null;
  if (!text && !file) return;
  chatBusy = true;
  chatUnderstood = "";                                       // a new message: its understood line is shown again
  $("chat-send").disabled = true;
  chatSay("me", [text, file ? `📎 ${file.name}` : ""].filter(Boolean).join("\n"));
  $("chat-text").value = "";
  try {
    chatFile = file;
    if (file) { $("chat-file").value = ""; chatMainScore = importedScore; }
    await chatTurn(text, !file && !state.song && importedScore !== chatMainScore ? importedScore : null, null, !!file);
  } catch (e) {
    chatSay("app", "문제가 생겼어요: " + e.message);
  } finally {
    chatBusy = false;
    chatFile = null;
    $("chat-send").disabled = false;
  }
}

// ---------- the background (#bg-notes, user request 2026-10-13): at night white notes and pieces of sheet music
// drift and twinkle in one band across the sky like the milky way; by day black notes wander over the page.  A
// canvas behind everything, never clickable; still with reduced motion, paused while the tab is hidden.
const BG_GLYPHS = ["\u266A", "\u266B", "\u2669", "\u266C"];          // eighth, beamed eighths, quarter, beamed sixteenths
function bgRandom(seed) {                                   // mulberry32: the same seed draws the same sky
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// Particles for a w x h window.  Night: along the line from the lower left to the upper right (x/w + y/h = 1) at a
// normal spread `s` across it, a tenth scattered as lone stars; day: anywhere, each with its own slow heading.
function bgParticles(w, h, mode, seed = 7) {
  const rnd = bgRandom(seed), area = w * h, out = [];
  const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) / 2 * 0.24;   // about a normal spread of 0.12
  const glyph = () => BG_GLYPHS[Math.floor(rnd() * BG_GLYPHS.length)];
  if (mode === "night") {
    const n = Math.max(60, Math.min(220, Math.round(area / 6000)));
    for (let i = 0; i < n; i++) {
      const lone = rnd() < 0.1, t = rnd(), s = lone ? rnd() * 2 - 1 : gauss();
      const staff = !lone && rnd() < 0.14;
      out.push({ kind: staff ? "staff" : "note", t, s, x: (t + s / 2) * w, y: (1 - t + s / 2) * h,
        size: staff ? 40 + rnd() * 80 : 9 + rnd() * 15, glyph: glyph(), heads: 2 + Math.floor(rnd() * 3),
        a: staff ? 0.18 + rnd() * 0.2 : 0.25 + rnd() * 0.55, tw: rnd() * Math.PI * 2, tws: 0.6 + rnd() * 1.6,
        rot: staff ? 0 : (rnd() - 0.5) * 0.6, speed: 0.004 + rnd() * 0.01, ink: "255,255,255" });
    }
  } else {
    const n = Math.max(25, Math.min(120, Math.round(area / 12000)));
    for (let i = 0; i < n; i++) {
      const dir = rnd() * Math.PI * 2, v = 1.5 + rnd() * 3.5;      // a few pixels a second: slow enough to follow with the eye
      out.push({ kind: "note", x: rnd() * w, y: rnd() * h, vx: Math.cos(dir) * v, vy: Math.sin(dir) * v,
        size: 16 + rnd() * 24, glyph: glyph(), a: 0.18 + rnd() * 0.24, rot: (rnd() - 0.5) * 0.8, vrot: (rnd() - 0.5) * 0.1,
        ink: "20,22,28" });
    }
  }
  return out;
}
let bg = null;
function bgMode() { return document.documentElement?.dataset?.theme === "light" ? "day" : "night"; }
function bgStep(dt) {
  const { w, h } = bg;
  for (const p of bg.parts) {
    if (bg.mode === "night") {                               // along the band, back to its start at the far end
      p.t += p.speed * dt;
      if (p.t > 1.1) p.t -= 1.2;
      p.x = (p.t + p.s / 2) * w;
      p.y = (1 - p.t + p.s / 2) * h;
    } else {                                                 // wander, and come back in on the other side
      const wrap = (v, span) => ((v + 40) % span + span) % span - 40;   // -40..span-40: off one edge, in at the other
      p.x = wrap(p.x + p.vx * dt, w + 80);
      p.y = wrap(p.y + p.vy * dt, h + 80);
      p.rot += p.vrot * dt;
    }
  }
}
function bgDraw(time) {
  const { ctx, w, h, mode } = bg;
  ctx.clearRect(0, 0, w, h);
  if (mode === "night") {                                    // the milky way's faint glow under the notes
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.rotate(Math.atan2(-h, w));
    const span = Math.hypot(w, h), g = ctx.createLinearGradient(0, -h * 0.28, 0, h * 0.28);
    g.addColorStop(0, "rgba(160,190,255,0)");
    g.addColorStop(0.5, "rgba(170,200,255,0.07)");
    g.addColorStop(1, "rgba(160,190,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(-span / 2, -h * 0.28, span, h * 0.56);
    ctx.restore();
  }
  for (const p of bg.parts) {
    const glow = mode === "night" ? 0.55 + 0.45 * Math.sin(p.tw + time * p.tws) : 1;
    ctx.save();
    ctx.globalAlpha = p.a * glow;
    ctx.translate(p.x, p.y);
    if (p.kind === "staff") {                                // a piece of sheet music: five lines and a few notes on them
      ctx.rotate(Math.atan2(-h, w));
      ctx.strokeStyle = ctx.fillStyle = `rgb(${p.ink})`;
      ctx.lineWidth = 0.8;
      const gap = p.size / 18;
      for (let k = 0; k < 5; k++) { ctx.beginPath(); ctx.moveTo(0, k * gap); ctx.lineTo(p.size, k * gap); ctx.stroke(); }
      for (let k = 0; k < p.heads; k++) {
        const hx = p.size * (k + 0.6) / (p.heads + 0.4), hy = gap * ((k * 3 + 1) % 5);
        ctx.beginPath(); ctx.ellipse(hx, hy, gap * 0.7, gap * 0.5, -0.4, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.moveTo(hx + gap * 0.6, hy); ctx.lineTo(hx + gap * 0.6, hy - gap * 3.2); ctx.stroke();
      }
    } else {
      ctx.rotate(p.rot);
      ctx.fillStyle = `rgb(${p.ink})`;
      ctx.font = `${p.size}px "Apple Symbols", "Segoe UI Symbol", "Noto Music", "Noto Sans Symbols 2", serif`;
      ctx.fillText(p.glyph, 0, 0);
    }
    ctx.restore();
  }
}
function bgResize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  bg.w = window.innerWidth;
  bg.h = window.innerHeight;
  bg.canvas.width = Math.round(bg.w * dpr);
  bg.canvas.height = Math.round(bg.h * dpr);
  bg.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bg.parts = bgParticles(bg.w, bg.h, bg.mode);
  if (bg.still) bgDraw(0);
}
function bgSetMode(mode) {
  if (!bg || bg.mode === mode) return;
  bg.mode = mode;
  bg.parts = bgParticles(bg.w, bg.h, mode);
  if (bg.still) bgDraw(0);
}
// Returns false when there is no 2D canvas (old browsers, the UI tests' fake page): the page works the same without it.
function startBackground() {
  const canvas = $("bg-notes"), ctx = canvas?.getContext?.("2d");
  if (!ctx || typeof window.innerWidth !== "number") return false;
  const still = !!window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  const mine = bg = { canvas, ctx, mode: bgMode(), parts: [], w: 0, h: 0, still, last: 0 };
  bgResize();
  window.addEventListener?.("resize", () => bg === mine && bgResize());
  if (still || typeof requestAnimationFrame !== "function") { bgDraw(0); return true; }
  const frame = (now) => {
    if (bg !== mine) return;
    const dt = Math.min(0.05, Math.max(0, (now - mine.last) / 1000));
    mine.last = now;
    if (!document.hidden) { bgStep(dt); bgDraw(now / 1000); }   // a hidden tab draws nothing
    requestAnimationFrame(frame);
  };
  requestAnimationFrame((now) => { mine.last = now; frame(now); });
  return true;
}

// Day / night (#btn-theme): the dark page is the default; "light" switches to the light palette (style.css
// :root[data-theme="light"]).  The choice is kept in this browser when storage is allowed (it may be refused).
const THEME_KEY = "instmate-theme";
function applyTheme(theme) {
  const root = document.documentElement;
  if (!root?.dataset) return;
  if (theme === "light") root.dataset.theme = "light"; else delete root.dataset.theme;
  $("btn-theme")?.setAttribute("aria-pressed", String(theme === "light"));
  bgSetMode(bgMode());                                       // the background follows: notes by day, the milky way at night
}
function initTheme() {
  let kept = null;
  try { kept = localStorage.getItem(THEME_KEY); } catch (e) { kept = null; }
  applyTheme(kept);
  if ($("btn-theme")) $("btn-theme").onclick = toggleTheme;
}
function toggleTheme() {
  const next = document.documentElement?.dataset?.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* not kept: the switch still works on this page */ }
}

// The page scrolls itself to the current step on load; the browser would then start Tab from there. The focus is put
// on the top of the page instead (without scrolling), so the first Tab reaches the skip link.
function focusPageTop() { $("page-top")?.focus?.({ preventScroll: true }); }

// Big text (#btn-text-size, after the kiosk chatbot's 큰 글씨): the type size variables grow (style.css
// :root[data-size="big"]); the choice is kept in this browser when storage allows.
const SIZE_KEY = "instmate-size";
function applySize(size) {
  const root = document.documentElement;
  if (!root?.dataset) return;
  if (size === "big") root.dataset.size = "big"; else delete root.dataset.size;
  $("btn-text-size")?.setAttribute("aria-pressed", String(size === "big"));
}
function initTextSize() {
  let kept = null;
  try { kept = localStorage.getItem(SIZE_KEY); } catch (e) { kept = null; }
  applySize(kept);
  if ($("btn-text-size")) $("btn-text-size").onclick = toggleTextSize;
}
function toggleTextSize() {
  const next = document.documentElement?.dataset?.size === "big" ? "normal" : "big";
  applySize(next);
  try { localStorage.setItem(SIZE_KEY, next); } catch (e) { /* not kept: it still works on this page */ }
}

// Studio look (#btn-look, user request 2026-10-18): flat near-black panels and a teal accent (workspace.css
// :root[data-style="studio"]); on by default, the old look kept in this browser when chosen and storage allows.
const STYLE_KEY = "instmate-style";
function applyStyle(style) {
  const root = document.documentElement;
  if (!root?.dataset) return;
  if (style === "studio") root.dataset.style = "studio"; else delete root.dataset.style;
  $("btn-look")?.setAttribute("aria-pressed", String(style === "studio"));
}
function initStyle() {
  let kept = null;
  try { kept = localStorage.getItem(STYLE_KEY); } catch (e) { kept = null; }
  applyStyle(kept === "plain" ? "plain" : "studio");     // studio is the default since 2026-10-19 (user: "스튜디오 기본으로")
  if ($("btn-look")) $("btn-look").onclick = () => {
    const next = document.documentElement?.dataset?.style === "studio" ? "plain" : "studio";
    applyStyle(next);
    try { localStorage.setItem(STYLE_KEY, next); } catch (e) { /* not kept: it still works on this page */ }
  };
}

// "← 대화로 돌아가 정정하기" on the song card: the conversation opens, scrolls into view and takes the cursor, so a
// correction ("도입부로 정정해줘") is one sentence away (user, 2026-10-13).
function backToChat() {
  setMode("chat");
  $("chat-card")?.scrollIntoView?.({ block: "start", behavior: "instant" });
  $("chat-text")?.focus?.();
}

function init() {
  initTheme();
  initStyle();
  initTextSize();
  for (const id of ['form-choice', 'seed-role', 'intro-player', 'melody-source', 'key-change', 'melody-shape',
    'melody-role', 'vocal-style', 'fit-voice', 'import-key']) {
    $(id)?.addEventListener?.('change', () => { $(id).dataset.userSet = '1'; });
  }
  startBackground();
  if ($("chat-form")) {
    $("chat-form").onsubmit = chatSend;
    $("mode-chat").onclick = () => setMode("chat");
    $("mode-form").onclick = () => setMode("form");
    setMode("chat");                          // the conversation is the first screen; the form is one tab away
    // with 씨앗이 on the page the greeting is its speech bubble (workspace.js, user 2026-10-19), not a chat line
    if (!$("mascot-btn")) chatSay("app", "안녕하세요. 멜로디가 담긴 악보 파일을 붙이고, 장르와 보컬(여성·남성)을 알려 주세요.");
  }
  $("pb-stop").onclick = stopPlayback;
  if ($("btn-back-chat")) $("btn-back-chat").onclick = backToChat;
  // (the "my own voice" recording was taken off the page, user decision 2026-10-07: the melody is sung by the
  // built-in guide vocal only; the page never sends vocal: true)
  const sb = $("sounds-box");             // the sound list loads the first time the panel is opened
  if (sb) sb.ontoggle = () => { if (sb.open && !$("sounds").childElementCount) loadSounds(); };
  const cb = $("credits-box");
  if (cb) cb.ontoggle = () => { if (cb.open) loadCredits(); };
  loadSoundOptions();
  const rm = $("release-mode");             // the switch re-reads everything that depends on the licence mode
  if (rm) rm.onchange = () => { loadSounds(); if ($("credits-box")?.open) loadCredits(); if ($("genre-box")?.open) loadGenreNotes(); };
  const gb = $("genre-box");
  if (gb) gb.ontoggle = () => { if (gb.open && !$("genre-notes").childElementCount) loadGenreNotes(); };
  const sel = $("input-tonic");
  NOTE_NAMES.forEach((n, i) => sel.append(new Option(`${n} (1 = ${n})`, i)));
  sel.onchange = () => { state.inputTonic = +sel.value; melodyChanged(); };

  document.querySelectorAll("[data-deg]").forEach((b) => (b.onclick = () => addNote(+b.dataset.deg)));
  $("btn-sharp").onclick = toggleSharp;
  $("btn-oct-up").onclick = octUp; $("btn-oct-down").onclick = octDown;
  if ($("btn-pick-all")) $("btn-pick-all").onclick = pickAllNotes;
  $("btn-del").onclick = delLast;
  const roll = $("piano-roll");
  roll.onpointerdown = prDown; roll.onpointermove = prMove; roll.onpointerup = prUp; roll.onpointercancel = () => { prDrag = null; renderPianoRoll(); };
  $("btn-play-mel").onclick = playMelodyOnly;
  $("btn-sample").onclick = loadSample;
  $("btn-clear").onclick = () => { state.tokens = []; melodyChanged(); };
  $("btn-generate").onclick = generate;
  $("btn-compose").onclick = startCompose;
  $("btn-rewrite").onclick = rewriteSection;
  $("style-tempo").onchange = setTempoFromInput;
  // While typing, only the source tag follows (the value is validated and applied on change).
  $("style-tempo").oninput = () => {
    const typed = $("style-tempo").value.trim() !== "";
    $("tempo-tag").textContent = typed ? SRC_TEXT.tempo.manual : "";
    $("tempo-tag").classList.remove("rec");
  };
  $("btn-play").onclick = () => state.result && playEvents(state.result.playback.notes, "반주 미리듣기 (간단한 소리)");
  $("btn-stop").onclick = stopPlayback;
  $("btn-undo").onclick = undoVersion;
  $("btn-mix").onclick = applyMix;
  $("btn-agent").onclick = agentRequest;
  $("btn-song").onclick = makeSong;
  $("btn-leadsheet").onclick = renderLeadsheet;
  $("btn-song-play").onclick = () => state.song && (routeNext = 0, playEvents(state.song.playback.notes, "곡 미리듣기 (간단한 소리)"));
  $("btn-song-stop").onclick = stopPlayback;
  if ($("import-file")) {
    $("import-file").onchange = readScoreFile;
    $("btn-import-seed").onclick = importSeed;
    $("btn-import-song").onclick = importSong;
    if ($("btn-make-score-song")) $("btn-make-score-song").onclick = importSong;
    if ($("btn-make-score-seed")) $("btn-make-score-seed").onclick = makeScoreSeedSong;
  }
  if ($("btn-step-next")) {
    for (const b of document.querySelectorAll("[data-go]")) b.onclick = (e) => {
      e.preventDefault?.();
      goStep(Number(b.dataset.go));
      if (b.dataset.jump) $(b.dataset.jump)?.scrollIntoView?.({ block: "start", behavior: "instant" });   // 4 YuE2 audio, 5 practice
    };
    for (const b of document.querySelectorAll("[data-start-go]")) b.onclick = () => setStart(b.dataset.startGo);
    if ($("btn-hum")) {
      $("btn-hum").onclick = humRecord;
      $("hum-file").onchange = humFromFile;
      $("btn-hum-use").onclick = humUse;
      $("btn-hum-drop").onclick = humDrop;
    }
    $("btn-step-prev").onclick = () => goStep(currentStep() - 1);
    $("btn-step-next").onclick = nextStep;
    setStart("score");
    goStep(1);
  }
  const eb = $("edit-box");
  if (eb) {
    eb.ontoggle = () => { if (eb.open) editSections(); };
    $("btn-edit-open").onclick = openPartEditor;
    $("edit-grid").onclick = (e) => {
      const r = $("edit-grid").querySelector?.("svg")?.getBoundingClientRect?.();
      if (!r) return;
      editCell(Math.floor((e.clientX - r.left - EG.label) / EG.cell), Math.floor((e.clientY - r.top) / EG.row));
    };
    $("edit-grid").onkeydown = (e) => {
      const map = { Delete: editDelete, Backspace: editDelete, ArrowLeft: () => editMove(-EDIT_STEP), ArrowRight: () => editMove(EDIT_STEP),
        ArrowUp: () => editPitch(1), ArrowDown: () => editPitch(-1) };
      if (map[e.key]) { e.preventDefault(); map[e.key](); }
    };
    $("btn-edit-del").onclick = editDelete;
    $("btn-edit-left").onclick = () => editMove(-EDIT_STEP);
    $("btn-edit-right").onclick = () => editMove(EDIT_STEP);
    $("btn-edit-up").onclick = () => editPitch(1);
    $("btn-edit-down").onclick = () => editPitch(-1);
    $("btn-edit-shorter").onclick = () => editLength(-EDIT_STEP);
    $("btn-edit-longer").onclick = () => editLength(EDIT_STEP);
    $("btn-edit-setlevel").onclick = editLevel;
    $("btn-edit-undo").onclick = () => editUndo(false);
    $("btn-edit-redo").onclick = () => editUndo(true);
    $("btn-edit-play").onclick = playEditPreview;
    $("btn-edit-stop").onclick = stopEditPreview;
    $("edit-with-band").onchange = () => { if (editDraft?.playing) scheduleEditPreview(); };
    $("btn-edit-apply").onclick = applyPartEdit;
    $("btn-edit-cancel").onclick = cancelPartEdit;
  }
  const bb = $("band-box");                                  // the band list is asked once, when the panel is first opened
  if (bb) {
    bb.ontoggle = () => { if (bb.open && !bandsLoaded) { bandsLoaded = true; loadBands().catch(() => {}); } };
    $("band-select").onchange = () => loadBandVersions().catch(() => {});
    $("btn-band-create").onclick = createBand;
    $("btn-band-save").onclick = saveBandVersion;
    if ($("btn-band-profile-save")) $("btn-band-profile-save").onclick = saveBandProfile;
    if ($("btn-band-profile-apply")) $("btn-band-profile-apply").onclick = applyBandProfile;
    $("member-file").onchange = (e) => importMemberPart(e.target.files[0]);
  }
  const yb = $("yue-box");                                   // status is asked once, when the panel is first opened
  if (yb) {
    $("btn-yue").onclick = () => makeYueDemo();
    if ($("yuesong-on")) {
      $("yuesong-on").onchange = () => $("yuesong-form").classList.toggle("hidden", !$("yuesong-on").checked);
      $("btn-yuesong").onclick = () => window.workspaceController?.openYue && $("yuesong-scope")?.value !== "asis" ? completeYueSong(state.song) : makeYueSong();
      if ($("yuesong-scope")) $("yuesong-scope").onchange = syncYueSections;
      if ($("yuesong-ins")) $("yuesong-ins").onchange = syncYueSections;
      if ($("btn-yuesong-sections-reset")) $("btn-yuesong-sections-reset").onclick = () => { yueSecs.touched = false; syncYueSections(); };
      $("btn-yuesong-adopt").onclick = adoptYueSong;
      $("btn-yuesong-yes").onclick = () => {
        if (!yueSongFirst?.abc || !yueSongFirst?.library_id) return showError("완성된 1절 정보를 찾을 수 없어요. '내 곡'에서 저장된 1절을 열고 '이 1절로 전곡 만들기'를 눌러 주세요.");
        return makeYueSong(yueSongFirst);
      };
      $("btn-yuesong-no").onclick = () => $("yuesong-ask").classList.add("hidden");
    }
    for (const a of $("views")?.querySelectorAll?.("a") || []) a.onclick = (e) => { e.preventDefault?.(); setView(a.dataset.view); };
    if ($("lib-list")) {
      $("lib-volume").oninput = () => { $("lib-audio").volume = Number($("lib-volume").value) / 100; };
      $("lib-audio").ontimeupdate = () => { if (routeLib && routeLib === libOpen) moveRouteDot($("lib-audio").currentTime || 0); };
      $("btn-lib-lyrics").onclick = () => $("lib-lyrics").classList.toggle("hidden");
      $("btn-lib-score").onclick = loadLibraryScore;
      if ($("btn-lib-full")) $("btn-lib-full").onclick = libraryFull;
      if ($("btn-lib-full-yes")) $("btn-lib-full-yes").onclick = libraryFullYes;
      if ($("btn-lib-full-no")) $("btn-lib-full-no").onclick = () => { libFullFirst = null; $("lib-full-ask").classList.add("hidden"); };
      $("lib-part").onchange = markMyPart;
      $("btn-lib-minus").onclick = async () => {        // practice track: the app's band without my part, YuE2's voice on top
        if (!libOpen) return showError("라이브러리에서 YuE2 곡을 먼저 열어 주세요. 왼쪽 메뉴의 라이브러리 목록에서 곡을 누르면 열려요.");
        const b = $("btn-lib-minus"), part = $("lib-part").value;
        b.disabled = true; b.textContent = "만드는 중…";
        try {
          const res = await fetch(`/api/yue/library/${libOpen}/minus?part=${part}`);
          if (!res.ok) throw new Error((await res.json()).detail || `실패 (${res.status})`);
          let timing = null;
          try { timing = JSON.parse(decodeURIComponent(res.headers.get("X-Instmate-Practice") || "{}"))?.beat_timing; } catch (_) { /* older servers have no measurements */ }
          showPracticeTiming(timing);
          $("lib-audio").src = URL.createObjectURL(await res.blob());
          $("lib-audio").play?.();
        } catch (e) { showError("내 파트 빼고 연습: " + e.message); }
        finally { b.disabled = false; b.textContent = "🎧 내 파트 빼고 연습"; }
      };
      $("btn-lib-close").onclick = () => { $("lib-audio").pause(); $("libbar").classList.add("hidden"); };
      // asked once when the page is served by the app (the UI tests count the page's first requests; they have no location)
      if (typeof location !== "undefined" && String(location.protocol).startsWith("http")) loadLibrary();
    }
    if ($("btn-go-practice")) $("btn-go-practice").onclick = () => { setView("practice"); focusPageTop?.(); };
    if ($("yuesong-random")) $("yuesong-random").onchange = syncRandomLyrics;
    syncRandomLyrics();
    if ($("btn-yuesong-cancel")) $("btn-yuesong-cancel").onclick = cancelYue;
    if ($("btn-yuesong-redo")) $("btn-yuesong-redo").onclick = redoYue;
    if ($("btn-song-ai")) $("btn-song-ai").onclick = playAiVocal;
    if ($("btn-song-overlay")) $("btn-song-overlay").onclick = playSongOverlay;
    if ($("btn-practice-score")) {
      $("btn-practice-score").onclick = showPracticeScore;
      $("btn-practice-xml").onclick = downloadPracticeScore;
      $("btn-practice-solo").onclick = () => playPractice("solo");
      $("btn-practice-minus").onclick = () => playPractice("minus");
      $("btn-practice-all").onclick = () => playPractice("all");
      $("btn-practice-stop").onclick = stopPlayback;
      $("btn-practice-print").onclick = () => printEngraved(false);
      $("btn-practice-print-all").onclick = () => printEngraved(true);
      $("btn-practice-edit").onclick = editPracticeScore;
      $("btn-practice-hq-solo").onclick = () => playPracticeHq("solo", $("btn-practice-hq-solo"));
      $("btn-practice-hq-minus").onclick = () => playPracticeHq("minus", $("btn-practice-hq-minus"));
    }
    yb.ontoggle = () => { if (yb.open && !yueLoaded) { yueLoaded = true; loadYueStatus(); } };
    $("vocal-style")?.addEventListener?.("change", syncYueVoice);
    for (const id of ["vocal-style", "seed-role", "melody-role", "fit-voice", "intro-player"]) {
      $(id)?.addEventListener?.("change", () => { if ($("make-style")) $("make-style").textContent = makeStyleText(); syncIntroPlayer(); });
      if (id === "vocal-style" || id === "melody-role") $(id)?.addEventListener?.("change", hideError);   // "pick a singer" is answered
    }
    syncIntroPlayer();
  }
  if ($("btn-bandpack")) $("btn-bandpack").onclick = downloadBandPack;
  $("btn-song-midi").onclick = () => state.song && downloadMidi(state.song, `jakgokmate_${state.song.genre}.mid`);
  $("btn-song-cover-png").onclick = saveCoverPng;
  $("btn-song-audio").onclick = () => state.song && downloadAudio(state.song, `jakgokmate_${state.song.genre}`, $("btn-song-audio"));
  $("song-sheet-box").ontoggle = renderSongSheet;
  $("btn-range-dark").onclick = () => rangeEdit("darker");
  $("btn-range-bright").onclick = () => rangeEdit("brighter");
  state.levelSelects = {};
  for (const [k, ko] of Object.entries(MIX_TRACKS)) {
    const label = document.createElement("label");
    label.textContent = `${ko} `;
    const sel = document.createElement("select");
    for (const lv of LEVELS) sel.append(new Option(`${lv}x`, String(lv)));
    sel.value = "1";
    label.append(sel);
    $("mix-levels").append(label);
    state.levelSelects[k] = sel;
  }
  $("btn-midi").onclick = () => downloadMidi();   // not the click event as `data`
  $("btn-audio").onclick = () => state.result && downloadAudio(state.result, `jakgokmate_accompaniment_${state.result.genre}`, $("btn-audio"));
  $("btn-hq").onclick = () => state.result && playHq(state.result, $("btn-hq"), "실제 악기 소리 · 반주");
  $("btn-song-hq").onclick = () => state.song && (routeNext = 0, playHq(state.song, $("btn-song-hq"), "실제 악기 소리 · 완성곡"));
  $("energy").onchange = invalidateResult;

  for (const [g, ko] of Object.entries(GENRES).filter(([g]) => BAND_GENRES.includes(g))) {   // band scope
    const b = document.createElement("button");
    b.textContent = ko; b.dataset.genre = g;
    b.onclick = () => {
      state.genre = g; state.src.genre = "manual"; state.style = null; invalidateResult(); syncStyleButtons();
      if (($("error")?.textContent || "").startsWith("장르를 아직")) hideError();     // the thing it asked for was just done
    };
    b.classList.toggle("on", g === state.genre);
    genreButtons[g] = b;
    $("genres").append(b);
  }
  for (const [k, ko] of Object.entries(INSTR)) {
    const b = document.createElement("button");
    b.textContent = ko; b.classList.toggle("on", state.instruments.has(k));   // default: band (strings off)
    b.onclick = () => { state.instruments.has(k) ? state.instruments.delete(k) : state.instruments.add(k); state.src.instr = "manual"; state.style = null; invalidateResult(); syncStyleButtons(); };
    instrButtons[k] = b;
    ((EXTRA_INSTR.includes(k) && $("instruments-extra")) || $("instruments")).append(b);
  }

  // The melody keys (digits, Backspace / Delete, arrows, [ ], Space) act on the page itself, never inside a text box,
  // a list, an editable area or a player (typing lyrics used to delete melody notes), nor on a key another control took.
  document.addEventListener("keydown", (e) => {
    const t = e.target || {};
    if (e.defaultPrevented || ["SELECT", "INPUT", "TEXTAREA", "AUDIO", "VIDEO"].includes(t.tagName) || t.isContentEditable
      || e.metaKey || e.ctrlKey || e.altKey) return;
    const m = /^(?:Digit|Numpad)([0-7])$/.exec(e.code);
    if (m) { e.preventDefault(); if (e.shiftKey && m[1] !== "0") state.sharpNext = true; addNote(+m[1]); return; }
    const actions = {
      ArrowUp: octUp, PageUp: octUp, ArrowDown: octDown, PageDown: octDown,
      BracketLeft: durHalf, BracketRight: durDouble, Backspace: delLast, Delete: delLast, Space: playMelodyOnly,
    };
    if (e.code === "Space" && (["BUTTON", "SUMMARY"].includes(t.tagName) || t.getAttribute?.("role") === "button")) return;   // Space presses the focused button or opens the fold-out
    const act = actions[e.code];
    if (act) { e.preventDefault(); act(); }
  });
  renderMelody();
  renderSourceTags();
  focusPageTop();
}
init();
if (typeof window !== "undefined") applyYueAvailability(window.INSTMATE_YUE_STATUS);
