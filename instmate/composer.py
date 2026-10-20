"""M4: seed melody -> 2+ minute song with the fine-tuned lead sheet model, code scoring and regeneration.

The agent is a small explicit state machine (no framework); every decision is a plain function call recorded in a
visible trace ({tool, input, output, ms}) and every rule is code, so the whole loop is testable with a fake
generator and still works when no model is available (the rule-based draft is used for every section).

Flow (`compose`)
  1. analyze_melody       key of the seed (ambiguous -> ask; a confirmed key is used as given)
  2. style_from_request   an already confirmed StyleAttrs dict fills genre / tempo / instruments (UI asks the user)
  3. plan_structure       song.plan_structure, Korean section names as in the lead sheet format
  4. rule draft           agent.run + song.build_song (what /api/song makes) -> lead sheet, moved to C / Am
  5. per unique section   generate_section (n candidates from the model, training "infill" prompt)
                          -> score_section (hard checks + penalties) -> best passing candidate is adopted;
                          failing sections go into the next round (max_rounds); still failing -> rule draft stays
  6. render               back to the seed key, render.leadsheet_to_song(key_change_last_chorus=True), song checks
  7. explain              short Korean summary built from the trace

The chorus is the user's seed melody and is never regenerated. Repeated sections (verse 2, pre-chorus 2, ...) reuse the
content of their first occurrence (pop repetition); the last chorus goes up a semitone in `render`.

Score of a candidate (0..1), see `score_candidate`
  hard fail (score 0): lead sheet validation fails / not 4/4 / not exactly one section named as requested with the
  requested bar count / no chords / fewer pitched notes than bars / the similarity policy says "copy"
  Instrumental sections (INSTRUMENTAL_SECTIONS = 인트로, 간주, 아웃트로) use a different rule: no note-count check;
  chords must be written in at least half of the bars; clash is averaged over bars that have notes only (none -> 0);
  range penalty only over pitched notes (none -> 0). Why: POP909's melody track is the vocal line, so these sections
  are mostly empty in the training targets. Measured on datasets/leadsheet/train.jsonl ('full' targets), melody notes
  per bar have median 0.00 for 인트로 (70% of intros have zero melody notes), 간주 (59% zero) and 아웃트로 (98% zero).
  With the real adapter the old rule rejected 45/45 intro and 42/45 outro candidates on 3 valid songs, so those
  sections always fell back to the rule draft. All other sections keep the old rule.
  score = 1 - W_CLASH_BARS * (bars over CLASH_MAX / bars) - W_CLASH_MEAN * mean clash
            - min(W_LEAP_CAP, W_LEAP * number of junction leaps > MAX_LEAP)
            - W_REPEAT * [same bar more than REPEAT_MAX times in a row]
            - W_RANGE * min(1, range violations / pitched notes)
  passing: not hard-failed and score >= PASS_SCORE.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import agent, check as chk, plagcheck, song as song_mod, style as style_mod
from .arrange import BAND, INSTRUMENTS, arrange_segments
from .genres import PROFILES
from .leadsheet import (KEYS, PITCH_MAX, PITCH_MIN, Bar, ChordMark, LeadNote, LeadSheet, LeadsheetError, Section,
                        parse_leadsheet, serialize_leadsheet, to_notes, validate_leadsheet)
from .lm_format import build_prompt, clean_completion
from .mix import resolve as resolve_mix
from .render import (KIND, NAME, _bar_marks, _shift_to, key_of, leadsheet_to_song, melody_notes, parse_chord,
                     section_style, song_to_leadsheet, transpose_symbol)
from .similarity import SimilarityIndex, notes_from_leadsheet, should_regenerate
from .theory import BAR, Key, Note, analyze_key, parse_number_input, split_by_bar

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- constants (score formula, generation)
CANDIDATES_DEFAULT = 5
MAX_ROUNDS_DEFAULT = 3
TEMPERATURE, TOP_P = 0.9, 0.95
MAX_TOKENS_CAP = 1500                 # hard cap per generated candidate (memory / time)
TOKENS_PER_BAR, TOKENS_OVERHEAD = 50, 60      # valid-set infill targets: p99 48 tokens/bar, header+tag ~30
MAX_PROMPT_TOKENS = 3400              # training used at most 4096 tokens of prompt + target
MODEL_GENRE = "발라드"                # every training sample carries this header genre (POP909 has no genre labels)
# Country-tagged models (v4, docs/training_log.md) were trained with [국가:...] headers: POP909 as 발라드 + 중국 and
# Hooktheory style groups as 케이팝/제이팝/뉴에이지.  An adapter declares this with instmate_tags.json {"country": true}.
COUNTRY_OF = {"kpop": "한국", "jpop": "일본", "newage": "기타"}      # every other arrangement genre: 중국 (POP909)
DEFAULT_COUNTRY = "중국"
MODEL_STYLE_GENRES = ("케이팝", "제이팝", "뉴에이지")
MIN_SONG_SECONDS, MAX_SONG_SECONDS = song_mod.TARGET_MIN_S, song_mod.TARGET_MAX_S      # 2-4 minutes, one definition
# Prefix forcing: the generation prompt ends with the exact start of an infill training target
# (header line + "\n[구조:NAME]\n"); candidate text = prefix + continuation. False = the old free-start path.
FORCE_SECTION_PREFIX = True
HF_PATTERNS = ["*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt",
               "*.jsonl", "*.jinja"]          # = mlx_lm.utils._download, so a snapshot fetched by mlx_lm is complete

PASS_SCORE = 0.75
W_CLASH_BARS = 0.5
W_CLASH_MEAN = 0.2
INSTRUMENTAL_SECTIONS = ("인트로", "간주", "아웃트로")      # melody may be sparse or empty (see module docstring)
W_LEAP = 0.1                          # per junction leap, at most W_LEAP_CAP in total
W_LEAP_CAP = 0.3
W_REPEAT = 0.3
W_RANGE = 0.4
REPEAT_MAX = 4                        # the same bar more than this many times in a row is penalized
# copying other sections: share of the candidate's melodic 4-grams (interval + duration) also found in the song's other
# sections; above COPY_FREE it costs up to W_COPY (a section that mostly re-sings another part sounds monotonous)
W_COPY, COPY_FREE, COPY_N = 0.25, 0.5, 4
# a near-total duplicate of another section (not just a shared motif/quotation) is rejected outright instead
# of only taking the W_COPY soft penalty below. LUNA repro 20261008 (song24): an exact 8-bar bridge copy of
# the verse (copy_share == 1.0) scored exactly 1 - W_COPY == 0.75 == PASS_SCORE and passed.
COPY_HARD_FAIL = 0.98
MELODY_SOFT_RANGE = (40, 96)          # melody notes outside count as range violations (besides instrument ranges)

# Training "infill" prompt (scripts/build_dataset.py INSTR_INFILL; tests/test_composer.py checks they are equal)
INSTR_INFILL = "다음 리드시트에서 채우기 표시가 있는 구간을 채워 그 구간의 리드시트를 작성하세요."

SPEC_TOOLS = ("analyze_melody", "style_from_request", "analyze_reference_audio", "plan_structure",
              "generate_section", "arrange_section", "score_section", "render", "explain")


# ---------------------------------------------------------------- trace
class Trace(list):
    """List of {tool, input, output, ms}; `progress(entry)` is called for every finished entry."""

    def __init__(self, progress=None):
        super().__init__()
        self.progress = progress
        self._lock = threading.Lock()

    @contextmanager
    def step(self, tool: str, inp: str):
        rec = {"tool": tool, "input": inp, "output": "", "ms": 0}
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec["output"] = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            rec["ms"] = round((time.perf_counter() - t0) * 1000)
            with self._lock:
                self.append(rec)
            if self.progress:
                self.progress(rec)


def _tr(trace):
    return trace if trace is not None else Trace()


# ---------------------------------------------------------------- generator (MLX, lazy)
class LeadsheetGenerator:
    """The fine-tuned lead sheet model (Qwen3 + LoRA adapter) through mlx-lm, loaded lazily and only once.

    Base model: env INSTMATE_LEADSHEET_BASE (default Qwen/Qwen3-1.7B), resolved in the LOCAL Hugging Face cache only.
    Adapter: env INSTMATE_LEADSHEET_ADAPTER (a directory with adapters.safetensors + adapter_config.json), default
    models/leadsheet_lora_1p7b when its adapters.safetensors exists, else none (-> `available()` is False).
    """
    BATCH = 5                      # candidates per mlx batch_generate call

    def __init__(self, base: str | None = None, adapter: str | os.PathLike | None = None):
        self.base = base or os.environ.get("INSTMATE_LEADSHEET_BASE") or "Qwen/Qwen3-1.7B"
        env = os.environ.get("INSTMATE_LEADSHEET_ADAPTER")
        if adapter is None and env:
            adapter = env
        if adapter is None:
            default = ROOT / "models" / "leadsheet_lora_1p7b"
            adapter = default if (default / "adapters.safetensors").exists() else None
        self.adapter = None if adapter is None else Path(adapter) if Path(adapter).is_absolute() else ROOT / adapter
        self.reason = ""
        self._lock = threading.Lock()
        self._model = self._tok = None
        self.last_call: dict | None = None
        self.calls: list[dict] = []

    # -- availability (cheap: no model load)
    def base_path(self) -> Path | None:
        p = Path(self.base)
        if p.exists():
            return p
        try:                    # same file patterns as mlx_lm's own download, so a snapshot made by it counts as complete
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(self.base, local_files_only=True, allow_patterns=HF_PATTERNS))
        except Exception:
            return None
        return path if (path / "config.json").exists() and list(path.glob("*.safetensors")) else None

    def available(self) -> bool:
        if importlib.util.find_spec("mlx_lm") is None:
            self.reason = "mlx_lm을 불러올 수 없어요"
        elif self.adapter is None:
            self.reason = "어댑터가 없어요 (INSTMATE_LEADSHEET_ADAPTER 또는 models/leadsheet_lora_1p7b/adapters.safetensors)"
        elif not (self.adapter / "adapters.safetensors").exists() or not (self.adapter / "adapter_config.json").exists():
            self.reason = f"어댑터 파일이 없어요: {self.adapter}"
        elif self.base_path() is None:
            self.reason = f"기반 모델 {self.base}이 로컬 캐시에 없어요 (내려받지 않아요)"
        else:
            self.reason = ""
        return not self.reason

    @property
    def country_tags(self) -> bool:
        """True when the adapter was trained with [국가:...] headers (its instmate_tags.json says {"country": true})."""
        if self.adapter is None:
            return False
        f = self.adapter / "instmate_tags.json"
        try:
            return f.is_file() and json.loads(f.read_text()).get("country") is True
        except (OSError, ValueError):
            return False

    def info(self) -> dict:
        return {"base": self.base, "adapter": None if self.adapter is None else str(self.adapter),
                "available": self.available(), "reason": self.reason, "country_tags": self.country_tags}

    # -- model
    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(self.reason)
        from mlx_lm import load
        self._model, self._tok = load(str(self.base_path()), adapter_path=str(self.adapter))

    def generate(self, input_text: str, n: int, temperature: float = TEMPERATURE, top_p: float = TOP_P,
                 max_tokens: int = MAX_TOKENS_CAP, prefix: str = "") -> list[str]:
        """`n` completions of the training prompt for `input_text` (cleaned at the first end token). A non-empty
        `prefix` is appended to the prompt (forced start of the answer); the returned texts are continuations only."""
        max_tokens = max(1, min(int(max_tokens), MAX_TOKENS_CAP))
        t0 = time.perf_counter()
        with self._lock:
            self._ensure_loaded()
            import mlx.core as mx
            from mlx_lm.sample_utils import make_sampler
            prompt = build_prompt(input_text, self._tok) + prefix    # the same builder as training
            ids = self._tok.encode(prompt, add_special_tokens=False)
            if len(ids) > MAX_PROMPT_TOKENS:
                raise RuntimeError(f"prompt too long: {len(ids)} tokens > {MAX_PROMPT_TOKENS}")
            sampler = make_sampler(temp=temperature, top_p=top_p)
            texts, mode, gen_tokens = [], "batch", 0
            try:
                from mlx_lm.generate import batch_generate
            except ImportError:
                batch_generate = None
            if batch_generate is None:
                mode = "sequential"
            remaining = n
            while remaining > 0:
                k = min(self.BATCH, remaining)
                if mode == "batch":
                    resp = batch_generate(self._model, self._tok, [ids] * k, max_tokens=max_tokens, sampler=sampler)
                    texts += list(resp.texts)
                    gen_tokens += int(resp.stats.generation_tokens)
                else:
                    from mlx_lm.generate import generate as one
                    for _ in range(k):
                        out = one(self._model, self._tok, prompt=ids, max_tokens=max_tokens, sampler=sampler)
                        texts.append(out)
                        gen_tokens += len(self._tok.encode(out, add_special_tokens=False))
                remaining -= k
                mx.clear_cache()
            peak = None
            try:
                peak = round(mx.get_peak_memory() / 1e9, 2)
            except Exception:
                pass
        self.last_call = {"ms": round((time.perf_counter() - t0) * 1000), "n": n, "prompt_tokens": len(ids),
                          "gen_tokens": gen_tokens, "mode": mode, "peak_gb": peak}
        self.calls.append(self.last_call)
        return [clean_completion(t) for t in texts]


_GENERATORS: dict[tuple, LeadsheetGenerator] = {}
_GEN_LOCK = threading.Lock()


def default_generator() -> LeadsheetGenerator:
    """One shared generator per (base, adapter) environment setting (the model is loaded once per process)."""
    key = (os.environ.get("INSTMATE_LEADSHEET_BASE"), os.environ.get("INSTMATE_LEADSHEET_ADAPTER"))
    with _GEN_LOCK:
        if key not in _GENERATORS:
            _GENERATORS[key] = LeadsheetGenerator()
        return _GENERATORS[key]


_INDEX_CACHE: dict = {}


def default_similarity_index() -> SimilarityIndex | None:
    path = ROOT / "models" / "similarity_index.npz"
    if "idx" not in _INDEX_CACHE:
        try:
            _INDEX_CACHE["idx"] = SimilarityIndex.load(path)
        except Exception:
            _INDEX_CACHE["idx"] = None
    return _INDEX_CACHE["idx"]


# ---------------------------------------------------------------- lead sheet helpers
def model_key_name(key: Key) -> str:
    """The training key of a mode: C major or A minor."""
    return "C" if key.mode == "major" else "Am"


def transpose_sheet(sheet: LeadSheet, shifts: list[int], key: str) -> LeadSheet:
    """Copy with section i moved by shifts[i] semitones (melody and chords); header key = `key`.

    Black-key chord roots use the app's mixed spelling for C / Am, flats or sharps by signature otherwise.
    """
    if len(shifts) != len(sheet.sections):
        raise ValueError("one shift per section is needed")
    signature = KEYS[key]
    flats = None if signature == 0 else signature < 0
    out = []
    for sec, sh in zip(sheet.sections, shifts):
        bars = []
        for bar in sec.bars:
            notes = []
            for n in bar.notes:
                p = None if n.pitch is None else n.pitch + sh
                if p is not None and not PITCH_MIN <= p <= PITCH_MAX:
                    raise ValueError(f"조옮김 후 음역을 벗어나요: {sec.name} MIDI {p}")
                notes.append(LeadNote(p, n.start, n.dur, n.tie))
            bars.append(Bar(notes, [ChordMark(transpose_symbol(c.symbol, sh, flats), c.start) for c in bar.chords]))
        out.append(Section(sec.name, bars))
    return LeadSheet(sheet.genre, key, sheet.meter, sheet.tempo, out, sheet.country)


def groups_of(sheet: LeadSheet) -> list[int]:
    """Index of the section whose content each section reuses: a 후렴 stands for itself, any other name reuses its
    first occurrence."""
    first: dict[str, int] = {}
    return [i if s.name == "후렴" else first.setdefault(s.name, i) for i, s in enumerate(sheet.sections)]


def unique_sections(sheet: LeadSheet) -> list[int]:
    """Sections the model writes, in song order: the first occurrence of every name except 후렴."""
    g = groups_of(sheet)
    return [i for i, s in enumerate(sheet.sections) if s.name != "후렴" and g[i] == i]


def _adapt_bars(bars: list[Bar], n: int) -> list[Bar]:
    """`n` bars from `bars` (cycled or cut); ties across a cut or a cycle boundary are dropped."""
    out = [copy.deepcopy(bars[i % len(bars)]) for i in range(n)]
    if n != len(bars):
        for i, b in enumerate(out):
            if i == n - 1 or i % len(bars) == len(bars) - 1:
                for note in b.notes:
                    note.tie = False
    return out


def apply_section(sheet: LeadSheet, uid: int, section: Section) -> LeadSheet:
    """Copy of `sheet` where every occurrence of section `uid`'s name carries `section`'s content."""
    g = groups_of(sheet)
    secs = [Section(s.name, _adapt_bars(section.bars, len(s.bars))) if g[i] == uid and s.name != "후렴" else s
            for i, s in enumerate(sheet.sections)]
    return LeadSheet(sheet.genre, sheet.key, sheet.meter, sheet.tempo, secs, sheet.country)


def _pitched(section: Section):
    return [n.pitch for b in section.bars for n in b.notes if n.pitch is not None]


def _first_pitch(section: Section):
    p = _pitched(section)
    return p[0] if p else None


def _last_pitch(section: Section):
    p = _pitched(section)
    return p[-1] if p else None


# ---------------------------------------------------------------- prompt (must equal the training "infill" input)
def header_line(sheet: LeadSheet) -> str:
    return (f"[장르:{sheet.genre}] [조:{sheet.key}] [박자:{sheet.meter}] [템포:{sheet.tempo}]"
            + (f" [국가:{sheet.country}]" if sheet.country else ""))


def model_head(sheet: LeadSheet, sections: list[Section] | None = None) -> LeadSheet:
    """The header the model sees: MODEL_GENRE and no country (v1), or, when the sheet carries a country, the style
    genre (케이팝/제이팝/뉴에이지, else MODEL_GENRE) and that country (v4)."""
    styled = sheet.country not in (None, DEFAULT_COUNTRY) and sheet.genre in MODEL_STYLE_GENRES
    genre = sheet.genre if styled else MODEL_GENRE
    return LeadSheet(genre, sheet.key, sheet.meter, sheet.tempo, list(sections or []), sheet.country)


def tag_for_generator(sheet: LeadSheet, genre: str, generator) -> LeadSheet:
    """Set (or clear) the sheet's country to match what the generator was trained with."""
    from dataclasses import replace
    wants = bool(getattr(generator, "country_tags", False)) if generator is not None else False
    return replace(sheet, country=COUNTRY_OF.get(genre, DEFAULT_COUNTRY) if wants else None)


def build_infill_input(sheet: LeadSheet, uid: int, chorus_bars: int) -> str:
    """Training-style infill prompt: the song (first occurrence of every section, the chorus cut to the seed length,
    which keeps the prompt near the training length) with section `uid` replaced by `<채우기: N마디>`."""
    seen, secs, pos = set(), [], None
    for i, s in enumerate(sheet.sections):
        if s.name in seen:
            continue
        seen.add(s.name)
        if i == uid:
            pos = len(secs)
        secs.append(Section(s.name, s.bars[:chorus_bars] if s.name == "후렴" else s.bars))
    if pos is None:
        raise ValueError(f"section {uid} is not the first occurrence of its name")
    ctx = model_head(sheet, secs)
    lines = serialize_leadsheet(ctx).splitlines()
    out, sec_i, i = [], -1, 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("[구조:"):
            sec_i += 1
            out.append(ln)
            if sec_i == pos:
                out.append(f"<채우기: {len(sheet.sections[uid].bars)}마디>")
                i += 1
                while i < len(lines) and not lines[i].startswith("[구조:"):
                    i += 1
                continue
        else:
            out.append(ln)
        i += 1
    return INSTR_INFILL + "\n" + "\n".join(out) + "\n"


def target_prefix(sheet: LeadSheet, uid: int) -> str:
    """How every infill training target starts: header line, newline, `[구조:NAME]`, newline."""
    return header_line(model_head(sheet)) + f"\n[구조:{sheet.sections[uid].name}]\n"


def max_tokens_for(bars: int, cap: int = MAX_TOKENS_CAP) -> int:
    return min(cap, TOKENS_PER_BAR * bars + TOKENS_OVERHEAD)


# ---------------------------------------------------------------- scoring
def parse_candidate(text: str, sheet: LeadSheet, uid: int) -> dict:
    """Validate one completion. Returns {grammar_ok, structure_ok, section, errors, text}.

    A completion is what training targets look like: header, `[구조:이름]`, bars. A missing header (and missing
    section tag) is added; the completion must then be a valid lead sheet with exactly one section of the requested
    name and bar count, in 4/4.
    """
    name, bars = sheet.sections[uid].name, len(sheet.sections[uid].bars)
    res = {"grammar_ok": False, "structure_ok": False, "section": None, "errors": [], "text": text}
    t = (text or "").strip("\n")
    if not t.strip():
        res["errors"].append("빈 출력")
        return res
    if not t.lstrip().startswith("[장르:"):
        head = header_line(model_head(sheet))
        t = head + "\n" + (t if t.lstrip().startswith("[구조:") else f"[구조:{name}]\n" + t)
    t = t.rstrip("\n") + "\n"
    res["text"] = t
    check = validate_leadsheet(t)
    if not check["ok"]:
        res["errors"] = check["errors"][:3]
        return res
    res["grammar_ok"] = True
    parsed = parse_leadsheet(t)
    if parsed.meter != "4/4":
        res["errors"].append(f"박자 {parsed.meter}")
    elif len(parsed.sections) != 1:
        res["errors"].append(f"구간 {len(parsed.sections)}개 (1개여야 함)")
    elif parsed.sections[0].name != name:
        res["errors"].append(f"구간 이름 {parsed.sections[0].name} (요청 {name})")
    elif len(parsed.sections[0].bars) != bars:
        res["errors"].append(f"마디 수 {len(parsed.sections[0].bars)} (요청 {bars})")
    else:
        res["structure_ok"] = True
        res["section"] = parsed.sections[0]
    return res


def arrange_section(trace, sheet: LeadSheet, section: Section, genre: str) -> dict:
    """Rule arrangement of one section in the sheet's key (instruments / energy of the genre template) plus its
    instrument-range and bar-length check. Returns {"events", "violations", "instruments", "energy"}."""
    key = key_of(sheet.key)
    one = LeadSheet(sheet.genre, sheet.key, sheet.meter, sheet.tempo, [section])
    kind = KIND[section.name]
    inst, energy = section_style(genre, kind)
    marks = _bar_marks(one)
    if not any(marks):
        return {"events": [], "violations": 0, "instruments": inst, "energy": energy}
    segs = [[(t, parse_chord(s, key)) for t, s in bar] for bar in marks]
    notes, _ = to_notes(one)
    events = arrange_segments(key, genre, notes, segs, inst, energy, sheet.meter)
    rv, lv = chk.range_violations(events), chk.length_violations(events, len(section.bars))
    out = {"events": events, "violations": len(rv) + len(lv), "instruments": inst, "energy": energy}
    if trace is not None:
        with trace.step("arrange_section", f"{section.name} {len(section.bars)}마디, {genre}") as st:
            st["output"] = (f"악기 {'+'.join(inst)}, 에너지 {energy}, 이벤트 {len(events)}개, "
                            f"음역·길이 위반 {out['violations']}건")
    return out


def _grams(section: Section, n: int = COPY_N) -> set:
    notes = [(nt.pitch, nt.dur) for b in section.bars for nt in b.notes if nt.pitch is not None]
    steps = [(b[0] - a[0], a[1]) for a, b in zip(notes, notes[1:])]
    return {tuple(steps[i:i + n]) for i in range(len(steps) - n + 1)}


def copy_share(section: Section, sheet: LeadSheet, uid: int) -> float:
    """Share of `section`'s melodic n-grams that also occur in the other sections of the song (0 = all new)."""
    mine = _grams(section)
    if not mine:
        return 0.0
    g = groups_of(sheet)
    others = set()
    for i, s in enumerate(sheet.sections):
        if g[i] != uid:
            others |= _grams(s)
    return round(len(mine & others) / len(mine), 3)


def copy_source(section: Section, sheet: LeadSheet, uid: int) -> dict | None:
    """The one other section that holds most of `section`'s melodic n-grams: {"section": name, "share"}, or None when
    nothing is shared.  Sections of the same group (a repeated verse) are the section itself and do not count."""
    mine = _grams(section)
    if not mine:
        return None
    g = groups_of(sheet)
    best = max(((round(len(mine & _grams(s)) / len(mine), 3), -i, s.name) for i, s in enumerate(sheet.sections) if g[i] != uid),
               default=None)
    return {"section": best[2], "share": best[0]} if best and best[0] > 0 else None


def score_candidate(sheet: LeadSheet, uid: int, cand: dict, genre: str, index: SimilarityIndex | None = None,
                    ref_index=None) -> dict:
    """Score one parsed candidate (see the module docstring). Returns a dict with `score`, `hard_fail` (reason or
    None), `passed` and the measured components."""
    r = {"score": 0.0, "hard_fail": None, "passed": False, "clash_mean": None, "bars_over": None, "leaps": None,
         "junctions": None, "max_repeat": None, "range_violations": None, "similar": None, "ref_level": None,
         "ref_notes": None, "copy_share": None, "copy_source": None}
    if not cand["grammar_ok"]:
        r["hard_fail"] = "문법: " + "; ".join(cand["errors"])
        return r
    if not cand["structure_ok"]:
        r["hard_fail"] = "구조: " + "; ".join(cand["errors"])
        return r
    section: Section = cand["section"]
    try:
        key = key_of(sheet.key)
        one = LeadSheet(sheet.genre, sheet.key, sheet.meter, sheet.tempo, [section])
        bars = len(section.bars)
        pitched = _pitched(section)
        instrumental = section.name in INSTRUMENTAL_SECTIONS
        marks = _bar_marks(one)
        if instrumental:
            with_chords = sum(1 for b in section.bars if b.chords)
            if 2 * with_chords < bars:
                r["hard_fail"] = f"코드가 있는 마디가 절반 미만이에요 ({with_chords}/{bars}마디)"
                return r
        else:
            if len(pitched) < bars:
                r["hard_fail"] = f"음이 너무 적어요 (음표 {len(pitched)}개 / {bars}마디)"
                return r
            if not any(marks):
                r["hard_fail"] = "코드가 없어요"
                return r
        # similarity policy: a copy of a known melody is a hard fail
        if index is not None and pitched:
            pitches, onsets = notes_from_leadsheet(one)
            res = index.check(pitches, onsets=onsets if index.mode == "rhythm" else None)
            r["similar"] = should_regenerate(res)
            if r["similar"]:
                r["hard_fail"] = f"유사도: 학습 데이터와 {res['max_run']}음 이상 연속 일치"
                return r
        # reference index (published melodies): an AI-written section with level "높음" is a hard fail (regenerated)
        if ref_index is not None and pitched:
            rr = plagcheck.section_level(ref_index, one)
            r["ref_level"], r["ref_notes"] = rr["level"], rr["notes"]
            if rr["level"] == plagcheck.LEVEL_HIGH:
                r["hard_fail"] = f"유사곡: '{rr['title']}' 과 {rr['notes']}음 연속 일치 ({rr['level']})"
                return r
        segs = [[(t, parse_chord(s, key)) for t, s in bar] for bar in marks]
        notes, _ = to_notes(one)
        rates = chk.clash_rates(key, notes, segs, sheet.meter) if pitched else []
        if len(rates) == bars:                      # clash only over bars that have notes (rest bars have no clash)
            rates = [x for x, b in zip(rates, section.bars) if any(n.pitch is not None for n in b.notes)]
        r["clash_mean"] = round(sum(rates) / len(rates), 3) if rates else 0.0
        r["bars_over"] = sum(x > chk.CLASH_MAX for x in rates)
        # junctions with the neighbours of every occurrence of this section
        g = groups_of(sheet)
        leaps = junctions = 0
        for i, s in enumerate(sheet.sections):
            if g[i] != uid or s.name == "후렴":
                continue
            fit = Section(section.name, _adapt_bars(section.bars, len(s.bars)))
            f, l = _first_pitch(fit), _last_pitch(fit)
            if i > 0 and f is not None and _last_pitch(sheet.sections[i - 1]) is not None:
                junctions += 1
                leaps += abs(_last_pitch(sheet.sections[i - 1]) - f) > song_mod.MAX_LEAP
            if i + 1 < len(sheet.sections) and l is not None and _first_pitch(sheet.sections[i + 1]) is not None:
                junctions += 1
                leaps += abs(_first_pitch(sheet.sections[i + 1]) - l) > song_mod.MAX_LEAP
        r["leaps"], r["junctions"] = leaps, junctions
        # repetition: the same bar (notes + chords) many times in a row
        sigs = [(tuple((n.pitch, n.start, n.dur) for n in b.notes), tuple((c.symbol, c.start) for c in b.chords))
                for b in section.bars]
        run = best = 1
        for a, b in zip(sigs, sigs[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        r["max_repeat"] = best
        r["copy_share"] = 0.0 if instrumental else copy_share(section, sheet, uid)
        r["copy_source"] = None if instrumental else copy_source(section, sheet, uid)
        # hard-fail only a near-total copy of one SPECIFIC other section (the pairwise share from copy_source).
        # copy_share() is a UNION over every other section: a mosaic that borrows a sub-threshold motif from two or
        # more different sections can push the union past COPY_HARD_FAIL without duplicating any single section, and
        # must still be allowed through on the soft W_COPY penalty below (LUNA 20261008 review blocker).
        src = r["copy_source"]
        if not instrumental and src is not None and src["share"] >= COPY_HARD_FAIL:
            r["hard_fail"] = f"복제: '{src['section']}' 구간과 거의 같아요 (일치율 {src['share']:.0%})"
            return r
        # ranges: instrument ranges of the rule arrangement + a soft melody range
        arr = arrange_section(None, sheet, section, genre)
        soft = sum(not MELODY_SOFT_RANGE[0] <= p <= MELODY_SOFT_RANGE[1] for p in pitched)
        r["range_violations"] = arr["violations"] + soft
    except (ValueError, LeadsheetError, KeyError) as exc:      # a candidate must never break the loop
        r["hard_fail"] = f"점수 계산 실패: {type(exc).__name__}: {exc}"
        return r
    penalty = (W_CLASH_BARS * r["bars_over"] / bars + W_CLASH_MEAN * r["clash_mean"]
               + min(W_LEAP_CAP, W_LEAP * leaps)
               + W_REPEAT * (best > REPEAT_MAX)
               + W_COPY * max(0.0, r["copy_share"] - COPY_FREE) / (1 - COPY_FREE)
               + W_RANGE * (min(1.0, r["range_violations"] / len(pitched)) if pitched else 0.0))
    r["score"] = round(max(0.0, 1.0 - penalty), 4)
    r["passed"] = r["score"] >= PASS_SCORE
    return r


# ---------------------------------------------------------------- tools (spec section 8)
def analyze_melody(trace, melody: list[Note], input_tonic: int = 0) -> dict:
    trace = _tr(trace)
    with trace.step("analyze_melody", f"음표 {len(melody)}개, {len(split_by_bar(melody))}마디") as st:
        analysis = analyze_key(melody, input_tonic)
        top = analysis["candidates"][0]["key"]
        st["output"] = f"1순위 {top['label']} (점수 차 {analysis['gap']})" + (", 애매함" if analysis["ambiguous"] else "")
    return analysis


def style_from_request(trace, style: dict | None = None, text: str | None = None) -> dict | None:
    """Settings from an already confirmed StyleAttrs dict. A free-text `text` only produces the proposal + question
    (rules only, no LLM here); confirmation happens in the UI, the composer never adopts an unconfirmed proposal."""
    trace = _tr(trace)
    if style is None and text is None:
        return None
    with trace.step("style_from_request", (f"확정된 스타일 {style.get('genre')}" if style else f"요청 '{text}'")) as st:
        if style is not None:
            settings = style_mod.apply_style(style)
            st["output"] = (f"장르 {settings['genre']}, 템포 {settings['bpm']}, 악기 {'+'.join(settings['instrument_list'])}, "
                            f"분위기 {settings['mood']}")
            return settings
        prop = style_mod.style_from_request(text, planner=False)
        st["output"] = "제안(확인 필요): " + prop["confirm_question"]
        return {"proposal": prop}


def plan_structure(trace, genre: str, bpm: int, motif_bars: int, form: str = "standard",
                   seed_role: str = "chorus") -> list[dict]:
    trace = _tr(trace)
    with trace.step("plan_structure", f"{genre}, 템포 {bpm}, 씨앗 {motif_bars}마디") as st:
        secs, notes = song_mod.plan_structure(genre, bpm, motif_bars, form, seed_role)
        plan = [{"kind": s["kind"], "name": NAME[s["kind"]], "bars": s["bars"], "label": s["label"]} for s in secs]
        st["output"] = " → ".join(f"{p['label']} {p['bars']}" for p in plan) + (f" (조정: {', '.join(notes)})" if notes else "")
    return plan


def generate_section(trace, generator, sheet: LeadSheet, uid: int, chorus_bars: int, n: int, rnd: int,
                     max_tokens_cap: int = MAX_TOKENS_CAP, force_prefix: bool | None = None) -> tuple[list[str], dict]:
    """`n` model candidates for section `uid`. Returns (texts, call info); a failing generator gives ([], info)."""
    trace = _tr(trace)
    sec = sheet.sections[uid]
    bars = len(sec.bars)
    force = FORCE_SECTION_PREFIX if force_prefix is None else force_prefix
    info = {"ms": 0, "prompt_tokens": None, "gen_tokens": None, "error": None, "mode": "prefix" if force else "free"}
    with trace.step("generate_section", f"{sec.name} {bars}마디, 후보 {n}개, {rnd}회차, {info['mode']}") as st:
        try:
            text_in = build_infill_input(sheet, uid, chorus_bars)
            t0 = time.perf_counter()
            kw = {}
            if force:
                prefix = target_prefix(sheet, uid)
                kw["prefix"] = prefix
            texts = list(generator.generate(text_in, n, temperature=TEMPERATURE, top_p=TOP_P,
                                            max_tokens=max_tokens_for(bars, max_tokens_cap), **kw))
            if force:
                texts = [prefix + t for t in texts]
            info["ms"] = round((time.perf_counter() - t0) * 1000)
            call = getattr(generator, "last_call", None) or {}
            info["prompt_tokens"], info["gen_tokens"] = call.get("prompt_tokens"), call.get("gen_tokens")
            st["output"] = (f"후보 {len(texts)}개 생성 (소스 model, {info['ms']}ms"
                            + (f", 생성 토큰 {info['gen_tokens']}" if info["gen_tokens"] is not None else "") + ")")
        except Exception as exc:
            texts, info["error"] = [], f"{type(exc).__name__}: {exc}"
            st["output"] = f"생성 실패: {info['error']}"
    return texts, info


def reject_kinds(scored: list[dict]) -> dict[str, int]:
    """{kind: count} of the candidates that did not pass: the word before the colon of a hard-fail reason (문법, 구조,
    복제, ...), or "점수 미달" for a candidate that was scored below the pass mark."""
    out: dict[str, int] = {}
    for c in scored:
        if c["passed"]:
            continue
        kind = c["hard_fail"].split(":", 1)[0].strip() if c["hard_fail"] else "점수 미달"
        out[kind] = out.get(kind, 0) + 1
    return dict(sorted(out.items()))


def score_section(trace, sheet: LeadSheet, uid: int, texts: list[str], genre: str,
                  index: SimilarityIndex | None, ref_index=None) -> list[dict]:
    """Parse + score every candidate. One dict per candidate: parse fields, `score`, `hard_fail`, `passed`."""
    trace = _tr(trace)
    sec = sheet.sections[uid]
    with trace.step("score_section", f"{sec.name} 후보 {len(texts)}개") as st:
        out = []
        for t in texts:
            cand = parse_candidate(t, sheet, uid)
            out.append({**cand, **score_candidate(sheet, uid, cand, genre, index, ref_index)})
        fails = [c["hard_fail"] for c in out if c["hard_fail"]]
        kinds = reject_kinds(out)
        st["output"] = (f"점수 {[c['score'] for c in out]}, 통과 {sum(c['passed'] for c in out)}/{len(out)}"
                        + (f", 실패 사유 예: {fails[0]}" if fails else "")
                        + (", 탈락 사유: " + ", ".join(f"{k} {n}" for k, n in kinds.items()) if kinds else ""))
    return out


def render(trace, sheet: LeadSheet, key_name: str, genre: str, levels: dict | None, flow: bool = True,
           legacy_lineup: bool = False, instruments: list[str] | None = None, key_change: bool | None = None,
           melody_role: str = "vocal", exact_back: bool = False) -> dict:
    """Assembled lead sheet (model key) -> seed key -> rule arrangement.  `key_change`: the last chorus goes up a
    semitone (True), stays as the user wrote it (False), or follows the genre's habit (None).
    `exact_back`: the sheet holds a user's melody that was moved into the model's key, so the way back is the exact
    opposite of that move (`_finish` asks for it).  Without it the nearest move is taken as before, which keeps the
    older research renders (scripts/make_listening_samples.py, docs/listen/round1) byte for byte."""
    trace = _tr(trace)
    with trace.step("render", f"{sum(len(s.bars) for s in sheet.sections)}마디 → {key_name}, {genre}") as st:
        text = serialize_leadsheet(sheet)
        shift = PROFILES[genre]["last_chorus_shift"] > 0 if genre in PROFILES else True   # hip-hop: no key change
        if key_change is not None:
            shift = bool(key_change) and shift             # the user's choice; a genre without key change keeps none
        # back to the seed's key by the exact opposite of the move that brought the seed into the sheet's key.  Worked
        # out anew as "the nearest move" it is the same for every key but the one six semitones away, where the nearest
        # move is down both ways and the user's melody came back an octave low (found 2026-10-12, F sharp / E flat minor).
        back = -_shift_to(key_of(key_name), key_of(sheet.key)) if exact_back else None
        result = leadsheet_to_song(text, key_name, genre, levels, key_change_last_chorus=shift, flow=flow,
                                   legacy_lineup=legacy_lineup, instruments=instruments, melody_role=melody_role,
                                   target_shift=back)
        flow = next((x["detail"] for x in result["trace"] if x["step"] == "곡 흐름"), None)
        st["output"] = f"{result['bars']}마디 {result['duration_sec']}초, 템포 {result['tempo']}" + (f". {flow}" if flow else "")
    return result


def explain(trace, stats: dict, checks: dict) -> str:
    """Short Korean summary built by code from the stats (no LLM)."""
    trace = _tr(trace)
    with trace.step("explain", "요약") as st:
        lines = [f"{stats['bars']}마디, {stats['duration_sec']}초 곡을 만들었어요."
                 + ("" if checks["duration_ok"] else f" ({checks['duration_note']})"),
                 "후렴은 씨앗 멜로디 그대로예요." if checks["chorus_equals_seed"] else
                 "후렴이 씨앗과 다르게 나와 규칙 곡으로 대체했어요." if checks.get("fallback_fixed", True) else
                 "후렴이 씨앗과 다르게 나와 규칙 곡으로 바꿔 봤지만 여전히 달라요."]
        if not stats["generator_available"]:
            lines.append("모델을 쓸 수 없어 모든 구간을 규칙 기반으로 만들었어요.")
        for s in stats["sections"]:
            if s["source"] == "model":
                lines.append(f"{s['name']}: 모델 후보 {s['candidates']}개 중 {s['passed']}개가 통과, 점수 {s['score']}인 후보를 "
                             f"골랐어요 ({s['rounds']}회차).")
            elif stats["generator_available"]:
                lines.append(f"{s['name']}: {s['rounds']}회 시도했지만 통과한 후보가 없어 규칙 기반 구간을 유지했어요.")
        g = stats["grammar"]
        if g["round1"]["candidates"]:
            lines.append(f"문법 통과율 {g['round1']['grammar_valid']}/{g['round1']['candidates']} (1회차), "
                         f"구간 통과 {g['section_pass_round1']}/{g['sections']} → 재생성 후 {g['section_pass_final']}/{g['sections']}.")
        text = " ".join(lines)
        st["output"] = text
    return text


# ---------------------------------------------------------------- the loop
def _new_stats(sheet: LeadSheet, uids: list[int]) -> dict:
    return {u: {"index": u, "name": sheet.sections[u].name, "bars": len(sheet.sections[u].bars), "rounds": 0,
                "candidates": 0, "grammar_valid": 0, "structure_ok": 0, "passed": 0, "score": None,
                "source": "rule", "latency_ms": 0, "gen_tokens": 0, "errors": [], "mode": None,
                "ref_rejected": 0, "rejects": {}} for u in uids}


def improve(trace, sheet: LeadSheet, uids: list[int], generator, index, genre: str, candidates: int, max_rounds: int,
            chorus_bars: int, max_tokens_cap: int = MAX_TOKENS_CAP,
            force_prefix: bool | None = None, ref_index=None) -> tuple[LeadSheet, dict, dict]:
    """Regenerate the sections `uids` (first occurrences, song order) until each has a passing candidate.

    Only sections without a passing candidate go into the next round; after `max_rounds` a failing section keeps its
    current (rule draft) content and the trace records `fallback: rule`. Returns (sheet, stats per section, grammar).
    """
    stats = _new_stats(sheet, uids)
    rounds = {}
    pending = list(uids)
    usable = generator is not None and getattr(generator, "available", lambda: True)()
    if usable:
        for rnd in range(1, max_rounds + 1):
            if not pending:
                break
            rs = rounds[rnd] = {"sections": len(pending), "candidates": 0, "grammar_valid": 0, "structure_ok": 0,
                                "sections_passed": 0}
            still = []
            for uid in pending:
                st = stats[uid]
                st["rounds"] = rnd
                texts, info = generate_section(trace, generator, sheet, uid, chorus_bars, candidates, rnd, max_tokens_cap,
                                             force_prefix)
                st["mode"] = info["mode"]
                st["latency_ms"] += info["ms"]
                st["gen_tokens"] += info["gen_tokens"] or 0
                if info["error"]:
                    st["errors"].append(info["error"])
                scored = score_section(trace, sheet, uid, texts, genre, index, ref_index)
                st["ref_rejected"] += sum(c.get("ref_level") == plagcheck.LEVEL_HIGH for c in scored)
                for kind, n in reject_kinds(scored).items():          # why candidates were turned down, per section
                    st["rejects"][kind] = st["rejects"].get(kind, 0) + n
                n_valid = sum(c["grammar_ok"] for c in scored)
                n_struct = sum(c["structure_ok"] for c in scored)
                n_pass = sum(c["passed"] for c in scored)
                for d in (st, rs):
                    d["candidates"] += len(scored)
                    d["grammar_valid"] += n_valid
                    d["structure_ok"] += n_struct
                st["passed"] += n_pass
                best = max((c for c in scored if c["passed"]), key=lambda c: c["score"], default=None)  # first wins ties
                if best is None:
                    still.append(uid)
                    continue
                st["score"], st["source"] = best["score"], "model"
                rs["sections_passed"] += 1
                sheet = apply_section(sheet, uid, best["section"])
                arrange_section(trace, sheet, sheet.sections[uid], genre)
            pending = still
    for uid in pending:
        stats[uid]["source"] = "rule"
        with _tr(trace).step("generate_section", f"{stats[uid]['name']} {stats[uid]['bars']}마디") as st:
            st["output"] = ("fallback: rule (모델을 쓸 수 없어요)" if not usable else
                            f"fallback: rule ({max_rounds}회 시도해도 통과한 후보가 없어 규칙 기반 구간 유지)")
    r1 = rounds.get(1, {"sections": 0, "candidates": 0, "grammar_valid": 0, "structure_ok": 0, "sections_passed": 0})
    cum = {k: sum(r[k] for r in rounds.values()) for k in ("candidates", "grammar_valid", "structure_ok")}
    grammar = {"round1": {k: r1[k] for k in ("candidates", "grammar_valid", "structure_ok")},
               "cumulative": cum, "sections": len(uids), "section_pass_round1": r1["sections_passed"],
               "section_pass_final": sum(s["source"] == "model" for s in stats.values()),
               "rounds": {str(k): v for k, v in rounds.items()}}
    return sheet, stats, grammar


# ---------------------------------------------------------------- song-level checks and finishing
ORIGIN_KO = {"input": "입력 그대로", "model": "AI가 새로 씀", "rule": "규칙 변형 (입력에서 만든 것)",
             "rest": "쉼 (멜로디 없음)", "none": "없음"}


def label_origins(result: dict, seed_kind: str = "chorus", sources: list[str | None] | None = None) -> None:
    """Write on every section who its melody is from: `origin` (input / model / rule / rest / none) and the Korean
    `melody_source` shown on screen.  A key-changed section that carries the input says so.
    `sources`: one "model" / "rule" per section in song order, for a result whose own statistics cover only part of
    the song (a one-section rewrite: the other sections keep what the parent song said)."""
    if sources is None:
        by_uid = {s["index"]: s["source"] for s in (result.get("compose_stats") or {}).get("sections") or []}
        groups = groups_of(parse_leadsheet(result["leadsheet_c"])) if result.get("leadsheet_c") else []
        sources = [by_uid.get(g) for g in groups]          # a repeated section has the source of its first occurrence
    notes = melody_notes(result)
    for i, sec in enumerate(result.get("sections") or []):
        lo, hi = sec["start_bar"] * BAR, (sec["start_bar"] + sec["bars"]) * BAR
        has_melody = any(lo <= n.start < hi for n in notes)
        src = sources[i] if i < len(sources) else None
        if sec["kind"] == seed_kind and has_melody:
            origin = "input"
        elif not has_melody:
            origin = "rest" if sec["kind"] in song_mod.MELODIC else "none"
        else:
            origin = "model" if src == "model" else "rule"
        sec["origin"] = origin
        sec["melody_source"] = ORIGIN_KO[origin] + (f" · {sec['shift']:+d}반음 전조" if origin == "input" and sec.get("shift") else "")


def chorus_matches_seed(result: dict, motif: list[Note], motif_bars: int, seed_kind: str = "chorus") -> tuple[bool, list[str]]:
    """Every section that carries the input (`seed_kind`: the chorus, or the verse) plays exactly the seed; a chorus
    with a key change is compared after undoing its shift."""
    notes = melody_notes(result)
    problems = []
    for sec in (s for s in result["sections"] if s["kind"] == seed_kind):
        lo = sec["start_bar"] * BAR
        expected = sorted((n.pitch, n.start, n.dur) for n in song_mod._pad(song_mod._tile(motif, motif_bars, sec["bars"]),
                                                                          sec["bars"]) if n.pitch is not None)
        got = sorted((n.pitch - sec["shift"], n.start - lo, n.dur) for n in notes
                     if lo <= n.start < lo + sec["bars"] * BAR)
        if got != expected:
            problems.append(f"{sec['label']}: 음 {len(got)}개 vs 씨앗 {len(expected)}개")
    return not problems, problems


def _finish(trace, sheet: LeadSheet, settings: dict, motif: list[Note], motif_bars: int, stats: dict,
            grammar: dict, generator_info: dict, t_start: float, rule_sheet: LeadSheet, ref_index=None,
            section_sources: list[str] | None = None) -> dict:
    key_name = settings["key"]["name"]
    genre = settings["genre"]
    result = render(trace, sheet, key_name, genre, settings.get("levels"), instruments=settings.get("instruments"),
                    key_change=settings.get("key_change"), melody_role=settings.get("melody_role") or "vocal",
                    exact_back=True)
    seed_kind = settings.get("seed_role") or "chorus"
    seed_name = NAME[seed_kind]
    ok, problems = chorus_matches_seed(result, motif, motif_bars, seed_kind)
    fallback = False
    if not ok:                          # never ship a song whose chorus differs from the seed: back to the rule draft
        with trace.step("generate_section", "곡 검사: 후렴이 씨앗과 다름") as st:
            st["output"] = "fallback: rule (" + "; ".join(problems) + ")"
        sheet, fallback = rule_sheet, True
        for s in stats.values():
            s["source"] = "rule"
        result = render(trace, sheet, key_name, genre, settings.get("levels"), instruments=settings.get("instruments"),
                    key_change=settings.get("key_change"), melody_role=settings.get("melody_role") or "vocal",
                    exact_back=True)
        ok, problems = chorus_matches_seed(result, motif, motif_bars, seed_kind)
    checks = {"chorus_equals_seed": ok, "chorus_problems": problems, "fallback_fixed": ok or not fallback,
              "duration_sec": result["duration_sec"], "min_duration_sec": MIN_SONG_SECONDS,
              "max_duration_sec": MAX_SONG_SECONDS,
              "duration_ok": MIN_SONG_SECONDS <= result["duration_sec"] <= MAX_SONG_SECONDS,
              "duration_note": song_mod.duration_note(result["duration_sec"], result["tempo"]),
              "range_violations": result["check"]["range_violations"], "length_violations": result["check"]["length_violations"],
              "leadsheet_valid": validate_leadsheet(serialize_leadsheet(sheet))["ok"]}
    section_stats = [stats[u] for u in sorted(stats)]
    compose_stats = {
        "bars": result["bars"], "duration_sec": result["duration_sec"], "sections": section_stats, "grammar": grammar,
        "model_sections": sum(s["source"] == "model" for s in section_stats),
        "rule_sections": sum(s["source"] == "rule" for s in section_stats),
        "generator_available": generator_info["available"], "generator": generator_info,
        "prefix_forcing": sorted({s["mode"] for s in section_stats if s["mode"]}) or None,
        "candidates_total": sum(s["candidates"] for s in section_stats),
        "generate_ms": sum(s["latency_ms"] for s in section_stats),
        "gen_tokens": sum(s["gen_tokens"] for s in section_stats),
    }
    similar = None
    if ref_index is not None:
        with trace.step("similar_check", "공개 곡 색인과 비교") as st:
            sheet_srcs = [("seed" if s.name == seed_name else stats[g]["source"] if g in stats else "rule")
                          for s, g in zip(sheet.sections, groups_of(sheet))]
            similar = plagcheck.check_song({"leadsheet_c": serialize_leadsheet(sheet)}, ref_index, sheet_srcs)
            similar["regenerated_candidates"] = sum(s.get("ref_rejected", 0) for s in section_stats)
            st["output"] = similar["summary"]
        compose_stats["similar_check"] = similar
    text = explain(trace, compose_stats, checks)
    played = song_to_leadsheet(result)
    compose_stats["total_ms"] = round((time.perf_counter() - t_start) * 1000)
    out = {**result, "status": "ok" if checks["duration_ok"] else "duration_out_of_range", "render_trace": result["trace"], "trace": list(trace), "compose_stats": compose_stats,
           "checks": checks, "explain": text, "leadsheet": played, "leadsheet_c": serialize_leadsheet(sheet),
           "settings": settings, "source": "compose"}
    if not ok:                              # still not the typed melody: said plainly, the status is left as it is
        shown = "; ".join(problems[:3])
        out["seed_warning"] = {"text": "입력 멜로디와 다르게 나왔어요. 후렴이 입력한 멜로디와 같지 않으니 악보를 확인해 주세요."
                                       + (f" 이유: {shown}" if shown else ""), "problems": problems[:10]}
    if similar is not None:
        out["similar_check"] = similar
    if fallback:
        out["fallback_rule_song"] = True
    return out


def _generator_info(generator) -> dict:
    if generator is None:
        return {"available": False, "reason": "모델 없음"}
    avail = bool(getattr(generator, "available", lambda: True)())
    info = getattr(generator, "info", None)
    base = info() if callable(info) else {}
    return {**base, "available": avail}


def compose(tokens: list[dict], input_tonic: int = 0, key: dict | None = None, genre: str | None = None,
            bpm: int | None = None, levels: dict | None = None, style: dict | None = None,
            candidates: int = CANDIDATES_DEFAULT, max_rounds: int = MAX_ROUNDS_DEFAULT, generator=None, progress=None,
            instruments: list[str] | None = None, energy: int = 0, prefs=None, index="auto",
            max_tokens_cap: int = MAX_TOKENS_CAP, force_prefix: bool | None = None,
            bias: dict[str, float] | None = None, ref_index=None, form: str = "standard",
            seed_chords: list[str | None] | None = None, key_change: bool | None = None,
            melody_role: str = "vocal", seed_role: str = "chorus", fill: str = "own",
            voice_range: tuple[int, int] | None = None, shape: bool = False) -> dict:
    """Seed melody -> song. `generator=None` means no model (every section from the rule draft).
    `shape` (option, off by default): the app-written pre-chorus ends open into the chorus and a bridge that copies
    verse bars is written anew (instmate/melodyshape.py); the input melody is not touched.
    `voice_range`: (lowest, highest) MIDI pitch of the singer; the sections the app writes are moved by octaves into it
    (the input melody is expected to have been fitted already, instmate/voicefit.py).
    `seed_role`: the section kind the input melody is ("chorus" or "verse"); `fill`: what the other melodic sections
    get with no model: "own" = rule variations of the input, "rest" = no melody (the band plays, nobody sings).
    The trained model was taught to fill a song around a given CHORUS, so with the input as the verse, and with
    "rest", the model is not used: a generator is refused rather than called outside what it was trained for.
    `key_change`: False keeps the last chorus in the user's key (the input melody is then identical in every chorus),
    True raises it a semitone where the genre does that, None = the genre's habit.
    `seed_chords`: chord symbols chosen for the seed's bars (agent.run); the chorus keeps them.

    `key`: {"tonic","mode"} confirmed by the user or None (the analysis decides; an ambiguous key returns
    {"status": "need_key", ...} so the caller can ask). `style`: an already confirmed StyleAttrs dict; it fills
    genre / bpm / instruments that were not given. `bias`: {roman: bonus} (e.g. style.chord_bias_for_key) that steers
    the chords of the RULE DRAFT only (the seed chorus progression and the rule-written sections; the bridge keeps its
    own DARK_BIAS on top). Sections the model writes keep the model's own chords. `index`: SimilarityIndex, None (no check) or "auto" (the stored one).
    """
    t_start = time.perf_counter()
    if not 1 <= candidates <= 10 or not 1 <= max_rounds <= 3:
        raise ValueError("candidates must be 1-10 and max_rounds 1-3")
    if seed_role not in song_mod.SEED_ROLES or fill not in song_mod.FILLS:
        raise ValueError(f"seed_role must be one of {song_mod.SEED_ROLES} and fill one of {song_mod.FILLS}")
    if generator is not None and (seed_role != "chorus" or fill == "rest"):
        raise ValueError("입력 멜로디를 벌스에 놓거나 나머지를 비워 둘 때는 AI로 채우기를 쓸 수 없어요 "
                         "(모델은 후렴이 주어진 곡을 채우도록 학습됐어요). \"내 멜로디 변형만\"으로 골라 주세요.")
    trace = Trace(progress)
    melody = parse_number_input(tokens, input_tonic)
    analysis = analyze_melody(trace, melody, input_tonic)
    top = analysis["candidates"][0]["key"]
    if key is None and analysis["ambiguous"]:
        return {"status": "need_key", "question": analysis["question"], "key_analysis": analysis, "trace": list(trace)}
    kc = {"tonic": int(key["tonic"]), "mode": key["mode"]} if key else {"tonic": top["tonic"], "mode": top["mode"]}
    key_obj = Key(kc["tonic"], kc["mode"])

    settings_style = style_from_request(trace, style=style)
    if settings_style:
        genre = genre or settings_style["genre"]
        bpm = bpm if bpm is not None else settings_style["bpm"]
        instruments = instruments or settings_style["instrument_list"]
    genre = genre or "ballad"
    instruments = list(instruments) if instruments else list(BAND)
    agent._validate(genre, instruments)
    mix = resolve_mix(genre, bpm, levels)          # validates genre-independent tempo / level ranges
    motif = parse_number_input(tokens, input_tonic)
    motif_bars = len(split_by_bar(motif))
    if seed_chords and len(seed_chords) > motif_bars:      # chords for bars the melody does not have (a silent last bar)
        seed_chords = list(seed_chords[:motif_bars])
    plan = plan_structure(trace, genre, mix["tempo"], motif_bars, form, seed_role)

    # --- rule draft: exactly what /api/song builds for the same inputs
    with trace.step("generate_section", "규칙 초안 (전체 구조)") as st:
        res = agent.run(tokens, input_tonic, genre, instruments, kc, energy, prefs, tempo=mix["tempo"], levels=levels,
                        bias=bias, seed_chords=seed_chords)
        seed_note = next((t["detail"] for t in res.get("trace", []) if t.get("step") == "코드 고정"), None)
        # the chorus is the user's melody and is never changed: a bar of it that fails the clash check gets the chord
        # that fits its strong-beat notes instead (chords the user chose for the seed stay as they are)
        chosen = {i for i, sym in enumerate(seed_chords or []) if sym}
        seed_romans, refit = agent.fit_seed_chords(key_obj, genre, motif, [c["roman"] for c in res["chords"]], chosen)
        state = {"tokens": tokens, "input_tonic": input_tonic, "genre": genre, "instruments": instruments,
                 "energy": energy, "key_choice": kc, "chords": seed_romans, "bias": bias,
                 "form": form, "seed_role": seed_role, "fill": fill}
        rule_song = song_mod.build_song(state, res)
        if [(s["kind"], s["bars"]) for s in rule_song["sections"]] != [(p["kind"], p["bars"]) for p in plan]:
            raise RuntimeError("rule draft does not follow the planned structure")
        rule_sheet = parse_leadsheet(song_to_leadsheet(rule_song))
        to_c = _shift_to(key_obj, key_of(model_key_name(key_obj)))
        sheet = transpose_sheet(rule_sheet, [to_c - s["shift"] for s in rule_song["sections"]], model_key_name(key_obj))
        st["output"] = (f"source rule: {len(sheet.sections)}구간 {sum(len(s.bars) for s in sheet.sections)}마디, "
                        f"{key_obj.name} → {sheet.key} ({to_c:+d}반음), 후렴 = 씨앗"
                        + ("; 씨앗 코드 조정 " + ", ".join(f"{c['bar'] + 1}마디 {c['from']}→{c['to']}" for c in refit) if refit else ""))

    settings = {"tokens": tokens, "input_tonic": input_tonic, "key": kc | {"name": key_obj.name}, "genre": genre,
                "tempo": mix["tempo"], "levels": levels, "instruments": instruments, "energy": energy,
                "style": style, "candidates": candidates, "max_rounds": max_rounds, "key_change": key_change,
                "melody_role": melody_role, "seed_role": seed_role, "fill": fill}
    idx = default_similarity_index() if isinstance(index, str) else index
    ref = plagcheck.default_reference_index() if isinstance(ref_index, str) else ref_index
    gen = generator if generator is not None and _generator_info(generator)["available"] else None
    ginfo = _generator_info(generator)
    sheet = tag_for_generator(sheet, genre, gen)
    uids = unique_sections(sheet)

    def in_range(sh):
        """The sections the app wrote, moved by octaves into the singer's range (the user's own melody is not moved)."""
        if not voice_range:
            return sh
        from .voicefit import fit_sections
        seed_name = NAME[seed_role]
        keep = {i for i, sec in enumerate(sh.sections) if sec.name.split()[0] == seed_name and (seed_role != "intro" or i == 0)}
        fitted, moved = fit_sections(sh, voice_range[0], voice_range[1], to_c, keep)
        if moved:
            with trace.step("fit_register", "가수 음역에 맞춰 구간 옥타브 조정") as st:
                st["output"] = ", ".join(f"{m['section']} {m['shift']:+d}" for m in moved)
        return fitted

    sheet = in_range(sheet)
    rule_sheet = sheet
    sheet, stats, grammar = improve(trace, sheet, uids, gen, idx, genre, candidates, max_rounds, motif_bars, max_tokens_cap,
                                  force_prefix, ref)
    sheet = in_range(sheet)                       # sections the model rewrote come back in its own register
    # notes the ear rests on, and section endings, land on their chord (the user's own melody is not touched)
    from .melodyfix import settle
    seed_name = NAME[seed_role]
    own = {i for i, sec in enumerate(sheet.sections) if sec.name.split()[0] == seed_name and (seed_role != "intro" or i == 0)}
    sheet, settled = settle(sheet, own, *((voice_range[0] + to_c, voice_range[1] + to_c) if voice_range else (None, None)))
    if settled:
        with trace.step("settle_melody", "앱이 쓴 구간의 긴 음과 끝 음을 코드에 맞춤") as st:
            st["output"] = f"{len(settled)}개 음을 가까운 코드 구성음으로 옮김 (최대 2반음, 입력 멜로디는 그대로)"
    shaped = []
    if shape:
        from .melodyshape import shape as shape_melody
        sheet, shaped = shape_melody(sheet, own, *((voice_range[0] + to_c, voice_range[1] + to_c) if voice_range else (None, None)))
        with trace.step("shape_melody", "앱이 쓴 프리코러스·브릿지 다듬기 (옵션)") as st:
            st["output"] = "; ".join(f"{c['name']}: {c['detail']}" for c in shaped) or "바꿀 곳 없음 (입력 멜로디는 그대로)"
        settings["melody_shape"] = True
    ginfo["similarity_index"] = idx is not None
    ginfo["reference_index"] = ref is not None
    out = _finish(trace, sheet, settings, motif, motif_bars, stats, grammar, ginfo, t_start, rule_sheet, ref)
    label_origins(out, seed_role)
    out["melody_settled"] = len(settled)
    if shape:
        out["melody_shaped"] = shaped
    if seed_chords:
        out["seed_chords"] = {"given": list(seed_chords), "note": seed_note}
    return out


def regenerate_section(settings: dict, leadsheet_c: str, section_index: int, generator, progress=None,
                       candidates: int | None = None, max_rounds: int | None = None, index="auto",
                       max_tokens_cap: int = MAX_TOKENS_CAP, force_prefix: bool | None = None, ref_index=None) -> dict:
    """Rewrite one section of a stored song (same loop with a single section). A 후렴 (the seed) cannot be rewritten;
    for a repeated section the first occurrence is rewritten and all occurrences follow (pop repetition)."""
    t_start = time.perf_counter()
    trace = Trace(progress)
    sheet = parse_leadsheet(leadsheet_c)
    if not 0 <= section_index < len(sheet.sections):
        raise ValueError(f"section_index must be 0-{len(sheet.sections) - 1}")
    if (settings.get("seed_role") or "chorus") != "chorus" or (settings.get("fill") or "own") == "rest":
        raise ValueError("이 곡은 AI로 구간을 다시 만들 수 없어요 (입력이 벌스이거나 나머지를 비워 둔 곡)")
    if sheet.sections[section_index].name == "후렴":
        raise ValueError("후렴은 씨앗 멜로디라 다시 만들지 않아요")
    uid = groups_of(sheet)[section_index]
    motif = parse_number_input(settings["tokens"], settings["input_tonic"])
    motif_bars = len(split_by_bar(motif))
    candidates = candidates or settings.get("candidates", CANDIDATES_DEFAULT)
    max_rounds = max_rounds or settings.get("max_rounds", MAX_ROUNDS_DEFAULT)
    idx = default_similarity_index() if isinstance(index, str) else index
    ref = plagcheck.default_reference_index() if isinstance(ref_index, str) else ref_index
    ginfo = _generator_info(generator)
    gen = generator if generator is not None and ginfo["available"] else None
    sheet = tag_for_generator(sheet, settings["genre"], gen)
    before = sheet
    sheet, stats, grammar = improve(trace, sheet, [uid], gen, idx, settings["genre"], candidates, max_rounds, motif_bars,
                                    max_tokens_cap, force_prefix, ref)
    ginfo["similarity_index"] = idx is not None
    ginfo["reference_index"] = ref is not None
    return _finish(trace, sheet, settings, motif, motif_bars, stats, grammar, ginfo, t_start, before, ref)
