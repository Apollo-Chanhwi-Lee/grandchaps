"""Style conversion: a free-text request ("성시경 희재 같은 느낌으로") -> style attributes -> user confirmation (M6).

Honesty rule: an artist or song name is only interpreted as a *style label* (genre, tempo range, instruments, mood,
structure). This module never fetches audio, never looks anything up online, and never copies or references the
melody, lyrics or chords of the named work. A local LLM is only asked "what style words does this phrase suggest?".

Public API
----------
StyleAttrs                  dataclass: genre, tempo_min, tempo_max, instruments, mood, structure_hint, source
                            .to_dict()
style_from_request(text, planner=None, timeout_s=15.0) -> dict
    {"attrs": StyleAttrs.to_dict(), "confirm_question": str (Korean, built by CODE), "method": "llm" | "rule",
     "notes": [str, ...]}   notes lists every correction / fallback / default that code applied.
    planner:
      None   -> try local Ollama (qwen3:14b, format=JSON schema, think=false, temperature 0); any failure -> rules
      False  -> rules only (no network at all)
      callable(text) -> dict (raw proposal), e.g. a fake in tests; exceptions and timeouts (> timeout_s) -> rules
    The LLM only proposes. Every field is validated and clamped by code (validate_attrs).
apply_style(attrs) -> dict
    Concrete generation settings: {"genre", "bpm", "tempo", "instruments" (name -> bool for every arrange.INSTRUMENTS),
    "instrument_list", "mood", "chord_bias": {"direction": "dark" | "bright" | None, "strength": float},
    "structure_hint"}.  `bpm` is the middle of the tempo range and is valid for mix.resolve(genre, tempo=bpm).
    "chord_bias" is a hint for the chord tools: use chord_bias_for_key(key, hint) to turn it into the
    {roman: bonus} dict that chords.progression(..., bias=) accepts (same mechanism as song.DARK_BIAS).
chord_bias_for_key(key, hint) -> dict[str, float]

Mood set (fixed): bright, dark, calm, energetic, sad, warm.  dark/sad bias toward "dark" chords, bright/energetic
toward "bright" chords, calm/warm give no chord bias.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field

from .arrange import INSTRUMENT_KO, INSTRUMENTS
from .mix import TEMPO_MAX, TEMPO_MIN
from .genres import PROFILES as _PROFILES
from .song import TEMPLATES

# genre lists, labels, default tempo ranges, instruments, moods and keywords: instmate/genres.py
_FIRST = ["ballad", "rock", "dance", "classical"]                  # the original order (LLM schema enum)
GENRES = _FIRST + [g for g in _PROFILES if g not in _FIRST]
GENRE_KO = {g: p["style_label"] for g, p in _PROFILES.items()}
MOODS = ["bright", "dark", "calm", "energetic", "sad", "warm"]
MOOD_KO = {"bright": "밝은", "dark": "어두운", "calm": "차분한", "energetic": "신나는", "sad": "슬픈", "warm": "따뜻한"}
STRUCTURE_KINDS = sorted({sec[0] for tpl in TEMPLATES.values() for sec in tpl})

DEFAULT_TEMPO = {g: tuple(p["tempo_range"]) for g, p in _PROFILES.items()}
DEFAULT_INSTRUMENTS = {g: list(p["instruments"]) for g, p in _PROFILES.items()}
DEFAULT_MOOD = {g: p["mood"] for g, p in _PROFILES.items()}
SLOW_RANGE, FAST_RANGE = (64, 76), (132, 156)
CHORD_BIAS_STRENGTH = 0.6                   # same size as song.DARK_BIAS
MOOD_BIAS = {"dark": "dark", "sad": "dark", "bright": "bright", "energetic": "bright", "calm": None, "warm": None}
LLM_TIMEOUT_S = 15.0

# ---------------------------------------------------------------- keyword table (deterministic fallback)
GENRE_WORDS = {g: list(p["words"]) for g, p in _PROFILES.items()}
# Song titles interpreted as a style label only (nothing about their melody is stored). Kept tiny on purpose.
STYLE_LABELS = {"희재": "ballad"}
SLOW_WORDS = ["느린", "느리게", "천천히", "slow"]
FAST_WORDS = ["빠른", "빠르게", "경쾌", "fast", "upbeat"]
MOOD_WORDS = {
    "energetic": ["신나는", "신나게", "energetic", "exciting"], "dark": ["어두운", "어둡", "dark", "우울"],
    "bright": ["밝은", "밝게", "bright", "cheerful"], "sad": ["슬픈", "슬프", "sad", "애절"],
    "calm": ["차분", "잔잔", "calm", "peaceful"], "warm": ["따뜻", "포근", "warm"],
}
INSTRUMENT_WORDS = {
    "piano": ["피아노", "piano"], "bass": ["베이스", "bass"], "drums": ["드럼", "drum"],
    "strings": ["스트링", "현악", "strings", "string"],
    "guitar": ["기타", "guitar", "일렉", "어쿠스틱"],
}
BAND_INSTRUMENTS = ["piano", "bass", "drums", "guitar"]       # "밴드": the default school-band line-up
NEGATION_AFTER = r"\s*(?:은|는|을|를)?\s*(?:없이|빼고|빼|제외|말고)"


@dataclass
class StyleAttrs:
    genre: str                       # one of GENRES
    tempo_min: int                   # TEMPO_MIN..TEMPO_MAX, tempo_min <= tempo_max
    tempo_max: int
    instruments: list[str]           # subset of arrange.INSTRUMENTS, app order, at least one
    mood: str                        # one of MOODS
    structure_hint: list[str] = field(default_factory=list)   # section kinds from song.TEMPLATES, may be empty
    source: str = ""                 # the user's original text

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- validation (code decides, LLM only proposes)
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _genre_from_word(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in GENRES:
        return v
    for g, words in GENRE_WORDS.items():
        if v == GENRE_KO[g] or v in words:
            return g
    return None


def _guess_genre(text: str) -> str | None:
    low = text.lower()
    hits = [(low.find(w), g) for g, words in GENRE_WORDS.items() for w in words if w in low]
    hits += [(low.find(k), g) for k, g in STYLE_LABELS.items() if k in low]
    return min(hits)[1] if hits else None


def validate_attrs(raw, source: str = "") -> tuple[StyleAttrs, list[str]]:
    """Clamp/repair a raw proposal (dict from an LLM or the rules) into StyleAttrs. Returns (attrs, notes)."""
    notes: list[str] = []
    if not isinstance(raw, dict):
        notes.append(f"제안이 객체가 아니라서 버렸어요({type(raw).__name__}): 기본값 사용")
        raw = {}

    genre = _genre_from_word(raw.get("genre"))
    if genre is None:
        genre = _guess_genre(source) or "ballad"
        notes.append(f"장르 {raw.get('genre')!r} 알 수 없음 -> {genre}로 대체")

    lo, hi = raw.get("tempo_min"), raw.get("tempo_max")
    if not (_is_num(lo) and _is_num(hi)):
        lo, hi = DEFAULT_TEMPO[genre]
        notes.append(f"템포 값이 숫자가 아님 -> {genre} 기본 {lo}~{hi}")
    else:
        lo_i, hi_i = int(round(lo)), int(round(hi))
        if lo_i != lo or hi_i != hi:
            notes.append("템포를 정수로 반올림")
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
            notes.append("템포 최소/최대가 뒤바뀌어 정렬")
        c_lo, c_hi = (min(max(v, TEMPO_MIN), TEMPO_MAX) for v in (lo_i, hi_i))
        if (c_lo, c_hi) != (lo_i, hi_i):
            notes.append(f"템포 {lo_i}~{hi_i}를 {TEMPO_MIN}~{TEMPO_MAX} 범위로 제한 -> {c_lo}~{c_hi}")
        lo, hi = c_lo, c_hi

    insts = raw.get("instruments")
    kept: list[str] = []
    if isinstance(insts, list):
        for i in insts:
            name = i.strip().lower() if isinstance(i, str) else None
            if name in INSTRUMENTS:
                if name not in kept:
                    kept.append(name)
            else:
                notes.append(f"알 수 없는 악기 {i!r} 제외")
    elif insts is not None:
        notes.append("악기 목록 형식이 잘못되어 무시")
    if not kept:
        kept = list(DEFAULT_INSTRUMENTS[genre])
        notes.append(f"악기가 비어 있음 -> {genre} 기본 편성 {kept}")
    kept = [i for i in INSTRUMENTS if i in kept]                 # app order

    mood = raw.get("mood")
    mood = mood.strip().lower() if isinstance(mood, str) else mood
    if mood not in MOODS:
        notes.append(f"분위기 {raw.get('mood')!r} 알 수 없음 -> {DEFAULT_MOOD[genre]}")
        mood = DEFAULT_MOOD[genre]

    hint_raw, hint = raw.get("structure_hint"), []
    if isinstance(hint_raw, list):
        for k in hint_raw:
            if isinstance(k, str) and k.strip().lower() in STRUCTURE_KINDS:
                hint.append(k.strip().lower())
            else:
                notes.append(f"알 수 없는 구조 요소 {k!r} 제외")
    elif hint_raw is not None:
        notes.append("구조 힌트 형식이 잘못되어 무시")

    return StyleAttrs(genre, int(lo), int(hi), kept, mood, hint, source), notes


# ---------------------------------------------------------------- confirmation question (code-generated)
def confirm_question(attrs: StyleAttrs) -> str:
    """One Korean sentence restating the validated attributes, e.g.
    "느린 피아노 발라드(템포 64~76), 스트링 포함으로 이해했어요. 맞나요?"."""
    if attrs.tempo_max <= 80:
        tempo_word = "느린 "
    elif attrs.tempo_min >= 120:
        tempo_word = "빠른 "
    else:
        tempo_word = ""
    mood_word = f"{MOOD_KO[attrs.mood]} " if attrs.mood != DEFAULT_MOOD[attrs.genre] else ""
    names = [INSTRUMENT_KO[i] for i in attrs.instruments]
    if attrs.instruments[0] == "piano":                     # "피아노 발라드(...), 스트링 포함"
        head = f"{tempo_word}{mood_word}{names[0]} {GENRE_KO[attrs.genre]}(템포 {attrs.tempo_min}~{attrs.tempo_max})"
        tail = f", {'·'.join(names[1:])} 포함" if len(names) > 1 else ""
    else:                                                   # "락(...), 베이스·드럼 편성"
        head = f"{tempo_word}{mood_word}{GENRE_KO[attrs.genre]}(템포 {attrs.tempo_min}~{attrs.tempo_max})"
        tail = f", {'·'.join(names)} 편성"
    if attrs.structure_hint:
        from .song import KO
        tail += f", 구조 {' - '.join(KO.get(k, k) for k in attrs.structure_hint)}"
    return f"{head}{tail}{_ro(head + tail)} 이해했어요. 맞나요?"


def _ro(text: str) -> str:
    """Korean particle 로/으로 chosen from the last character of `text` ("스타일로", "포함으로", "아웃트로로")."""
    last = text[-1]
    if not ("가" <= last <= "힣"):
        return " 스타일로"                                  # ends with ")" or a digit: avoid guessing the reading
    jong = (ord(last) - 0xAC00) % 28
    return "로" if jong in (0, 8) else "으로"


# ---------------------------------------------------------------- deterministic keyword fallback
def _negated(low: str, word: str) -> bool:
    w = re.escape(word)
    return bool(re.search(w + NEGATION_AFTER, low) or re.search(r"(?:without|no)\s+" + w, low))


def rule_proposal(text: str) -> tuple[dict, list[str]]:
    """Keyword table -> raw proposal dict (+ notes about what was defaulted)."""
    low, notes = text.lower(), []
    genre = _guess_genre(text)
    if genre is None:
        genre = "ballad"
        notes.append("장르 단서 없음 -> 발라드 기본값")
    slow, fast = any(w in low for w in SLOW_WORDS), any(w in low for w in FAST_WORDS)
    if slow and not fast:
        lo, hi = SLOW_RANGE
    elif fast and not slow:
        lo, hi = FAST_RANGE
    else:
        lo, hi = DEFAULT_TEMPO[genre]
        if slow and fast:
            notes.append("느리다/빠르다 단서가 함께 있어 기본 템포 사용")
    hits = [(low.find(w), m) for m, words in MOOD_WORDS.items() for w in words if w in low]
    mood = min(hits)[1] if hits else DEFAULT_MOOD[genre]
    insts = []
    for name, words in INSTRUMENT_WORDS.items():
        found = [w for w in words if w in low]
        if found and not any(_negated(low, w) for w in found):
            insts.append(name)
    if any(w in low for w in ("밴드", "band")):
        insts += [i for i in BAND_INSTRUMENTS if i not in insts]
    for name, words in INSTRUMENT_WORDS.items():                   # explicit "빼고" beats the band default
        if name in insts and any(_negated(low, w) for w in words):
            insts.remove(name)
    if not insts:
        insts = [i for i in DEFAULT_INSTRUMENTS[genre]
                 if not any(_negated(low, w) for w in INSTRUMENT_WORDS[i])]   # "드럼 없이" also trims the default set
        notes.append("악기 단서 없음 -> 장르 기본 편성" + ("(요청한 악기 제외)" if len(insts) < len(DEFAULT_INSTRUMENTS[genre]) else ""))
    return {"genre": genre, "tempo_min": lo, "tempo_max": hi, "instruments": insts, "mood": mood,
            "structure_hint": []}, notes


# ---------------------------------------------------------------- local LLM proposal (Ollama, structured output)
SCHEMA = {
    "type": "object",
    "properties": {
        "genre": {"type": "string", "enum": GENRES},
        "tempo_min": {"type": "integer"}, "tempo_max": {"type": "integer"},
        "instruments": {"type": "array", "items": {"type": "string", "enum": INSTRUMENTS}},
        "mood": {"type": "string", "enum": MOODS},
        "structure_hint": {"type": "array", "items": {"type": "string", "enum": STRUCTURE_KINDS}},
    },
    "required": ["genre", "tempo_min", "tempo_max", "instruments", "mood"],
}
SYSTEM = (
    "You convert a Korean or English music-style request into style attributes as JSON. "
    "If the request names an artist or a song, treat it ONLY as a style label: describe the typical genre, tempo range "
    "(BPM), instrumentation and mood of that style. Never recall or reproduce melody, chords or lyrics. "
    f"genre: one of {GENRES}. tempo_min/tempo_max: BPM integers in {TEMPO_MIN}..{TEMPO_MAX}, a narrow range of about 10-20. "
    f"instruments: 2-4 typical for that style, subset of {INSTRUMENTS}. mood: one of {MOODS}. "
    f"structure_hint: leave empty [] unless the user explicitly names song sections; allowed {STRUCTURE_KINDS}."
)


class OllamaStylePlanner:
    """callable(text) -> raw dict, via Ollama /api/chat with a JSON schema. Raises on any failure (caller falls back)."""

    def __init__(self, model: str | None = None, base_url: str = "http://127.0.0.1:11434",
                 timeout: float = LLM_TIMEOUT_S, client=None):
        import httpx
        from .llm import OLLAMA_KEEP_ALIVE, OLLAMA_MODEL
        self.model = model or os.environ.get("INSTMATE_OLLAMA_MODEL", OLLAMA_MODEL)
        self.url = base_url.rstrip("/") + "/api/chat"
        self.keep_alive = OLLAMA_KEEP_ALIVE
        self.client = client or httpx.Client(timeout=timeout)

    def __call__(self, text: str) -> dict:
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": self.keep_alive,
                "format": SCHEMA, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
        r = self.client.post(self.url, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama 응답 {r.status_code}")
        return json.loads((r.json().get("message") or {}).get("content") or "")


def _run_planner(planner, text: str, timeout_s: float):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(planner, text)
    try:
        return fut.result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False)                                    # a hung planner must not block the caller


def style_from_request(text: str, planner=None, timeout_s: float = LLM_TIMEOUT_S) -> dict:
    """See the module docstring. Never raises for LLM problems; ValueError only for an empty request."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("스타일 요청이 비어 있어요")
    text = text.strip()
    notes: list[str] = []
    raw, method = None, "rule"
    if planner is not False:
        try:
            raw = _run_planner(planner if planner is not None else OllamaStylePlanner(timeout=timeout_s),
                               text, timeout_s)
            method = "llm"
        except FutureTimeout:
            notes.append(f"LLM이 {timeout_s:g}초 안에 답하지 않아 규칙 방식으로 전환")
        except Exception as e:                                     # unavailable, bad JSON, planner bug
            notes.append(f"LLM 사용 불가({type(e).__name__}) -> 규칙 방식으로 전환")
    if method == "llm":
        attrs, vnotes = validate_attrs(raw, text)
        notes += vnotes
    else:
        raw, rnotes = rule_proposal(text)
        attrs, vnotes = validate_attrs(raw, text)
        notes += rnotes + vnotes
    return {"attrs": attrs.to_dict(), "confirm_question": confirm_question(attrs), "method": method, "notes": notes}


# ---------------------------------------------------------------- concrete generation settings
def _as_attrs(attrs) -> StyleAttrs:
    if isinstance(attrs, StyleAttrs):
        return attrs
    checked, _ = validate_attrs(attrs, attrs.get("source", "") if isinstance(attrs, dict) else "")
    return checked


def apply_style(attrs) -> dict:
    """StyleAttrs (or its dict) -> settings usable by the generation pipeline. See the module docstring."""
    a = _as_attrs(attrs)
    bpm = (a.tempo_min + a.tempo_max) // 2
    direction = MOOD_BIAS[a.mood]
    return {"genre": a.genre, "bpm": bpm, "tempo": bpm,
            "instruments": {i: i in a.instruments for i in INSTRUMENTS},
            "instrument_list": list(a.instruments), "mood": a.mood,
            "chord_bias": {"direction": direction, "strength": CHORD_BIAS_STRENGTH if direction else 0.0},
            "structure_hint": list(a.structure_hint)}


def chord_bias_for_key(key, hint: dict) -> dict[str, float]:
    """{roman: bonus} for chords.progression(..., bias=): +strength on every dark (or bright) chord of the key's vocabulary."""
    from . import chords as ch
    if not hint or not hint.get("direction"):
        return {}
    flag = "dark" if hint["direction"] == "dark" else "bright"
    return {c.roman: float(hint["strength"]) for c in ch.vocab(key) if getattr(c, flag)}
