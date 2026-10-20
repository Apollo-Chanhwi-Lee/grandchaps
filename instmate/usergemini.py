"""A visitor's own Gemini key (user decision 2026-10-19, option 2: each person uses their own free quota).

The page sends the key in the `X-Gemini-Key` header of one request; it is used for that request only and is never
stored, logged or put in a response.  Any failure (limit reached, network, bad key, odd answer) falls back to the local
model (qwen) or the plain rules, and `events` collects one line the page shows ("qwen local llm으로 조정됩니다").
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

from . import llm as _llm

KEY_RE = re.compile(r"^[A-Za-z0-9._-]{20,128}$")      # newer keys start "AQ." and hold a dot
# gemini-3.8-flash thinks at "medium" by default and took 52 s for a greeting (measured 2026-10-20); the flash-lite model is
# free, answers at "minimal" thinking by default and is fast enough for reading one sentence (ai.google.dev model + thinking pages)
MODEL = os.environ.get("INSTMATE_USER_GEMINI_MODEL", "gemini-3.5-flash-lite")
THINKING = "minimal"
GEMINI_TIMEOUT_S = 10.0             # Netlify's proxy cuts a request at 26 s with an empty body: Gemini + the local fallback must fit
COOLDOWN_S = 300.0                  # after a 429 this key is not tried again for a while (its free quota refills over time)
_cool: dict[str, float] = {}        # sha256(key)[:16] -> time before which Gemini is skipped (the key itself is not kept)
NOTE_LIMIT = "Gemini 무료 한도를 다 써서 qwen local llm으로 조정됩니다."
NOTE_FAIL = "Gemini에 연결하지 못해({why}) qwen local llm(또는 기본 규칙)으로 조정됩니다."   # why: an HTTP status or error name, never the key


def valid(key) -> bool:
    return isinstance(key, str) and bool(KEY_RE.match(key))


def _id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ask(key: str, system: str, user: str, timeout: float = 20.0, client=None) -> str:
    """One Gemini answer as text (Interactions API, like llm.GeminiPlanner).  Raises llm.LLMError."""
    import httpx
    try:
        r = (client or httpx).post(_llm.GeminiPlanner.URL, headers={"x-goog-api-key": key}, timeout=timeout,
                                   json={"model": MODEL, "system_instruction": system, "input": user,
                                         "generation_config": {"thinking_level": THINKING}})
    except Exception as e:
        raise _llm.LLMError(f"Gemini 요청 실패: {type(e).__name__}") from None
    if r.status_code != 200:
        raise _llm.LLMError(f"Gemini 응답 {r.status_code}")       # the body is not copied: it can echo request details
    try:
        return _llm.parse_steps(r.json())["text"]
    except (ValueError, AttributeError):
        raise _llm.LLMError("Gemini 응답을 읽지 못했어요") from None


def try_ask(key, events: list | None, system: str, user: str, timeout: float = 20.0, client=None) -> str | None:
    """Gemini's text, or None when there is no (usable) key, the key is cooling down, or the call failed; a failure adds
    one note to `events`.  The caller then uses the local model."""
    if not valid(key):
        return None
    if time.time() < _cool.get(_id(key), 0.0):
        return None
    try:
        got = ask(key, system, user, min(timeout, GEMINI_TIMEOUT_S), client)
    except _llm.LLMError as e:
        limited = " 429" in str(e)
        if limited:
            _cool[_id(key)] = time.time() + COOLDOWN_S
        if events is not None and not events:
            events.append(NOTE_LIMIT if limited else NOTE_FAIL.format(why=str(e).replace("Gemini ", "")[:60]))
        return None
    return got or None


def json_object(text: str) -> dict:
    """The first {...} of an answer that may wrap it in words or a code fence; ValueError when there is none."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object")
    got = json.loads(text[a:b + 1])
    if not isinstance(got, dict):
        raise ValueError("not an object")
    return got
