"""LLM agent layer: a natural-language request -> tool calls -> checks -> next action (2차관문 3-3).

The LLM never writes music. It calls tools; each call is validated here, executed by the existing code on a
working copy, and its checked result is sent back so the next call can depend on it. Only a request that ends
with `finish` is adopted, as one version. A question, the budget, or an LLM error adopts nothing.

Code-enforced rules in run_request:
  * The LLM can never waive the clash check. `allow_clash` is not in the tool schema, and a call that sends it anyway
    is run without it (note "allow_clash 무시"). A failed check is not consent: when a modify_bars check fails
    (need_input / budget_exceeded), the request ends as need_input with nothing adopted and the failed proposal
    ({bars (0-based), mood, can_force, question}) in `pending`. Forcing the least-clashing chord happens only through
    an explicit user confirmation (app.py POST /api/agent/confirm), bound to the history, the base version, the bars
    and the mood of that proposal.
  * an ok modify_bars result says in Korean that the check passed and no confirmation is needed.
  * ask_user after at least one applied modify_bars and with no failed check in this request is not a question for the
    user (there is nothing to confirm): it returns status "redirected" telling the model to call finish. It counts
    against the call budget and is used once per request; a second ask_user ends the request as need_input.
  * the final message of an ok request starts with a line built from the real base-vs-final difference (`diff_summary`);
    the model's own summary follows as "모델 설명: ...". `changed` is that difference.
  * Time: one absolute deadline (DEADLINE_S) for the whole request. It is checked before and after every planner call,
    after every tool execution and before any success end (finish or a plain text answer). The planner's HTTP timeout
    is set to the remaining time (`set_timeout`). Over the deadline -> budget_exceeded with the base state and nothing
    adopted. A tool that is already running is not interrupted (the music tools are synchronous and short); the
    deadline only guarantees that a late result is never adopted.
  * Planner output is untrusted: its shape and argument types are checked before any dict method is used
    (`_clean_reply`, `_validate_call`). A malformed call gets a structured "rejected" result; a malformed reply is an
    llm_error. Nothing is adopted in either case unless the request still ends with a valid finish.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import agent
from .mix import LEVEL_MAX, LEVEL_MIN, LEVEL_TRACKS, TEMPO_MAX, TEMPO_MIN

MAX_TOOL_CALLS = 8
DEADLINE_S = 60.0                     # includes loading the local model into memory on a cold start
DEFAULT_MODEL = "gemini-3.8-flash"          # free tier on the pricing page (checked 2026-09-30)

TOOLS = [
    {"type": "function", "name": "modify_bars",
     "description": "선택한 마디들의 코드를 더 어둡게 또는 더 밝게 바꾼다. 조성 안의 코드만 쓰고, 멜로디와 부딪히는지 검사한 결과를 돌려준다.",
     "parameters": {"type": "object", "properties": {
         "bars": {"type": "array", "items": {"type": "integer"}, "description": "바꿀 마디 번호 (1부터)"},
         "mood": {"type": "string", "enum": ["darker", "brighter"]}},
         "required": ["bars", "mood"]}},
    {"type": "function", "name": "set_mix",
     "description": "템포(BPM)나 악기별 세기를 바꾼다. 코드와 음표는 바뀌지 않는다.",
     "parameters": {"type": "object", "properties": {
         "tempo": {"type": "integer", "description": f"{TEMPO_MIN}~{TEMPO_MAX}"},
         "levels": {"type": "object", "description": f"악기별 세기 배율 {LEVEL_MIN}~{LEVEL_MAX} (1은 그대로)",
                    "properties": {t: {"type": "number"} for t in LEVEL_TRACKS}}}}},
    {"type": "function", "name": "ask_user",
     "description": "요청이 모호하거나 도구로 할 수 없는 부분이 있거나 검사가 실패했을 때만 사용자에게 질문한다. "
                    "검사를 통과한 변경을 확인받으려고 쓰지 않는다. 이번 요청의 변경은 적용되지 않는다.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    {"type": "function", "name": "finish",
     "description": "요청을 끝낸다. 실제로 실행된 도구 결과만 근거로 무엇을 바꿨는지 요약한다.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

SYSTEM = (
    "너는 작곡메이트의 편곡 에이전트다. 사용자의 요청을 도구 호출로만 수행한다. 음표나 코드를 직접 지어내지 않는다.\n"
    "규칙: 마디 번호는 1부터. 도구 결과(status, 충돌률)를 보고 다음 행동을 정한다. "
    "modify_bars가 status ok를 돌려주면 충돌 검사를 통과한 것이라 사용자 확인이 필요 없다. 통과한 변경을 확인받으려고 "
    "ask_user를 부르지 않는다. ask_user는 검사가 실패했을 때(need_input, budget_exceeded), 요청이 모호할 때, "
    "도구로 할 수 없는 요청(가수 목소리, 새 악기, 멜로디 변경 등)일 때만 쓰고, 할 수 없는 것은 그렇게 알리며 가능한 대안을 제시한다. "
    "충돌 검사를 건너뛰는 도구나 인자는 없다. 검사가 실패하면 사용자에게 알리고 끝낸다(적용 여부는 사용자가 버튼으로 정한다). "
    "모든 변경이 끝나면 finish로 실제 결과만 요약한다."
)


def _chat_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}} for tool in tools]


class LLMError(RuntimeError):
    pass


def _http_timeout(default: float, call_timeout: float | None) -> float:
    """The HTTP timeout of one planner call: the planner's own limit, cut to the time left in the request budget."""
    return default if call_timeout is None else max(0.001, min(default, call_timeout))


def load_api_key(root: Path) -> str | None:
    """GEMINI_API_KEY from the environment, else from <root>/.env (never committed, never copied to the snapshot)."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = root / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "GEMINI_API_KEY" and v.strip():
                return v.strip().strip('"').strip("'")
    return None


class GeminiPlanner:
    """Gemini Interactions API (POST /v1beta/interactions). Conversation state is chained by interaction id."""

    URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 30.0, client=None):
        import httpx
        self.api_key, self.model, self.prev = api_key, model, None
        self.client = client or httpx.Client(timeout=timeout)
        self.tools = TOOLS
        self.timeout, self.call_timeout = timeout, None

    def set_timeout(self, seconds: float | None) -> None:
        """Time left in the request's overall budget; the next HTTP call never waits longer than that."""
        self.call_timeout = seconds

    def _post(self, body: dict) -> dict:
        body = {"model": self.model, "tools": self.tools, "system_instruction": SYSTEM, **body}
        if self.prev:
            body["previous_interaction_id"] = self.prev
        try:
            r = self.client.post(self.URL, headers={"x-goog-api-key": self.api_key}, json=body,
                                 timeout=_http_timeout(self.timeout, self.call_timeout))
        except Exception as e:                                   # network, timeout
            raise LLMError(f"Gemini 요청 실패: {type(e).__name__}") from None
        if r.status_code != 200:
            raise LLMError(f"Gemini 응답 {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except ValueError:
            raise LLMError("Gemini 응답이 JSON이 아니에요") from None
        if not isinstance(data, dict):
            raise LLMError("Gemini 응답이 객체가 아니에요")
        self.prev = data.get("id") or self.prev
        return parse_steps(data)

    def start(self, text: str) -> dict:
        return self._post({"input": text})

    def send_results(self, results: list[dict]) -> dict:
        return self._post({"input": [{"type": "function_result", "name": r["name"], "call_id": r["call_id"],
                                      "result": [{"type": "text", "text": json.dumps(r["result"], ensure_ascii=False)}]}
                                     for r in results]})


OLLAMA_URL = os.environ.get("INSTMATE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
# Apache-2.0, tool calling in Ollama. Measured 2026-09-30 on this Mac: qwen3:14b answered in 5.8-20.5 s with correct
# tool calls; qwen3:4b wrote 3-5k chars of reasoning before each call despite think=false (22-105 s), so 14b is default.
OLLAMA_MODEL = os.environ.get("INSTMATE_OLLAMA_MODEL", "qwen3:14b")
OLLAMA_KEEP_ALIVE = os.environ.get("INSTMATE_OLLAMA_KEEP_ALIVE", "30m")  # isolated low-memory servers can unload after a reply


def unload(client=None) -> None:
    """Free qwen's memory now (keep_alive 0) so YuE2 has the RAM; a stopped Ollama has nothing to free."""
    import httpx
    try:
        (client or httpx).post(OLLAMA_URL + "/api/generate", json={"model": OLLAMA_MODEL, "keep_alive": 0}, timeout=10.0)
    except Exception:
        pass


class OllamaPlanner:
    """Local LLM through Ollama's /api/chat with tools. Stateless API: the message history is kept here."""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL, timeout: float = DEADLINE_S, client=None):
        import httpx
        self.model, self.url = model, base_url.rstrip("/") + "/api/chat"
        self.client = client or httpx.Client(timeout=timeout)
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.tools = _chat_tools(TOOLS)
        self.timeout, self.call_timeout = timeout, None

    def set_timeout(self, seconds: float | None) -> None:
        """Time left in the request's overall budget; the next HTTP call never waits longer than that."""
        self.call_timeout = seconds

    def _chat(self) -> dict:
        body = {"model": self.model, "messages": self.messages, "tools": self.tools, "stream": False,
                "think": False, "keep_alive": OLLAMA_KEEP_ALIVE, "options": {"temperature": 0}}
        try:
            r = self.client.post(self.url, json=body, timeout=_http_timeout(self.timeout, self.call_timeout))
        except Exception as e:
            raise LLMError(f"Ollama 요청 실패: {type(e).__name__} (ollama serve 실행 중인지 확인)") from None
        if r.status_code != 200:
            raise LLMError(f"Ollama 응답 {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except ValueError:
            raise LLMError("Ollama 응답이 JSON이 아니에요") from None
        if not isinstance(data, dict):
            raise LLMError("Ollama 응답이 객체가 아니에요")
        msg = data.get("message")
        msg = {} if msg is None else msg
        if not isinstance(msg, dict):
            raise LLMError("Ollama 응답의 message가 객체가 아니에요")
        self.messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})
        raw_calls = msg.get("tool_calls")
        raw_calls = [] if raw_calls is None else raw_calls
        if not isinstance(raw_calls, list):
            raise LLMError("Ollama 응답의 tool_calls가 배열이 아니에요")
        calls = []
        for i, tc in enumerate(raw_calls):
            if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                calls.append(tc)                           # kept as is: run_request turns it into a structured rejection
                continue
            fn = tc["function"]
            args = fn["arguments"] if "arguments" in fn else {}      # a missing field is {}; null/list/number stay as sent
            if isinstance(args, str):                      # some models return a JSON string
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_unparsed": args}
            calls.append({"id": f"{fn.get('name')}-{len(self.messages)}-{i}", "name": fn.get("name"), "args": args})
        content = msg.get("content")
        return {"calls": calls, "text": content.strip() if isinstance(content, str) else content}

    def start(self, text: str) -> dict:
        self.messages.append({"role": "user", "content": text})
        return self._chat()

    def send_results(self, results: list[dict]) -> dict:
        for r in results:
            self.messages.append({"role": "tool", "tool_name": r["name"],
                                  "content": json.dumps(r["result"], ensure_ascii=False)})
        return self._chat()


def ollama_status(base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, client=None) -> str | None:
    """None if Ollama is running and has `model`; otherwise a message for the user."""
    import httpx
    try:
        r = (client or httpx).get(base_url.rstrip("/") + "/api/tags", timeout=2.0)
        names = {m.get("name") for m in r.json().get("models", [])}
    except Exception:
        return ("글로 요청하는 기능은 이 컴퓨터의 언어 모델(Ollama)이 켜져 있어야 해요(터미널에서 `ollama serve`). "
                "꺼져 있어도 아래 마디별 \"어둡게 / 밝게\" 버튼과 믹스 조절은 그대로 쓸 수 있어요.")
    if model not in names:
        return f"로컬 모델 {model}이(가) 없어요. `ollama pull {model}`로 받아 주세요."
    return None


def parse_steps(data: dict) -> dict:
    """{"calls": [{"id", "name", "args"}], "text": str} from an Interactions API response. Argument types are not
    judged here (run_request does); only the container shapes are, so that iterating them cannot raise."""
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        raise LLMError("Gemini 응답의 steps가 배열이 아니에요")
    calls, texts = [], []
    for s in steps:
        if not isinstance(s, dict):
            raise LLMError("Gemini 응답의 step이 객체가 아니에요")
        if s.get("type") == "function_call":
            calls.append({"id": s.get("id"), "name": s.get("name"), "args": s["arguments"] if "arguments" in s else {}})
        elif s.get("type") == "model_output":
            content = s.get("content", [])
            if not isinstance(content, list):
                raise LLMError("Gemini 응답의 content가 배열이 아니에요")
            texts += [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
    return {"calls": calls, "text": "\n".join(t for t in texts if isinstance(t, str) and t)}


# ---------------------------------------------------------------- tool execution on a working copy
def _describe(state: dict, result: dict) -> str:
    chords = " | ".join(f"{c['bar'] + 1}:{c['symbol']}({c['roman']})" for c in result["chords"])
    clash = ", ".join(f"{i + 1}:{r}" for i, r in enumerate(result["check"]["clash_rates"]))
    return (f"조: {result['key']['label']}, 장르: {state['genre']}, 템포: {result['tempo']}, "
            f"세기: {json.dumps(result['levels'], ensure_ascii=False)}\n마디 수: {len(result['chords'])}\n"
            f"코드: {chords}\n마디별 충돌률(기준 {result['check']['threshold']} 이하): {clash}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _clean_reply(reply) -> dict:
    """The planner's reply as {"calls": [{"id", "name", "args", "error"?}], "text": str}. The reply is untrusted JSON:
    a reply that is not an object, whose `calls` is not an array or whose `text` is not a string raises LLMError;
    a single bad call (not an object, no string name, args present but not an object) is kept with an `error` so that
    run_request answers it with a structured "rejected" result instead of crashing. A missing `args` means {}."""
    if not isinstance(reply, dict):
        raise LLMError(f"planner 응답이 객체가 아니에요 ({type(reply).__name__})")
    raw = reply.get("calls")
    raw = [] if raw is None else raw
    if not isinstance(raw, list):
        raise LLMError(f"planner 응답의 calls가 배열이 아니에요 ({type(raw).__name__})")
    text = reply.get("text")
    text = "" if text is None else text
    if not isinstance(text, str):
        raise LLMError(f"planner 응답의 text가 문자열이 아니에요 ({type(text).__name__})")
    calls = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            calls.append({"id": f"bad-{i}", "name": "", "args": {}, "error": f"도구 호출은 객체여야 해요 ({type(c).__name__})"})
            continue
        cid = c.get("id")
        cid = cid if isinstance(cid, str) and cid else f"call-{i}"
        name = c.get("name")
        args = c["args"] if "args" in c else {}
        if not isinstance(name, str) or not name:
            calls.append({"id": cid, "name": "", "args": {}, "error": "도구 이름(name)이 문자열이 아니에요"})
        elif not isinstance(args, dict):
            calls.append({"id": cid, "name": name, "args": {},
                          "error": f"args는 JSON 객체여야 해요 ({type(args).__name__}로 받았어요)"})
        else:
            calls.append({"id": cid, "name": name, "args": args})
    return {"calls": calls, "text": text}


def _validate_call(name: str, args, n_bars: int) -> str | None:
    """Error text for the model, or None. The existing tools validate again; this gives the model a clear reason.
    Types are checked before any dict method is used (args may be anything the model sent)."""
    if not isinstance(args, dict):
        return f"args는 JSON 객체여야 해요 ({type(args).__name__}로 받았어요)"
    if name == "modify_bars":
        bars, mood = args.get("bars"), args.get("mood")
        if not isinstance(bars, list) or not bars or not all(_is_int(b) for b in bars):
            return "bars는 1개 이상의 정수 목록이어야 해요"
        if not all(1 <= b <= n_bars for b in bars):
            return f"마디 번호는 1~{n_bars}예요"
        if mood not in ("darker", "brighter"):
            return "mood는 darker 또는 brighter예요"
    elif name == "set_mix":
        if "tempo" not in args and "levels" not in args:
            return "tempo나 levels 중 하나는 있어야 해요"
        if "tempo" in args and not _is_int(args["tempo"]):
            return f"tempo는 {TEMPO_MIN}~{TEMPO_MAX} 정수예요"
        if "levels" in args:
            lv = args["levels"]
            if not isinstance(lv, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
                                                   for k, v in lv.items()):
                return "levels는 {악기: 숫자} 객체예요"
    elif name == "ask_user":
        q = args.get("question")
        if not isinstance(q, str) or not q.strip():
            return "question은 비어 있지 않은 문자열이어야 해요"
    elif name == "finish":
        if "summary" in args and args["summary"] is not None and not isinstance(args["summary"], str):
            return "summary는 문자열이에요"
    else:
        return f"없는 도구예요: {name}"
    return None


def _modify(state: dict, result: dict, args: dict):
    """modify_bars on the working copy. allow_clash is never passed: the LLM cannot waive the check."""
    bars = [b - 1 for b in args["bars"]]
    r = agent.modify(state["tokens"], state["input_tonic"], state["genre"], state["instruments"],
                     state["key_choice"], state["chords"], bars, args["mood"], state["energy"], None,
                     base_events=result["events"], allow_clash=False,
                     tempo=result["tempo"], levels=result["levels"])
    if r["status"] not in agent.ADOPTABLE:
        return None, {"status": r["status"], "question": r["question"], "can_force": r["can_force"]}
    new_state = {**state, "chords": [c["roman"] for c in r["chords"]]}
    th = r["check"]["threshold"]
    msg = (f"검사 통과: 바꾼 마디의 충돌률이 모두 기준 {th} 이하예요. 사용자 확인은 필요 없어요. "
           "더 바꿀 것이 없으면 finish를 호출하세요.")
    return (new_state, r), {"status": r["status"], "message": msg,
                            "changed": {f"{b + 1}마디": [f, t] for b, f, t in
                                        zip(r["change"]["bars"], r["change"]["from_all"], r["change"]["to_all"])},
                            "clash_rates": r["check"]["clash_rates"]}


def _mix(state: dict, result: dict, args: dict):
    r = agent.remix(state, result, args.get("tempo"), args.get("levels"))
    return (state, r), {"status": "ok", "changed": r["tool_result"]["changed"]}


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def diff_summary(base: dict, final: dict) -> tuple[str, bool]:
    """(Korean text, changed) built only from the base and final results: chords (old -> new), tempo, levels."""
    parts = [f"{i + 1}마디 {a['symbol']}({a['roman']}) -> {b['symbol']}({b['roman']})"
             for i, (a, b) in enumerate(zip(base["chords"], final["chords"])) if a["roman"] != b["roman"]]
    bits = []
    if parts:
        bits.append("코드 " + ", ".join(parts))
    if base["tempo"] != final["tempo"]:
        bits.append(f"템포 {base['tempo']} -> {final['tempo']}")
    lv = [f"{t} {_fmt(base['levels'][t])} -> {_fmt(final['levels'][t])}"
          for t in final["levels"] if final["levels"][t] != base["levels"].get(t)]
    if lv:
        bits.append("세기 " + ", ".join(lv))
    if not bits:
        return "실제 변경: 없음 (곡이 요청 전과 같아요)", False
    return "실제 변경: " + "; ".join(bits), True


def run_request(planner, state: dict, result: dict, request: str,
                max_calls: int = MAX_TOOL_CALLS, deadline_s: float = DEADLINE_S, clock=time.monotonic) -> dict:
    """Drive the planner over tools.
    Returns {"status", "message", "log", "state", "result", "changed", "pending"}.

    status: ok (finished; changed or not) | need_input (the model asked, or a check failed) | budget_exceeded | llm_error.
    Only "ok" with changed=True should be adopted; the returned state/result are the final working copy.
    For "ok" the message starts with a line built by code from the actual base-vs-final difference (diff_summary), and
    `changed` is that difference (a request whose edits cancel out is not a change). The model's own text is an optional
    second line, labelled as the model's; it can never replace or contradict the first line.
    `pending` (need_input only): the modify_bars proposals whose check failed and that the user may force:
    [{"bars": 0-based list as asked, "mood", "can_force": True, "question"}]. The caller stores them against the base
    version; only an explicit user confirmation may apply them (the LLM can not).
    A check that failed in this request always ends it as need_input, even if the model then calls finish or answers in
    text, unless a later successful modify_bars covered the same bars.
    """
    start = clock()
    deadline = start + deadline_s
    work_state, work_result = state, result
    log, calls = [], 0
    blocked_bars: set[int] = set()        # bars whose modify ended in need_input/budget_exceeded in this request
    failed: list[dict] = []               # modify_bars proposals whose check failed and no later successful edit covered
    modify_ok = 0                         # modify_bars calls that passed the check and were applied to the working copy
    redirected = False                    # the passed-check ask_user redirect is used at most once per request
    set_timeout = getattr(planner, "set_timeout", None)

    def done(status, message, with_pending=False):
        pend = [dict(f) for f in failed if f["can_force"]] if with_pending else []
        if status == "ok":
            summary, diff = diff_summary(result, work_result)
            note = (message or "").strip()
            message = summary + (f"\n모델 설명: {note}" if note else "")
            return {"status": status, "message": message, "model_note": note, "log": log, "changed": diff,
                    "state": work_state, "result": work_result, "pending": []}
        return {"status": status, "message": message, "model_note": message, "log": log, "changed": False,
                "state": state, "result": result, "pending": pend}     # nothing is adopted unless the request finished ok

    def over_budget():
        return done("budget_exceeded", f"도구 호출 {max_calls}회 또는 {int(deadline_s)}초 예산을 넘어 멈췄어요. "
                                       "이번 요청의 변경은 적용하지 않았어요.")

    def ask_planner(fn, *a):
        """(reply, None) or (None, final result): the deadline is checked before and after the call, the HTTP timeout is
        the time left, and the reply is shape-checked."""
        left = deadline - clock()
        if left <= 0:
            return None, over_budget()
        if set_timeout:
            set_timeout(left)
        try:
            reply = fn(*a)
        except LLMError as e:
            return None, (over_budget() if clock() > deadline else done("llm_error", str(e)))
        if clock() > deadline:
            return None, over_budget()
        try:
            return _clean_reply(reply), None
        except LLMError as e:
            return None, done("llm_error", str(e))

    def finish_ok(text):
        """Every success end goes through here: late -> budget_exceeded; a failed check not resolved -> need_input."""
        if clock() > deadline:
            return over_budget()
        if failed:
            return done("need_input", " ".join(f["question"] for f in failed)
                        + " 이번 요청의 변경은 아직 아무것도 적용하지 않았어요.", with_pending=True)
        return done("ok", text)

    reply, early = ask_planner(planner.start, f"현재 곡 상태:\n{_describe(state, result)}\n\n사용자 요청: {request}")
    if early:
        return early
    while True:
        if not reply["calls"]:                                   # plain text answer: treat as finish
            return finish_ok(reply["text"])
        results = []
        for call in reply["calls"]:
            name, args = call["name"], call["args"]
            if calls >= max_calls or clock() > deadline:
                log.append({"tool": name, "args": args, "result": {"status": "budget_exceeded"}})
                return over_budget()
            calls += 1
            err = call.get("error") or _validate_call(name, args, len(work_result["chords"]))
            if err:
                out = {"status": "rejected", "error": err}
            elif name == "ask_user" and modify_ok and not blocked_bars and not redirected:
                # every change passed its check and no check failed in this request, so there is nothing to confirm:
                # a question here (clash/검사 confirmation) must not end the request as need_input. Once per request.
                redirected = True
                out = {"status": "redirected", "message": "이번 요청의 변경은 모두 충돌 검사를 통과했고 검사 실패는 없었어요. "
                       "사용자 확인이 필요 없어서 질문을 전달하지 않았어요. 변경을 적용하려면 finish를 호출하세요."}
            elif name == "ask_user":
                log.append({"tool": name, "args": args, "result": {"status": "need_input"}})
                return done("need_input", args["question"], with_pending=True)
            elif name == "finish":
                end = finish_ok(args.get("summary") or "")
                log.append({"tool": name, "args": args, "result": {"status": end["status"]}})
                return end
            else:
                note = None
                if name == "modify_bars" and "allow_clash" in args:
                    # the model may never waive the check: the argument is not in the schema and is dropped
                    args = {k: v for k, v in args.items() if k != "allow_clash"}
                    note = "allow_clash 무시: 충돌 검사는 모델이 건너뛸 수 없어요. 검사가 실패하면 사용자가 직접 결정해요"
                try:
                    new, out = (_modify if name == "modify_bars" else _mix)(work_state, work_result, args)
                except ValueError as e:                          # invalid values caught by the music tools
                    new, out = None, {"status": "rejected", "error": str(e)}
                if new:
                    work_state, work_result = new
                if name == "modify_bars" and out.get("status") in ("need_input", "budget_exceeded"):
                    blocked_bars |= set(args["bars"])
                    entry = {"bars": sorted(b - 1 for b in args["bars"]), "mood": args["mood"],
                             "can_force": bool(out.get("can_force")), "question": out["question"]}
                    if all((f["bars"], f["mood"]) != (entry["bars"], entry["mood"]) for f in failed):
                        failed.append(entry)
                elif name == "modify_bars" and out.get("status") in agent.ADOPTABLE:
                    modify_ok += 1
                    covered = {b - 1 for b in args["bars"]}
                    failed[:] = [f for f in failed if not set(f["bars"]) <= covered]   # a later successful edit resolves it
                if note:
                    out = {**out, "note": note}
                if clock() > deadline:                           # a tool that ran past the deadline: its result is not adopted
                    log.append({"tool": name, "args": args, "result": out})
                    return over_budget()
            log.append({"tool": name, "args": args, "result": out})
            results.append({"name": name, "call_id": call["id"], "result": out})
        reply, early = ask_planner(planner.send_results, results)
        if early:
            return early
