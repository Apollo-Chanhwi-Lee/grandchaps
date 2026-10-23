"""Constrained workspace assistant. A reply proposes settings; it never runs generation."""
from __future__ import annotations

import re

CHOICES = {
    "melody_role": ("vocal", "lead_guitar", "acoustic_guitar", "keys"),
    "form": ("standard", "chorus_first"),
    "seed_role": ("chorus", "verse", "intro"),
    "melody_source": ("ai", "own", "rest"),
    "intro_player": ("guitar", "vocal"),
}
FLAGS = ("fit_voice", "key_change", "melody_shape")
SETTING_KEYS = {"tempo", "genre", "voice", "lyrics_mode", "key_shift", *CHOICES, *FLAGS}


def ollama_plan(text: str, context: dict, history: list, client=None, key: str | None = None, events: list | None = None) -> dict:
    """Local JSON planner with bounded context, no tools/execution, and no resident model."""
    import json
    import os
    import httpx
    from .chords import GENRES
    from .yuesong import VOICES
    from .llm import OLLAMA_MODEL, OLLAMA_URL
    model = os.environ.get("INSTMATE_WORKSPACE_MODEL", OLLAMA_MODEL)
    url = os.environ.get("INSTMATE_WORKSPACE_OLLAMA_URL", OLLAMA_URL).rstrip("/")
    fields = {
        "tempo": {"type": ["integer", "null"], "minimum": 40, "maximum": 200},
        "genre": {"type": ["string", "null"], "enum": [*GENRES, None]},
        "voice": {"type": ["string", "null"], "enum": [*VOICES, None]},
        "lyrics_mode": {"type": ["string", "null"], "enum": ["ah", "model", None]},
        "key_shift": {"type": ["integer", "null"], "minimum": -4, "maximum": 0},
    }
    fields.update({key: {"type": ["string", "null"], "enum": [*values, None]} for key, values in CHOICES.items()})
    fields.update({key: {"type": ["boolean", "null"]} for key in FLAGS})
    schema = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    voice_names = {key: value["label"] for key, value in VOICES.items()}
    from .chat import GENRE_LABEL
    genre_names = {g: GENRE_LABEL[g] for g in GENRES if g in GENRE_LABEL}
    system = (
        "너는 작곡메이트의 설정 추천 담당이다. 사용자가 원하는 변경안 또는 느낌에 어울리는 목표 설정을 JSON으로 제안한다. "
        "제안과 실제 적용은 다르다: 아직 적용하지 말라는 말은 변경안을 제안하지 말라는 뜻이 아니다. "
        f"출력은 {list(fields)} 필드를 가진 JSON 객체다. "
        "제안할 필드는 목표 값을, 유지하거나 관련 없는 필드는 null을 쓴다. 자유로운 설명이나 실행 결과는 쓰지 않는다. "
        f"genre는 {list(GENRES)} 중에서 선택. 장르 이름: {genre_names}. 말에 나온 이름과 정확히 같은 것을 고른다"
        "(감성록은 emo_rock이지 rock이 아니고, 모던록은 modern_rock, 하드록은 hard_rock, 피아노 록은 piano_rock이다). "
        f"목소리 의미: {voice_names}. "
        "melody_role은 멜로디 담당(vocal 보컬, lead_guitar 일렉 기타, acoustic_guitar 어쿠스틱 기타, keys 피아노). "
        "form은 standard 기본 구성, chorus_first 후렴 먼저. seed_role은 입력 멜로디의 위치(chorus 후렴, verse 벌스, intro 인트로). "
        "melody_source는 ai 모델 제안, own 규칙 변형, rest 나머지 비우기. verse/intro 위치에는 ai가 불가능하므로 own을 함께 제안한다. "
        "intro_player는 인트로 담당 guitar/vocal. fit_voice는 보컬 음역으로 조 옮기기, key_change는 마지막 후렴 전조, melody_shape는 새 구간 다듬기다. "
        "예: 현재 jpop 148 BPM 남성이며 잔잔한 발라드와 따뜻한 여성, 82 BPM을 원하면 "
        '{"tempo":82,"genre":"ballad","voice":"mezzo","lyrics_mode":null,"key_shift":null}. '
        "음표 수정·파일 처리·곡 생성은 할 수 없으므로 그런 요청에는 모든 필드를 null로 둔다. "
        "질문뿐이거나 설정 변경을 원하지 않으면 모든 필드를 null로 둔다. "
        "이 객체는 실제 실행이 아닌 확인 대기 중인 변경안이며, 음원이나 음표를 바꾸지 않는다. /no_think")
    messages = [{"role": "system", "content": system}]
    for item in history[-6:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:500]})
    messages.append({"role": "user", "content": json.dumps({"current_settings": context, "request": text}, ensure_ascii=False)})
    from . import usergemini as ug
    proposed, via = None, "ollama"
    said = ug.try_ask(key, events, system, messages[-1]["content"] if len(messages) < 3 else json.dumps(messages[1:], ensure_ascii=False), 30.0)
    if said is not None:
        try:
            got = ug.json_object(said)
            proposed = got if set(got) == set(fields) else None     # a partial answer is not adopted: the local model asks again
            via = "gemini" if proposed else via
        except ValueError:
            pass
    if proposed is None:
        response = (client or httpx).post(url + "/api/chat", timeout=40.0, json={
            "model": model, "messages": messages, "stream": False, "think": False, "format": schema,
            "keep_alive": 0, "options": {"temperature": 0, "num_predict": 256, "num_ctx": 4096}})
        response.raise_for_status()
        message = response.json().get("message", {})
        proposed = json.loads(message.get("content", ""))
    if not isinstance(proposed, dict) or set(proposed) != set(fields):
        raise ValueError("Model did not return the complete proposal schema")
    from .chat import read_text
    said = read_text(text).get("genre")                       # the rules read the named genre exactly; a model that kept only part of
    given = proposed.get("genre")                              # the name (감성록 -> 록) is corrected, nothing else is overridden
    if said and given and given != said and GENRE_LABEL.get(said, "").endswith(GENRE_LABEL.get(given, "\0")):
        proposed["genre"] = said
    proposed_patch = validate_patch({key: value for key, value in proposed.items() if value is not None})
    names = {"tempo": "템포", "genre": "장르", "voice": "보컬", "lyrics_mode": "가사 방식", "key_shift": "키 낮춤",
             "melody_role": "멜로디 담당", "form": "곡 구성", "seed_role": "입력 위치", "melody_source": "나머지 멜로디",
             "intro_player": "인트로 담당", "fit_voice": "보컬 음역 맞춤", "key_change": "마지막 후렴 전조", "melody_shape": "새 구간 다듬기"}
    values = [f"{names[key]}: {voice_names[value] if key == 'voice' else value}" for key, value in proposed_patch.items()]
    reply = "설정 변경안(미적용): " + " · ".join(values) if values else "바꿀 설정을 찾지 못했어요. 장르·템포(BPM)·보컬처럼 바꾸고 싶은 것을 말해 주세요. 예: \"82 BPM 잔잔한 발라드로\""
    # Execution acknowledgements come from code after adoption, never from model-written narration.
    return {"reply": reply, "patch": proposed_patch, "via": via}


def connection_status(client=None) -> dict:
    """Discovery only: do not start/load a model on a page visit."""
    import os
    import httpx
    from .llm import OLLAMA_MODEL, OLLAMA_URL
    model = os.environ.get("INSTMATE_WORKSPACE_MODEL", OLLAMA_MODEL)
    url = os.environ.get("INSTMATE_WORKSPACE_OLLAMA_URL", OLLAMA_URL).rstrip("/")
    status = {"available": False, "provider": "ollama", "model": model, "reason": None}
    if os.environ.get("INSTMATE_WORKSPACE_LLM", "auto").lower() == "off":
        return {**status, "reason": "언어 모델은 꺼져 있습니다. 빠른 설정 수정은 가능합니다."}
    try:
        response = (client or httpx).get(url + "/api/tags", timeout=2.0)
        response.raise_for_status()
        names = {m.get("name") for m in response.json().get("models", []) if isinstance(m, dict)}
        if model not in names:
            return {**status, "reason": f"로컬 언어 모델 {model}이 없습니다. 빠른 설정 수정은 가능합니다."}
        return {**status, "available": True}
    except Exception:
        return {**status, "reason": "로컬 언어 모델 서버에 연결되지 않았습니다. 빠른 설정 수정은 가능합니다."}


def validate_patch(patch: dict) -> dict:
    """All-or-nothing allowlist; model output has no execution or melody authority."""
    from .chords import GENRES
    from .yuesong import VOICES
    if not isinstance(patch, dict) or set(patch) - SETTING_KEYS:
        raise ValueError("Unsupported patch")
    for key, value in patch.items():
        if key in CHOICES and (not isinstance(value, str) or value not in CHOICES[key]):
            raise ValueError("Invalid choice")
        if key in FLAGS and type(value) is not bool:
            raise ValueError("Invalid flag")
        if key == "tempo" and (type(value) is not int or not 40 <= value <= 200):
            raise ValueError("Invalid tempo")
        if key == "key_shift" and (type(value) is not int or value not in (0, -1, -2, -3, -4)):
            raise ValueError("Invalid key shift")
        if key == "genre" and (not isinstance(value, str) or value not in GENRES):
            raise ValueError("Invalid genre")
        if key == "voice" and (not isinstance(value, str) or value not in VOICES):
            raise ValueError("Invalid voice")
        if key == "lyrics_mode" and value not in ("ah", "model"):
            raise ValueError("Invalid lyrics mode")
    return dict(patch)


def _tempo_command(text: str, current: int) -> dict | None:
    """Only complete affirmative command forms get immediate authority; other language goes to proposals."""
    clean = text.strip().lower()
    if re.search(r"(?:바꾸지|올리지|내리지|변경하지|하지)\s*마|don't\s+change|do\s+not\s+change", clean):
        return {"reply": "설정을 변경하지 않겠습니다.", "patch": {}, "confirmation_required": False, "source": "rules"}
    name = r"(?:bpm|템포)\s*(?:을|를|은|는)?\s*"
    number = r"(\d+)"
    verb = r"(?:바꿔|변경해|설정해|맞춰|올려|높여|내려|낮춰|줄여|빠르게\s*해|느리게\s*해)(?:\s*(?:줘요?|주세요))?"
    unsupported = re.search(name + r"([-+]?\d[\d.,eE+-]*)", clean)
    if unsupported:
        raw = unsupported.group(1)
        tail = clean[unsupported.end():]
        suffix = tail and (tail[0] in '/*%^_' or (tail[0].isascii() and tail[0].isalpha() and not re.match(r'bpm\b', tail)))
        if not raw.isdecimal() or suffix:
            # Non-canonical spelling is a proposal, never an immediate interpretation.
            return None
    transition = re.fullmatch(name + number + r"\s*에서\s*" + number + r"\s*(?:으로|로)\s*" + verb, clean)
    absolute = re.fullmatch(name + number + r"\s*(?:bpm\s*)?(?:으로|로)\s*" + verb, clean)
    before_unit = re.fullmatch(number + r"\s*bpm\s*(?:으로|로)?\s*" + verb, clean)
    simple = re.fullmatch(name + number + r"\s*(?:bpm\s*)?(?:" + verb + r")?", clean)
    direction_only = re.fullmatch(name + r"(?:올려|높여|내려|낮춰|줄여|빠르게\s*해|느리게\s*해)(?:\s*(?:줘요?|주세요))?", clean)
    confirm = False
    down = bool(re.search(r"내려|낮춰|줄여|느리게", clean))
    if transition:
        if not 40 <= int(transition.group(1)) <= 200:
            return {"reply": "출발 템포와 목표 템포를 40~200 BPM 범위로 확인해 주세요.", "patch": {}, "confirmation_required": False, "source": "rules"}
        target = int(transition.group(2))
    elif absolute or before_unit:
        target = int((absolute or before_unit).group(1))
    elif simple:
        amount = int(simple.group(1))
        relative = bool(re.search(r"올려|높여|내려|낮춰|줄여|빠르게|느리게", clean))
        target = current + (-amount if down else amount) if relative else amount
    elif direction_only:
        target = max(40, min(200, current + (-10 if down else 10)))
        confirm = True
    else:
        return None
    if not 40 <= target <= 200:
        return {"reply": "지원하는 템포는 40~200 BPM입니다. 설정은 변경하지 않았습니다.", "patch": {}, "confirmation_required": False, "source": "rules"}
    reply = f"현재 {current} BPM에서 {target} BPM으로 바꿀까요? 적용 전에 확인해 주세요." if confirm else f"템포 설정을 {target} BPM으로 바꿀게요. 기존 음원은 그대로이며 다시 만들 때 적용됩니다."
    return {"reply": reply, "patch": {"tempo": target}, "confirmation_required": confirm, "source": "rules"}


def respond(text: str, context: dict | None = None, planner=None, history=None) -> dict:
    """Return a small UI patch, leaving all note tokens and audio untouched."""
    current = (context or {}).get("tempo", 90)
    if type(current) is not int or not 40 <= current <= 200:
        current = 90
    fast = _tempo_command(text, current) or _genre_command(text)
    if fast is not None:
        return fast
    if planner is not None:
        safe_context = {k: v for k, v in (context or {}).items()
                        if k in SETTING_KEYS}
        try:
            answer = planner(text, safe_context, history or [])
            if not isinstance(answer, dict):
                raise ValueError("Expected an object")
            patch = validate_patch(answer.get("patch", {}))
            reply = answer.get("reply", "설정 변경안을 제안합니다.")
            if not isinstance(reply, str):
                raise ValueError("Expected text")
        except Exception:
            return {"reply": "언어 모델의 응답을 안전하게 적용할 수 없어 설정은 변경하지 않았습니다.",
                    "patch": {}, "confirmation_required": False, "source": "error"}
        return {"reply": reply[:1000], "patch": patch,
                "confirmation_required": bool(patch), "source": answer.get("via", "ollama")}
    return {"reply": "원하시는 변경을 구체적으로 말씀해 주세요.", "patch": {},
            "confirmation_required": False, "source": "rules", "requires_llm": True}


def _genre_command(text: str) -> dict | None:
    """"감성록으로 바꿔줘": a complete command naming exactly one genre.  The rules read the genre (longest name wins, so 감성록 is
    never 록), a proposal still waits for the user's confirmation.  Anything looser goes to the model."""
    from .chat import GENRES as WORDS, GENRE_LABEL, normalize
    # "감성록", "감성록으로", "감성록 느낌", "이 멜로디로 감성록 곡 만들어줘", "감성록으로 바꿔줘": only the name is read, so any extra word falls through
    m = re.fullmatch(r"\s*(?:이\s*멜로디(?:를|로)?\s*)?(.+?)\s*(?:으로|로|를|을)?\s*(?:느낌|스타일|장르)?\s*(?:으로|로)?\s*(?:곡\s*)?"
                     r"(?:(?:바꿔|변경해|설정해|맞춰|만들어|해)(?:\s*(?:줘요?|주세요))?)?\s*[.!]?\s*", text or "")
    if not m:
        return None
    said = normalize(m.group(1))
    hits = [g for g, words in WORDS.items() if said in {normalize(w) for w in words}]
    if len(hits) != 1:
        return None
    g = hits[0]
    return {"reply": f"설정 변경안(미적용): 장르: {GENRE_LABEL[g]}", "patch": {"genre": g}, "confirmation_required": True, "source": "rules"}


def app_facts() -> str:
    """What the app can do, as plain lines the answering model may use (and nothing else).  Built from the code's own tables."""
    from .chat import GENRE_LABEL
    from .yuesong import VOICES
    voices = ", ".join(v["label"] for v in VOICES.values())
    return (f"- 장르: {', '.join(GENRE_LABEL.values())}.\n- 보컬: {voices}.\n"
            "- 멜로디 넣기: 악보 파일(MIDI, MusicXML, PDF), 허밍·녹음(음높이만 읽어 멜로디로 바꿈), 왼쪽 고급 도구에서 음표 직접 찍기.\n"
            "- 곡 만들기: 올린 악보 전체를 그대로 편곡하고, 이어서 YuE2가 노래로 만든다. YuE2는 한 곡에 약 10분, 한 번에 한 곡만 만들 수 있다.\n"
            "- YuE2 음원은 비상업(CC BY-NC 4.0)이라 수업·연습·비영리 용도로만 쓴다. 상업 이용은 직접 편곡하거나 직접 연주한다.\n"
            "- 느낌(슬픈, 신나는 등)을 말하면 장조·단조를 추천한다. 악보의 음은 바뀌지 않고 코드와 조 이름만 바뀐다.\n"
            "- 설정 도우미는 장르, 템포(BPM), 보컬 같은 설정 변경안을 만든다(적용은 사용자가 확인한 뒤).\n"
            "- 내 Gemini 키를 저장하면 대화를 이해할 때 Gemini를 먼저 쓰고, 그 답에는 [Gemini]가 붙는다. 한도가 차면 qwen이나 기본 규칙으로 넘어간다.\n"
            "- 만든 곡은 왼쪽 메뉴의 내 곡에 모이고, 파트별 악보와 연습은 결과 화면에 있다.")


def answer_question(text: str, context: dict | None, key: str | None, events: list | None = None) -> str | None:
    """A short answer to a question about the app from the visitor's Gemini key, or None (no key, a failure, no facts for it)."""
    import json
    from . import usergemini as ug
    system = ("너는 작곡메이트의 안내원이다. 아래 '앱 사실'과 '현재 설정'에 적힌 내용만으로 한국어 해요체 두세 문장으로 답한다. "
              "사실에 없는 것은 지어내지 말고 \"그건 잘 모르겠어요\"라고 하고, 할 수 있는 것을 한 가지 알려 준다. "
              "곡을 만들거나 설정을 바꾸는 일은 하지 않는다.\n앱 사실:\n" + app_facts()
              + "\n현재 설정: " + json.dumps(context or {}, ensure_ascii=False))
    said = ug.try_ask(key, events, system, text, 30.0)
    return said.strip()[:600] if said and said.strip() else None
