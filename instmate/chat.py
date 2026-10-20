"""Conversation that leads from a score file to a finished song.

    step(session, text, score=None, event=None, llm=None) -> {"session", "say": [...], "action": {...} | None, "ask": ...}

The ORDER of the work is fixed here in code: a score (POST /api/score/read) -> the user's consent when the file was
not read exactly -> genre -> voice -> key (asked when the analysis cannot tell) -> compose -> listen -> rewrite a
section.  Words only fill in the answers: what the user wrote is read by plain rules first and, when a rule finds
nothing for the question at hand, by the language model if one is given (`llm(text, ask, choices) -> dict`); its
answer is checked against the allowed values, so it can never start a step or skip one.  Everything works without it.

The session travels with the page (the server keeps nothing); `action` tells the page which existing API to call
next ("compose": POST /api/compose with `body`, "regenerate": POST /api/compose/{song_id}/section).

Once a song exists, a message that is not a new setting is a request on that song.  It is carried out through
function calling (`TOOLS`): plain rules pick the tool when the words are clear ("브릿지 다시 써줘", "구간별 편성 켜줘",
a question), otherwise the language model picks one (`tools(text, song, sound_options) -> {"name", "args"}`).  Either
way the call is checked here (`_valid_call`) before anything runs, and the tools can only rewrite a section the app
wrote, switch a sound option, or look up an explanation (instmate/explain.py).  No tool writes a note or a chord.

Unclear words are asked back, never guessed (the course's function calling rule: "Don't make assumptions about what
values to plug into functions. Ask for clarification if a user request is ambiguous.").  The model is told so and
calls `ask_user`; the question and the request it belongs to wait in `session["pending"]`, and the user's answer goes
back to the model together with them.  A change the MODEL chose from indirect words (not the rules from clear ones) is
not run either: it is put to the user as a yes / no question first.
"""
from __future__ import annotations

import re

from . import composer, explain, scoreimport, sectionfind
from .leadsheet import KEYS
from .theory import NOTE_NAMES, parse_number_input

GENRES = {"ballad": ("발라드",), "rock": ("록", "락", "밴드", "rock"), "dance": ("댄스", "edm", "클럽"), "kpop": ("케이팝", "k-pop", "kpop", "k팝"),
          "jpop": ("제이팝", "j-pop", "jpop", "j팝"), "newage": ("뉴에이지",), "hiphop": ("힙합",),
          "piano_rock": ("피아노 록", "피아노록", "피아노 락"), "pop_punk": ("팝펑크", "팝 펑크"), "hard_rock": ("하드록", "하드 록", "하드락"),
          "emo_rock": ("감성록", "감성 록", "감성락", "이모"), "modern_rock": ("모던록", "모던 록", "모던락"), "classical": ("클래식",)}
GENRE_LABEL = {g: names[0] for g, names in GENRES.items()}
VOICES = {"female": ("여성", "여자", "여보컬", "여가수", "소프라노", "알토"), "male": ("남성", "남자", "남보컬", "남가수", "바리톤", "테너")}
VOICES_EN = {"female": ("female", "woman", "girl"), "male": ("male", "man", "boy")}     # whole English words only: "female" holds "male"
YES = ("네", "예", "응", "좋아", "진행", "그래", "ㅇㅇ", "yes", "ok", "오케이", "괜찮", "넵", "넹", "ㅇㅋ", "오키", "그럼", "당연", "맞아")
NO = ("아니", "취소", "안 돼", "안돼", "no", "그만", "아냐", "아뇨", "ㄴㄴ", "싫어", "하지마")
SECTIONS = {"verse": ("벌스", "verse", "1절", "2절"), "pre": ("프리코러스", "프리", "pre"), "bridge": ("브릿지", "브리지", "bridge"),
            "chorus": ("후렴", "코러스", "chorus", "사비"), "intro": ("인트로", "intro"), "outro": ("아웃트로", "outro")}
EDIT_WORDS = ("다시", "바꿔", "바꾸", "바꿨", "고쳐", "고치", "새로", "수정")
# Words right after a section name that make it a landmark or an exception, not the section to work on
# ("후렴 앞에 나오는 부분", "브릿지 말고", "후렴은 그대로 두고", "후렴처럼").
LANDMARK = ("앞", "뒤", "전에", "전의", "다음", "이후", "처럼", "같이", "같은", "말고", "빼고", "그대로", "보다", "까지", "부터")
CHOICE = ("중에", "중 하나", "아무")           # the user leaves the choice open: asked back, not picked for them
MAX_ASKS = 2                                   # times the same open question is put before the app stops asking
NOTE_KO = {"다": 0, "라": 2, "마": 4, "바": 5, "사": 7, "가": 9, "나": 11}
NOTE_EN = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
USES = {"whole": ("곡 전체", "전체", "통째", "전곡", "처음부터 끝까지", "전부", "모두", "그대로 해", "악보 그대로", "자동", "풀버전", "풀곡"),
        "intro": ("도입", "인트로", "전주", "intro", "앞부분", "오프닝"),
        "chorus": ("후렴", "하이라이트", "사비", "코러스", "chorus", "훅")}
USE_LABEL = {"whole": "곡 전체", "intro": "도입", "chorus": "후렴(하이라이트)"}
PARTS = {"drums": ("드럼", "drum"), "bass": ("베이스", "bass"), "guitar": ("기타", "guitar"), "piano": ("피아노", "건반", "키보드", "piano")}
PART_LABEL = {"drums": "드럼", "bass": "베이스", "guitar": "기타", "piano": "피아노"}
ASKS = ("file", "consent", "use", "genre", "voice", "mood", "key")
ON_WORDS = ("켜", "켠", "넣어", "on")
OFF_WORDS = ("꺼", "끄", "빼", "off")
# What the model may call on a finished song.  The same shape as instmate/llm.py's TOOLS (JSON schema per function).
TOOLS = [
    {"type": "function", "name": "rewrite_section",
     "description": "앱이 쓴 구간 하나의 멜로디를 다시 쓴다. 사용자가 올린 멜로디가 들어간 구간은 다시 쓰지 못한다.",
     "parameters": {"type": "object", "properties": {
         "section": {"type": "string", "enum": list(SECTIONS), "description": "구간 종류"},
         "nth": {"type": "integer", "description": "그 종류의 몇 번째 구간인지 (1부터)"}},
         "required": ["section"]}},
    {"type": "function", "name": "set_sound_option",
     "description": "소리 옵션 하나를 켜거나 끈다. 음표와 코드는 바뀌지 않는다.",
     "parameters": {"type": "object", "properties": {
         "id": {"type": "string", "description": "옵션 id (지금 쓸 수 있는 옵션 목록 중 하나)"},
         "on": {"type": "boolean"}},
         "required": ["id", "on"]}},
    {"type": "function", "name": "explain",
     "description": "이 곡이나 앱이 한 일에 대한 질문(왜, 무엇, 어떻게)에 앱의 설명 자료를 찾아 답한다.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    {"type": "function", "name": "ask_user",
     "description": "요청이 모호하거나 위 도구로 할 수 없을 때만 사용자에게 되묻거나, 할 수 없다고 알린다.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
]
HINT = ("구간을 고치려면 \"브릿지 다시 써줘\"처럼, 새 곡은 악보 파일을 다시 붙여 주세요. "
        "세부 조정(마디별 코드, 악기, 템포)은 \"직접 설정\" 탭에 있어요.")


def new_session() -> dict:
    return {"score": None, "consent": False, "use": None, "genre": None, "voice": None, "key": None, "mood": None, "source": "ai",
            "song": None, "pending": None}     # an open question on the finished song: {"call": ...} to confirm or {"request", "question"}


SPELLINGS = (("락", "록"), ("롹", "록"), ("콜", "콜"))      # one spelling for the same word (after the kiosk's catalog.normalize)


def normalize(text: str) -> str:
    """Spaces and marks removed and the spelling made one: "감성 락" and "감성록", "K - POP" and "kpop" read the same."""
    t = re.sub(r"[\s+&()\[\]/,.·'\"!?~\-_]", "", (text or "").lower())
    for src, dst in SPELLINGS:
        t = t.replace(src, dst)
    return t


def _has(text: str, words) -> bool:
    """Any of the words in the text, as written or with spaces and spelling folded (the words fold the same way)."""
    folded = normalize(text)
    return any(w in text or (normalize(w) and normalize(w) in folded) for w in words)


def _english(text: str, words) -> bool:
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text) for w in words)


def fuzzy_genre(text: str) -> str | None:
    """The genre a short answer to the open genre question comes nearest to ("발라두" -> ballad), when one stands out;
    never used outside that question (a near miss in a long sentence is no answer)."""
    from difflib import SequenceMatcher
    key = normalize(text)
    if not 2 <= len(key) <= 12:
        return None
    scored = sorted(((max(SequenceMatcher(None, key, normalize(w)).ratio() for w in names), g) for g, names in GENRES.items()
                     if any(len(normalize(w)) >= 3 for w in names)), reverse=True)
    if scored and scored[0][0] >= 0.6 and (len(scored) < 2 or scored[0][0] - scored[1][0] >= 0.1):
        return scored[0][1]
    return None


CORRECT_CUES = ("정정", "로해", "로만들", "로바꿔", "로바꾸", "로다시", "말고", "아니", "아냐", "틀렸", "이야", "입니다")


def use_correction(text: str) -> str | None:
    """What the score is, when the words correct it ("정정할게. 도입부로 해줘", "아니 곡 전체로 해줘"): a use word together
    with a cue of correction.  A rewrite of a named section ("후렴 말고 브릿지 다시 써줘") and a remark about one
    ("후렴 어때?") are not corrections."""
    n = normalize(text)
    if not any(c in n for c in CORRECT_CUES):
        return None
    if _has(text or "", EDIT_WORDS) and named_section(text)[1]:
        return None
    hits = [(len(normalize(w)), u) for u, words in USES.items() for w in words if normalize(w) in n]
    return max(hits)[1] if hits else None


KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def understood(found: dict) -> list[str]:
    """What the rules took from the message, as short Korean lines the page shows under it (nothing found: [])."""
    out = []
    if found.get("genre") in GENRE_LABEL:
        out.append(f"장르: {GENRE_LABEL[found['genre']]}")
    if found.get("voice") in VOICES:
        out.append(f"보컬: {'여성' if found['voice'] == 'female' else '남성'}")
    if found.get("use") in USE_LABEL:
        out.append(f"쓰임: {USE_LABEL[found['use']]}")
    if isinstance(found.get("key"), dict):
        out.append(f"조: {KEY_NAMES[found['key']['tonic'] % 12]} {'장조' if found['key']['mode'] == 'major' else '단조'}")
    if isinstance(found.get("yes"), bool):
        out.append("확인: " + ("네" if found["yes"] else "아니요"))
    return out


def named_section(text: str) -> tuple[str | None, bool]:
    """(the one section kind the words point at, whether a section was named at all).  A name followed by a landmark
    word is where something else is, not the target; two targets or an open choice give no answer."""
    t = (text or "").lower()
    spans: list[tuple[int, int, str]] = []
    for size, start, kind in sorted(((len(w), m.start(), k) for k, words in SECTIONS.items() for w in words
                                     for m in re.finditer(re.escape(w), t)), reverse=True):
        if all(start >= e or start + size <= b for b, e, _ in spans):      # the longest name wins: "프리코러스" is not "코러스"
            spans.append((start, start + size, kind))
    targets = {k for _, e, k in spans if not _has(t[e:e + 5], LANDMARK)}
    if len(targets) != 1 or _has(t, CHOICE):
        return None, bool(spans)
    return targets.pop(), True


def read_text(text: str) -> dict:
    """What the plain rules find in a message: any of genre, voice, key ({"tonic", "mode"}), yes (True / False),
    source ("own"), edit (a section kind), nth (which one of that kind, from 1), edit_unclear (True: a rewrite is
    wanted but the words do not settle which section)."""
    t = (text or "").lower().strip()
    out: dict = {}
    folded = normalize(t)
    hits = [(len(normalize(w)), g) for g, words in GENRES.items() for w in words if w in t or normalize(w) in folded]
    if hits:
        out["genre"] = max(hits)[1]                      # the longest name wins: "하드록" is not "록"
    voices = [v for v, words in VOICES.items() if _has(t, words) or _english(t, VOICES_EN[v])]
    if len(voices) == 1:                                 # both named: not decided here, asked back
        out["voice"] = voices[0]
    m = re.search(r"([a-g])\s*([#b♯♭]?)\s*(장조|단조|major|minor|메이저|마이너)", t) or None
    if m:
        pc = (NOTE_EN[m.group(1)] + {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(m.group(2), 0)) % 12
        out["key"] = {"tonic": pc, "mode": "major" if m.group(3) in ("장조", "major", "메이저") else "minor"}
    else:
        m = re.search(r"(올림|내림)?\s*([다라마바사가나])\s*(장조|단조)", t)
        if m:
            pc = (NOTE_KO[m.group(2)] + {"올림": 1, "내림": -1}.get(m.group(1), 0)) % 12
            out["key"] = {"tonic": pc, "mode": "major" if m.group(3) == "장조" else "minor"}
        elif "장조" in t or _has(t, MOOD_WORDS["major"]):
            out["mode"] = "major"
        elif "단조" in t or _has(t, MOOD_WORDS["minor"]):
            out["mode"] = "minor"
    if _has(t, NO):
        out["yes"] = False
    elif _has(t, YES):
        out["yes"] = True
    if _has(t, ("ai 없이", "ai없이", "내 멜로디만", "변형만")):
        out["source"] = "own"
    if not _has(t, EDIT_WORDS):                      # "후렴 다시 써줘" is an edit, not a statement about the file
        uses = [(len(normalize(w)), u) for u, words in USES.items() for w in words if w in t or normalize(w) in folded]
        if uses:
            out["use"] = max(uses)[1]
    parts = [(len(w), p) for p, words in PARTS.items() for w in words if w in t]
    if parts:
        out["part"] = max(parts)[1]
    if _has(t, EDIT_WORDS):
        kind, _ = named_section(t)
        if kind is None:
            out["edit_unclear"] = True                   # the action is clear, the section is not: asked, never guessed
        else:
            out["edit"] = kind
            n = re.search(r"([1-9])\s*(?:번째|절)|(첫|두|세)\s*번째", t)
            if n:
                out["nth"] = int(n.group(1)) if n.group(1) else {"첫": 1, "두": 2, "세": 3}[n.group(2)]
    m = re.search(r"\b([1-9])\s*번", t) or re.fullmatch(r"\s*([1-9])\s*", t)   # "2번", or a bare "2" for a numbered question
    if m and "nth" not in out:
        out["pick"] = int(m.group(1))
    m = re.search(r"(\d+)\s*마디", t)
    if m:
        out["bars_said"] = int(m.group(1))             # "49마디입니다": the whole score when it is the score's length
    return out


def _from_llm(llm, text: str, ask: str, choices: list[str]) -> dict:
    """The model's reading of `text` for the open question, reduced to allowed values; {} on any problem."""
    if llm is None or not (text or "").strip():
        return {}
    try:
        got = llm(text, ask, choices)
    except Exception:
        return {}
    if not isinstance(got, dict):
        return {}
    out = {}
    if got.get("genre") in GENRES:
        out["genre"] = got["genre"]
    if got.get("voice") in VOICES:
        out["voice"] = got["voice"]
    if isinstance(got.get("yes"), bool):
        out["yes"] = got["yes"]
    if got.get("use") in USES:
        out["use"] = got["use"]
    if type(got.get("pick")) is int and 1 <= got["pick"] <= max(1, len(choices)):
        out["pick"] = got["pick"]
    return out


# feeling words -> the mode they suggest (user request 2026-10-19: ask the feeling, recommend major or minor)
MOOD_WORDS = {"minor": ("슬프", "슬픈", "우울", "어둡", "어두운", "쓸쓸", "외로", "아련", "그리운", "이별", "애절", "서글", "몽환",
                        "진지", "신비", "긴장", "박진", "강렬", "정열", "비장", "sad", "dark", "gloomy"),
              "major": ("밝", "신나", "신난", "경쾌", "희망", "설레", "행복", "즐거", "상큼", "따뜻", "산뜻", "힘찬",
                        "평화", "평온", "당당", "감동", "활기", "유쾌", "기쁜", "happy", "bright")}


def _mood_key(session: dict) -> dict | None:
    """The key recommended for the felt mode: the file's key or its relative (same key signature, same notes)."""
    name = (session["score"] or {}).get("key")
    if name not in KEYS or session.get("mood") not in ("major", "minor"):
        return None
    pair = [k for k, n in KEYS.items() if n == KEYS[name]]
    pick = next(k for k in pair if k.endswith("m") == (session["mood"] == "minor"))
    tonic = NOTE_NAMES.index(_ENHARM.get(pick.rstrip("m"), pick.rstrip("m")))
    return {"tonic": tonic, "mode": session["mood"], "name": pick}


_ENHARM = {"C#": "C#", "Db": "C#", "D#": "Eb", "G#": "Ab", "A#": "Bb", "Gb": "F#", "Cb": "B"}


def _key_question(session: dict) -> dict | None:
    """The key analysis of the seed when it cannot tell: {"question", "candidates": [key, ...]}; None when it can."""
    tok = scoreimport.to_tokens(session["score"], 0, min(8, int(session["score"]["bars"])))
    analysis = composer.analyze_melody(None, parse_number_input(tok["tokens"], tok["input_tonic"]), tok["input_tonic"])
    if not analysis["ambiguous"]:
        return None
    return {"question": analysis["question"], "candidates": [c["key"] for c in analysis["candidates"][:2]]}


def _next_ask(session: dict) -> str | None:
    if session["score"] is None:
        return "file"
    if not session["score"].get("exact", True) and not session["consent"]:
        return "consent"
    if session.get("mood") is None and session["key"] is None:
        return "mood"                                  # the feeling first: it recommends major or minor (before key and split)
    if session["use"] is None:
        # a long score is the whole song (user, 2026-10-19: nobody should have to choose an 8-bar piece); it is only
        # taken as an intro or a chorus when the words say so ("도입부야", use_correction) or the melody is short
        session["use"] = "whole" if int(session["score"].get("bars") or 0) > scoreimport.SEED_MAX_BARS else "chorus"
    if session["genre"] is None:
        return "genre"
    if session["voice"] is None:
        return "voice"
    if session["use"] != "whole" and session["key"] is None and _key_question(session):
        return "key"                                   # a whole song keeps the key its score says
    return None


def choices(session: dict, ask: str | None) -> list[str]:
    """The answers the page shows as buttons under the question; each is a text the rules above read."""
    if ask == "use":
        return [USE_LABEL["whole"], "도입", "후렴"]
    if ask == "consent":
        return ["네", "아니요"]
    if ask == "voice":
        return ["여성", "남성"]
    if ask == "mood":
        return ["슬픈", "우울한", "신나는", "밝은", "상관없어요"]
    if ask == "genre":
        return [GENRE_LABEL[g] for g in ("ballad", "rock", "emo_rock", "jpop", "kpop", "modern_rock")]
    if ask == "key":
        return ["1번", "2번"] if _key_question(session) else []
    if ask is None and session.get("song"):             # a finished song and nothing open: what could be said next
        kinds = [x.get("kind") for x in session["song"].get("sections", [])]
        offers = ["왜 조를 옮겼어?", "소리 옵션 보여줘"]
        for kind, word in (("bridge", "브릿지"), ("pre", "프리코러스"), ("verse", "벌스")):
            if kind in kinds:
                offers.insert(0, f"{word} 다시 써줘")
                break
        offers.append("이 곡은 어떻게 만들었어?")
        if session.get("score") is not None and session.get("use"):      # the score may have been taken for the wrong thing
            offers.append("후렴으로 정정해줘" if session["use"] == "intro" else "도입으로 정정해줘")
        return offers
    return []


def _choices(session: dict, ask: str | None) -> list[str]:
    if ask == "use":
        return list(USES)
    if ask == "genre":
        return list(GENRES)
    if ask == "voice":
        return list(VOICES)
    if ask == "key":
        q = _key_question(session)
        return [k["label"] for k in q["candidates"]] if q else []
    return []


def _question(session: dict, ask: str) -> str:
    if ask == "file":
        return "멜로디가 담긴 악보 파일(MIDI, MusicXML, PDF)을 붙여 주세요. 장르와 보컬(여성·남성)도 함께 적어 주시면 바로 만들어요."
    if ask == "consent":
        changes = session["score"].get("changes") or []
        shown = "; ".join(f"{c.get('bar', '?')}마디 {c.get('note', '')} {c.get('text') or c.get('what', '')}".strip() for c in changes[:3])
        return (f"파일을 그대로 옮기지 못한 곳이 {len(changes)}군데 있어요" + (f" ({shown})" if shown else "")
                + ". 이대로 진행할까요? (네 / 아니요)")
    if ask == "use":
        return (f"올린 악보가 {session['score'].get('bars')}마디예요. 이 악보는 무엇인가요? "
                "1 곡 전체(악보 그대로 편곡) / 2 도입(처음 8마디를 인트로로) / 3 후렴(처음 8마디를 후렴·하이라이트로). "
                "번호나 이름으로 답하거나 아래 버튼을 눌러 주세요.")
    if ask == "genre":
        return "어떤 장르로 만들까요? 예: " + ", ".join(GENRE_LABEL[g] for g in ("ballad", "rock", "emo_rock", "jpop", "kpop", "modern_rock"))
    if ask == "voice":
        return "멜로디를 부를 보컬은 여성과 남성 중 어느 쪽인가요?"
    if ask == "mood":
        return "어떤 느낌의 곡인가요? 예: 슬픈 록, 우울한, 신나는, 밝은. 느낌에 맞춰 장조·단조를 추천해 드려요."
    q = _key_question(session)
    return f"{q['question']} (1번 {q['candidates'][0]['label']} / 2번 {q['candidates'][1]['label']})"


def _take(session: dict, found: dict, ask: str | None) -> None:
    """Put what was found into the session: stated facts always, a bare yes / number only for the open question."""
    for k in ("genre", "voice", "source"):
        if k in found:
            session[k] = found[k]
    if "use" in found and (ask == "use" or session["use"] is None):
        session["use"] = found["use"]
    elif ask == "use" and found.get("pick") in (1, 2, 3):
        session["use"] = ("whole", "intro", "chorus")[found["pick"] - 1]
    elif ask == "use" and session["score"] and found.get("bars_said", 0) >= (session["score"].get("bars") or 10 ** 9):
        session["use"] = "whole"                         # the length of the whole score was given: the whole score
    if ask == "consent" and found.get("yes") is True:
        session["consent"] = True
    if found.get("mode") or ask == "mood":                  # any other answer to the mood question: keep the file's key
        session["mood"] = found.get("mode") or "any"
        rec = _mood_key(session) if session["key"] is None else None
        if rec:
            session["key"] = {"tonic": rec["tonic"], "mode": rec["mode"]}
            session["key_note"] = (f"{'슬프거나 어두운' if rec['mode'] == 'minor' else '밝은'} 느낌이라 "
                                   f"{NOTE_NAMES[rec['tonic']]} {'단조' if rec['mode'] == 'minor' else '장조'}를 추천해요. "
                                   "조표가 같은 짝이라 멜로디 음은 바뀌지 않아요.")
    if ask == "key" or "key" in found:
        q = _key_question(session) if session["score"] is not None and (session["score"].get("exact", True) or session["consent"]) else None
        cands = q["candidates"] if q else []
        if "key" in found:
            session["key"] = {"tonic": found["key"]["tonic"], "mode": found["key"]["mode"]}
        elif found.get("pick") in (1, 2) and cands:
            c = cands[found["pick"] - 1]
            session["key"] = {"tonic": c["tonic"], "mode": c["mode"]}
        elif found.get("mode") and cands:
            hit = [c for c in cands if c["mode"] == found["mode"]]
            if len(hit) == 1:
                session["key"] = {"tonic": hit[0]["tonic"], "mode": hit[0]["mode"]}


def song_body(session: dict, sections: list[dict] | None = None) -> dict:
    """The POST /api/score/song body for a score that is the whole song: the melody as written, the app's first
    split into sections (or `sections`, found from what repeats), the band arranged around it (no model writes a
    section)."""
    score = session["score"]
    return {"score": score, "sections": sections or scoreimport.default_sections(int(score["bars"])), "genre": session["genre"],
            "accept_changes": True, "melody_role": "vocal", "fit_voice": session["voice"],
            "key": (_mood_key(session) or {}).get("name") or score.get("key")}


def compose_body(session: dict) -> dict:
    """The POST /api/compose body for the session (the same fields the form sends)."""
    tok = scoreimport.to_tokens(session["score"], 0, min(8, int(session["score"]["bars"])))
    if session.get("use") == "intro":                  # the input is the intro: the model is not used for that
        return {"tokens": tok["tokens"], "input_tonic": tok["input_tonic"], "genre": session["genre"], "key_choice": session["key"],
                "melody_source": "own", "fit_voice": session["voice"], "key_change": False, "melody_role": "vocal",
                "seed_role": "intro", "intro_player": "guitar", "tempo": tok.get("bpm"), "melody_shape": True}
    return {"tokens": tok["tokens"], "input_tonic": tok["input_tonic"], "genre": session["genre"], "key_choice": session["key"],
            "melody_source": session["source"], "fit_voice": session["voice"], "key_change": False, "melody_role": "vocal",
            "seed_role": "chorus", "tempo": tok.get("bpm"),
            "melody_shape": True}                 # chosen by ear on 2026-10-10 (pair 3); the API's own default stays off


def _edit(session: dict, found: dict) -> dict:
    song = session["song"]
    rows = [(i, s) for i, s in enumerate(song["sections"]) if s.get("kind") == found["edit"]]
    label = SECTIONS[found["edit"]][0]
    if not rows:
        return {"say": [f"이 곡에는 {label} 구간이 없어요. 구간: " + ", ".join(s.get("label", s.get("kind", "")) for s in song["sections"])], "action": None}
    i, sec = rows[min(found.get("nth", 1), len(rows)) - 1]
    if sec.get("source") == "seed":
        return {"say": [f"{sec.get('label', label)}은 올려 주신 멜로디라 앱이 고치지 않아요. 벌스·프리코러스·브릿지는 다시 쓸 수 있어요."], "action": None}
    if song.get("melody_source") == "own":
        return {"say": ["AI 없이 만든 곡은 구간을 다시 쓰지 않아요. \"직접 설정\" 탭에서 마디별로 고칠 수 있어요."], "action": None}
    return {"say": [f"{sec.get('label', label)}만 다시 만들게요. 다른 구간은 그대로 둬요."],
            "action": {"type": "regenerate", "song_id": song["song_id"], "section_index": i}}


def _label(option: dict) -> str:
    return re.sub(r"\s*\(.*?\)", "", str(option.get("label", ""))).strip()


def _sound_rule(text: str, sound_options: list[dict]) -> dict | None:
    """The tool call plain words ask for: an option named by its label together with on / off."""
    t = (text or "").lower().replace(" ", "")
    hits = [o for o in sound_options if _label(o) and _label(o).lower().replace(" ", "") in t]
    on, off = _has(t, ON_WORDS), _has(t, OFF_WORDS)
    if len(hits) != 1 or on == off:
        return None
    return {"name": "set_sound_option", "args": {"id": hits[0]["id"], "on": on}}


def _valid_call(call, sound_options: list[dict]) -> dict | None:
    """The call reduced to what a tool accepts, or None.  Nothing the model sends is used unchecked."""
    if not isinstance(call, dict) or not isinstance(call.get("args"), dict):
        return None
    name, args = call.get("name"), call["args"]
    if name == "rewrite_section" and args.get("section") in SECTIONS:
        nth = args.get("nth", 1)
        return {"name": name, "args": {"section": args["section"], "nth": nth if type(nth) is int and 1 <= nth <= 20 else 1}}
    if name == "set_sound_option" and isinstance(args.get("on"), bool) and any(o.get("id") == args.get("id") for o in sound_options):
        return {"name": name, "args": {"id": args["id"], "on": args["on"]}}
    if name in ("explain", "ask_user") and isinstance(args.get("question"), str) and args["question"].strip():
        return {"name": name, "args": {"question": args["question"].strip()[:300]}}
    return None


def _run_call(session: dict, call: dict, sound_options: list[dict], explainer) -> dict:
    """Carry out a checked call: {"say", "action", "tool"}."""
    name, args = call["name"], call["args"]
    if name == "rewrite_section":
        return {**_edit(session, {"edit": args["section"], "nth": args["nth"]}), "tool": call}
    if name == "set_sound_option":
        label = _label(next(o for o in sound_options if o["id"] == args["id"]))
        return {"say": [f"{label} 옵션을 {'켰' if args['on'] else '껐'}어요. 다음에 만드는 미리듣기부터 적용돼요. 음표와 코드는 그대로예요."],
                "action": {"type": "sound_option", "id": args["id"], "on": args["on"]}, "tool": call}
    if name == "explain":
        return {"say": [explainer(args["question"], session.get("song"))], "action": None, "tool": call}
    return {"say": [args["question"]], "action": None, "tool": call}


def _confirm(session: dict, call: dict, sound_options: list[dict]) -> dict:
    """A change the model chose from indirect words: asked back as a yes / no question, run only on a yes."""
    if call["name"] == "rewrite_section":
        tried = _edit(session, {"edit": call["args"]["section"], "nth": call["args"]["nth"]})
        if tried["action"] is None:                        # cannot be done anyway (the user's melody, no such section)
            return {**tried, "tool": call}
        label = session["song"]["sections"][tried["action"]["section_index"]].get("label", "")
        question = f"{label} 구간의 멜로디를 다시 쓸까요? 지금 멜로디는 새 것으로 바뀌고 다른 구간은 그대로예요. (네 / 아니요)"
    else:
        option = next(o for o in sound_options if o["id"] == call["args"]["id"])
        about = str(option.get("help", "")).split(". ")[0].rstrip(".")
        question = f"{_label(option)} 옵션을 {'켤' if call['args']['on'] else '끌'}까요?" + (f" ({about}.)" if about else "") + " (네 / 아니요)"
    session["pending"] = {"call": call, "question": question}
    return {"say": [question], "action": None, "tool": call}


def _which_section(session: dict, request: str, tries: int) -> dict:
    """The rules' own question when a rewrite is wanted and no section is settled (no model, or it gave nothing)."""
    song = session["song"]
    free = [x.get("label", "") for x in song.get("sections", []) if x.get("source") != "seed"]
    kept = [x.get("label", "") for x in song.get("sections", []) if x.get("source") == "seed"]
    if not free or song.get("melody_source") == "own" or tries > MAX_ASKS:
        return {"say": [HINT], "action": None, "tool": None}
    question = (f"어느 구간을 다시 쓸까요? {', '.join(free)} 중에서 골라 주세요."
                + (f" ({', '.join(kept)}은 올려 주신 멜로디라 고치지 않아요.)" if kept else ""))
    session["pending"] = {"request": request, "question": question, "want": "edit", "tries": tries}
    return {"say": [question], "action": None, "tool": None}


def _on_song(session: dict, text: str, tools, sound_options: list[dict], explainer, pending: dict | None = None,
             found: dict | None = None) -> dict:
    """A request on the finished song: the rules' call, else the model's, checked and carried out; else the hint.
    `pending`: the question this message may be the answer to."""
    asked_section = isinstance(pending, dict) and pending.get("want") == "edit"
    want_edit = bool((found or {}).get("edit_unclear")) or asked_section
    if asked_section:                                      # the answer to "which section?": a name alone settles it
        kind, _ = named_section(text)
        if kind is not None:
            return {**_edit(session, {"edit": kind, "nth": (found or {}).get("nth", 1)}), "tool": None}
    if isinstance(pending, dict) and isinstance(pending.get("call"), dict):          # a yes / no on a proposed change
        call = _valid_call(pending["call"], sound_options)
        if (found or {}).get("yes") is True and call is not None:
            return _run_call(session, call, sound_options, explainer)
        if (found or {}).get("yes") is False:
            return {"say": ["알겠어요. 그대로 둘게요."], "action": None, "tool": None}
        pending = None                                     # something else was said: a new request
    call = _sound_rule(text, sound_options)
    if call is None and not want_edit and explain.is_question(text):
        call = {"name": "explain", "args": {"question": text}}
    by_rule = call is not None
    if call is None and tools is not None and (text or "").strip():
        history = None
        if isinstance(pending, dict) and isinstance(pending.get("request"), str) and isinstance(pending.get("question"), str):
            history = [{"role": "user", "content": pending["request"][:1000]}, {"role": "assistant", "content": pending["question"][:300]}]
        try:
            call = tools(text, session.get("song"), sound_options, history) if history else tools(text, session.get("song"), sound_options)
        except Exception:
            call = None
    call = _valid_call(call, sound_options)
    if call is None and want_edit:                         # no model, or nothing usable from it: the rules ask
        first = pending["request"] if asked_section and isinstance(pending.get("request"), str) else text
        return _which_section(session, first, (pending.get("tries", 1) + 1) if asked_section else 1)
    if call is None:
        return {"say": [HINT], "action": None, "tool": None}
    if want_edit and call["name"] == "rewrite_section" and \
            _edit(session, {"edit": call["args"]["section"], "nth": call["args"]["nth"]})["action"] is None:
        # the model settled on a section that cannot be rewritten (the user's own melody, or none such): its guess is
        # not an answer to "which section", so the question is put instead of a refusal
        first = pending["request"] if asked_section and isinstance(pending.get("request"), str) else text
        return _which_section(session, first, (pending.get("tries", 1) + 1) if asked_section else 1)
    if call["name"] == "ask_user":                         # asked back: the answer comes with this request next turn
        session["pending"] = {"request": text, "question": call["args"]["question"]}
    if not by_rule and call["name"] in ("rewrite_section", "set_sound_option"):
        return _confirm(session, call, sound_options)
    return _run_call(session, call, sound_options, explainer)


def _found(text: str, ask: str | None) -> dict:
    """read_text, and for the open genre question a near miss of a genre name (a typo)."""
    found = read_text(text)
    if ask == "genre" and "genre" not in found:
        near = fuzzy_genre(text)
        if near:
            found["genre"] = near
    return found


def step(session: dict | None, text: str = "", score: dict | None = None, event: dict | None = None, llm=None,
         attached: bool = False, tools=None, sound_options: list[dict] | None = None, explainer=None,
         find_sections: bool = False, answer_any: bool = False) -> dict:
    """One turn (see `_step`), with `understood`: what the rules took from the message."""
    prior = {**new_session(), **{k: v for k, v in (session or {}).items() if k in new_session()}}
    out = _step(session, text, score, event, llm, attached, tools, sound_options, explainer, find_sections, answer_any)
    out["understood"] = understood(_found(text, _next_ask(prior) if score is None else None)) if not event else []
    return out


def _step(session: dict | None, text: str = "", score: dict | None = None, event: dict | None = None, llm=None,
          attached: bool = False, tools=None, sound_options: list[dict] | None = None, explainer=None,
          find_sections: bool = False, answer_any: bool = False) -> dict:
    """One turn.  `score`: what POST /api/score/read returned for a file attached to this message.
    `event`: {"type": "composed", "song": {"song_id", "melody_source", "sections": [{"label", "kind", "source"}]}} from
    the page when a compose or rewrite finished.  Returns the new session, what to say, the next action for the page
    (or None) and the open question ("file" | "consent" | "genre" | "voice" | "key" | None).
    `tools`, `sound_options` ([{"id", "label", "help"}]) and `explainer` (question, song -> text) serve requests on a finished
    song (see the module text); `find_sections` asks for the sections of a whole score to be found from its repeats;
    `answer_any`: a question is answered even before any file is attached (a visitor with a Gemini key)."""
    sound_options = sound_options or []
    explainer = explainer or (lambda question, song: explain.say(explain.answer(question, song)))
    s = {**new_session(), **{k: v for k, v in (session or {}).items() if k in new_session()}}
    say: list[str] = []
    if event and event.get("type") == "composed" and isinstance(event.get("song"), dict):
        s["song"] = event["song"]
        if isinstance(s["song"].get("seed_warning"), str) and s["song"]["seed_warning"].strip():
            say.append(s["song"]["seed_warning"].strip()[:400])        # the chorus is not the typed melody: said first
        names = " → ".join(x.get("label", "") for x in s["song"].get("sections", []))
        say.append(f"밴드 편곡을 완성했어요. 구성: {names}. 파트별 악보는 아래 3단계에 있어요. 악보를 다른 것으로 잘못 알았다면 \"도입부로 정정해줘\"처럼 말해 주세요. 완성본은 \"AI 보컬 얹기\" 버튼으로 만들어요(AI 가수 YuE2가 밴드 반주 위에서 부른 하이라이트 약 1분, 2~3분 걸려요). 다 되면 재생 버튼이 나와요. "
                   "마음에 들지 않는 구간은 \"브릿지 다시 써줘\"처럼 말해 주세요.")
        return {"session": s, "say": say, "action": None, "ask": None}
    if attached and score is None:                        # a file came with this message and has not been read yet
        part = read_text(text).get("part")
        if part:                                           # a band part: it replaces that part of the finished song
            if not s["song"]:
                return {"session": s, "ask": _next_ask(s) or None, "action": None,
                        "say": [f"{PART_LABEL[part]} 파트는 곡을 만든 뒤에 넣을 수 있어요. 먼저 멜로디 악보로 곡을 만들어 주세요. "
                                "그다음 이 파일을 다시 붙이고 \"이건 " + PART_LABEL[part] + " 파트야\"라고 말해 주세요."]}
            return {"session": s, "ask": None, "action": {"type": "part_import", "part": part},
                    "say": [f"이 파일을 {PART_LABEL[part]} 파트로 넣을게요. 연주한 음은 그대로 두고 그 파트만 바꿔요."]}
        return {"session": s, "ask": None, "say": [], "action": {"type": "read_score"}}     # a melody: read it first
    if score is not None:                                  # a new file starts a new song
        s.update(score=score, consent=False, key=None, song=None, use=read_text(text).get("use"))
        say.append(f"악보를 읽었어요: {score.get('bars', '?')}마디, {score.get('meter', '')}, {score.get('key', '')} 조, 템포 {score.get('bpm', '?')}.")
        if score.get("meter") != "4/4":
            s["score"] = None
            return {"session": s, "say": say + ["대화로 만들기는 4/4 박자 멜로디만 받아요. \"직접 설정\" 탭의 \"이 악보로 곡 만들기\"를 써 주세요."],
                    "action": None, "ask": "file"}
    found = _found(text, _next_ask(s) if score is None else None)
    fix = use_correction(text) if s["song"] and score is None and s["score"] is not None else None
    if fix and fix != s["use"]:                           # the score was something else: the old song is let go, a new one follows
        s.update(use=fix, song=None, pending=None)
        say.append(f"알겠어요, 정정할게요. 악보를 {USE_LABEL[fix]}(으)로 보고 곡을 다시 만들게요.")
        found = {**found, "use": fix}
    pending, s["pending"] = s.get("pending"), None        # an open question lasts one turn
    if s["song"] and score is None and "edit" in found:
        return {"session": s, "ask": None, **_edit(s, found)}
    if s["song"] and score is None and isinstance(pending, dict) and "call" in pending and "yes" in found:
        return {"session": s, "ask": None, **_on_song(s, text, tools, sound_options, explainer, pending, found)}
    ask = _next_ask(s)
    if ask and ask != "file":
        missing = (ask == "genre" and "genre" not in found) or (ask == "voice" and "voice" not in found) or \
                  (ask == "use" and not ({"use", "pick", "bars_said"} & set(found))) or \
                  (ask == "consent" and "yes" not in found) or (ask == "key" and not ({"key", "pick", "mode"} & set(found)))
        if missing:
            found = {**_from_llm(llm, text, ask, _choices(s, ask)), **found}
    if ask == "consent" and found.get("yes") is False:
        s["score"] = None
        return {"session": s, "say": ["알겠어요. 진행하지 않을게요. 다른 파일을 붙여 주시면 다시 읽어요."], "action": None, "ask": "file"}
    _take(s, found, ask)
    ask = _next_ask(s)
    if ask == "key":                                       # an answer given together with the facts above
        _take(s, found, "key")
        ask = _next_ask(s)
    if note := s.pop("key_note", None):
        say.append(note)
    if ask:
        if (ask != "file" or answer_any) and score is None and not found and (explain.is_question(text) or (answer_any and text.strip())):   # a question about the question
            say.append(explainer(text, None))
        return {"session": s, "say": say + [_question(s, ask)], "action": None, "ask": ask}
    if s["song"] and score is None and not (set(found) & {"genre", "voice", "key", "source"}):   # a bare yes / number is no new song
        return {"session": s, "ask": None, **_on_song(s, text, tools, sound_options, explainer, pending, found)}
    bars = int(s["score"].get("bars") or 0)
    if s["use"] in ("chorus", "intro") and bars > scoreimport.SEED_MAX_BARS and not 0 < found.get("bars_said", 0) <= scoreimport.SEED_MAX_BARS:
        # a long score said to be "the chorus" / "the intro" (user, 2026-10-19): every bar is that section, and the song
        # goes on from there; it is not cut to an 8-bar seed
        name = "후렴" if s["use"] == "chorus" else "인트로"
        say.append(f"{GENRE_LABEL[s['genre']]}, {VOICES[s['voice']][0]} 보컬로 악보 전체({bars}마디)를 {name}으로 넣고 만들게요. "
                   "멜로디는 올린 그대로예요.")
        return {"session": s, "say": say, "ask": None,
                "action": {"type": "score_song", "body": song_body(s, [{"name": name, "start_bar": 0, "bars": bars}])}}
    if s["use"] == "whole":
        found_sections = sectionfind.find(s["score"]) if find_sections else None
        if found_sections:
            split = "구간은 멜로디에서 반복되는 부분을 찾아 나눴어요: " + sectionfind.describe(found_sections) + ". 틀렸으면 \"직접 설정\" 탭에서 고칠 수 있어요."
        elif find_sections:
            split = "반복되는 부분을 찾지 못해 구간은 앞 절반을 벌스, 뒤 절반을 후렴으로 나눠요."
        else:
            split = "구간은 앱이 먼저 나눠요."
        say.append(f"{GENRE_LABEL[s['genre']]}, {VOICES[s['voice']][0]} 보컬로 악보 전체({s['score'].get('bars')}마디)를 그대로 편곡할게요. "
                   "멜로디는 올린 그대로이고 " + split)
        return {"session": s, "say": say, "action": {"type": "score_song", "body": song_body(s, found_sections)}, "ask": None}
    body = compose_body(s)
    key = f", {next((k['label'] for k in (_key_question(s) or {'candidates': []})['candidates'] if k['tonic'] == s['key']['tonic'] and k['mode'] == s['key']['mode']), '고른 조')}" if s["key"] else ""
    where = "인트로" if s["use"] == "intro" else "후렴"
    more = " (8마디가 넘는 악보라 처음 8마디를 씨앗으로 써요)" if int(s["score"].get("bars") or 0) > scoreimport.SEED_MAX_BARS else ""
    say.append(f"{GENRE_LABEL[s['genre']]}, {VOICES[s['voice']][0]} 보컬{key}로 만들기 시작할게요. 올려 주신 멜로디는 {where}에 그대로 둬요{more}.")
    return {"session": s, "say": say, "action": {"type": "compose", "body": body}, "ask": None}


def ollama_reader(timeout: float = 20.0, key: str | None = None, events: list | None = None):
    """`llm` for `step`, backed by the local dialogue model (instmate/llm.py's Ollama model), or None when Ollama is
    not running or does not have the model.  The model only fills in a small JSON object; `step` checks it."""
    import json
    from . import llm as _llm, usergemini as _ug
    if _llm.ollama_status() is not None and not _ug.valid(key):       # a visitor's Gemini key works without the local model
        return None

    def read(text: str, ask: str, choices: list[str]) -> dict:
        import httpx
        labels = ", ".join(f"{g}={GENRE_LABEL[g]}" for g in GENRES)
        system = ("너는 작곡 도우미의 말 읽기 담당이다. 사용자의 말에서 작곡 설정만 골라 JSON 하나로 답한다. "
                  "모르는 항목은 넣지 않는다. 짐작해서 채우지 않는다. 다른 말은 쓰지 않는다.\n"
                  f"항목: genre ({', '.join(GENRES)}; {labels}), voice (female, male), yes (true, false), "
                  "use (whole, intro, chorus) = 올린 악보가 곡 전체인지, 도입(처음 8마디)인지, 후렴(처음 8마디)인지, "
                  "pick (보기 번호, 1부터).\n"
                  f"지금 묻고 있는 것: {ask}. 보기: {', '.join(choices) or '없음'}.\n"
                  "사용자는 정해진 단어로 답하지 않는다. 띄어쓰기, 오타, 줄임말, 돌려 하는 말을 알아듣는다.\n"
                  "예: \"그냥 쭉 다 해줘요\" -> {\"use\": \"whole\"} / \"앞에 8마디만 인트로\" -> {\"use\": \"intro\"} / "
                  "\"후렴이에요\" -> {\"use\": \"chorus\"} / \"여자분이 불렀으면\" -> {\"voice\": \"female\"} / "
                  "\"감성 락 느낌\" -> {\"genre\": \"emo_rock\"} / \"넵 그렇게요\" -> {\"yes\": true} / "
                  "\"아니 그건 좀\" -> {\"yes\": false} / \"두번째로\" -> {\"pick\": 2} / \"몰라요\" -> {}")
        said = _ug.try_ask(key, events, system, text, timeout)
        if said is not None:
            try:
                return _ug.json_object(said)
            except ValueError:
                pass                                       # an unreadable answer: the local model reads it instead
        r = httpx.post(_llm.OLLAMA_URL + "/api/chat", timeout=timeout, json={
            "model": _llm.OLLAMA_MODEL, "stream": False, "think": False, "format": "json", "keep_alive": _llm.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]})
        return json.loads(r.json()["message"]["content"])
    return read


def ollama_tools(timeout: float = 20.0, key: str | None = None, events: list | None = None):
    """`tools` for `step`: the local model picks one of TOOLS for a request on the finished song (function calling
    through Ollama's /api/chat), or None when Ollama is not running.  `step` checks the call before it runs."""
    import json
    from . import llm as _llm, usergemini as _ug
    if _llm.ollama_status() is not None and not _ug.valid(key):
        return None

    def pick(text: str, song: dict | None, sound_options: list[dict], history: list[dict] | None = None) -> dict | None:
        import httpx
        parts = ", ".join(f"{x.get('label', '')}" for x in (song or {}).get("sections", []))
        options = "\n".join(f"- {o['id']}: {_label(o)}. {o.get('help', '')}" for o in sound_options)
        system = ("너는 작곡메이트의 대화 도우미다. 완성된 곡에 대한 사용자의 요청을 도구 하나를 불러 처리한다. 음표나 코드를 "
                  "직접 지어내지 않는다. 구간의 멜로디를 다시 쓰라는 요청은 rewrite_section, 소리 옵션이 하는 일을 바라는 "
                  "요청은 그 옵션을 set_sound_option으로 켜고(on=true), 그만하라는 요청은 끈다(on=false). 질문은 explain. "
                  "새 악기나 솔로 추가, 가사, 음표를 직접 고치는 일은 할 수 없으니 ask_user로 할 수 없다고 알리고 할 수 있는 "
                  "것(구간 다시 쓰기, 소리 옵션)을 제안한다. 함수에 넣을 값을 추측하지 않는다: 어느 구간인지, 어느 옵션인지, "
                  "켜는지 끄는지가 말에서 분명하지 않으면 ask_user로 보기를 들어 한 가지만 되묻는다. 앞선 대화가 있으면 "
                  "처음 요청과 사용자의 답을 함께 읽어 도구를 고른다.\n"
                  f"이 곡의 구간: {parts or '모름'}.\n쓸 수 있는 소리 옵션:\n{options or '없음'}")
        names = [t["name"] for t in TOOLS]
        said = _ug.try_ask(key, events, system + "\n도구 목록(JSON): " + json.dumps(TOOLS, ensure_ascii=False)
                           + '\n도구를 하나 골라 {"name": 도구 이름, "args": {...}} JSON 하나로만 답한다.',
                           json.dumps([*(history or []), {"user": text}], ensure_ascii=False), timeout)
        if said is not None:
            try:
                got = _ug.json_object(said)
                if got.get("name") in names and isinstance(got.get("args", {}), dict):
                    return {"name": got["name"], "args": got.get("args", {})}
            except ValueError:
                pass                                       # not a usable pick: the local model picks instead
        r = httpx.post(_llm.OLLAMA_URL + "/api/chat", timeout=timeout, json={
            "model": _llm.OLLAMA_MODEL, "stream": False, "think": False, "keep_alive": _llm.OLLAMA_KEEP_ALIVE,
            "tools": _llm._chat_tools(TOOLS), "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": text}]})
        message = r.json()["message"]
        calls = message.get("tool_calls") or []
        if not calls:                                      # it asked back in plain words instead of calling ask_user
            said = (message.get("content") or "").strip()
            return {"name": "ask_user", "args": {"question": said}} if said else None
        fn = calls[0].get("function") or {}
        args = fn.get("arguments")
        return {"name": fn.get("name"), "args": json.loads(args) if isinstance(args, str) else args}
    return pick


def ollama_text(timeout: float = 30.0, key: str | None = None, events: list | None = None):
    """`llm(system, user) -> str` for instmate/explain.py, backed by the local model; None when Ollama is not running."""
    from . import llm as _llm, usergemini as _ug
    if _llm.ollama_status() is not None and not _ug.valid(key):
        return None

    def write(system: str, user: str) -> str:
        import httpx
        said = _ug.try_ask(key, events, system, user, timeout)
        if said is not None:
            return said
        r = httpx.post(_llm.OLLAMA_URL + "/api/chat", timeout=timeout, json={
            "model": _llm.OLLAMA_MODEL, "stream": False, "think": False, "keep_alive": _llm.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        return r.json()["message"]["content"]
    return write
