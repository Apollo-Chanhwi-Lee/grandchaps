"""Answers to "why" and "what is" questions, looked up in the app's own explanations (configs/explain.yaml).

    answer(question, song=None, llm=None) -> {"text", "sources": [title, ...], "found": bool}

Retrieval-augmented generation in its smallest form:
  1. index     every passage (title + the words people ask with + text) as weighted character pairs
  2. retrieve  the passages nearest to the question (cosine of those weights), none when nothing is near
  3. augment   the passages and a few facts of the open song are put before the question
  4. generate  the language model answers from them only (`llm(system, user) -> str`)
  5. check     the answer is shown only when its wording stays on that material (`grounded`); an answer that adds
               its own reasons is replaced by the nearest passage as written, and "not in the material" is passed on

Without a model, or when it fails, the nearest passage is given as written, so this works with no model.  The
answer always names the passages it rests on.  Character pairs, not word vectors: Korean words change their endings,
and no embedding model has to be installed.  Nothing here changes a song.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "explain.yaml"
TOP = 3                 # passages given to the model
NEAR = 0.12             # cosine below which a passage is not about the question
ASK_WORDS = ("왜", "뭐야", "뭔가요", "무엇", "무슨", "어떻게", "알려", "설명", "궁금", "뜻", "?", "인가요", "나요", "까요", "어디까지", "어떤", "종류", "있는데", "있나", "뭐가", "뭘")
GROUNDED = 0.6          # share of an answer's character pairs that must come from the passages, the song and the question
SYSTEM = ("너는 작곡메이트의 안내원이다. 아래 '자료'에 적힌 내용만으로 사용자의 질문에 한국어 두세 문장, 해요체로 답한다. "
          "'왜'라는 질문에는 자료에 적힌 사실을 이유로 들어 답한다. 자료의 낱말을 그대로 쓰고, 자료에 없는 이유나 추측은 "
          "덧붙이지 않는다. 자료가 질문과 관계없으면 \"자료에 없어요\"라고만 답한다. 음표나 코드를 새로 지어내지 않는다.")


class ExplainError(ValueError):
    """configs/explain.yaml is not usable (the message names the entry)."""


def _pairs(text: str) -> Counter:
    t = re.sub(r"[^0-9a-z가-힣]+", " ", (text or "").lower())
    return Counter(w[i:i + 2] for w in t.split() for i in range(len(w) - 1)) if t.strip() else Counter()


@lru_cache(maxsize=1)
def index() -> tuple:
    """(passages, weight of every character pair, one weighted vector per passage)."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    passages = data.get("passages") if isinstance(data, dict) else None
    if not isinstance(passages, list) or not passages:
        raise ExplainError("configs/explain.yaml에 passages 목록이 없어요")
    seen = set()
    for n, p in enumerate(passages):
        if not isinstance(p, dict) or not all(isinstance(p.get(k), str) and p[k].strip() for k in ("id", "title", "text")):
            raise ExplainError(f"{n + 1}번째 설명에 id, title, text가 필요해요")
        if p["id"] in seen:
            raise ExplainError(f"설명 id가 겹쳐요: {p['id']}")
        seen.add(p["id"])
    counts = [_pairs(" ".join((p["title"], p.get("ask") or "", p["text"]))) for p in passages]
    df = Counter(g for c in counts for g in c)
    idf = {g: math.log(1 + len(passages) / n) for g, n in df.items()}
    vectors = [{g: (1 + math.log(n)) * idf[g] for g, n in c.items()} for c in counts]
    return passages, idf, vectors


def _cosine(a: dict, b: dict) -> float:
    dot = sum(v * b[g] for g, v in a.items() if g in b)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, top: int = TOP) -> list[dict]:
    """[{"id", "title", "text", "score"}] nearest first; empty when no passage is near the question."""
    passages, idf, vectors = index()
    q = {g: (1 + math.log(n)) * idf[g] for g, n in _pairs(question).items() if g in idf}
    scored = sorted(((_cosine(q, v), i) for i, v in enumerate(vectors)), reverse=True)
    return [{**{k: passages[i][k] for k in ("id", "title", "text")}, "score": round(s, 3)} for s, i in scored[:top] if s >= NEAR]


def is_question(text: str) -> bool:
    return any(w in (text or "") for w in ASK_WORDS)


def facts(song: dict | None) -> str:
    """What is known about the open song, as lines for the model (only what the page's summary carries)."""
    if not song:
        return ""
    lines = []
    for label, key in (("조", "key"), ("템포", "tempo"), ("장르", "genre")):
        if song.get(key):
            lines.append(f"{label}: {song[key]}")
    names = [f"{s.get('label', '')}({'올린 멜로디' if s.get('source') == 'seed' else '앱이 씀'})" for s in song.get("sections", [])]
    if names:
        lines.append("구성: " + " → ".join(names))
    return "\n".join(lines)


def grounded(text: str, material: str) -> float:
    """Share of the answer's character pairs that also occur in the material it was given (0..1)."""
    said, given = _pairs(text), _pairs(material)
    total = sum(said.values())
    return sum(n for g, n in said.items() if g in given) / total if total else 0.0


def answer(question: str, song: dict | None = None, llm=None) -> dict:
    found = retrieve(question)
    if not found:
        return {"text": "그 질문에 맞는 설명을 찾지 못했어요. 코드, 조, 구간, 보컬, 옵션에 대해 물어보실 수 있어요.", "sources": [], "found": False}
    sources = [p["title"] for p in found]
    text = None
    if llm is not None:
        material = "\n".join(f"[{p['title']}] {p['text']}" for p in found)
        about = facts(song)
        user = f"자료:\n{material}\n" + (f"\n지금 곡:\n{about}\n" if about else "") + f"\n질문: {question}"
        try:
            got = llm(SYSTEM, user)
            if isinstance(got, str) and "자료에 없" in got:      # near passages, but none answers this question
                return {"text": "그 부분은 설명 자료에 없어요. 코드, 조, 구간, 보컬, 옵션에 대해 물어보실 수 있어요.", "sources": [], "found": False}
            if isinstance(got, str) and got.strip() and grounded(got, user) >= GROUNDED:
                text = got.strip()[:600]                  # only an answer that stays on the material is shown
        except Exception:
            text = None
    if text is None:                                       # no model: the nearest passage as written
        text, sources = found[0]["text"], sources[:1]
    return {"text": text, "sources": sources, "found": True}


def say(result: dict) -> str:
    return result["text"] + (f" (근거: {', '.join(result['sources'])})" if result["sources"] else "")
