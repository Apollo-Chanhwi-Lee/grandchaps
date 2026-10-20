"""Agent loop: analyze -> choose tool -> generate -> self-check -> regenerate problem bars -> output."""
from __future__ import annotations

import base64
from typing import Callable

from . import chords as ch
from .arrange import INSTRUMENTS, Event, arrange
from . import check as chk
from .check import report
from .export import playback, to_midi, to_musicxml
from .mix import apply_levels, resolve as resolve_mix
from .prefs import Prefs
from .theory import BAR, Key, analyze_key, parse_number_input, split_by_bar

MAX_ROUNDS = 3

# Generation policy (decided 2026-09-30, before any test-split use): generate once and only report the clash
# check. The rule fixed in advance ("max val maj/min accuracy, mean clash <= ground truth 10.5%") picked
# "no regeneration" (runs/selfcheck_threshold_val.json). The check-driven loop runs on user edits (modify).
GENERATE_REGENERATE = False

# Chord-generation tools. The rule engine is always available; a trained model registers itself here later.
ChordTool = Callable[..., list[ch.ChordDef]]
CHORD_TOOLS: dict[str, ChordTool] = {"rule": ch.progression}


def register_chord_tool(name: str, fn: ChordTool) -> None:
    CHORD_TOOLS[name] = fn


def _pick_tool(trace: list[dict]) -> tuple[str, ChordTool]:
    name = "dl" if "dl" in CHORD_TOOLS else "rule"
    reason = "학습된 코드 모델이 등록되어 있어 사용" if name == "dl" else "학습된 코드 모델이 없어 규칙 기반 엔진 사용"
    trace.append({"step": "도구 선택", "tool": name, "detail": reason})
    return name, CHORD_TOOLS[name]


def _generate_with_checks(key, genre, melody, tool, trace, forced=None, banned=None, bias=None):
    banned = {i: set(s) for i, s in (banned or {}).items()}
    best = None
    for rnd in range(MAX_ROUNDS + 1):
        prog = tool(key, genre, melody, banned=banned, forced=forced, bias=bias)
        rep = report(key, melody, prog, [])
        romans = [c.roman for c in prog]
        trace.append({"step": "코드 생성" if rnd == 0 else f"재생성 {rnd}회차", "tool": "chords",
                      "detail": f"{' - '.join(romans)} / 충돌률 평균 {rep['mean_clash']}"})
        if best is None or rep["mean_clash"] < best[1]["mean_clash"]:
            best = (prog, rep)
        problems = [i for i in rep["problem_bars"] if not (forced and i in forced)]
        if not problems:
            trace.append({"step": "자체 검사", "tool": "check", "detail": "모든 마디가 기준 통과"})
            break
        if rnd == MAX_ROUNDS:
            trace.append({"step": "자체 검사", "tool": "check",
                          "detail": f"{MAX_ROUNDS}회 재생성 후에도 마디 {[p + 1 for p in problems]} 기준 초과, 가장 좋은 결과 사용"})
            break
        trace.append({"step": "자체 검사", "tool": "check",
                      "detail": f"마디 {[p + 1 for p in problems]} 충돌률 {[rep['clash_rates'][p] for p in problems]}"
                                f" > {rep['threshold']}, 해당 마디만 다시 생성"})
        for p in problems:
            banned.setdefault(p, set()).add(romans[p])
            if len(banned[p]) >= len(ch.vocab(key)):   # never ban every chord
                banned[p].discard(romans[p])
    return best[0]


def _finish(key, genre, melody, prog, instruments, energy, trace, analysis, events=None, mix=None):
    """Assemble a result. `events` stay canonical; the mix (tempo, levels) is applied only to what is rendered."""
    mix = mix or resolve_mix(genre)
    if events is None:
        events = arrange(key, genre, melody, prog, instruments, energy)
    rep = report(key, melody, prog, events)
    if rep["range_violations"] or rep["length_violations"]:
        raise RuntimeError(f"arrangement check failed: {rep['range_violations'][:3]} {rep['length_violations'][:3]}")
    trace.append({"step": "편곡 검사", "tool": "check", "detail": "악기 음역과 마디 길이 모두 정상"})
    bars = split_by_bar(melody)
    return {
        "status": "ok",
        "key": key.to_dict(),
        "key_analysis": analysis,
        "genre": genre,
        "instruments": instruments,
        "chords": [{"bar": i, "roman": c.roman, "symbol": ch.chord_symbol(key, c),
                    "reason": ch.explain_bar(key, c, bars[i]),
                    "alternatives": ch.alternatives(key, genre, melody, i)} for i, c in enumerate(prog)],
        "check": rep,
        "trace": trace,
        "tempo": mix["tempo"],
        "levels": dict(mix["levels"]),
        "playback": playback(apply_levels(events, mix["levels"]), genre, mix["tempo"]),
        "musicxml": to_musicxml(key, genre, melody, prog, mix["tempo"]),
        "midi_b64": base64.b64encode(to_midi(apply_levels(events, mix["levels"]), genre, mix["tempo"])).decode(),
        "events": [e.to_dict() for e in events],   # exact tick-level arrangement; the base for later bar edits
    }


def _validate(genre, instruments):
    if genre not in ch.GENRES:
        raise ValueError(f"genre must be one of {ch.GENRES}")
    bad = set(instruments) - set(INSTRUMENTS)
    if bad:
        raise ValueError(f"unknown instruments {sorted(bad)}")


def merge_bias(*biases: dict[str, float] | None) -> dict[str, float]:
    """Sum several {roman: bonus} dicts (preference + style + bridge darkness combine, none replaces another)."""
    out: dict[str, float] = {}
    for b in biases:
        for roman, v in (b or {}).items():
            out[roman] = out.get(roman, 0) + v
    return out


def forced_from_symbols(key: Key, symbols: list[str | None], tool, trace: list[dict]) -> dict[int, str]:
    """{bar: roman} for the chosen chord symbols that are chords of this key (and that the tool can be held to)."""
    by_symbol = {}
    for c in ch.vocab(key) + ch.ext_vocab(key):
        by_symbol.setdefault(ch.chord_symbol(key, c), c.roman)
    if tool is ch.progression:
        expressible = lambda r: True
    else:                                        # a learned tool only knows the 24 major / minor classes
        from .ml import chord_to_class
        expressible = lambda r: chord_to_class(ch.chord_by_roman(key, r)) is not None
    forced, free = {}, []
    for i, sym in enumerate(symbols):
        roman = by_symbol.get(sym) if sym else None
        if roman is not None and expressible(roman):
            forced[i] = roman
        elif sym:
            free.append(f"{i + 1}마디 {sym}")
    trace.append({"step": "코드 고정", "tool": "user", "detail": f"고른 코드 {len(forced)}마디를 그대로 사용"
                  + (f"; 이 조의 코드가 아니어서 앱이 다시 고르는 마디: {', '.join(free)}" if free else "")})
    return forced


# chords a section (and the song) may end on, in order of preference; "Iadd9" is tried where the vocabulary has it
CLOSING = {"major": ("I", "Iadd9", "vi"), "minor": ("i", "iadd9", "VI")}


def fit_seed_chords(key: Key, genre: str, melody: list[Note], romans: list[str],
                    fixed: set[int] | frozenset[int] = frozenset()) -> tuple[list[str], list[dict]]:
    """The seed's chords with every bar that fails the clash check given the chord of this key and genre that fits
    the bar's strong-beat melody notes best: (romans, [{"bar", "from", "to", "rate_from", "rate_to"}]).
    The melody is never touched.  A bar in `fixed` (a chord the user chose) and a bar that already passes stay; a
    bar is changed only when another chord brings it under the check's limit.  Among equally good chords the one
    that holds most of the bar's melody as chord notes wins, then the order of the key's vocabulary.
    The seed's LAST bar ends every chorus and, with the last chorus, the song: it is not open to any chord.  Only
    the tonic family is tried there, in the order of CLOSING (the first that passes), and when none passes the bar
    keeps its chord and its clash: a song that ends away from home was judged worse than one reported clash.
    Used for the song's chorus (composer.compose); the plain accompaniment keeps its report-only policy."""
    from .check import CLASH_MAX, clash_rates
    from .theory import Note
    bars = split_by_bar(melody)
    out, changes = list(romans), []
    cands = ch.vocab_for(key, genre)
    for i, notes in enumerate(bars[:len(out)]):
        if i in fixed:
            continue
        bar = [Note(n.pitch, n.start % BAR, n.dur) for n in notes]
        rate = lambda c: clash_rates(key, bar, [c])[0]
        now = rate(ch.chord_by_roman(key, out[i]))
        if now <= CLASH_MAX:
            continue
        held = lambda c: sum(n.dur for n in bar if n.pitch is not None and ch.note_role(key, c, n.pitch % 12) == "chord")
        if i == len(out) - 1:
            closing = [c for r in CLOSING[key.mode] for c in cands if c.roman == r and rate(c) <= CLASH_MAX]
            if not closing:
                continue
            best = cands.index(closing[0])
        else:
            best = min(range(len(cands)), key=lambda j: (rate(cands[j]), -held(cands[j]), j))
        if rate(cands[best]) <= CLASH_MAX:
            changes.append({"bar": i, "from": out[i], "to": cands[best].roman, "rate_from": round(now, 3),
                            "rate_to": round(rate(cands[best]), 3)})
            out[i] = cands[best].roman
    return out, changes


def run(tokens: list[dict], input_tonic: int, genre: str, instruments: list[str],
        key_choice: dict | None = None, energy: int = 0, prefs: Prefs | None = None,
        tempo: int | None = None, levels: dict | None = None, bias: dict[str, float] | None = None,
        seed_chords: list[str | None] | None = None) -> dict:
    """`bias`: extra {roman: bonus} (e.g. a confirmed style's chord_bias) added to the stored preference bias.
    `seed_chords`: chord symbols the user chose for the seed's bars (e.g. the score comparison's AI chords); a bar whose
    symbol is a chord of this key is fixed to it, the others (None, or a chord outside the key's vocabulary) are left to
    the chord tool and listed in the trace."""
    _validate(genre, instruments)
    extra_bias = bias
    mix = resolve_mix(genre, tempo, levels)
    trace: list[dict] = []
    melody = parse_number_input(tokens, input_tonic)
    trace.append({"step": "입력", "tool": "parse", "detail": f"음표 {len(melody)}개, {len(split_by_bar(melody))}마디"})
    analysis = analyze_key(melody, input_tonic)
    top = analysis["candidates"][0]["key"]
    trace.append({"step": "조성 분석", "tool": "key",
                  "detail": f"1순위 {top['label']} (점수 차 {analysis['gap']})"})
    if key_choice is None:
        if analysis["ambiguous"]:
            trace.append({"step": "되묻기", "tool": "ask", "detail": analysis["question"]})
            return {"status": "need_key", "question": analysis["question"], "key_analysis": analysis, "trace": trace}
        key = Key(top["tonic"], top["mode"])
    else:
        key = Key(int(key_choice["tonic"]), key_choice["mode"])
        trace.append({"step": "조성 확정", "tool": "user", "detail": f"사용자 선택 {key.label_ko}"})
    bias = prefs.bias(genre, key.mode) if prefs else {}
    if bias:
        trace.append({"step": "취향 반영", "tool": "memory", "detail": f"이전 수정 기록 {bias}"})
    if extra_bias:
        bias = merge_bias(bias, extra_bias)
        trace.append({"step": "스타일 반영", "tool": "style", "detail": f"코드 쏠림 {extra_bias}"})
    _, tool = _pick_tool(trace)
    forced = forced_from_symbols(key, seed_chords, tool, trace) if seed_chords else None
    if GENERATE_REGENERATE:
        prog = _generate_with_checks(key, genre, melody, tool, trace, forced=forced, bias=bias)
    else:
        prog = tool(key, genre, melody, forced=forced, bias=bias)
        rep = report(key, melody, prog, [])
        trace.append({"step": "코드 생성", "tool": "chords",
                      "detail": f"{' - '.join(c.roman for c in prog)} / 충돌률 평균 {rep['mean_clash']}"})
        probs = rep["problem_bars"]
        trace.append({"step": "자체 검사 (관찰)", "tool": "check",
                      "detail": (f"마디 {[p + 1 for p in probs]} 충돌률 {[rep['clash_rates'][p] for p in probs]} > "
                                 f"{rep['threshold']}. 생성 정책상 재생성하지 않음, 마디 수정으로 고칠 수 있어요")
                      if probs else "모든 마디가 기준 통과"})
    return _finish(key, genre, melody, prog, instruments, energy, trace, analysis, mix=mix)


MOOD_KO = {"darker": "어둡게", "brighter": "밝게"}
MODIFY_ATTEMPTS = 4          # tool calls per modify request (budget)

# Completion status of a modify request. Only "ok" and "partial" results may be adopted as a version.
ADOPTABLE = {"ok", "partial"}


def _in_bar(e: Event, bar: int) -> bool:
    return e.start // BAR == bar


def _all_romans(key: Key) -> set[str]:
    """Every chord name any tool can return (rule vocabulary + the learned models' 24 classes): the ban universe."""
    from .ml import N_CLASSES, class_to_chord
    return {c.roman for c in ch.vocab(key)} | {class_to_chord(key, cid).roman for cid in range(N_CLASSES)}


def mood_candidates(key: Key, mood: str, current: str) -> list[str]:
    """In-key chords (rule vocabulary) with the mood's flag, minus the current chord.

    Chromatic chords (e.g. D or E major in C major) are never offered for "brighter", whichever tool runs.
    """
    flag = "dark" if mood == "darker" else "bright"
    return [c.roman for c in ch.vocab(key) if getattr(c, flag) and c.roman != current]


def _event_key(e: dict) -> tuple:
    return (e["track"], e["pitch"], e["start"], e["dur"], e["velocity"])


def verify_locks(base: list[dict], result: dict, current: list[str], bars) -> list[str]:
    """Check the returned result, not the intermediate list it was built from.

    Locked: melody events, every event starting outside `bars`, and every chord outside `bars`.
    The bar of an event is computed here from its start tick (not via the assembly helper), so an assembly bug
    cannot also blind this check.
    """
    bars = {bars} if isinstance(bars, int) else set(bars)
    label = ", ".join(str(b + 1) for b in sorted(bars))
    out = []
    outside = lambda evs: sorted(_event_key(e) for e in evs if e["start"] // BAR not in bars)
    melody = lambda evs: sorted(_event_key(e) for e in evs if e["track"] == "melody")
    if outside(base) != outside(result["events"]):
        out.append(f"마디 {label} 밖의 반주 이벤트가 기준 버전과 다름")
    if melody(base) != melody(result["events"]):
        out.append("멜로디 이벤트가 기준 버전과 다름")
    after = [c["roman"] for c in result["chords"]]
    if [r for i, r in enumerate(after) if i not in bars] != [r for i, r in enumerate(current) if i not in bars]:
        out.append(f"마디 {label} 밖의 코드가 바뀜")
    return out


def _normalize_bars(bar, n: int) -> list[int]:
    bars = [bar] if isinstance(bar, int) else list(bar)
    if not bars:
        raise ValueError("no bar selected")
    if any(isinstance(b, bool) or not isinstance(b, int) for b in bars):
        raise ValueError("bars must be integers")
    if len(set(bars)) != len(bars):
        raise ValueError("duplicate bar")
    if not all(0 <= b < n for b in bars):
        raise ValueError(f"bar must be 0-{n - 1}")
    return sorted(bars)


def modify(tokens: list[dict], input_tonic: int, genre: str, instruments: list[str], key_choice: dict,
           current: list[str], bar, mood: str, energy: int = 0, prefs: Prefs | None = None,
           base_events: list[dict] | None = None, allow_clash: bool = False,
           tempo: int | None = None, levels: dict | None = None) -> dict:
    """Change the chords of one bar or a set of bars (`bar`: int or list) to darker/brighter ones.

    All selected bars are decoded together with the same chord tool as generation (so chords inside the range
    connect), every other bar is fixed. After each tool call every still-open bar is clash-checked: a passing
    bar is fixed for later attempts, a failing bar's candidate is banned and only those bars are asked again.
    Outcomes (result["status"]):
      ok              - every selected bar changed and passed the check; nothing outside the selection changed
      partial         - changed although some bars still clash (only when the user sent allow_clash)
      need_input      - a bar has no in-key mood chord left that the tool can return; nothing adopted, the user is asked
      budget_exceeded - MODIFY_ATTEMPTS tool calls used with bars still failing; nothing adopted
    Arrangement events outside the selection are copied from `base_events` (the adopted version's events), or
    from a fresh arrangement of `current` when no base is given.
    """
    _validate(genre, instruments)
    mix = resolve_mix(genre, tempo, levels)
    if mood not in MOOD_KO:
        raise ValueError("mood must be 'darker' or 'brighter'")
    melody = parse_number_input(tokens, input_tonic)
    key = Key(int(key_choice["tonic"]), key_choice["mode"])
    n = len(split_by_bar(melody))
    if len(current) != n:
        raise ValueError(f"current has {len(current)} chords but melody has {n} bars")
    bars = _normalize_bars(bar, n)
    for r in current:
        ch.chord_by_roman(key, r)            # ValueError for an unknown chord name
    label = ", ".join(str(b + 1) for b in bars)
    universe = _all_romans(key)
    allowed = {b: mood_candidates(key, mood, current[b]) for b in bars}
    banned = {b: universe - set(allowed[b]) for b in bars}
    trace = [{"step": "수정 요청", "tool": "user", "detail": f"마디 {label}을(를) {MOOD_KO[mood]}"
                                                         + (" (충돌 허용)" if allow_clash else "")}]
    tool_name, tool = _pick_tool(trace)
    if tool is ch.progression:
        expressible = lambda r: True
    else:
        # A learned tool only knows 24 maj/min classes; it cannot be forced to e.g. vii°. Such bars are left free
        # for decoding and restored afterwards (only the selected bars may change).
        from .ml import chord_to_class
        expressible = lambda r: chord_to_class(ch.chord_by_roman(key, r)) is not None
    outside = {i: r for i, r in enumerate(current) if i not in bars and expressible(r)}

    tried: dict[int, list[dict]] = {b: [] for b in bars}
    fixed: dict[int, str] = {}               # bars that already passed: kept for later attempts
    status = None
    for attempt in range(MODIFY_ATTEMPTS):
        open_bars = [b for b in bars if b not in fixed]
        exhausted = [b for b in open_bars if not set(allowed[b]) - banned[b]]
        if exhausted:
            status = "need_input"
            break
        forced = {**outside, **{b: r for b, r in fixed.items() if expressible(r)}}
        try:
            out = tool(key, genre, melody, banned={b: set(banned[b]) for b in open_bars}, forced=forced)
        except ch.NoCandidate:                  # the tool cannot return any remaining candidate
            status = "need_input"
            break
        cand = [ch.chord_by_roman(key, fixed[i]) if i in fixed else out[i] if i in bars
                else ch.chord_by_roman(key, current[i]) for i in range(n)]
        rates = report(key, melody, cand, [])["clash_rates"]
        passed, failed = [], []
        for b in open_bars:
            tried[b].append({"roman": cand[b].roman, "symbol": ch.chord_symbol(key, cand[b]), "clash": rates[b]})
            if rates[b] <= chk.CLASH_MAX:
                fixed[b] = cand[b].roman
                passed.append(f"{b + 1}마디 {cand[b].roman}({rates[b]})")
            else:
                banned[b].add(cand[b].roman)
                failed.append(f"{b + 1}마디 {cand[b].roman}({rates[b]})")
        trace.append({"step": "자체 검사", "tool": "check", "detail":
                      (f"통과·고정: {', '.join(passed)}" if passed else "")
                      + ("; " if passed and failed else "")
                      + (f"기준 {chk.CLASH_MAX} 초과, 다른 후보 시도: {', '.join(failed)}" if failed else "")})
        if len(fixed) == len(bars):
            status = "ok"
            break
    else:
        status = "budget_exceeded"

    failing = [b for b in bars if b not in fixed]
    best = {b: min(tried[b], key=lambda t: t["clash"]) for b in failing if tried[b]}   # ties: earlier attempt
    can_force = bool(failing) and len(best) == len(failing)
    if status != "ok" and allow_clash and can_force:
        status = "partial"
        trace.append({"step": "사용자 판단 반영", "tool": "user", "detail": "충돌을 허용해, 시도한 후보 중 가장 덜 부딪히는 코드 적용: "
                      + ", ".join(f"{b + 1}마디 {best[b]['roman']}({best[b]['clash']})" for b in failing)})
        fixed.update({b: best[b]["roman"] for b in failing})

    if status not in ADOPTABLE:
        parts = []
        for b in failing:
            if not allowed[b]:
                parts.append(f"{b + 1}마디는 {MOOD_KO[mood]} 바꿀 조성 안의 코드가 없어요")
            elif status == "need_input" and not set(allowed[b]) - banned[b]:
                parts.append(f"{b + 1}마디는 쓸 수 있는 코드 {len(allowed[b])}개를 모두 시도했지만 멜로디와 부딪혀요")
            else:
                parts.append(f"{b + 1}마디는 후보 {len(allowed[b])}개 중 {len(tried[b])}개를 시도했지만 기준을 통과하지 못했어요")
        question = ". ".join(parts) + "."
        if status == "budget_exceeded":
            question = f"{MODIFY_ATTEMPTS}번 시도했지만 끝나지 않았어요. " + question
        if can_force:
            question += " 시도한 것 중 가장 덜 부딪히는 코드(" + ", ".join(
                f"{b + 1}마디 {best[b]['symbol']} {best[b]['clash']}" for b in failing) + ")로 그래도 바꿀까요?"
        trace.append({"step": "되묻기" if status == "need_input" else "예산 초과로 중단", "tool": "ask", "detail": question})
        return {"status": status, "question": question, "can_force": can_force, "trace": trace,
                "tool_result": {"tool": tool_name, "bars": bars, "changed_bars": [], "before": {"chords": list(current)},
                                "after": {"chords": list(current)}, "tried": tried, "candidates": allowed,
                                "problem_bars": failing, "lock_violations": [], "status": status}}

    prog = [ch.chord_by_roman(key, fixed[i] if i in fixed else current[i]) for i in range(n)]
    new = {b: prog[b].roman for b in bars}
    trace.append({"step": "코드 교체", "tool": tool_name,
                  "detail": ", ".join(f"{b + 1}마디 {current[b]} → {new[b]}" for b in bars)})
    if base_events is None:
        prior = arrange(key, genre, melody, [ch.chord_by_roman(key, r) for r in current], instruments, energy)
    else:
        prior = [Event(**e) for e in base_events]
    fresh = arrange(key, genre, melody, prog, instruments, energy)
    # Trade-off: bars after the selection keep their old voicing, so the voice leading out of it may jump.
    events = [e for e in prior if not any(_in_bar(e, b) for b in bars)] + \
             [e for e in fresh if any(_in_bar(e, b) for b in bars)]
    result = _finish(key, genre, melody, prog, instruments, energy, trace, None, events, mix=mix)
    locks = verify_locks([e.to_dict() for e in prior], result, current, bars)
    if locks:
        raise RuntimeError(f"lock violation: {locks}")
    result["trace"].append({"step": "범위 유지 검사", "tool": "check",
                            "detail": f"최종 결과의 마디 {label} 밖 반주·멜로디·코드가 기준 버전과 동일"})
    result["status"] = status
    if prefs:   # only an adopted result teaches the preference
        for b in bars:
            prefs.record(genre, key.mode, new[b])
    result["change"] = {"bar": bars[0], "bars": bars, "from": current[bars[0]], "to": new[bars[0]],
                        "from_all": [current[b] for b in bars], "to_all": [new[b] for b in bars], "mood": mood}
    result["tool_result"] = {"tool": tool_name, "bars": bars, "changed_bars": bars,
                             "before": {"chords": list(current)}, "after": {"chords": [c.roman for c in prog]},
                             "tried": tried, "problem_bars": result["check"]["problem_bars"],
                             "lock_violations": locks, "status": status}
    return result


def remix(state: dict, base_result: dict, tempo: int | None = None, levels: dict | None = None) -> dict:
    """Change only the mix (tempo, per-track levels) of an adopted version; nothing is re-arranged.

    `state` is the version's input state (tokens, input_tonic, genre, instruments, energy, key_choice, chords);
    `base_result` its stored result. Unspecified settings keep the base version's values.
    """
    genre = state["genre"]
    old = {"tempo": base_result["tempo"], "levels": dict(base_result["levels"])}
    mix = resolve_mix(genre, old["tempo"] if tempo is None else tempo, {**old["levels"], **(levels or {})})
    changed = {}
    if mix["tempo"] != old["tempo"]:
        changed["tempo"] = [old["tempo"], mix["tempo"]]
    lv = {t: [old["levels"][t], v] for t, v in mix["levels"].items() if v != old["levels"][t]}
    if lv:
        changed["levels"] = lv
    if not changed:
        raise ValueError("mix unchanged")
    melody = parse_number_input(state["tokens"], state["input_tonic"])
    key = Key(int(state["key_choice"]["tonic"]), state["key_choice"]["mode"])
    prog = [ch.chord_by_roman(key, r) for r in state["chords"]]
    events = [Event(**e) for e in base_result["events"]]
    parts = []
    if "tempo" in changed:
        parts.append(f"템포 {changed['tempo'][0]} → {changed['tempo'][1]}")
    for track, (a, b) in changed.get("levels", {}).items():
        parts.append(f"{track} 세기 {a} → {b}")
    trace = [{"step": "믹스 변경", "tool": "mix", "detail": ", ".join(parts)}]
    result = _finish(key, genre, melody, prog, state["instruments"], state["energy"], trace,
                     base_result.get("key_analysis"), events, mix=mix)
    locks = []
    if result["events"] != base_result["events"]:
        locks.append("편곡 이벤트가 바뀜")
    if [c["roman"] for c in result["chords"]] != state["chords"]:
        locks.append("코드가 바뀜")
    if locks:
        raise RuntimeError(f"lock violation: {locks}")
    result["trace"].append({"step": "범위 유지 검사", "tool": "check", "detail": "코드·멜로디·편곡 이벤트는 기준 버전과 동일, 믹스만 변경"})
    result["tool_result"] = {"tool": "mix", "changed": changed, "changed_bars": [], "lock_violations": locks,
                             "status": "ok"}
    return result
