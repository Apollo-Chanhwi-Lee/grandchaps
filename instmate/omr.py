"""A score PDF read as a melody: PDF -> page images (poppler) -> MusicXML (homr, optical music recognition) -> the
app's score (scoreimport.read), pages joined in order.

    read_pdf(pdf, work) -> score      the same dict scoreimport.read returns, plus "omr" and exact = False

Recognition is a guess from a picture: notes can be missed or misread.  So the result is never "the file as written":
it is a converted version (`exact` False) that the user has to look at and accept before anything is made from it.

homr (AGPL-3.0) runs as its own program in its own environment (models/omr/homr_venv, `INSTMATE_OMR_BIN`); it is not
part of the app and is not shipped with it.  Without it, or without poppler's pdftoppm / pdfinfo, PDFs are refused
with the reason and MIDI / MusicXML keep working.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import scoreimport

ROOT = Path(__file__).resolve().parent.parent
MAX_PAGES = 30                     # a long score: every page is read in turn (tens of seconds each)
MAX_BYTES = 30 * 1024 * 1024
DPI = 200
PAGE_TIMEOUT = 240                 # seconds for one page (the first page also loads the models)
NOTE = ("PDF 악보를 그림으로 읽어 인식한 결과예요. 빠지거나 틀리게 읽힌 음이 있을 수 있어요. "
        "아래에 마디별로 읽은 음 이름을 적었어요. 악보와 비교해 확인한 뒤 진행해 주세요. 템포가 악보에 없으면 120으로 둬요.")


class OmrError(scoreimport.ScoreImportError):
    """The PDF cannot be read as a score (Korean message)."""


def _bin() -> Path:
    return Path(os.environ.get("INSTMATE_OMR_BIN") or ROOT / "models" / "omr" / "homr_venv" / "bin" / "homr")


def unavailable_reason() -> str | None:
    if not shutil.which("pdftoppm") or not shutil.which("pdfinfo"):
        return "PDF를 그림으로 바꾸는 도구(poppler)가 없어요. MIDI나 MusicXML로 올려 주세요."
    if not _bin().exists():
        return "악보 인식 프로그램(homr)이 설치돼 있지 않아요. MIDI나 MusicXML로 올려 주세요."
    return None


def page_count(pdf: Path) -> int:
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise OmrError("PDF를 여는 데 너무 오래 걸려요.") from None
    pages = [line.split(":", 1)[1].strip() for line in out.stdout.splitlines() if line.startswith("Pages:")]
    if out.returncode != 0 or len(pages) != 1 or not pages[0].isdigit() or int(pages[0]) < 1:
        raise OmrError("PDF를 열지 못했어요. 암호가 걸렸거나 손상된 파일일 수 있어요.")
    return int(pages[0])


def to_images(pdf: Path, work: Path) -> list[Path]:
    """One PNG per page, in page order."""
    try:
        run = subprocess.run(["pdftoppm", "-r", str(DPI), "-png", str(pdf), str(work / "page")],
                             capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise OmrError("PDF를 그림으로 바꾸는 데 너무 오래 걸려요.") from None
    pages = sorted(work.glob("page-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    if run.returncode != 0 or not pages:
        raise OmrError("PDF를 그림으로 바꾸지 못했어요.")
    return pages


def recognize(image: Path) -> Path | None:
    """The page's MusicXML next to the image, or None when no staff was found on the page."""
    out = image.with_suffix(".musicxml")
    try:
        subprocess.run([str(_bin()), str(image)], capture_output=True, text=True, timeout=PAGE_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise OmrError("악보 인식이 너무 오래 걸려 멈췄어요. 쪽수를 줄여 다시 올려 주세요.") from None
    return out if out.exists() and out.stat().st_size > 0 else None


def join(pages: list[dict]) -> dict:
    """Scores of consecutive pages as one score: page n starts on the bar after page n - 1."""
    if not pages:
        raise OmrError("읽을 수 있는 악보가 없어요.")
    first = pages[0]
    if any(p["meter"] != first["meter"] or p["bar_ticks"] != first["bar_ticks"] for p in pages):
        raise OmrError("쪽마다 박자가 다르게 읽혔어요. 박자가 바뀌는 곡은 MusicXML로 올려 주세요.")
    notes, changes, bar0 = [], [], 0
    for p in pages:
        off = bar0 * first["bar_ticks"]
        notes += [{**n, "start": n["start"] + off} for n in p["notes"]]
        changes += [{**c, "bar": c["bar"] + bar0} for c in p.get("changes") or []]
        bar0 += p["bars"]
    whole = all(isinstance(p.get("chords"), list) for p in pages)
    return {**first, "notes": notes, "bars": bar0,
            "chords": [c for p in pages for c in p["chords"]] if whole else None,
            "chords_note": first.get("chords_note") if whole else "PDF에서 코드 기호를 읽지 못했어요. 코드는 앱이 붙여요.",
            "changes": changes[:scoreimport.MAX_CHANGES],
            "changes_total": sum(int(p.get("changes_total") or 0) for p in pages),
            "snapped": sum(int(p.get("snapped") or 0) for p in pages),
            "dropped": sum(int(p.get("dropped") or 0) for p in pages),
            "range": [min(n["pitch"] for n in notes), max(n["pitch"] for n in notes)]}


def read_pdf(pdf: Path, work: Path) -> dict:
    """See the module docstring.  `work`: an empty directory for the page images (the caller deletes it)."""
    reason = unavailable_reason()
    if reason:
        raise OmrError(reason)
    with open(pdf, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise OmrError("PDF 파일이 아니에요.")
    count = page_count(pdf)
    if count > MAX_PAGES:
        raise OmrError(f"PDF가 {count}쪽이에요. 한 번에 {MAX_PAGES}쪽까지 읽을 수 있어요. 멜로디가 있는 쪽만 남겨 올려 주세요.")
    scores, empty = [], []
    for i, image in enumerate(to_images(pdf, work), 1):
        xml = recognize(image)
        if xml is None:
            empty.append(i)
            continue
        try:
            scores.append(scoreimport.read(xml))
        except scoreimport.ScoreImportError:
            empty.append(i)                        # a page with staves but no readable melody (e.g. a title page)
    if not scores:
        raise OmrError("이 PDF에서 악보를 읽지 못했어요. 스캔이 흐리거나 손으로 쓴 악보는 인식이 어려워요. "
                       "MIDI나 MusicXML로 올려 주세요.")
    score = join(scores)
    score["exact"] = False                          # a recognition result is never the file note for note
    score["omr"] = {"engine": "homr", "pages": count, "pages_read": len(scores), "pages_empty": empty, "note": NOTE}
    score["key_note"] = f"{score.get('key_note') or ''} · PDF 인식".strip(" ·")
    return score


__all__ = ["read_pdf", "join", "unavailable_reason", "OmrError", "MAX_PAGES", "MAX_BYTES", "NOTE"]
