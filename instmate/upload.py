"""Streaming multipart/form-data reader for ONE uploaded file (no python-multipart dependency).

`save_first_file(chunks, content_type, dest_dir, max_bytes, allowed_ext)` reads an async iterable of body chunks,
writes the first file part to `dest_dir/<uuid><ext>` while it streams (never the whole body in memory) and returns
the path. Anything that goes wrong raises `UploadError(message, status)` and removes the partial file.

The parser is a small state machine (`_Parser.feed`) that does not depend on how the body is cut into chunks; the
boundary may be split across chunks anywhere. Only the extension of the client's file name is used (a fresh uuid is
the real name), so a hostile file name cannot point outside `dest_dir`.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

HEADER_MAX = 16 * 1024          # one part's header block
OVERHEAD = 1024 * 1024          # multipart framing + other fields allowed on top of max_bytes


class UploadError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def boundary_of(content_type: str | None) -> bytes:
    m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type or "", re.I)
    if not (content_type or "").lower().startswith("multipart/form-data") or not m:
        raise UploadError("multipart/form-data 형식으로 파일을 보내 주세요.")
    return (m.group(1) or m.group(2)).encode("latin-1")


def _filename(headers: bytes) -> str | None:
    """File name from a part's Content-Disposition, or None when the part is not a file."""
    for line in headers.decode("utf-8", "replace").split("\r\n"):
        if line.lower().startswith("content-disposition:"):
            m = re.search(r'filename\*?=(?:"([^"]*)"|([^;]+))', line, re.I)
            if m:
                return (m.group(1) if m.group(1) is not None else m.group(2)).strip()
    return None


class _Parser:
    PREAMBLE, AFTER_DELIM, HEADERS, BODY, DONE = range(5)

    def __init__(self, boundary: bytes, open_sink, max_bytes: int):
        self.delim = b"\r\n--" + boundary
        self.buf = b"\r\n"                       # the first boundary has no CRLF before it: add a virtual one
        self.state = self.PREAMBLE
        self.open_sink = open_sink               # name -> writable (or raises UploadError)
        self.max_bytes = max_bytes
        self.sink = None
        self.file_done = False
        self.written = 0

    def _emit(self, data: bytes) -> None:
        if self.sink is None or not data:
            return
        self.written += len(data)
        if self.written > self.max_bytes:
            raise UploadError(f"파일이 너무 커요(최대 {self.max_bytes // (1024 * 1024)}MB).", 413)
        self.sink.write(data)

    def feed(self, data: bytes) -> None:
        self.buf += data
        while self.state != self.DONE:
            if self.state == self.PREAMBLE:
                i = self.buf.find(self.delim)
                if i < 0:
                    self.buf = self.buf[-(len(self.delim) - 1):]
                    return
                self.buf = self.buf[i + len(self.delim):]
                self.state = self.AFTER_DELIM
            elif self.state == self.AFTER_DELIM:
                if len(self.buf) < 2:
                    return
                if self.buf[:2] == b"--":
                    self.state = self.DONE
                elif self.buf[:2] == b"\r\n":
                    self.buf = self.buf[2:]
                    self.state = self.HEADERS
                else:                                # transport padding or garbage after the boundary
                    raise UploadError("multipart 본문이 올바르지 않아요.")
            elif self.state == self.HEADERS:
                i = self.buf.find(b"\r\n\r\n")
                if i < 0:
                    if len(self.buf) > HEADER_MAX:
                        raise UploadError("multipart 헤더가 너무 길어요.")
                    return
                name = _filename(self.buf[:i])
                self.buf = self.buf[i + 4:]
                if name is not None and not self.file_done:
                    self.sink = self.open_sink(name)
                self.state = self.BODY
            else:                                    # BODY
                i = self.buf.find(self.delim)
                if i >= 0:
                    self._emit(self.buf[:i])
                    self.buf = self.buf[i + len(self.delim):]
                    if self.sink is not None:
                        self.file_done, self.sink = True, None
                    self.state = self.AFTER_DELIM
                else:
                    keep = len(self.delim) - 1
                    if len(self.buf) > keep:
                        self._emit(self.buf[:-keep])
                        self.buf = self.buf[-keep:]
                    return

    def finish(self) -> None:
        if self.state != self.DONE or not self.file_done:
            raise UploadError("파일을 받지 못했어요. 파일 하나를 골라 다시 올려 주세요.")


async def save_first_file(chunks, content_type: str | None, dest_dir: Path, max_bytes: int,
                          allowed_ext: set[str], content_length: int | None = None) -> Path:
    """Stream the first file part of a multipart body into `dest_dir`. See the module docstring."""
    boundary = boundary_of(content_type)
    if content_length is not None and content_length > max_bytes + OVERHEAD:
        raise UploadError(f"파일이 너무 커요(최대 {max_bytes // (1024 * 1024)}MB).", 413)
    dest_dir.mkdir(parents=True, exist_ok=True)
    opened: list = []
    path_box: list[Path] = []

    def open_sink(name: str):
        ext = Path(name.replace("\\", "/")).suffix.lower()
        if ext not in allowed_ext:
            raise UploadError(f"지원하지 않는 형식이에요({ext or '확장자 없음'}). {', '.join(sorted(allowed_ext))} 파일을 올려 주세요.")
        path = dest_dir / f"{uuid.uuid4().hex}{ext}"
        f = open(path, "wb")
        path_box.append(path)
        opened.append(f)
        return f

    parser = _Parser(boundary, open_sink, max_bytes)
    total = 0
    try:
        async for chunk in chunks:
            total += len(chunk)
            if total > max_bytes + OVERHEAD:
                raise UploadError(f"파일이 너무 커요(최대 {max_bytes // (1024 * 1024)}MB).", 413)
            parser.feed(chunk)
        parser.finish()
    except BaseException:
        for f in opened:
            f.close()
        for p in path_box:
            p.unlink(missing_ok=True)
        raise
    for f in opened:
        f.close()
    return path_box[0]
