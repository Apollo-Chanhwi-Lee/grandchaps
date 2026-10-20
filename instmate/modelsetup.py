"""First-run model download of the locally installed app (user request 2026-10-20: models are downloaded on the first run,
with size, terms, progress and failure shown; generation runs on the recipient's own Mac).

Only the local app (start_mac.command sets INSTMATE_LOCAL_APP=1) may download; a public or demo server answers "off".  Nothing is
downloaded without the recipient's explicit yes to the terms.  What is fetched is pinned: one revision of the public MLX port, an
exact file list, exact sizes, and sha256 for the large weight files (the same revision and weights this project ran, checked
2026-10-20).  A model counts as installed only when every listed file has its exact size; a download that is running or failed is
never reported as done (review F04, F08).  The download runs in one background thread and can be asked again after a failure
(huggingface_hub resumes partial files).  The downloader is injectable so tests never touch the network.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "ahmadw/YuE2-3B-MLX"                  # public MLX port of m-a-p/YuE2-3B
REVISION = "fe0a9050fd658257b486b880422d8872ee1f81e3"     # pinned: files below are exactly this revision
# name -> (bytes, sha256 or None).  The runner needs generate.py, yue2_model.py, yue2_vae.py and the 8bit folder.
FILES = {
    "generate.py": (14916, None), "yue2_model.py": (8983, None), "yue2_vae.py": (2945, None),
    "8bit/config.json": (1042, None), "8bit/qwen.tiktoken": (2561218, None), "8bit/vae_config.json": (1378, None),
    "8bit/yue2_generation_config.json": (466, None),
    "8bit/vae.safetensors": (265441814, "dc7bea8e4f8e764be6ee2ce3291606a5381aaa20f5c4eb26e86ca81dd1eeac6e"),
    "8bit/model.safetensors": (4264488042, "bb6ff9895114be4aaa6e1b3a7d272eee55b54c9e619716c67355b45673fe3d7c"),
}
LICENSE = "CC BY-NC 4.0 (비상업: 수업·연습·비영리 용도로만)"
NEED_FREE_FACTOR = 1.25                      # free disk wanted before starting

_lock = threading.Lock()
_job = {"state": "idle", "phase": None, "error": None, "started": None, "finished": None}


def expected_bytes() -> int:
    return sum(size for size, _ in FILES.values())


def target() -> Path:
    return Path(os.environ.get("INSTMATE_YUE_DIR") or ROOT / "models" / "yue2_mlx" / "repo")


def enabled() -> bool:
    return os.environ.get("INSTMATE_LOCAL_APP") == "1"


def installed(path: Path | None = None) -> bool:
    """Every pinned file is present with its exact size.  (Content hashes are checked once, right after a download.)"""
    d = path or target()
    for name, (size, _) in FILES.items():
        p = d / name
        try:
            if not p.is_file() or p.stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def _bytes_in(path: Path) -> int:
    total = 0
    if path.is_dir():
        for p in path.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                pass
    return total


def status() -> dict:
    """What the page shows on the first run: size, terms, the pinned version and the progress of a running download."""
    with _lock:
        job = dict(_job)
    running = job["state"] == "running"
    done = installed() and not running and job["state"] != "error"
    total = expected_bytes()
    have = _bytes_in(target()) if running else (total if done else 0)
    state = "running" if running else "error" if job["state"] == "error" else "done" if done else "idle"
    return {"enabled": enabled(), "installed": done, "state": state, "phase": job["phase"] if running else None, "error": job["error"],
            "model": "YuE2 노래 모델 (8비트)", "source": f"huggingface.co/{REPO}", "revision": REVISION[:7], "license": LICENSE,
            "expected_bytes": total, "bytes_done": min(have, total), "percent": min(100, int(100 * have / total)) if total else 0,
            "note": "모델 파일과 실행 코드(.py 3개)를 함께 받아요(고정된 버전 %s). 크기는 약 4.5GB예요. 인터넷이 끊기면 멈추고, "
                    "다시 누르면 이어서 받아요. 이 모델은 비상업 조건이에요." % REVISION[:7]}


def _default_downloader(repo: str, path: Path) -> None:
    from huggingface_hub import snapshot_download      # installed with requirements.txt
    snapshot_download(repo, revision=REVISION, allow_patterns=sorted(FILES), local_dir=str(path))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> str | None:
    """None when every file has its exact size and the large weights their sha256; else a plain sentence for the page."""
    if not installed(path):
        return "받기는 끝났지만 모델 파일이 모두 있지 않거나 크기가 달라요. 다시 눌러 주세요."
    for name, (_, want) in FILES.items():
        if want and _sha256(path / name) != want:
            try:
                (path / name).unlink()                       # so that the next try fetches it again instead of trusting it
            except OSError:
                pass
            return f"받은 파일({name})이 검증값과 달라서 지웠어요. 다시 눌러 주세요."
    return None


def _reason(exc: BaseException) -> str:
    """A plain sentence for the page; never the exception text (it can carry URLs or request details)."""
    name = type(exc).__name__
    if isinstance(exc, (ConnectionError, TimeoutError)) or "Connection" in name or "Timeout" in name or "Offline" in name:
        return "인터넷 연결이 끊겼어요. 연결을 확인하고 다시 눌러 주세요(받던 곳에서 이어서 받아요)."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "저장 공간이 모자라요. 공간을 비우고 다시 눌러 주세요."
    return f"모델을 받지 못했어요({name}). 다시 눌러 보고, 계속 안 되면 이 문장을 알려 주세요."


def _run(downloader, path: Path) -> None:
    err = None
    try:
        downloader(REPO, path)
        with _lock:
            _job["phase"] = "verify"
        err = verify(path)
    except BaseException as exc:                      # the thread must always leave a state behind
        err = _reason(exc)
    with _lock:
        _job.update(state="error" if err else "done", phase=None, error=err, finished=time.time())


def start(accept_terms: bool, downloader=None, path: Path | None = None) -> dict:
    """Begin the download.  Raises PermissionError (not the local app), ValueError (terms not accepted, no room) or
    RuntimeError (already running or already installed); all messages are Korean sentences for the page."""
    if not enabled():
        raise PermissionError("이 서버에서는 모델을 받을 수 없어요. 내 Mac에 설치한 앱에서만 받아요.")
    if accept_terms is not True:
        raise ValueError("비상업(CC BY-NC 4.0) 조건에 동의하고 받기를 눌러 주세요.")
    dest = path or target()
    with _lock:
        failed = _job["state"] == "error"
    if installed(dest) and not failed:                       # after a failed check the files are fetched again, whatever their sizes
        raise RuntimeError("모델이 이미 있어요.")
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    need = expected_bytes() * NEED_FREE_FACTOR
    if free < need:
        raise ValueError(f"저장 공간이 모자라요. 약 {int(need / 1e9)}GB가 필요하고 지금 {free / 1e9:.1f}GB가 비어 있어요.")
    with _lock:
        if _job["state"] == "running":
            raise RuntimeError("이미 받고 있어요.")
        _job.update(state="running", phase="download", error=None, started=time.time(), finished=None)
    threading.Thread(target=_run, args=(downloader or _default_downloader, dest), daemon=True, name="yue2-first-run").start()
    return status()


def wait(timeout: float = 5.0) -> None:
    """For tests: block until the background download has left the running state."""
    end = time.time() + timeout
    while time.time() < end:
        with _lock:
            if _job["state"] != "running":
                return
        time.sleep(0.01)
    raise TimeoutError("download thread still running")


def reset() -> None:
    """For tests."""
    with _lock:
        _job.update(state="idle", phase=None, error=None, started=None, finished=None)
