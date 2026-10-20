"""One-step setup for an Apple Silicon Mac: checks, .venv, packages, and the optional extras (each asked first).

    python3 scripts/install_mac.py [--core] [--yes] [--dry-run]

Standard library only, so the Mac's own python3 can run it.  --core is what start_mac.command runs on the first launch: the
programs only (.venv and requirements.txt), no questions.  Without it the optional sound tools are asked first (--yes answers
yes).  --dry-run prints what would run and changes nothing.

No model is fetched here.  The YuE2 singing model is downloaded inside the app on the first run, after the recipient accepts
its terms (instmate/modelsetup.py).  The lead sheet model (trained on non-commercial data, docs/LICENSES.md) is not handed
out; without it the app writes songs from the user's melody with rules.  The qwen chat model is optional (9 GB).
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
SAFE = re.compile(r"^(?!-)[A-Za-z0-9._-]+(?<!\.)$")


def problems(path: Path = ROOT, system: str | None = None, machine: str | None = None) -> list[str]:
    """Why this folder or machine cannot run the app (Korean lines); empty when it can."""
    out = []
    if (system or platform.system()) != "Darwin" or (machine or platform.machine()) != "arm64":
        out.append("Apple Silicon Mac(M1 이상)에서만 설치할 수 있어요. Windows는 아직 지원하지 않아요.")
    bad = [p for p in path.resolve().parts[1:] if not SAFE.match(p)]
    if bad:
        out.append("폴더 이름에 한글·공백·특수문자가 있어요: " + ", ".join(bad)
                   + ". 영문·숫자·밑줄(_)·하이픈(-)만 쓰는 폴더(예: ~/Documents/instmate)로 옮긴 뒤 다시 실행해 주세요.")
    return out


READY = ROOT / ".venv" / ".instmate_ready"      # written only after every package installed; holds the requirements.txt hash


def requirements_hash(root: Path | None = None) -> str:
    import hashlib
    return hashlib.sha256(((root or ROOT) / "requirements.txt").read_bytes()).hexdigest()


def plan(have_uv: bool, have_brew: bool, extras: dict, have_venv: bool = False) -> list[list[str]]:
    """The commands in order.  `extras`: {"sound": bool} the answer to the optional question (no model is fetched here).
    `have_venv`: a .venv from an earlier, unfinished run is reused; only the package step is repeated (review F03)."""
    py = str(VENV_PY)
    cmds = [] if have_venv else [["uv", "venv", "--python", "3.13", ".venv"] if have_uv else [sys.executable, "-m", "venv", ".venv"]]
    cmds.append(["uv", "pip", "install", "--python", py, "-r", "requirements.txt"] if have_uv
                else [py, "-m", "pip", "install", "-r", "requirements.txt"])
    if extras.get("sound") and have_brew:
        cmds.append(["brew", "install", "fluid-synth", "ffmpeg"])
    return cmds


def ask(question: str, yes: bool) -> bool:
    if yes:
        return True
    return input(question + " [y/N] ").strip().lower() in ("y", "yes", "ㅛ")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    yes, dry, core = "--yes" in args, "--dry-run" in args, "--core" in args
    found = problems()
    if found:
        print("\n".join("설치할 수 없어요: " + p for p in found))
        return 1
    have_uv, have_brew = bool(shutil.which("uv")), bool(shutil.which("brew"))
    if not have_uv and sys.version_info < (3, 11):
        print("uv가 없고 이 python3가 오래됐어요. 터미널에서 `brew install uv` 한 뒤 다시 실행해 주세요.")
        return 1
    extras = {"sound": have_brew and not core and ask("악보를 소리로 듣는 도구(fluid-synth, ffmpeg)를 설치할까요? 없어도 악보·MIDI는 돼요.", yes)}
    for cmd in plan(have_uv, have_brew, extras, VENV_PY.exists()):
        print("$", " ".join(cmd))
        if not dry and subprocess.run(cmd, cwd=ROOT).returncode:
            print("여기서 멈췄어요. 위 명령의 오류 문장을 그대로 알려 주세요. 같은 파일을 다시 열면 멈춘 곳부터 이어서 설치해요.")
            return 1
    if dry:
        print("\n--dry-run: 위 명령은 실행하지 않았어요. 아무것도 설치하지 않았어요.")
        return 0
    READY.write_text(requirements_hash(), encoding="utf-8")        # the finish mark: start_mac.command trusts the .venv only with it
    print("\n프로그램 설치 끝." + ("" if core else " start_mac.command를 더블클릭하면 앱이 열려요.")
          + "\n노래 모델은 앱을 처음 켠 화면에서 받아요(약 4.5GB, 비상업 조건 확인 후)."
          + "\n리드시트 모델은 나눠 드리지 않아요. 곡은 올린 멜로디를 바탕으로 규칙으로 만들어요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
