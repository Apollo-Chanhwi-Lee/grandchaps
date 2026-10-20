"""What the locally installed app ships (one list, used by the release build and by the closure test).

    python3 scripts/package_files.py <root>     prints the problems of that tree (an unpacked app or this source tree); exit 1 if any

The closure rule (review F01, 2026-10-20): every script, config or static file that the app's own code launches or loads must be in
the shipped tree.  Folders the recipient's Mac fills in later (models, data) are not shipped and are not checked.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EXACT = {
    "app.py", "pyproject.toml", "requirements.txt", "README.md", "LICENSE", "start_mac.command",
    "configs/explain.yaml", "configs/instruments.yaml",
    # the scripts the app starts as separate programs (instmate/yuesong.py RUNNER, vst.py WORKER, yueblend.py SEPARATE) and the setup tools
    "scripts/yue_continue.py", "scripts/vst_worker.py", "scripts/demucs_separate.py",
    "scripts/install_mac.py", "scripts/build_mac_installer.py", "scripts/package_files.py",
    "docs/LICENSES.md", "docs/설치_Mac.md", "docs/GitHub_배포_준비.md",
    "tests/test_install_mac.py", "tests/test_mac_installer.py", "tests/test_modelsetup.py", "tests/test_package_closure.py",
    "tests/test_local_app_start.py",
}
RUNTIME_DIRS = {"models", "data", "work"}      # created or filled on the recipient's Mac


def allowed(path: str) -> bool:
    """Whether a tracked file belongs in the package."""
    if path in EXACT:
        return True
    return (path.startswith("instmate/") and path.endswith(".py")) or path.startswith("static/")


# "scripts/x.py" written out, and ROOT / "scripts" / "x.py" (or Path(__file__)...parent.parent / "scripts" / "x.py")
_LITERAL = re.compile(r"[\"']((?:scripts|configs)/[A-Za-z0-9_.-]+\.(?:py|yaml))[\"']")
_CHAIN = re.compile(r"/\s*[\"'](scripts|configs)[\"']\s*/\s*[\"']([A-Za-z0-9_.-]+\.(?:py|yaml))[\"']")


def referenced_files(root: Path) -> dict[str, str]:
    """{"scripts/x.py": "instmate/y.py", ...}: files the app's code names, with the first file that names them."""
    out: dict[str, str] = {}
    for src in [root / "app.py", *sorted((root / "instmate").glob("*.py"))]:
        if not src.is_file():
            continue
        text = src.read_text(encoding="utf-8")
        for rel in [m.group(1) for m in _LITERAL.finditer(text)] + [f"{a}/{b}" for a, b in _CHAIN.findall(text)]:
            out.setdefault(rel, str(src.relative_to(root)))
    return out


def static_references(root: Path) -> list[str]:
    """/static/... files the page and its scripts load."""
    index = root / "static" / "index.html"
    if not index.is_file():
        return []
    found = set(re.findall(r"[\"'(]/static/([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)", index.read_text(encoding="utf-8")))
    for js in ("app.js", "workspace.js"):
        p = root / "static" / js
        if p.is_file():
            found |= set(re.findall(r"[\"'(]/static/([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)", p.read_text(encoding="utf-8")))
    return sorted(f"static/{n}" for n in found)


def problems(root: Path) -> list[str]:
    out = []
    for rel, by in referenced_files(root).items():
        if not allowed(rel):
            out.append(f"{by} uses {rel}, which the package list does not include")
        if not (root / rel).is_file():
            out.append(f"{by} uses {rel}, which is not in this tree")
    for rel in static_references(root):
        if not (root / rel).is_file():
            out.append(f"the page loads /{rel}, which is not in this tree")
    return out


if __name__ == "__main__":
    found = problems(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent))
    print("\n".join(found) or "closure ok")
    sys.exit(1 if found else 0)
