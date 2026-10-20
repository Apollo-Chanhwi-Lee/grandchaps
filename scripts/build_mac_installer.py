"""Build the single-file Mac installer for GitHub Releases from the review ZIP (user request 2026-10-20).

    python3 scripts/build_mac_installer.py <app.zip> <out_installer.command>

The installer is a zsh script with the ZIP appended after a marker.  A recipient downloads this one file and double-clicks it
(no git clone, no developer setup): it checks macOS on Apple Silicon and a safe folder name, verifies the payload's SHA-256,
unpacks into ~/instmate_app (code is replaced on an update, .venv / models / data stay), then runs start_mac.command: first-run
program setup, a loopback-only local server, the browser opens, models are downloaded inside the app after consent.

Not done here, on purpose: code signing and notarization (a "from an unidentified developer" prompt is expected: right-click, Open),
bundling Python (Python 3.11+ or uv must exist), any model weights.  Standard library only.
"""
from __future__ import annotations

import hashlib
import stat
import sys
import zipfile
from pathlib import Path

MARK = "__ARCHIVE_BELOW__"
HEADER = r'''#!/bin/zsh
# 작곡메이트 (Grandchaps InstMate) installer for macOS on Apple Silicon: double-click, or run `zsh <this file>`.
# Payload sha256: @SHA@   (the ZIP appended below the marker line)
#   INSTMATE_HOME  install folder (default ~/instmate_app; English letters, digits . _ - only)
#   INSTMATE_UNAME_S / INSTMATE_UNAME_M  override the platform check (tests)
pause() { read -k 1 "?아무 키나 누르면 닫혀요"; }
fail() { echo "설치할 수 없어요: $1"; pause; exit 1; }
[ "${INSTMATE_UNAME_S:-$(uname -s)}" = "Darwin" ] && [ "${INSTMATE_UNAME_M:-$(uname -m)}" = "arm64" ] \
  || fail "Apple Silicon Mac(M1 이상)에서만 설치할 수 있어요. Windows와 Intel Mac은 아직 지원하지 않아요."
DEST="${INSTMATE_HOME:-$HOME/instmate_app}"
case "$DEST" in *[!A-Za-z0-9._/-]*) fail "설치 폴더 이름에 한글·공백·특수문자가 있어요: $DEST (영문·숫자·. _ - 만 쓰는 폴더로 바꿔 주세요)";; esac
SELF="$0"; [ -f "$SELF" ] || fail "설치 파일 위치를 찾지 못했어요."
LINE=$(awk '/^@MARK@$/ { print NR + 1; exit 0 }' "$SELF")
[ -n "$LINE" ] || fail "설치 파일이 손상됐어요. 다시 내려받아 주세요."
WORK=$(mktemp -d "${TMPDIR:-/tmp}/instmate_install.XXXXXX") || fail "임시 폴더를 만들지 못했어요."
trap 'rm -rf "$WORK"' EXIT
tail -n +"$LINE" "$SELF" > "$WORK/app.zip"
GOT=$(shasum -a 256 "$WORK/app.zip" | awk '{print $1}')
[ "$GOT" = "@SHA@" ] || fail "설치 파일이 손상됐거나 바뀌었어요(검사값이 달라요). 아무것도 설치하지 않았어요. 다시 내려받아 주세요."
echo "설치 위치: $DEST"
unzip -q -o "$WORK/app.zip" -d "$WORK/x" || fail "압축을 풀지 못했어요."
mkdir -p "$DEST" || fail "설치 폴더를 만들지 못했어요: $DEST"
cp -R "$WORK/x/instmate_local/." "$DEST/" || fail "파일을 복사하지 못했어요."
chmod +x "$DEST/start_mac.command" "$DEST/scripts/install_mac.py"
echo "설치 파일 풀기 끝. 앱을 시작해요."
rm -rf "$WORK"; trap - EXIT                    # exec does not run the EXIT trap
exec zsh "$DEST/start_mac.command"
exit 0
'''


def build(zip_path: Path, out_path: Path) -> str:
    """Write the installer; returns the sha256 of the embedded ZIP.  Refuses to overwrite and refuses a ZIP that is not the app."""
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; an earlier release is never overwritten")
    data = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    need = {"instmate_local/start_mac.command", "instmate_local/scripts/install_mac.py", "instmate_local/app.py"}
    if not need <= set(names):
        raise ValueError(f"not the app ZIP (missing {sorted(need - set(names))})")
    sha = hashlib.sha256(data).hexdigest()
    head = HEADER.replace("@SHA@", sha).replace("@MARK@", MARK).encode("utf-8")
    if MARK.encode() in data:
        raise ValueError("the marker occurs inside the payload")
    out_path.write_bytes(head + MARK.encode() + b"\n" + data)
    out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return sha


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(build(Path(sys.argv[1]), Path(sys.argv[2])))
