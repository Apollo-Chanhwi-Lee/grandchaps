"""scripts/build_mac_installer.py: the single-file installer verifies its payload, refuses the wrong Mac or folder, unpacks without
touching an installed .venv / models / data, and hands over to start_mac.command.  A stand-in app ZIP is used here; the real app ZIP
is exercised by the release verification, not by this test."""
import importlib.util
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_mac_installer", ROOT / "scripts/build_mac_installer.py")
bmi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bmi)
needs_zsh = pytest.mark.skipif(shutil.which("zsh") is None or shutil.which("shasum") is None, reason="needs zsh and shasum")


def stand_in_zip(path: Path, version="one"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("instmate_local/app.py", f"# app {version}\n")
        z.writestr("instmate_local/scripts/install_mac.py", "# installer stand-in\n")
        z.writestr("instmate_local/start_mac.command", '#!/bin/zsh\ncd "$(dirname "$0")"\necho started-$(cat app.py | tail -c 8) > "$(dirname "$0")/STARTED.txt"\n')
    return path


def run(installer, home, tmp, **extra):
    env = {**os.environ, "INSTMATE_HOME": str(home), "TMPDIR": str(tmp), **extra}
    return subprocess.run(["zsh", str(installer)], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def test_the_builder_embeds_the_zip_once_and_refuses_overwrites_and_other_zips(tmp_path):
    z = stand_in_zip(tmp_path / "a.zip")
    out = tmp_path / "installer.command"
    sha = bmi.build(z, out)
    data = out.read_bytes()
    assert sha in data.split(b"__ARCHIVE_BELOW__\n")[0].decode() and data.count(b"\n__ARCHIVE_BELOW__\n") == 1
    assert data.split(b"__ARCHIVE_BELOW__\n", 1)[1] == z.read_bytes() and out.stat().st_mode & 0o100
    with pytest.raises(FileExistsError):
        bmi.build(z, out)
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as o:
        o.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="not the app ZIP"):
        bmi.build(other, tmp_path / "o.command")
    assert not (tmp_path / "o.command").exists()


@needs_zsh
def test_install_unpacks_keeps_the_installed_environment_and_starts_the_app(tmp_path):
    out = tmp_path / "i.command"
    bmi.build(stand_in_zip(tmp_path / "a.zip", "one"), out)
    home, tmpdir = tmp_path / "instmate_app", tmp_path / "tmp"
    tmpdir.mkdir()
    r = run(out, home, tmpdir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (home / "app.py").read_text() == "# app one\n" and "started-" in (home / "STARTED.txt").read_text()
    assert os.access(home / "start_mac.command", os.X_OK) and os.access(home / "scripts/install_mac.py", os.X_OK)
    assert list(tmpdir.iterdir()) == [], "the temporary folder is removed before the app takes over"
    for keep in (".venv/bin/python", "models/yue2_mlx/repo/generate.py", "data/prefs.db"):    # an update leaves these alone
        (home / keep).parent.mkdir(parents=True, exist_ok=True)
        (home / keep).write_text("kept")
    out2 = tmp_path / "i2.command"
    bmi.build(stand_in_zip(tmp_path / "b.zip", "two"), out2)
    assert run(out2, home, tmpdir).returncode == 0
    assert (home / "app.py").read_text() == "# app two\n"
    assert all((home / k).read_text() == "kept" for k in (".venv/bin/python", "models/yue2_mlx/repo/generate.py", "data/prefs.db"))


@needs_zsh
def test_a_changed_payload_is_refused_and_nothing_is_installed(tmp_path):
    out = tmp_path / "i.command"
    bmi.build(stand_in_zip(tmp_path / "a.zip"), out)
    raw = bytearray(out.read_bytes())
    raw[-30] ^= 0xFF                                      # one byte inside the appended ZIP
    out.write_bytes(bytes(raw))
    home = tmp_path / "instmate_app"
    r = run(out, home, tmp_path)
    assert r.returncode == 1 and "검사값이 달라요" in r.stdout and "아무것도 설치하지 않았어요" in r.stdout
    assert not home.exists()


@needs_zsh
def test_other_machines_and_unsafe_folders_are_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "i.command"
    bmi.build(stand_in_zip(tmp_path / "a.zip"), out)
    for extra, word in (({"INSTMATE_UNAME_M": "x86_64"}, "Apple Silicon"), ({"INSTMATE_UNAME_S": "Linux"}, "Apple Silicon")):
        home = tmp_path / "instmate_app"
        r = run(out, home, tmp_path, **extra)
        assert r.returncode == 1 and word in r.stdout and "Windows" in r.stdout and not home.exists(), r.stdout
    for bad in ("작곡 메이트", "has space", "semi;colon"):
        home = tmp_path / bad
        r = run(out, home, tmp_path)
        assert r.returncode == 1 and "설치 폴더 이름" in r.stdout and not home.exists(), (bad, r.stdout)
