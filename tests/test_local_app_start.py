"""start_mac.command, run for real with a controlled set of tools on PATH: reviews F02 (uv only, no Python, too-old Python) and F03
(an install that stopped halfway or whose requirements changed is repeated), and the normal start with the browser opener.
No package is installed and no browser is opened: uv and Python are small stand-ins that record how they were called."""
import hashlib
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs = pytest.mark.skipif(not all(shutil.which(t) for t in ("zsh", "lsof", "curl", "shasum", "cut", "cat", "seq", "sleep")),
                           reason="needs zsh and the usual tools")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def tool_dir(tmp_path, **fakes):
    """A bin folder with only the tools the script needs (symlinks) plus the given stand-ins ({name: shell body})."""
    d = tmp_path / "bin"
    d.mkdir()
    for t in ("lsof", "curl", "shasum", "cut", "cat", "seq", "sleep", "kill", "perl", "xcode-select"):
        real = shutil.which(t)
        if real:
            (d / t).symlink_to(real)
    for name, body in fakes.items():
        name = name.replace("_", ".") if name.startswith("python3_") else name        # python3_12 -> python3.12
        f = d / name
        f.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    return d


def app_dir(tmp_path, requirements="fastapi\n"):
    a = tmp_path / "app"
    (a / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "start_mac.command", a / "start_mac.command")
    (a / "scripts" / "install_mac.py").write_text("# stand-in\n", encoding="utf-8")
    (a / "requirements.txt").write_text(requirements, encoding="utf-8")
    (a / "app.py").write_text("# stand-in\n", encoding="utf-8")
    return a


def run(app, bin_dir, log, extra_env=None, timeout=40):
    env = {"PATH": str(bin_dir), "HOME": str(app.parent), "INSTMATE_PORT": str(free_port()), "INSTMATE_OPEN": "true", **(extra_env or {})}
    r = subprocess.run(["/bin/zsh", str(app / "start_mac.command")], env=env, capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL, cwd=str(app.parent))      # NOT the app folder: the script must find its own
    return r, (log.read_text(encoding="utf-8") if log.exists() else "")


def fake_venv(app, mark=None):
    py = app / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")                   # the app stand-in dies at once: the start is not the point here
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    if mark is not None:
        (app / ".venv" / ".instmate_ready").write_text(mark, encoding="utf-8")


@needs
def test_a_mac_with_only_uv_is_set_up_through_uv(tmp_path):
    log = tmp_path / "calls.log"
    bin_dir = tool_dir(tmp_path, uv=f'echo "uv $@" >> {log}')
    app = app_dir(tmp_path)
    r, calls = run(app, bin_dir, log)
    assert calls.strip() == "uv run --python 3.13 --no-project scripts/install_mac.py --core", calls
    assert r.returncode == 1 and "프로그램 준비가 끝나지 않았어요" in r.stdout, r.stdout      # the stand-in made no venv: it stops, it does not start


@needs
def test_a_usable_python_runs_the_installer_and_an_old_or_missing_one_gets_exact_advice(tmp_path):
    log = tmp_path / "calls.log"
    new = tool_dir(tmp_path, python3_12=f'[ "$1" = "-c" ] && exit 0; echo "python3.12 $@" >> {log}')
    app = app_dir(tmp_path)
    r, calls = run(app, new, log)
    assert calls.strip() == "python3.12 scripts/install_mac.py --core", calls
    # a Python that fails the 3.11 test is not used, and with no uv the message says what to install
    shutil.rmtree(tmp_path / "bin")
    old = tool_dir(tmp_path, python3='exit 1')
    log.unlink()
    r, calls = run(app, old, log)
    assert calls == "" and r.returncode == 1
    assert "brew install uv" in r.stdout and "Python 3.11" in r.stdout and "python.org" in r.stdout, r.stdout
    shutil.rmtree(tmp_path / "bin")
    r, calls = run(app, tool_dir(tmp_path), log)                               # nothing at all
    assert calls == "" and r.returncode == 1 and "uv" in r.stdout and not (app / ".venv").exists()


@needs
def test_an_install_that_stopped_halfway_or_whose_requirements_changed_is_repeated(tmp_path):
    log = tmp_path / "calls.log"
    bin_dir = tool_dir(tmp_path, uv=f'echo "uv $@" >> {log}')
    app = app_dir(tmp_path, "fastapi\nmido\n")
    fake_venv(app)                                                              # .venv exists but no finish mark: a run that stopped
    r, calls = run(app, bin_dir, log)
    assert "install_mac.py --core" in calls, "an unfinished .venv must not be trusted"
    log.unlink()
    fake_venv_mark = hashlib.sha256(b"fastapi\n").hexdigest()                   # a mark of OTHER requirements
    (app / ".venv" / ".instmate_ready").write_text(fake_venv_mark, encoding="utf-8")
    r, calls = run(app, bin_dir, log)
    assert "install_mac.py --core" in calls, "changed requirements.txt repeats the package step"
    log.unlink()
    (app / ".venv" / ".instmate_ready").write_text(hashlib.sha256(b"fastapi\nmido\n").hexdigest(), encoding="utf-8")
    r, calls = run(app, bin_dir, log)
    assert calls == "", "a finished install with the same requirements is not repeated"
    assert "앱이 켜지지 못했어요" in r.stdout                                    # it went on to start the (stand-in) app


def test_the_installer_reuses_an_unfinished_venv_and_writes_the_finish_mark_only_at_the_end(tmp_path, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("install_mac_f03", ROOT / "scripts/install_mac.py")
    inst = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inst)
    assert [c[1] if c[0] == "uv" else c[1:3] for c in inst.plan(True, False, {})] == ["venv", "pip"]
    assert [c[1] for c in inst.plan(True, False, {}, have_venv=True)] == ["pip"], "an existing .venv is kept, only the packages are redone"
    monkeypatch.setattr(inst, "ROOT", tmp_path)
    monkeypatch.setattr(inst, "READY", tmp_path / ".venv" / ".instmate_ready")
    monkeypatch.setattr(inst, "VENV_PY", tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(inst, "problems", lambda *a, **k: [])
    monkeypatch.setattr(inst.shutil, "which", lambda n: "/bin/" + n if n == "uv" else None)
    (tmp_path / "requirements.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    ran = []
    monkeypatch.setattr(inst.subprocess, "run", lambda cmd, cwd=None: ran.append(cmd) or type("R", (), {"returncode": 1 if len(ran) == 1 else 0})())
    assert inst.main(["--core"]) == 1 and not inst.READY.exists(), "a failed package step leaves no finish mark"
    assert "다시 열면 멈춘 곳부터" in capsys.readouterr().out
    assert inst.main(["--core"]) == 0 and inst.READY.read_text() == hashlib.sha256(b"a\n").hexdigest()
    ran.clear()
    assert inst.main(["--core", "--dry-run"]) == 0 and ran == [], "a dry run runs nothing"


@needs
def test_the_normal_start_launches_a_loopback_only_server_and_calls_the_opener(tmp_path):
    """Runs the real start_mac.command with this test's Python and the real app; the opener is a recording stand-in."""
    port = free_port()
    opened = tmp_path / "opened.txt"
    opener = tmp_path / "fake_open.sh"
    opener.write_text(f'#!/bin/sh\necho "$1" > "{opened}"\n', encoding="utf-8")
    opener.chmod(opener.stat().st_mode | stat.S_IXUSR)
    env = {**os.environ, "INSTMATE_PYTHON": sys.executable, "INSTMATE_PORT": str(port), "INSTMATE_OPEN": str(opener),
           "INSTMATE_DATA": str(tmp_path / "data"), "INSTMATE_VST": "off", "INSTMATE_YUE_DIR": str(tmp_path / "yue"), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen(["zsh", str(ROOT / "start_mac.command")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        end = time.time() + 90
        while time.time() < end and not opened.exists():
            assert proc.poll() is None, proc.stdout.read().decode()
            time.sleep(0.5)
        assert opened.exists(), "the browser opener was never called"
        assert opened.read_text().strip() == f"http://127.0.0.1:{port}"
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/models/status", timeout=10) as r:
            assert b'"enabled":true' in r.read(), "the local app mode is on"
        listen = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
        assert f"127.0.0.1:{port}" in listen and "*:" not in listen, listen          # this Mac only
    finally:
        os.killpg(proc.pid, 15)
        proc.wait(timeout=20)
    end = time.time() + 15
    while time.time() < end and subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout:
        time.sleep(0.2)
    assert subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout == "", "the port stayed open"
