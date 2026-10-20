"""scripts/install_mac.py: refuses an unsafe folder or machine, and plans exactly the commands it should."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_mac", ROOT / "scripts/install_mac.py")
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)


def test_only_apple_silicon_and_ascii_folders_pass(tmp_path):
    ok = tmp_path / "instmate_1"
    ok.mkdir()
    assert inst.problems(ok, "Darwin", "arm64") == []
    assert len(inst.problems(ok, "Windows", "AMD64")) == 1 and len(inst.problems(ok, "Darwin", "x86_64")) == 1
    for name in ("작곡 메이트", "has space", "-lead", "dot."):
        bad = tmp_path / name
        bad.mkdir()
        found = inst.problems(bad, "Darwin", "arm64")
        assert len(found) == 1 and name in found[0], name


def test_plan_has_the_commands_in_order_and_extras_only_when_asked():
    base = inst.plan(True, True, {})
    assert [c[0:2] for c in base] == [["uv", "venv"], ["uv", "pip"]]
    full = inst.plan(True, True, {"sound": True, "yue": True})              # no model is ever fetched by the installer
    assert full[2] == ["brew", "install", "fluid-synth", "ffmpeg"] and len(full) == 3
    assert not any("snapshot_download" in " ".join(c) or "huggingface" in " ".join(c) for c in full)
    assert len(inst.plan(True, False, {"sound": True})) == 2                  # no brew: the sound tools are skipped
    assert inst.plan(False, False, {})[0][1:3] == ["-m", "venv"]               # no uv: the plain venv


def test_dry_run_prints_the_commands_and_runs_none(monkeypatch, capsys):
    monkeypatch.setattr(inst, "problems", lambda *a, **k: [])
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a command")))
    assert inst.main(["--dry-run", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "$ uv venv" in out and "brew install fluid-synth" in out and "YuE2-3B-MLX" not in out
    assert "설치 끝" not in out and "아무것도 설치하지 않았어요" in out        # a dry run never says it installed


def test_an_unsafe_folder_stops_before_any_command(monkeypatch, capsys):
    monkeypatch.setattr(inst, "problems", lambda *a, **k: ["폴더 이름에 한글"])
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a command")))
    assert inst.main(["--yes"]) == 1 and "설치할 수 없어요" in capsys.readouterr().out


def test_core_mode_asks_nothing_installs_only_the_programs_and_never_a_model(monkeypatch, capsys):
    monkeypatch.setattr(inst, "problems", lambda *a, **k: [])
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(inst, "ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked a question")))
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a command")))
    assert inst.main(["--core", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("$ ") == 2 and "brew" not in out and "huggingface" not in out and "아무것도 설치하지 않았어요" in out
