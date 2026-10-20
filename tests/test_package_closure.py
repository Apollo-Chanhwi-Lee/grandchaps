"""Review F01 (2026-10-20): the package list and the files the app launches must agree, in the source tree and in an unpacked app."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_files", ROOT / "scripts/package_files.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


def test_every_script_the_app_launches_is_listed_and_present():
    refs = pf.referenced_files(ROOT)
    assert "scripts/yue_continue.py" in refs, "the generation runner must be found by the scan (it was missed once)"
    assert {"scripts/vst_worker.py", "scripts/demucs_separate.py", "configs/instruments.yaml"} <= set(refs), sorted(refs)
    assert pf.problems(ROOT) == []


def test_the_scan_finds_both_spellings_and_reports_a_gap(tmp_path):
    (tmp_path / "instmate").mkdir()
    (tmp_path / "app.py").write_text('X = "scripts/one.py"\n', encoding="utf-8")
    (tmp_path / "instmate" / "a.py").write_text('R = Path(__file__).parent.parent / "scripts" / "two.py"\nC = ROOT / "configs" / "t.yaml"\n', encoding="utf-8")
    assert pf.referenced_files(tmp_path) == {"scripts/one.py": "app.py", "scripts/two.py": "instmate/a.py", "configs/t.yaml": "instmate/a.py"}
    found = pf.problems(tmp_path)
    assert len(found) == 6 and all("one.py" in f or "two.py" in f or "t.yaml" in f for f in found), found     # unlisted and missing, each


def test_the_package_list_excludes_what_must_not_ship():
    for bad in ("CLAUDE.md", ".env", "data/prefs.db", "models/x.safetensors", "datasets/a.csv", "work/public_demo/invite.txt",
                "scripts/public_demo.py", "public_site/index.html", "notebooks/a.ipynb", "samples/a.mid"):
        assert not pf.allowed(bad), bad
    for good in ("scripts/yue_continue.py", "instmate/modelsetup.py", "static/app.js", "LICENSE", "start_mac.command"):
        assert pf.allowed(good), good
