"""First-run model download of the locally installed app: refused unless local and accepted, a pinned revision and exact files,
progress, failure, retry and an honest "done"; the local app answers only to its own address.  No test downloads a model or opens
a browser; the real start script is exercised in tests/test_local_app_start.py."""
import hashlib
import shutil
import sys
import threading
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from instmate import modelsetup

ROOT = Path(__file__).resolve().parents[1]
GOOD = b"m" * 20
TINY = {"generate.py": (10, None), "8bit/model.safetensors": (20, hashlib.sha256(GOOD).hexdigest())}


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTMATE_LOCAL_APP", "1")
    monkeypatch.setenv("INSTMATE_YUE_DIR", str(tmp_path / "yue"))
    monkeypatch.setattr(modelsetup, "FILES", dict(TINY))
    monkeypatch.setattr(modelsetup.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 0, 10**12))
    modelsetup.reset()
    yield tmp_path / "yue"
    modelsetup.reset()


def fake_install(repo, path, weights=GOOD):
    assert repo == modelsetup.REPO
    (path / "8bit").mkdir(parents=True, exist_ok=True)
    (path / "generate.py").write_bytes(b"g" * 10)
    (path / "8bit" / "model.safetensors").write_bytes(weights)


def test_the_pinned_table_matches_what_the_runner_needs_and_the_public_revision():
    files = modelsetup.FILES.keys()
    assert {"generate.py", "yue2_model.py", "yue2_vae.py", "8bit/model.safetensors", "8bit/vae.safetensors"} <= set(files)
    assert modelsetup.REVISION == "fe0a9050fd658257b486b880422d8872ee1f81e3"
    big = {n: v for n, v in modelsetup.FILES.items() if v[1]}
    assert set(big) == {"8bit/model.safetensors", "8bit/vae.safetensors"} and all(len(h) == 64 for _, h in big.values())
    assert 4.4e9 < modelsetup.expected_bytes() < 4.7e9


def test_nothing_is_downloaded_unless_this_is_the_local_app_and_the_terms_are_accepted(local, monkeypatch):
    calls = []
    monkeypatch.delenv("INSTMATE_LOCAL_APP")
    with pytest.raises(PermissionError):
        modelsetup.start(True, downloader=lambda *a: calls.append(a))
    assert modelsetup.status()["enabled"] is False
    monkeypatch.setenv("INSTMATE_LOCAL_APP", "1")
    for bad in (False, None, "yes", 1):
        with pytest.raises(ValueError):
            modelsetup.start(bad, downloader=lambda *a: calls.append(a))
    assert calls == [] and (not local.exists() or not any(local.iterdir()))


def test_status_tells_size_terms_version_and_source_before_anything_is_downloaded(local):
    s = modelsetup.status()
    assert s["enabled"] and not s["installed"] and s["state"] == "idle" and s["percent"] == 0 and s["error"] is None
    assert "CC BY-NC" in s["license"] and s["revision"] == "fe0a905" and "실행 코드" in s["note"] and "4.5GB" in s["note"]
    assert s["source"].endswith("ahmadw/YuE2-3B-MLX")


def test_default_downloader_asks_for_the_pinned_revision_and_only_the_listed_files(monkeypatch, tmp_path):
    seen = {}
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda repo, **kw: seen.update(repo=repo, **kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    modelsetup._default_downloader(modelsetup.REPO, tmp_path)
    assert seen["repo"] == "ahmadw/YuE2-3B-MLX" and seen["revision"] == modelsetup.REVISION
    assert seen["allow_patterns"] == sorted(modelsetup.FILES) and seen["local_dir"] == str(tmp_path)
    assert not any("*" in p for p in seen["allow_patterns"]), "no wildcard: nothing outside the list is fetched or executed"


def test_download_runs_in_the_background_shows_progress_verifies_and_finishes(local):
    gate, started = threading.Event(), threading.Event()
    def slow(repo, path):
        (path / "8bit").mkdir(parents=True, exist_ok=True)
        (path / "8bit" / "model.safetensors").write_bytes(b"m" * 12)           # 12 of 30 bytes: partly there
        started.set()
        gate.wait(5)
        fake_install(repo, path)
    first = modelsetup.start(True, downloader=slow)
    assert first["state"] == "running" and started.wait(5)
    mid = modelsetup.status()
    assert mid["state"] == "running" and mid["installed"] is False and mid["bytes_done"] == 12 and mid["percent"] == 40, mid
    with pytest.raises(RuntimeError):                                          # a second click does not start a second download
        modelsetup.start(True, downloader=slow)
    gate.set()
    modelsetup.wait()
    done = modelsetup.status()
    assert done["installed"] and done["state"] == "done" and done["percent"] == 100 and done["error"] is None
    with pytest.raises(RuntimeError):                                          # already there
        modelsetup.start(True, downloader=slow)


def test_partial_files_and_running_or_failed_jobs_are_never_reported_as_done(local):
    """Review F04: two files with the right names but not the right content/size were shown as finished."""
    (local / "8bit").mkdir(parents=True)
    (local / "generate.py").write_bytes(b"g" * 10)
    (local / "8bit" / "model.safetensors").write_bytes(b"m" * 7)               # cut short
    s = modelsetup.status()
    assert s["installed"] is False and s["state"] == "idle"
    gate = threading.Event()
    def stuck(repo, path):
        gate.wait(5)
        raise ConnectionError("offline")
    modelsetup.start(True, downloader=stuck)                                   # a retry is allowed: the files are incomplete
    (local / "8bit" / "model.safetensors").write_bytes(GOOD)                   # complete-looking files while the job runs
    running = modelsetup.status()
    assert running["state"] == "running" and running["installed"] is False, running
    gate.set()
    modelsetup.wait()
    failed = modelsetup.status()                                               # the failure wins over the files that look complete
    assert failed["state"] == "error" and failed["installed"] is False and "인터넷" in failed["error"], failed
    modelsetup.start(True, downloader=fake_install)                            # and the same button works again
    modelsetup.wait()
    assert modelsetup.status()["installed"] is True


def test_a_download_with_the_right_size_but_the_wrong_content_is_deleted_and_fetched_again(local):
    modelsetup.start(True, downloader=lambda repo, path: fake_install(repo, path, weights=b"x" * 20))   # same size, other bytes
    modelsetup.wait()
    s = modelsetup.status()
    assert s["state"] == "error" and not s["installed"] and "검증값과 달라서 지웠어요" in s["error"], s
    assert not (local / "8bit" / "model.safetensors").exists(), "the bad file is removed so the retry fetches it again"
    modelsetup.start(True, downloader=fake_install)
    modelsetup.wait()
    assert modelsetup.status()["installed"] and modelsetup.status()["error"] is None


def test_a_failure_is_explained_without_the_raw_error_and_can_be_retried(local):
    def offline(repo, path):
        raise ConnectionError("https://example.invalid/model?token=SECRET-VALUE failed")
    modelsetup.start(True, downloader=offline)
    modelsetup.wait()
    s = modelsetup.status()
    assert s["state"] == "error" and "인터넷" in s["error"] and "SECRET" not in s["error"] and "example.invalid" not in s["error"]
    modelsetup.start(True, downloader=fake_install)
    modelsetup.wait()
    assert modelsetup.status()["installed"]
    modelsetup.reset()
    shutil.rmtree(local)
    def odd(repo, path):
        raise KeyError("internal detail")
    modelsetup.start(True, downloader=odd)
    modelsetup.wait()
    assert "KeyError" in modelsetup.status()["error"] and "internal detail" not in modelsetup.status()["error"]
    modelsetup.reset()
    modelsetup.start(True, downloader=lambda repo, path: None)                 # finished but nothing arrived: not called done
    modelsetup.wait()
    s = modelsetup.status()
    assert s["state"] == "error" and not s["installed"] and "크기가 달라요" in s["error"]


def test_too_little_disk_space_is_refused_before_downloading(local, monkeypatch):
    monkeypatch.setattr(modelsetup.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**10, 9 * 10**9, 5))
    with pytest.raises(ValueError, match="저장 공간"):
        modelsetup.start(True, downloader=lambda *a: pytest.fail("must not download"))


def test_the_endpoints_answer_403_422_409_and_200_and_report_missing_tools(local, monkeypatch):
    import app as app_mod
    c = TestClient(app_mod.app, base_url="http://127.0.0.1:8000")
    st = c.get("/api/models/status").json()
    assert st["enabled"] is True and set(st["features"]) == {"pdf", "pdf_reason", "practice_parts", "practice_reason"}
    assert c.post("/api/models/yue2/download", json={}).status_code == 422
    assert c.post("/api/models/yue2/download", json={"accept_terms": False}).status_code == 422
    monkeypatch.setattr(modelsetup, "_default_downloader", fake_install)
    ok = c.post("/api/models/yue2/download", json={"accept_terms": True})
    assert ok.status_code == 200 and ok.json()["state"] in ("running", "done")
    modelsetup.wait()
    assert c.get("/api/models/status").json()["installed"] is True
    assert c.post("/api/models/yue2/download", json={"accept_terms": True}).status_code == 409
    monkeypatch.delenv("INSTMATE_LOCAL_APP")
    other = TestClient(app_mod.app)                                            # a server that is not the local app
    assert other.get("/api/models/status").json()["enabled"] is False
    assert other.post("/api/models/yue2/download", json={"accept_terms": True}).status_code == 403


def test_the_local_app_answers_only_to_its_own_address_and_its_own_page(local, monkeypatch):
    """Review (Origin/Host): other host names (DNS rebinding) and requests from other sites are refused."""
    import app as app_mod
    monkeypatch.setattr(modelsetup, "_default_downloader", fake_install)
    c = TestClient(app_mod.app, base_url="http://127.0.0.1:8000")
    assert c.get("/api/models/status").status_code == 200
    assert c.get("/", headers={"Host": "localhost:8000"}).status_code == 200
    for host in ("evil.example", "evil.example:8000", "127.0.0.1.evil.example", "192.168.0.5:8000"):
        r = c.get("/api/models/status", headers={"Host": host})
        assert r.status_code == 403 and "127.0.0.1" in r.json()["detail"], host
    body = {"accept_terms": False}
    assert c.post("/api/models/yue2/download", json=body).status_code == 422                        # no Origin: curl and tools
    assert c.post("/api/models/yue2/download", json=body, headers={"Origin": "http://127.0.0.1:8000"}).status_code == 422   # its own page
    for origin in ("http://evil.example", "https://127.0.0.1:8000", "http://127.0.0.1:9999", "null"):
        assert c.post("/api/models/yue2/download", json=body, headers={"Origin": origin}).status_code == 403, origin
    assert c.post("/api/models/yue2/download", json=body, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert modelsetup.status()["state"] == "idle", "nothing was started by the refused requests"
    monkeypatch.delenv("INSTMATE_LOCAL_APP")                                   # without the flag the guard is off (public and test servers)
    assert TestClient(app_mod.app).get("/api/models/status", headers={"Host": "evil.example"}).status_code == 200


def test_the_first_run_card_is_a_collapsed_row_hidden_until_the_server_says_local():
    page = (ROOT / "static/index.html").read_text(encoding="utf-8")
    card = page[page.index('id="firstrun-card"'):page.index('id="chat-card"')]
    assert '<details class="card hidden" id="firstrun-card"' in page and "4.5GB" in card and "CC BY-NC 4.0" in card
    assert "악보·MIDI는 지금 바로 돼요" in card and "고정된 버전" in card and "실행 코드" in card
    assert 'id="firstrun-terms"' in card and 'id="firstrun-start" class="primary" disabled' in card and '<progress id="firstrun-bar"' in card
    assert "open" not in card.split(">")[0].replace("details", ""), "collapsed by default: the melody input stays in view"
    js = (ROOT / "static/workspace.js").read_text(encoding="utf-8")
    assert "/api/models/status" in js and "/api/models/yue2/download" in js and "accept_terms: terms.checked" in js
    assert "if (!shown) { card.classList.add('hidden'); return; }" in js         # demo or no server: stays hidden
    assert "앱과 연결이 끊겼어요" in js and "window.loadYueStatus" in js          # a lost connection is not "done"; done refreshes YuE2
    assert "o.disabled = !f.practice_parts" in js and "pdf-note" in js           # missing tools are said before a click fails
    builder = ROOT / "scripts/build_demo.py"                                   # not in the installed app, only in the full source
    if builder.exists():
        assert "#firstrun-card" in builder.read_text(encoding="utf-8")        # the static demo never shows it
