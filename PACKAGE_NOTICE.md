# PACKAGE_NOTICE: private review copy of the locally installed app (route A, release 11)

This is the app that the single-file Mac installer carries: **test version 0.1.1**, a newer build of the 0.1.0 test version published on GitHub Releases on 2026-10-20. It is a test version: it was not checked on a clean Mac, it is not signed, and the limits below apply.

## What it is
- The Grandchaps songwriting helper (InstMate), installed and run **on the recipient's own Mac**. Flow: installer file -> first-run program setup -> a local server that answers on this Mac only (127.0.0.1) -> the browser opens the app -> the singing model is downloaded inside the app on the first run, after the recipient accepts its terms.
- Generation runs on the recipient's Mac. Nothing here connects to the author's computer, a tunnel or a remote GPU.
- Source: branch `e3-kv8-1017`, commit `5b3252d`. Uncommitted files included from the worktree: LICENSE, app.py, configs/explain.yaml, docs/GitHub_배포_준비.md, docs/설치_Mac.md, instmate/chat.py, instmate/explain.py, instmate/modelsetup.py, instmate/workspace_assistant.py, pyproject.toml, scripts/build_mac_installer.py, scripts/install_mac.py, scripts/package_files.py, start_mac.command, static/app.js, static/index.html, static/workspace.js, tests/test_install_mac.py, tests/test_local_app_start.py, tests/test_mac_installer.py, tests/test_modelsetup.py, tests/test_package_closure.py.
- Supported: **macOS on Apple Silicon (arm64) only.** Windows, Intel Macs and Linux are not supported (the singing and composing models are MLX only). Python 3.11 or newer, or uv, must already exist.
- A clean-machine installation, the real 4.5 GB model download and a full song generation were **not** verified. See the build report for what was actually run and what was stood in for.

## What is NOT in it
- Model weights of any kind (YuE2, lead sheet model, chat model), sound fonts, training data, reference music, scores, user songs and uploads.
- Secrets, invite codes, API keys, logs, git history, browser or local settings, dependency environments, caches.
- Other tests, notebooks and the other documents of the project.

## Rights and licence limits you must know (nothing here clears any right)
1. **YuE2 model**: downloaded only after the recipient ticks the consent for its CC BY-NC 4.0 (non-commercial) terms on the first-run card. Output made with it is for class, practice and non-profit use. Generated WAV files carry this notice.
2. **requirements.txt** lists packages such as `pedalboard` (GPL-3.0) and `torch`. Only the requirement lines are shipped; the packages are installed on the recipient's machine. The effect of GPL-3.0 on handing out an installed app has not been reviewed (see `docs/LICENSES.md`).
3. **Lead sheet model**: trained on data with non-commercial or research-only terms (see `docs/LICENSES.md`); it is not shipped. Without it the app writes songs from the given melody with rules.
4. **Mascot pictures in `static/mascot/`**: made by the project owner with external AI image tools. No rights or terms review is recorded in `docs/LICENSES.md`, and the owner raised a concern that the character looks like an existing game character. On 2026-10-20 the owner decided to publish them as they are. If a problem is reported, replace the pictures and publish a new release.
5. **Fonts and browser libraries** in `static/vendor/`: Pretendard (SIL OFL 1.1, `LICENSE.txt`), Tone.js (MIT, `Tone.LICENSE.md`), OpenSheetMusicDisplay (BSD-3-Clause, `opensheetmusicdisplay.LICENSE`). Their licence files are included.
6. The source is under the **PolyForm Noncommercial License 1.0.0** (`LICENSE`, official text unchanged). It covers this project's own source only; fonts, libraries and models keep their own terms.
7. The installer is **not signed or notarized** (owner decision); macOS shows an "unidentified developer" prompt. Allow it once: right-click, Open, or System Settings > Privacy and Security > Open Anyway, or run it from Terminal.
8. README links to some documents (`docs/윈도우_이어하기.md` and others) that are not in this ZIP.

## How to start
Read `docs/설치_Mac.md`. From this ZIP: unzip into a folder whose path uses only English letters and digits and run `zsh start_mac.command`.
