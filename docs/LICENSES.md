# 데이터·모델 라이선스

작곡메이트가 학습·평가에 쓰는 외부 자료와 라이선스입니다. 원본 파일은 `datasets/`에 두며 저장소에 포함하지 않습니다 (`.gitignore`).
음원(오디오)은 학습에 쓰지 않습니다. 멜론·유튜브 등에서 곡을 내려받지 않으며, 사용자가 직접 올린 참고 오디오는 분석 후 삭제합니다.

## 사용 중

| 자료 | 용도 | 라이선스 | 위치 |
|---|---|---|---|
| POP909 (Wang et al., 2020) | 코드 모델(BiLSTM, HMM) 학습·평가, **그리고** 리드시트 변환(`scripts/build_dataset.py`)과 그 데이터로 한 Qwen3-1.7B LoRA 학습 (M1~M4) | MIT (로컬 `LICENSE` 파일 확인) | `datasets/POP909-Dataset` |
| POP909 구조 레이블 (Dai, Zhang, Dannenberg, 2020, `hierarchical-structure-analysis`) | M1 리드시트 데이터 (구간 이름 추정) | MIT (Copyright 2020 Shuqi Dai, 로컬 `LICENSE` 파일 확인) | `datasets/pop909_structure` |
| Hooktheory (SheetSage 공개본, `Hooktheory.json.gz`, 26,175 클립) | 리드시트 데이터 v2 (`scripts/build_dataset_v2.py`, 멜로디+코드만, 장르 태그 `팝`)와 그 데이터로 할 다음 LoRA 학습 | CC BY-NC-SA 3.0 (비영리, 동일조건 변경허락). 학습에 쓰게 되었으므로 이 조건이 적용된다. 가중치 배포는 검토 필요 (아래) | `datasets/hooktheory`. sha256 `917b7cd5...698e0c` |
| Nottingham Music Database (ABC 변환본, `datasets/nottingham/ABC_cleaned`) | 리드시트 데이터 v2 (장르 태그 `민요`) | 변환 저장소는 GPLv3 (로컬 `LICENSE.md`). 원자료(Nottingham Music Database, Ashover 자료는 Mick Peat 등)의 권리는 확인하지 못했다. 사용자의 결정으로 비영리 과제용으로 포함. 어떤 배포 전에도 권리 검토 필요 | `datasets/nottingham` |
| Bach 코랄 (music21 내장 코퍼스) | 리드시트 데이터 v2 (장르 태그 `클래식`) | 작곡은 퍼블릭 도메인. music21 코퍼스 사본은 music21과 함께 설치된 로컬 파일이며 music21은 BSD-3-Clause (코퍼스 파일별 편집본의 표기는 확인하지 않았다) | music21 패키지 안 (`music21/corpus/bach`) |
| Lakh MIDI Dataset v0.1 (Raffel, 2016; `lmd_full`/`lmd_matched`, `match_scores.json`) + MSD 곡 이름·연도(`unique_tracks.txt`, `tracks_per_year.txt`) + tagtraum 장르 라벨(CD1/CD2) | 리드시트 데이터 `datasets/leadsheet_lakh` (멜로디·코드를 MIDI에서 자동 추출, 장르 태그 `팝`, 실제 장르는 meta에만) | 데이터셋 자체는 CC BY 4.0 (배포 페이지 원문: "The Lakh MIDI Dataset is distributed with a CC-BY 4.0 license"). 그러나 MIDI는 저작권이 있는 실제 곡의 채보이고 채보자 표기가 없다. tagtraum 라벨은 "Research only, strictly non-commercial". **연구·비영리 전용(원장 결정 `research_review`). 어떤 배포 전에도 학습 입력에서 제외해야 한다** | `datasets/lakh`, `datasets/leadsheet_lakh`. 원장 `datasets/lakh/ledger.csv`, 상세 `docs/lakh_stage1.md`, `docs/lakh_stage2.md` |
| PDMX v9 (Long et al., ICASSP 2025; Zenodo 10.5281/zenodo.15571083) | 리드시트 데이터 `datasets/leadsheet_pdmx` (`scripts/build_dataset_pdmx.py`, 멜로디+화음, 장르 태그는 PDMX `genres`에서 온 출처 표시) | 데이터셋 전체는 Zenodo 표기 CC BY 4.0. 개별 악보는 MuseScore 공개 표기 `publicdomain` 또는 `cc-zero`만 사용. 제공자가 보고한 라이선스 표기 충돌(`license_conflict`) 31,221곡은 쓰지 않고 장부에서 `hold`. 변환 코드의 원 저장소는 MIT(코드 라이선스일 뿐 악보 권리가 아님). 곡 단위 장부 `datasets/pdmx/ledger.csv`(원본 id, 라이선스, 충돌 여부, 파일 sha256, 사용/보류/제외와 이유). 법률 보증이 아니며 개별 악보의 권리 적법성은 확인하지 못했다. 저작자 표기와 CC BY 조건은 배포 시 따로 지킨다 | `datasets/pdmx` (원본, 저장소에 포함하지 않음), `datasets/leadsheet_pdmx` |

## 받았지만 아직 쓰지 않음

(현재 없음. Hooktheory는 리드시트 데이터 v2에 쓰게 되어 위 "사용 중" 표로 옮겼다. 과거 기록: 이전에는 구간 이름이 없어 M1에서 제외했고, 데이터 변환·학습·평가 코드에 입력으로 연결되어 있지 않았다.)

## 검토 후 보류·제외

| 자료 | 라이선스 | 판단 |
|---|---|---|
| Chordonomicon | 데이터 CC BY-NC 4.0, 코드 Apache-2.0 | 코드 진행만 있고 멜로디가 없어 보류 |
| Nottingham 민요 데이터 | (이전 판단) 일반 라이선스 확인 못 함 (Ashover 자료는 Mick Peat 권리) | 이전에는 권리 확인 전까지 제외했으나, 사용자의 결정으로 비영리 과제용 학습에 포함했다. 위 "사용 중" 표 참고 |

## 실행 모델

| 모델 | 라이선스 | 비고 |
|---|---|---|
| Qwen3 14B (Ollama) | Apache-2.0 | 자연어 요청 해석. 로컬 실행 |
| Qwen3 1.7B (`Qwen/Qwen3-1.7B`, Hugging Face) | Apache-2.0 | 리드시트 모델의 기반 모델. LoRA 어댑터(`models/leadsheet_lora_1p7b`)는 POP909(MIT 표기)와 구조 레이블(MIT)로만 학습. 어댑터의 배포 조건은 아래 "배포 전 검토" 참고 |
| EXAONE 3.5 | 비상업 라이선스 | 기본 모델에서 제외 |
| DPO 어댑터 (`models/leadsheet_dpo_v1_mlx`, v1 위) | 배포 안 함 | 기반은 v1(POP909, 원곡 권리 미확인). 학습 쌍은 v2 모델(Hooktheory CC BY-NC-SA 3.0 포함)이 만든 후보라 비상업 조건을 이어받는다고 보고 과제 발표 이외로 쓰지 않는다 |
| v3 어댑터 (`runs/lora_v3_1p7b_20261003`) | 배포 안 함, 채택 안 함 | 학습 입력에 Lakh(연구 전용)와 Hooktheory(비상업) 포함 |

## 사용 조건: 확인한 원문 조건과 배포 전 검토가 필요한 사항

**확인한 원문 조건 (자료별).** POP909와 구조 레이블은 로컬 `LICENSE` 파일이 MIT입니다. Hooktheory 공개본은 SheetSage의 안내에 따라 CC BY-NC-SA 3.0, Chordonomicon은 데이터 CC BY-NC 4.0과 코드 Apache-2.0, Qwen3 1.7B/14B는 모델 카드의 Apache-2.0입니다 (위 표). 이 표는 자료별 표기를 모은 것이고, 아래 사항을 해결했다는 뜻이 아닙니다.

**배포 전에 검토가 필요한 사항.**

- 라이선스 표기만으로 원곡의 권리가 해결되지는 않습니다. POP909는 실제 대중가요를 채보한 자료이므로, 데이터셋의 MIT 표기와 별개로 원곡·채보의 권리 범위를 따로 확인해야 합니다. 이 문서는 그 판단을 하지 않았습니다.
- LoRA 가중치(`models/leadsheet_lora_1p7b`)와 생성된 곡을 학습 자료의 조건 아래 어떻게 배포할 수 있는지는 확정하지 않았습니다. 현재 1.7B 어댑터의 학습 입력은 POP909와 구조 레이블(둘 다 MIT 표기)뿐이지만, 이것이 가중치와 생성곡의 배포 조건을 정해 주지는 않습니다. 다음 학습(데이터 v2, Qwen3-4B 기반 LoRA 예정)의 입력에는 Hooktheory(CC BY-NC-SA 3.0)와 Nottingham(원자료 권리 미확인), Bach 코랄이 들어갑니다. 그 가중치와 생성곡에 비영리·동일조건 등이 어떻게 적용되는지는 정해지지 않았으므로, 그 어댑터는 이 검토를 마칠 때까지 과제 제출·발표 이외로 배포하지 않습니다. Qwen3-4B 모델 카드의 라이선스는 이 환경에서 확인하지 못했습니다.
- 과제 제출·발표(비영리 교육 목적) 이외의 사용, 즉 공개 저장소, 가중치 공개, 생성곡 공개, 상업 이용은 모두 위 검토를 마친 뒤에만 합니다.
- 외부에 올릴 때는 데이터 원본을 올리지 않고, 이 문서와 출처 표기만 포함합니다.

## 주요 라이브러리

| 라이브러리 | 라이선스 | 용도 |
|---|---|---|
| mlx, mlx-lm | MIT | LoRA 학습과 생성 |
| librosa | ISC | 레퍼런스 음원 분석 |
| music21 | BSD-3-Clause | 악보(MusicXML) |
| Salamander Grand Piano V3 SF2 (원작 Alexander Holm, SF2 변환 FreePats/roberto@zenvoid.org) | CC BY 3.0 (출처 표시 필요) | 피아노 음색 (`models/soundfonts/SalamanderGrandPiano-V3+20200602.sf2`, freepats.zenvoid.org에서 2026-10-04 받음, tar.xz sha256 `15edb061...e213`). GM 음색 위에 겹쳐 피아노(프로그램 0)만 바꾼다 |
| Pedalboard 0.9.25 (Spotify) | GPL-3.0 | VST3/AU 악기로 특정 파트 연주 (`instmate/vst.py`). 코드를 배포하면 GPL 소스 공개 의무가 생김 |
| Surge XT 1.3.4 | GPL-3.0 | 신디사이저 (사용자 플러그인 폴더에 설치, `scripts/vst_pick.py`로 파트에 연결) |
| Dexed 1.0.1 | GPL-3.0 | FM(DX7) 일렉트릭 피아노 계열 |
| sfizz 1.2.3 (`sfizz_render`) | BSD-2-Clause | SFZ 악기 렌더링 도구 (`~/.local/bin`, 아직 앱에 연결하지 않음) |
| pyworld 0.3.5 (WORLD 보코더, Masanori Morise) | pyworld MIT, WORLD 수정 BSD | 허밍 가이드 보컬: 사용자가 녹음한 자기 목소리를 멜로디 음높이로 다시 합성 (`instmate/vocal.py`). 녹음은 `data/voice/`에만 저장 |
| FastAPI | MIT | 서버 |
| Tone.js, OpenSheetMusicDisplay (CDN) | MIT, BSD-3-Clause | 브라우저 재생·악보 표시 |

## 음원 렌더링 (M7, `instmate/audio.py`)

외부 프로그램을 subprocess로 실행할 뿐 코드에 링크하거나 저장소에 포함하지 않습니다. 실행 중 외부 API는 쓰지 않습니다.
사운드폰트는 저장소에 넣지 않으며(`models/`는 `.gitignore`), 받은 파일의 경로를 `INSTMATE_SOUNDFONT`로 지정하거나
`models/soundfonts/GeneralUser-GS.sf2` 또는 `models/soundfonts/FluidR3_GM.sf2`에 둡니다.

| 자료·프로그램 | 라이선스 (확인한 원문) | 용도 |
|---|---|---|
| GeneralUser GS v2.0.3 (S. Christian Collins), 권장 사운드폰트 | "GeneralUser GS v2.0.3 License v2.0". 원문: "You may use GeneralUser GS without restriction for your own music creation, private or commercial. ... Please feel free to use it in your software projects, and to modify the SoundFont bank or its packaging to suit your needs." 샘플에 대해서는 "all of which allow full use in music production" 이지만, 저자 스스로 "I cannot be 100% sure where all of the samples originated"라고 적었다. 웹에 소개할 때는 저자의 다운로드 파일에 직접 링크하지 말라는 요청이 있다. (출처: `github.com/mrbumpy409/GeneralUser-GS`의 `documentation/LICENSE.txt`, 2026-10-03 확인). **받아서 사용 중**: 2026-10-03 같은 저장소 `main`의 `GeneralUser-GS.sf2` (32,319,396 bytes, git blob `298b552d…` 일치, sha256 `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe`), 라이선스 원문 사본 `models/soundfonts/GeneralUser-GS_LICENSE.txt` | General MIDI 악기 소리 |
| FluidR3_GM (Frank Wen), 대안 사운드폰트 | MIT. 원문(README 인용): "I hereby release Fluid under the MIT license, as described in COPYING." Copyright 2000-2002, 2008 Frank Wen. (출처: Debian `fluid-soundfont` 3.1-5.3 패키지의 `debian/copyright`, 2026-10-03 확인) | General MIDI 악기 소리 |
| FluidSynth 2.6.1 | LGPL-2.1-or-later (Homebrew formula 표기) | MIDI -> WAV 오프라인 합성 |
| ffmpeg (선택) + LAME | ffmpeg GPL-3.0-or-later (Homebrew 빌드 표기), LAME LGPL-2.0-or-later | WAV -> MP3 변환. 없으면 WAV만 |

렌더링한 음원은 생성곡의 녹음이므로, 위 "배포 전 검토" 사항(학습 자료 조건과 생성곡의 배포)이 그대로 적용됩니다.
사운드폰트 라이선스는 소리 샘플의 사용 조건일 뿐 생성곡의 배포 조건을 정해 주지 않습니다.

## 화면 라이브러리 (static/vendor, 2026-10-05 추가)

발표장에서 인터넷 없이 화면이 열리도록 CDN 대신 저장소에 넣은 파일이다. 내용은 고치지 않았다.

| 파일 | 버전 | 라이선스 | 받은 곳 | SHA-256 (앞 16자) |
|---|---|---|---|---|
| `static/vendor/Tone.js` | Tone.js 14.8.49 | MIT (`Tone.LICENSE.md`) | cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js | 1261cdd3331d8262 |
| `static/vendor/opensheetmusicdisplay.min.js` | OpenSheetMusicDisplay 1.8.9 | BSD-3-Clause (`opensheetmusicdisplay.LICENSE`) | cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.9 | 9c4b03dd965857a1 |

## 가이드 보컬의 기본 목소리 (2026-10-05 추가)

녹음한 목소리가 없을 때 '나나나'/'라라라'는 이 맥의 한국어 음성(`say -v Yuna`)이 낸 "나아아…"/"라아아…" 소리를 분석해 멜로디 높이로
다시 합성한다(`instmate/vocal.py: system_voice`). 음성은 실행할 때마다 이 컴퓨터에서 만들어 메모리에서만 쓰고, 저장소나 과제 폴더에
넣지 않는다. macOS 시스템 음성은 개인적·비상업적 용도의 콘텐츠 제작에 한해 쓸 수 있으므로, 이 기능으로 만든 음원은 발표·시연용으로만
쓰고 배포하지 않는다. `say`가 없는 컴퓨터에서는 포먼트로 만든 합성 목소리(`builtin_voice`)로 대체된다.

## AI 보컬 데모: YuE2 (2026-10-05 추가, 선택 기능)

- 모델: YuE2-3B (Multimodal Art Projection, https://github.com/multimodal-art-projection/YuE). 코드 Apache-2.0, 가중치 CC BY-NC 4.0
  + 개인 창작자 추가 허용(개인 사용자·창작자·음악가는 생성한 곡을 공개·배포·판매할 수 있음, 출처 표기는 권장, 기만적 사용 금지).
- 실행: 맥에서는 커뮤니티 MLX 이식판 `ahmadw/YuE2-3B-MLX`의 8비트 가중치(같은 CC BY-NC 4.0)를 `models/yue2_mlx/repo`에 받아 쓴다
  (저장소에는 넣지 않음). 실행 전 스크립트(`generate.py`, `yue2_model.py`, `yue2_vae.py`)에 외부 통신·프로세스 실행·파일 삭제가 없는 것을
  읽어서 확인했다. 추가 패키지: tiktoken.
- 쓰는 방식: 작곡메이트 리드시트의 멜로디와 코드를 ABC 악보로 넘기고 가사는 음표마다 'na'/'la' 한 음절(`instmate/yue.py`).
  반주는 YuE2가 새로 만든다(앱의 편곡이 아님). 앱의 다른 기능은 이 모델 없이 그대로 동작한다.
- 한계: 최대 메모리 약 18GB(24GB 맥에서 다른 무거운 작업과 같이 못 돌림), 생성마다 결과가 다름, 한국어 가사 지원은 확인되지 않음.
- 라이선스 확인 (2026-10-10 정리, 근거는 2026-10-05에 받아 둔 사본 `models/yue2_mlx/license_snapshot_20261005/MODEL_LICENSE`, 해시는 같은 폴더 `SHA256.txt`):
  - 가중치(YuE2-3B, YuE2-Vae, YuE2-Vae-legacy): CC BY-NC 4.0. 출처 표기 대상: https://huggingface.co/m-a-p/YuE2-3B , https://huggingface.co/m-a-p/YuE2-Vae , https://huggingface.co/m-a-p/YuE2-Vae-legacy
  - 학술 이용 조항(사본의 "Academic use"): 학생·교육자·연구자·학술 기관은 비상업 연구와 교육에 무료로 쓸 수 있다. 이 과제(수업)의 사용은 여기에 해당한다. 이 조항이 기관에 상업 이용권을 주지는 않는다.
  - 코드: Apache-2.0 (사본 `LICENSE`).
  - 표기: 화면의 AI 보컬 옆에 "AI 보컬: YuE2 (m-a-p)"라고 적는다. 배포 모드에서는 YuE2를 쓰지 않는다(위 등급 B).
- AI 보컬 얹기 (2026-10-10 추가, `instmate/yueblend.py`, 버튼을 눌러야만 시작): YuE2가 부른 하이라이트에서 목소리만 분리해 앱의 반주에 얹는다.
  분리에는 Demucs(htdemucs)를 쓴다. Demucs는 앱과 따로 설치하는 별도 프로그램이다(`models/demucs/venv`, 저장소와 배포물에 넣지 않음).
  Demucs 코드는 MIT 라이선스(https://github.com/adefossez/demucs)이고, 내려받는 사전학습 가중치의 조건은 이 문서 작성 시점에 원문으로 확인하지 못했다. 그래서 등급 B로 두고 배포 모드에서는 쓰지 않는다.

## 악기 라이선스 등급과 배포 모드 (2026-10-06 추가, 지시서 `App/악기.md`)

등급은 `configs/instruments.yaml`의 `tier`에 적고, 앱의 "악기 출처·라이선스" 목록(`GET /api/credits`)에 그대로 나온다.

- **A**: 악기 파일·플러그인을 앱과 함께 배포할 수 있다 (CC0, CC-BY, MIT, BSD, GPL). 데모·배포 모드 모두에서 쓴다.
- **B**: 그 악기로 만든 음원만 나눌 수 있고 악기는 각자 설치한다. 데모 모드에서만 쓴다. 원문을 확인하지 않은 것도 B로 취급한다.
- 배포 모드(화면의 스위치, 또는 `INSTMATE_MODE=release`)에서는 A만 쓴다. B 악기는 `release`에 적은 A 악기로 바뀌고, 없으면 기본 사운드폰트 소리로 바뀐다. `scripts/vst_pick.py <part>`로 직접 고른 플러그인(`models/vst/tracks.json`)은 라이선스 기록이 없으므로 배포 모드에서 쓰지 않는다.
- 악기 소리는 실행 중에 인터넷에서 받지 않는다. 이 컴퓨터에 설치된 것만 쓰고, 없으면 GeneralUser GS로 대체한다.

| 지시서 ID | 악기 | 등급 | 라이선스 | 이 앱에서의 상태 |
|---|---|---|---|---|
| gm_default | GeneralUser GS v2.0.3 | A | GeneralUser GS License v2.0 (위 "음원 렌더링" 표, 2026-10-03 원문 확인) | 사용 중. 모든 대체의 기준 |
| piano_salamander | Salamander Grand Piano V3 (SF2 변환본) | A | CC BY 3.0 (위 "주요 라이브러리" 표). 출처 표기를 앱 목록에 넣었다 | 사용 중 (SF2. SFZ 원본은 쓰지 않는다) |
| synth_surge | Surge XT 1.3.4 | A | GPL-3.0 | 사용 중: `surge_pad` (배포 모드에서 Vital 대신). 소리는 초기 패치에서 `scripts/make_surge_pad.py`로 직접 만든 것이고 공장 프리셋이 아니다 |
| epiano_dexed | Dexed 1.0.1 | A | GPL-3.0 | 표에 있음 (`dexed_epiano`). 저장된 소리 파일이 없어 지금은 사운드폰트로 대체된다 |
| pad_vital | Vital (무료판) | B | 무료판 약관. 등급은 지시서 기준이며 약관 원문을 이번에 다시 확인하지 않았다 | 데모 모드에서만 사용 |
| strings_vsco | VSCO-2 Community Edition (Versilian Studios) | A | CC0-1.0. 2026-10-06 확인: `github.com/sgossner/VSCO-2-CE`의 `LICENSE` 파일 원문이 "CC0 1.0 Universal"이고 GitHub 라이선스 표시도 CC0-1.0 | 사용 중: `vsco_strings` (발라드·J-pop 스트링). `SFZ` 브랜치 커밋 `6dd651d5`에서 바이올린·비올라·첼로 합주 지속음 폴더만 받았다 (343 MB, `models/sfz/VSCO-2-CE`, 저장소에 올리지 않음). 세 섹션을 음역별로 이어 붙이고 샘플 음량을 고르게 맞춘 SFZ는 `scripts/make_vsco_strings.py`가 만든다 |
| amp_nam | Neural Amp Modeler 0.7.13 | 플러그인 A, **캡처 파일은 아직 없어 B로 취급** | 플러그인 MIT (2026-10-06 확인: `github.com/sdatkinson/NeuralAmpModelerPlugin`의 GitHub 라이선스 표시 MIT) | 플러그인만 설치됨: 공식 릴리스 v0.7.13의 `NeuralAmpModeler-v0.7.13-mac.dmg` (sha256 `e9cea7a4...39173`)에서 설치 프로그램을 실행하지 않고 VST3 묶음만 사용자 플러그인 폴더에 복사했다 (서명 없는 패키지). **캡처를 고른 설정 파일이 없어 지금은 내장 이펙트가 쓰인다.** 캡처는 사용자가 플러그인 창에서 고르고(`scripts/vst_pick.py --fx crunch`), 파일별 라이선스·확인 날짜·URL을 여기에 적은 뒤에만 등급을 올린다 |
| strings_vpo | Virtual Playing Orchestra | 미확인 | 샘플마다 다름 | **미사용** |
| drums_avl | AVL Drumkits | 미확인 | | **미사용** |
| drums_drumgizmo | DrumGizmo 킷 | 미확인 | | **미사용** |
| ir_openair | OpenAIR IR | 미확인 | | **미사용** |
| synth_odin, fx_airwindows, fx_dragonfly | Odin 2, Airwindows, Dragonfly Reverb | (지시서 기준 A) | | **미사용** (설치되어 있지 않고 연결하지 않았다) |
| labs_spitfire, amp_roots, amp_amplitube, drums_mtpower | Spitfire LABS, Amped Roots Free, AmpliTube 5 CS, MT Power Drum Kit 2 | (지시서 기준 B) | | **미사용** (설치되어 있지 않고 연결하지 않았다) |

배포 모드는 보컬에도 적용된다 (2026-10-06): 가이드 보컬의 맥 시스템 음성(B) 대신 내장 합성 음성을 쓰고, AI 보컬 데모(YuE2, 가중치 CC BY-NC, B)는 요청을 거절한다. 사용자가 녹음한 자기 목소리는 그대로 쓴다.

GPL에 대하여: Pedalboard, Surge XT, Dexed는 GPL-3.0이다. 이 앱을 자기 컴퓨터나 서버에서 돌려 웹 화면으로만 쓰게 할 때는 소스 공개 의무가 생기지 않지만, 설치형 앱으로 묶어 배포하면 앱 전체에 GPL이 적용되는지 검토해야 한다. 이것은 법률 자문이 아니다.

PDF 악보 인식 (2026-10-06): homr 0.7.0 (AGPL-3.0)을 `models/omr/homr_venv`에 따로 설치해 별도 프로그램으로 실행한다. 앱 코드에 포함하거나 함께 배포하지 않으며(B), 없으면 PDF만 거절되고 MIDI·MusicXML은 그대로 동작한다. PDF를 그림으로 바꾸는 poppler(GPL)도 같은 방식의 외부 프로그램이다. 이것은 법률 자문이 아니다.

악보 조판 (2026-10-07): Verovio 6.3.0 (LGPL-3.0)을 파이썬 패키지로 설치해 파트별 악보를 A4 쪽에 조판한다. 앱은 이 라이브러리를 고치지 않고 불러 쓰기만 하며, 없으면 브라우저의 간단한 악보(OpenSheetMusicDisplay)로 대체된다. 조판에 쓰는 음악 기호 글꼴(Leipzig, Bravura 등)은 Verovio에 포함된 것이다. 이것은 법률 자문이 아니다.

## 큰 사운드폰트의 보관과 재배치 (2026-10-07 기록)

`SalamanderGrandPiano-V3+20200602.sf2` (1,266,462,334바이트)는 Git 저장소와 제출 스냅샷에 넣지 않는다. GitHub는 100MiB를 넘는 일반 파일을 받지 않고, 과정 폴더 스냅샷에도 1.27GB를 싣지 않기로 했다(`scripts/sync_snapshot.py`의 `EXCLUDE_NAMES`).

| 항목 | 내용 |
|---|---|
| 보관 위치 | 이 Mac의 `models/soundfonts/SalamanderGrandPiano-V3+20200602.sf2` (저장소 밖, `.gitignore`의 `models/`) |
| 파일 SHA-256 | `712d0e681efbe5203a8014e9b3e84168f1908c82f2f6fb13bd2c77d6d72c70b7` (2026-10-07 이 Mac에서 계산) |
| 받은 곳 | FreePats 프로젝트의 피아노 페이지 `https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html` (2026-10-04 받음. 2026-10-07에 페이지가 열리는 것만 확인했고, 파일을 다시 받아 대조하지는 않았다) |
| 다시 놓는 법 | 위 페이지에서 "Salamander Grand Piano V3, SF2" 묶음(tar.xz)을 받아 풀고, `.sf2` 파일을 `models/soundfonts/`에 둔 뒤 앱을 다시 켠다. 위 SHA-256과 같은지 확인한다 |
| 사용·재배포 권한 | CC BY 3.0. 출처를 표시하면 사용과 재배포가 가능하다. 원작 Alexander Holm, SF2 변환 FreePats(roberto@zenvoid.org). 앱의 "악기 출처·라이선스" 목록에 표기돼 있다 |
| 없을 때 | 피아노가 기본 사운드폰트(GeneralUser GS)의 피아노로 난다. 다른 기능은 그대로다 |
| Colab | 확인하지 않았다. Colab 실행 환경의 파일은 세션이 끝나면 사라지므로, 쓰려면 실행할 때마다 위 페이지에서 받아야 한다. 노트북에는 이 파일을 받는 칸을 넣지 않았다 |

기본 사운드폰트 `GeneralUser-GS.sf2` (32,319,396바이트, SHA-256 `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe`)는 스냅샷에는 들어가고 Git에는 들어가지 않는다.

## 글꼴 (2026-10-19)
- Pretendard 1.3.9 가변 글꼴 `static/vendor/pretendard/PretendardVariable.woff2` (SIL Open Font License 1.1, 원문 `static/vendor/pretendard/LICENSE.txt`, 출처 https://github.com/orioncactus/pretendard). 스튜디오 화면의 본문 글꼴. SHA-256 9599f12fd42fc0bce1cd50b47a0c022e108d7aa64dd0d1bb0ed44f3282d900b4.
