# GitHub 배포 준비 (공개 저장소 + Releases)

사용자 결정(2026-10-20): 공개 저장소에는 소스와 설치 안내, GitHub Releases에는 지인이 받는 Mac 설치 파일 하나, 기존 사이트는 시연·소개와 GitHub 다운로드 버튼. 생성은 각자 컴퓨터에서 합니다.

## 준비된 것 (로컬에만 있음, 아무것도 공개하지 않았음)
| 항목 | 위치 | 상태 |
|---|---|---|
| 설치 파일 빌더 | `scripts/build_mac_installer.py` | 단일 파일(.command)에 ZIP을 붙이고 검사값 확인, 풀기, 시작까지 하는 설치 파일을 만듭니다. 시험 있음 |
| 첫 실행 모델 받기 | `instmate/modelsetup.py`, 앱 첫 화면 카드 | 약관 동의, 크기, 진행률, 실패와 재시도. 로컬 앱 모드에서만 켜짐. 가짜 다운로더로 시험 |
| 시작 스크립트 | `start_mac.command` | 처음이면 프로그램 설치, 로컬 서버(127.0.0.1) 시작, 켜지면 브라우저 열기 |
| 공개 저장소 내보내기 폴더 | 릴리스 폴더의 `public_export/` | 소스 허용 목록 + 공개용 README. git 기록 없음 |
| 사이트 다운로드 버튼 | `scripts/build_demo.py --download-url <GitHub Releases 주소>` | github.com의 releases 주소만 받음. 주소가 정해지기 전에는 "공개 준비 중" 문구 |

## 사용자 결정 (2026-10-20)
1. **저장소:** 이미 있는 공개 저장소 `Apollo-Chanhwi-Lee/grandchaps`를 그대로 쓰고, 새 내용으로 덮어씁니다(기존에는 옛 정적 데모 `index.html`과 `.nojekyll`만 있었음). 기존 비공개 `instmate` 저장소의 공개 전환이나 기록 공개는 하지 않습니다.
2. **소스 라이선스:** 비영리 조건. PolyForm Noncommercial License 1.0.0(공식 원문 그대로 `LICENSE`). 이 라이선스는 이 프로젝트의 소스에만 적용되고, 외부 구성요소(글꼴, 라이브러리, 모델)는 각자의 조건을 따릅니다(`docs/LICENSES.md`).
3. **마스코트 그림:** 사용자 결정으로 **그대로 공개**합니다. 외부 AI 도구로 만들어 권리 기록이 없고 닮았다는 우려가 있었던 그림입니다. 문제가 되면 그림만 바꿔 다시 올립니다(이미 공개된 기록은 남습니다).
4. **서명:** 하지 않고 배포합니다. 받는 사람은 "확인되지 않은 개발자" 경고에서 우클릭 후 열기를 누릅니다.

## 아직 남은 게이트
4. **GPL 검토:** `requirements.txt`의 `pedalboard`(GPL-3.0)가 설치된 앱의 배포에 미치는 영향을 검토해야 합니다.
5. **모델 권리:** YuE2는 CC BY-NC 4.0이라 첫 실행 화면에서 동의를 받고 받게 했습니다. 리드시트 모델과 학습 데이터는 올리지 않습니다(`docs/LICENSES.md`).
6. **PolyForm Noncommercial과 GPL의 관계:** 4번과 함께 보세요. 이 라이선스는 소스에만 적용되며 `pedalboard`에는 영향이 없습니다.
7. **깨끗한 Mac 시험:** 새 Mac(또는 새 사용자 계정)에서 Release 파일 하나로 처음부터 끝까지 한 번 해 보기 전에는 "설치된다"고 공개 안내하지 않습니다.

## 게이트가 풀린 뒤의 순서 (아직 실행하지 않음, 각 단계는 별도 승인)
1. 기존 `Apollo-Chanhwi-Lee/grandchaps`에 `public_export/` 올리기(덮어쓰기: `index.html`만 교체, 나머지는 추가).
2. Release 올리기: `gh release create v0.1.0 Instmate-Mac-Installer.command Instmate-Mac-App.zip --repo Apollo-Chanhwi-Lee/grandchaps --title "..." --notes-file ...`.
3. 사이트에 버튼 넣기: `python scripts/build_demo.py --out public_site --download-url https://github.com/Apollo-Chanhwi-Lee/grandchaps/releases/latest` 후 Netlify 배포.
4. 임시 터널과 서버는 더 이상 쓰지 않습니다. 사이트는 정적 시연이라 작성자 Mac이 필요하지 않습니다.
