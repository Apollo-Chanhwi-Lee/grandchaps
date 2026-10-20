# 작곡메이트 (Grandchaps InstMate)

멜로디(악보 파일)를 넣고 말로 요청하면 밴드 곡 초안(악보, 파트별 MIDI, 미리듣기)을 만들어 주는 작곡 도우미입니다. 밴드가 받아서 고치고 직접 연주하는 것이 목표이고, 음원은 미리듣기입니다.

## 내려받기 (Mac)
**[Releases](https://github.com/Apollo-Chanhwi-Lee/grandchaps/releases/latest)에서 `Instmate-Mac-Installer.command` 파일 하나를 받으세요.** (직접 풀어서 쓰려면 같은 곳의 `Instmate-Mac-App.zip`을 받으세요.)

1. 파일을 더블클릭합니다. 서명되지 않은 파일이라 "확인되지 않은 개발자" 경고가 나옵니다. **우클릭 후 열기**를 누르거나, 막히면 **시스템 설정 > 개인정보 보호 및 보안 > 그래도 열기**를 누르세요. 터미널에서 `zsh Instmate-Mac-Installer.command`로 실행해도 됩니다.
2. 처음 실행하면 앱이 설치되고, 내 Mac에서 서버가 켜지고(내 Mac에서만 접속 가능한 주소), 브라우저에 작곡메이트가 열립니다.
3. 노래(YuE2) 모델은 앱 첫 화면에서 받습니다. 약 4.5GB, **비상업(CC BY-NC 4.0)** 조건에 동의해야 받을 수 있습니다.
4. 곡 만들기는 모두 **내 Mac에서** 계산합니다. 작성자의 컴퓨터나 서버를 거치지 않습니다.

자세한 안내와 한계: [`docs/설치_Mac.md`](docs/설치_Mac.md)

## 지원 환경
- **macOS, Apple Silicon(M1 이상)만.** Windows, Intel Mac, Linux는 지원하지 않습니다.
- Python 3.11 이상 또는 `uv`가 미리 있어야 합니다.

## 소스에서 실행하기
```bash
zsh start_mac.command      # 처음이면 프로그램을 설치하고, 앱을 켜고, 브라우저를 엽니다
```
설치 파일을 직접 만들려면 이 저장소를 `instmate_local/` 폴더 이름으로 ZIP으로 묶은 뒤 `python3 scripts/build_mac_installer.py <ZIP> <출력 파일>`을 실행합니다.

## 권리와 라이선스 (읽어 주세요)
- 이 저장소에는 **모델 가중치, 학습 데이터, 키, 사용자 곡이 들어 있지 않습니다.** 리드시트 모델은 나눠 드리지 않으며, 없어도 올린 멜로디를 바탕으로 규칙으로 곡을 만듭니다.
- YuE2 모델의 음원은 비상업 조건입니다. 수업·연습·비영리 용도로만 쓰세요.
- 소스 코드는 **PolyForm Noncommercial License 1.0.0**(비영리 조건)입니다(`LICENSE`). 글꼴, 라이브러리, 모델 같은 외부 구성요소는 각자의 조건을 따릅니다(`docs/LICENSES.md`).
- 마스코트 그림, GPL 패키지(`pedalboard`) 등 검토가 끝나지 않은 항목은 `PACKAGE_NOTICE.md`와 `docs/LICENSES.md`를 보세요.
