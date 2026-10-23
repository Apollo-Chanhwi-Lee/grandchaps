#!/bin/zsh
# Double-click to start 작곡메이트 on this Mac: first run sets up the programs, then the app starts its own server (this Mac only,
# 127.0.0.1) and opens in the browser.  Models are downloaded inside the app on the first run.  Close this window to stop the app.
#   INSTMATE_PYTHON  python to use instead of .venv/bin/python (tests)     INSTMATE_PORT  port (default 8000)
#   INSTMATE_OPEN    command that opens the address (default: open)
cd "${0:A:h}" || exit 1                 # this file's own folder (zsh), no external tool needed
PY="${INSTMATE_PYTHON:-.venv/bin/python}"
pause() { read -k 1 "?아무 키나 누르면 닫혀요"; }
stop_here() { echo "$1"; pause; exit 1; }

# A usable Python 3.11+ (not the system stub that asks to install the developer tools).
find_python() {                                  # (not named "path": in zsh that variable is tied to PATH)
  local name found
  for name in python3.13 python3.12 python3.11 python3; do
    found=$(command -v "$name") || continue
    [ "$found" = /usr/bin/python3 ] && ! xcode-select -p >/dev/null 2>&1 && continue
    "$found" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null && { echo "$found"; return 0; }
  done
  return 1
}

# The programs are set up when .venv exists AND the finish mark holds the hash of requirements.txt: a run that stopped halfway
# leaves a .venv without it, so the next start repeats the package step instead of launching a broken app.
NEED=0
if [ -z "${INSTMATE_PYTHON:-}" ]; then
  HASH=$(shasum -a 256 requirements.txt 2>/dev/null | cut -d' ' -f1)
  { [ -x "$PY" ] && [ -n "$HASH" ] && [ "$(cat .venv/.instmate_ready 2>/dev/null)" = "$HASH" ]; } || NEED=1
fi
if [ "$NEED" = 1 ]; then
  echo "처음 실행이거나 설치가 끝나지 않았어요. 앱에 필요한 프로그램을 준비해요(몇 분 걸려요, 인터넷이 필요해요)."
  if command -v uv >/dev/null 2>&1; then
    uv run --python 3.13 --no-project scripts/install_mac.py --core || stop_here "설치가 멈췄어요. 위 오류 문장을 알려 주세요. 같은 파일을 다시 열면 멈춘 곳부터 이어서 해요."
  elif BASE=$(find_python); then
    "$BASE" scripts/install_mac.py --core || stop_here "설치가 멈췄어요. 위 오류 문장을 알려 주세요. 같은 파일을 다시 열면 멈춘 곳부터 이어서 해요."
  else
    stop_here "Python 3.11 이상 또는 uv가 필요해요. 터미널에서 'brew install uv'를 실행하거나(Homebrew가 없다면 https://docs.astral.sh/uv/ 의 설치 방법을 따라 하세요), https://www.python.org/downloads/ 에서 Python을 설치한 뒤 이 파일을 다시 열어 주세요."
  fi
  [ -x "$PY" ] || stop_here "프로그램 준비가 끝나지 않았어요. 위 오류 문장을 알려 주세요."
fi

PORT="${INSTMATE_PORT:-8000}"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  stop_here "포트 $PORT 를 다른 프로그램이 쓰고 있어요. 앱이 이미 켜져 있다면 http://127.0.0.1:$PORT 를 열어 보세요. 다른 포트는 INSTMATE_PORT=8001 처럼 정해요."
fi
export INSTMATE_LOCAL_APP=1 INSTMATE_PORT="$PORT"
"$PY" app.py &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' INT TERM EXIT
READY=0
for i in $(seq 1 90); do                       # open the browser only when the server answers
  kill -0 $SERVER 2>/dev/null || stop_here "앱이 켜지지 못했어요. 위 오류 문장을 알려 주세요."
  curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/" && { READY=1; break; }
  sleep 1
done
[ "$READY" = 1 ] || stop_here "앱이 90초 안에 켜지지 않았어요. 위 오류 문장을 알려 주세요."
echo "작곡메이트가 켜졌어요: http://127.0.0.1:$PORT  (이 창을 닫으면 꺼져요)"
# Its own app window when pywebview works (closing the window ends the app); the browser otherwise, or when INSTMATE_OPEN is set (tests).
if [ -z "${INSTMATE_OPEN:-}" ] && [ -f scripts/app_window.py ] && "$PY" -c 'import webview' 2>/dev/null; then
  "$PY" scripts/app_window.py "http://127.0.0.1:$PORT" && exit 0
fi
${INSTMATE_OPEN:-open} "http://127.0.0.1:$PORT"
wait $SERVER
