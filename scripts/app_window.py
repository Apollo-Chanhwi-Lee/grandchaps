"""Open the running local server in its own app window (no address bar, no tabs), with the seed mascot as the icon.

    python scripts/app_window.py http://127.0.0.1:8000

Exit code 0 when the window was shown and closed by the user; 2 when no window can be made (no pywebview, no display), so that
start_mac.command falls back to the browser.  The page's saved settings (theme, the user's own Gemini key) live in a folder of their own
under ~/Library/Application Support/Grandchaps, so they survive closing the window (pywebview forgets everything by default).
"""
import os
import sys
from pathlib import Path

ICON = Path(__file__).resolve().parent.parent / "static" / "mascot" / "eyes_open.png"
STORE = Path.home() / "Library" / "Application Support" / "Grandchaps" / "webview"


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else "http://127.0.0.1:8000"
    from urllib.parse import urlsplit
    try:
        u = urlsplit(url)
        local = u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost") and u.port is not None and u.username is None
    except ValueError:
        local = False
    if not local:
        print("this window opens only the local app:", url, file=sys.stderr)
        return 2
    try:
        import webview
        STORE.mkdir(parents=True, exist_ok=True)
        webview.create_window("작곡메이트", url, width=1280, height=860, min_size=(900, 640))
        webview.start(_set_icon, private_mode=False, storage_path=str(STORE))
    except Exception as exc:                      # no pywebview, no display, a broken WebKit: the caller opens the browser instead
        print("앱 창을 열지 못했어요:", exc, file=sys.stderr)
        return 2
    return 0


def _set_icon() -> None:
    try:
        from AppKit import NSApplication, NSImage
        image = NSImage.alloc().initWithContentsOfFile_(str(ICON))
        if image is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:                             # the icon is a nicety; the window works without it
        pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))
