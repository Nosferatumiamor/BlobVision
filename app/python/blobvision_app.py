#!/usr/bin/env python3
"""BlobVision desktop launcher — wraps the Gradio UI in a native PyWebView window.

No browser tab, no URL bar: a real OS window with title bar, taskbar icon, and
clean shutdown on close. The Gradio server runs in-process (non-blocking) and
serves the same UI blobvision_ui.py builds.
"""
import argparse
import atexit
import os
import socket
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(APP_DIR)
os.chdir(APP_DIR)
for p in (SCRIPT_DIR, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("GRADIO_SSR_MODE", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_PROXY = "127.0.0.1,localhost,<local>"
for _k in ("NO_PROXY", "no_proxy"):
    if "127.0.0.1" not in os.environ.get(_k, ""):
        os.environ[_k] = "{},{}".format(os.environ.get(_k, ""), _PROXY).strip(",")

ICON_PATH = os.path.join(APP_DIR, "assets", "logo", "blob-logo.ico")
DEFAULT_TITLE = "BlobVision"


def _wait_for_port(host, port, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def _import_webview():
    try:
        import webview
        return webview
    except ImportError as exc:
        print(
            "PyWebView n'est pas installe dans le venv.\n"
            "Run: venv\\Scripts\\python.exe -m pip install pywebview\n",
            flush=True,
        )
        raise SystemExit(2) from exc


def main():
    parser = argparse.ArgumentParser(description="BlobVision desktop app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--lazy", action="store_true", help="Open UI without loading models until Start")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--open-browser", action="store_true", help="Ignored — native window opens automatically")
    parser.add_argument("--debug", action="store_true", help="Open dev tools in the webview")
    args = parser.parse_args()

    print("BlobVision — construction de l'interface Gradio...", flush=True)
    import blobvision_ui as ui

    atexit.register(ui._cleanup_on_exit)
    demo = ui.build_ui()
    demo.queue(default_concurrency_limit=1)
    if args.lazy:
        ui._load["stopped"] = True
        ui._load["stage"] = "Disconnected"
    else:
        threading.Thread(target=ui._warmup_worker, daemon=True).start()

    from blobvision_paths import gradio_allowed_paths

    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        ssr_mode=False,
        theme=__import__("gradio").themes.Soft(),
        css=ui.CSS,
        allowed_paths=gradio_allowed_paths(),
        inbrowser=False,
        prevent_thread_lock=True,
    )
    print("BlobVision — demarrage du serveur local (port {})...".format(args.port), flush=True)
    demo.launch(**launch_kwargs)

    url = "http://{}:{}?_t={}".format(args.host, args.port, int(time.time()))
    if not _wait_for_port(args.host, args.port, timeout=120):
        print("Le serveur Gradio n'a pas repondu a temps — ouverture navigateur de secours.", flush=True)
        import webbrowser
        webbrowser.open(url)
        input("Appuye sur Entree pour quitter...")
        return

    webview = _import_webview()
    print("BlobVision — ouverture de la fenetre native sur {}".format(url), flush=True)
    window = webview.create_window(
        DEFAULT_TITLE,
        url=url,
        width=args.width,
        height=args.height,
        min_size=(960, 600),
        text_select=False,
        confirm_close=False,
    )
    try:
        webview.start(
            debug=args.debug,
            icon=ICON_PATH if os.path.isfile(ICON_PATH) else None,
        )
    except Exception as exc:
        print("PyWebView a echoue ({}) — ouverture navigateur de secours.".format(exc), flush=True)
        import webbrowser
        webbrowser.open(url)
        input("Appuye sur Entree pour quitter...")
    finally:
        try:
            demo.close()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()
