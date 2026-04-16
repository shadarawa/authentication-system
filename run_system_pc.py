# run_system_pc.py
import argparse, os, subprocess, sys, time, webbrowser
from urllib.request import urlopen

from tts_utils import speak, summarize_cmd

PY = sys.executable
def wait_any_health(base_url, timeout_s=90.0):
    health_paths = ["/api/health", "/health", "/docs", "/openapi.json"]
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        for p in health_paths:
            try:
                with urlopen(base_url.rstrip("/") + p, timeout=2) as r:
                    if 200 <= r.status < 400:
                        return True, p
            except Exception as e:
                last_err = e
        time.sleep(0.5)
    return False, repr(last_err)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_port", type=int, default=8000)
    ap.add_argument("--dash_port", type=int, default=8501)
    ap.add_argument("--server_host", default="0.0.0.0")
    ap.add_argument("--server_url", default=None)
    ap.add_argument("--speak_terminal", action="store_true",
                    help="Speak important terminal messages (best-effort TTS)")
    ap.add_argument("--speak_full_cmd", action="store_true",
                    help="If used with --speak_terminal, also speak the full command line")
    args = ap.parse_args()

    def say(text: str) -> None:
        if args.speak_terminal:
            speak(text)

    def say_cmd(cmd) -> None:
        if not args.speak_terminal:
            return
        if args.speak_full_cmd:
            speak("Command: " + " ".join(cmd))
        else:
            msg = summarize_cmd(cmd)
            if msg:
                speak(msg)

    server_url = args.server_url or "http://127.0.0.1:%d" % args.server_port
    env = os.environ.copy()
    env["SERVER_URL"] = server_url

    procs = []

    backend_cmd = [PY, "-m", "uvicorn", "main:app", "--host", args.server_host, "--port", str(args.server_port)]
    print("[RUN] Starting AI backend:", " ".join(backend_cmd))
    say_cmd(backend_cmd)
    p_backend = subprocess.Popen(backend_cmd, env=env)
    procs.append(p_backend)

    ok, info = wait_any_health(server_url, timeout_s=120.0)
    if not ok:
        print("[ERROR] Backend not healthy. Last error:", info)
        say("Backend not healthy.")
        for p in procs:
            try: p.terminate()
            except: pass
        sys.exit(1)

    print("[OK] Backend healthy via:", info)
    say("Backend is ready.")

    dash_cmd = [PY, "-m", "streamlit", "run", "dashboard.py",
                "--server.port", str(args.dash_port),
                "--server.headless", "false"]
    print("[RUN] Starting dashboard:", " ".join(dash_cmd))
    say_cmd(dash_cmd)
    p_dash = subprocess.Popen(dash_cmd, env=env)
    procs.append(p_dash)

    dash_url = "http://127.0.0.1:%d" % args.dash_port
    print("[INFO] Dashboard:", dash_url)
    print("[INFO] Backend  :", server_url)
    say(f"Dashboard is available on port {args.dash_port}.")
    webbrowser.open(dash_url)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[RUN] Shutting down...")
        say("Shutting down.")
        for p in procs[::-1]:
            try: p.terminate()
            except: pass

if __name__ == "__main__":
    main()
