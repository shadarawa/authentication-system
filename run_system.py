#run_system.py
# python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang en
import argparse
import os
import subprocess
import sys
import time
import webbrowser
from urllib.request import urlopen

from tts_utils import speak

PY = sys.executable
def wait_any_health(base_url: str, timeout_s: float = 180.0):
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
    ap.add_argument("--server_url", default=None, help="default http://127.0.0.1:<server_port>")
    ap.add_argument("--no_pc", action="store_true", help="Start backend + dashboard only")

    ap.add_argument("--speak_terminal", action="store_true",
                    help="Speak important terminal messages (best-effort TTS)")
    ap.add_argument("--speak_full_cmd", action="store_true",
                    help="If used with --speak_terminal, also speak the full command line")

    ap.add_argument("--stt_name", action="store_true", help="Enable STT for capturing teacher name")
    ap.add_argument("--vosk_model", default="vosk-model-small-en-us-0.15", help="Path to Vosk model folder")
    ap.add_argument("--name_seconds", type=float, default=3.0, help="Seconds to record name audio for STT")

    ap.add_argument("--pc_ui_lang", choices=["en", "ar"], default="en",
                    help="Language for spoken prompts in pc_client (laptop user)")
    ap.add_argument("--pc_mute", action="store_true",
                    help="Disable spoken prompts in pc_client")
    ap.add_argument("--cooldown", type=float, default=6.0, help="pc_client cooldown")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")

    args = ap.parse_args()

    def say(text: str) -> None:
        if args.speak_terminal:
            speak(text)

    def say_cmd(cmd, label: str = "Command"):
        if not args.speak_terminal:
            return
        if args.speak_full_cmd:
            speak(f"{label}: " + " ".join(cmd))
        else:
            speak(f"Starting {label}.")

    server_url = args.server_url or f"http://127.0.0.1:{args.server_port}"
    env = os.environ.copy()
    env["SERVER_URL"] = server_url

    procs = []

    backend_cmd = [PY, "-m", "uvicorn", "main:app", "--host", args.server_host, "--port", str(args.server_port)]
    print("[RUN] Starting AI backend:", " ".join(backend_cmd))
    say_cmd(backend_cmd)
    p_backend = subprocess.Popen(backend_cmd, env=env)
    procs.append(p_backend)

    ok, info = wait_any_health(server_url, timeout_s=240.0)
    if not ok:
        print("[ERROR] Backend did not become healthy in time. Last error:", info)
        say("Backend did not become healthy in time.")
        for p in procs:
            try: p.terminate()
            except Exception: pass
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

    dash_url = f"http://127.0.0.1:{args.dash_port}"
    print(f"[INFO] Dashboard: {dash_url}")
    print(f"[INFO] Backend  : {server_url}")
    say(f"Dashboard is available on port {args.dash_port}.")
    try:
        webbrowser.open(dash_url)
    except Exception:
        pass

    def shutdown():
        print("\n[RUN] Shutting down...")
        say("Shutting down.")
        for p in procs[::-1]:
            try: p.terminate()
            except Exception: pass

    try:
        if args.no_pc:
            print("[RUN] Backend + Dashboard running. Press Ctrl+C to stop.")
            say("Backend and dashboard are running. Press Control C to stop.")
            while True:
                time.sleep(1.0)

        print("[RUN] Starting PC client loop (press 'q' in camera window to quit)...")
        say("Starting the PC client. Press Q in the camera window to quit.")
        pc_cmd = [PY, "pc_client.py", "--server", server_url, "--cooldown", str(args.cooldown), "--cam", str(args.cam),
                  "--register_on_reject"]
        if args.stt_name:
            pc_cmd += ["--stt_name", "--vosk_model", args.vosk_model, "--name_seconds", str(args.name_seconds)]
        if getattr(args, "pc_ui_lang", None):
            pc_cmd += ["--ui_lang", str(args.pc_ui_lang)]
        if getattr(args, "pc_mute", False):
            pc_cmd += ["--mute"]
        if args.speak_terminal:
            if args.speak_full_cmd:
                speak("Command: " + " ".join(pc_cmd))
            else:
                speak("Starting PC client")
        subprocess.call(pc_cmd, env=env)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()

if __name__ == "__main__":
    main()
