# run_system_robot.py (Python 3.x on PC)
import argparse
import webbrowser
import os, subprocess, sys, time
import socket
from urllib.request import urlopen

from tts_utils import speak, summarize_cmd

PY = sys.executable

def wait_health(url, timeout_s=45.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def detect_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--server_port", type=int, default=8000)
    ap.add_argument("--dash_port", type=int, default=8501)
    ap.add_argument("--server_host", default="0.0.0.0")
    ap.add_argument("--server_url", default=None, help="Must be reachable by robot/PC (e.g. http://PC_IP:8000)")

    ap.add_argument("--py27", default=r"C:\Python27\python.exe")

    ap.add_argument("--nao_ip", required=True)
    ap.add_argument("--nao_port", type=int, default=9559)

    ap.add_argument("--pscp_path", default=r"C:\Program Files\PuTTY\pscp.exe")
    ap.add_argument("--robot_user", default="nao")
    ap.add_argument("--robot_pass", default="nao")
    ap.add_argument("--local_cache", default="nao_cache")

    ap.add_argument("--trigger", choices=["face", "touch"], default="face")
    ap.add_argument("--cooldown", type=float, default=8.0)

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

    det_ip = detect_local_ip()
    server_url = args.server_url or f"http://{det_ip}:{args.server_port}"
    env = os.environ.copy()
    env["SERVER_URL"] = server_url

    procs = []

    backend_cmd = [PY, "-m", "uvicorn", "main:app", "--host", args.server_host, "--port", str(args.server_port)]
    print("[RUN] Backend:", " ".join(backend_cmd))
    say_cmd(backend_cmd)
    p_backend = subprocess.Popen(backend_cmd, env=env)
    procs.append(p_backend)

    if not wait_health(server_url + "/api/health"):
        print("[ERROR] Backend not healthy.")
        say("Backend not healthy.")
        for p in procs:
            try: p.terminate()
            except: pass
        sys.exit(1)

    say("Backend is ready.")

    dash_cmd = [PY, "-m", "streamlit", "run", "dashboard.py",
                "--server.port", str(args.dash_port),
                "--server.headless", "false"]
    print("[RUN] Dashboard:", " ".join(dash_cmd))
    say_cmd(dash_cmd)
    p_dash = subprocess.Popen(dash_cmd, env=env)
    procs.append(p_dash)

    dash_url = f"http://127.0.0.1:{args.dash_port}"
    print("[INFO] Dashboard:", dash_url)
    print("[INFO] Backend  :", server_url)
    say(f"Dashboard is available on port {args.dash_port}.")
    webbrowser.open(dash_url)

    nao_cmd = [
        args.py27, "nao_client.py",
        "--nao_ip", args.nao_ip,
        "--nao_port", str(args.nao_port),
        "--server", server_url,
        "--trigger", args.trigger,
        "--cooldown", str(args.cooldown),
        "--pscp_path", args.pscp_path,
        "--robot_user", args.robot_user,
        "--robot_pass", args.robot_pass,
        "--local_cache", args.local_cache,
    ]
    print("[RUN] NAO client:", " ".join(nao_cmd))
    say_cmd(nao_cmd)

    try:
        subprocess.call(nao_cmd, env=env)
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
