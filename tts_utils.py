# tts_utils.py
from __future__ import annotations

import os
import platform
import subprocess
import threading

_PLAY_LOCK = threading.Lock()

_ENGINE = None
_ENGINE_LOCK = threading.Lock()

def _assets_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", "prompts")

def _wav_path(key: str, lang: str) -> str:
    return os.path.join(_assets_dir(), lang, f"{key}.wav")

def _play_wav_sync(path: str) -> bool:
    try:
        if platform.system().lower().startswith("win"):
            import winsound

            snd_filename = getattr(winsound, "SND_FILENAME", 0x00020000)
            snd_sync = getattr(winsound, "SND_SYNC", 0x0000)
            winsound.PlaySound(path, snd_filename | snd_sync)
            return True

        import soundfile as sf
        import sounddevice as sd

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        sd.play(data, sr)
        sd.wait()
        return True
    except Exception as e:
        print(f"[TTS] WAV play failed: {e}")
        return False

def _speak_powershell_sync(text: str) -> bool:
    try:
        ps = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 0
$s.Volume = 100
$s.Speak($args[0])
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception as e:
        print(f"[TTS] PowerShell speak failed: {e}")
        return False

def _speak_pyttsx3_sync(text: str) -> bool:
    global _ENGINE
    try:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                import pyttsx3

                _ENGINE = pyttsx3.init()
                try:
                    _ENGINE.setProperty("volume", 1.0)
                except Exception:
                    pass
        _ENGINE.say(text)
        _ENGINE.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS] pyttsx3 failed: {e}")
        return False

def speak(
    text: str,
    key: str | None = None,
    lang: str = "en",
    enabled: bool = True,
    extra_after: str | None = None,
) -> None:

    if not enabled or not text:
        return

    def _worker():
        with _PLAY_LOCK:
            if key:
                wp = _wav_path(key, lang)
                if os.path.exists(wp):
                    if _play_wav_sync(wp):
                        if extra_after:
                            if platform.system().lower().startswith("win"):
                                if _speak_powershell_sync(extra_after):
                                    return
                            _speak_pyttsx3_sync(extra_after)
                        return

            if platform.system().lower().startswith("win"):
                if _speak_powershell_sync(text):
                    return
            _speak_pyttsx3_sync(text)

    threading.Thread(target=_worker, daemon=True).start()

def speak_sync(
    text: str,
    key: str | None = None,
    lang: str = "en",
    enabled: bool = True,
    extra_after: str | None = None,
) -> None:

    if not enabled or not text:
        return

    with _PLAY_LOCK:
        if key:
            wp = _wav_path(key, lang)
            if os.path.exists(wp):
                if _play_wav_sync(wp):
                    if extra_after:
                        if platform.system().lower().startswith("win"):
                            if _speak_powershell_sync(extra_after):
                                return
                        _speak_pyttsx3_sync(extra_after)
                    return

        if platform.system().lower().startswith("win"):
            if _speak_powershell_sync(text):
                return
        _speak_pyttsx3_sync(text)

def summarize_cmd(cmd) -> str:
    try:
        if not isinstance(cmd, (list, tuple)):
            return "Starting command."

        head = [str(x) for x in cmd[:6]]
        tail = " ..." if len(cmd) > 6 else ""

        s = " ".join(head).lower()
        if "uvicorn" in s and "main:app" in s:
            return "Starting backend server."
        if "streamlit" in s and "dashboard.py" in s:
            return "Starting dashboard."
        if "pc_client.py" in s:
            return "Starting PC client."

        return "Starting: " + " ".join(head) + tail
    except Exception:
        return "Starting command."
