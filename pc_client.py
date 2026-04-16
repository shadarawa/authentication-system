# pc_client.py
import argparse
import base64
import json
import os
import io
import wave
import re

from tts_utils import speak as _tts_speak, speak_sync as _tts_speak_sync

_SPEAK_ENABLED = True
_UI_LANG = "en"

_UI = {
    "en": {
        "ready": "System is ready. Please stand in front of the camera.",
        "face_detected": "Face detected. Please stay still and look at the camera.",
        "starting_verification": "Starting verification now.",
        "recording_voice": "Recording your voice now. Please speak clearly.",
        "processing": "Processing. Please wait.",
        "verify_failed": "Network or server error. Please try again.",
        "welcome_granted": "Welcome {name}. Access granted. Please proceed.",
        "access_denied": "Access denied.",
        "not_recognized_say_name": "I could not recognize you. Please say your full name for administrator approval.",
        "not_recognized_type_name": "I could not recognize you. Please type your full name for administrator approval.",
        "say_full_name_now": "Please say your full name now.",
        "could_not_understand_type": "Sorry, I could not understand. Please type your full name.",
        "request_sent": "Thank you. Your request has been sent to the administrator for approval.",
        "cooldown": "Next person, please.",
        "please_speak_louder": "I could not hear you well. Please speak louder and try again.",
    },
    "ar": {
        "ready": "النظام جاهز. تفضل بالوقوف أمام الكاميرا.",
        "face_detected": "تم اكتشاف وجه. من فضلك ابقَ ثابتاً وانظر إلى الكاميرا.",
        "starting_verification": "سأبدأ التحقق الآن.",
        "recording_voice": "سأسجل صوتك الآن. تكلّم بوضوح.",
        "processing": "جاري المعالجة. انتظر من فضلك.",
        "verify_failed": "حدث خطأ في الاتصال أو في الخادم. حاول مرة أخرى.",
        "welcome_granted": "أهلاً {name}. تم السماح بالدخول. تفضل.",
        "access_denied": "تم رفض الدخول.",
        "not_recognized_say_name": "لم أتعرف عليك. من فضلك قل اسمك الكامل للموافقة من الإدارة.",
        "not_recognized_type_name": "لم أتعرف عليك. من فضلك اكتب اسمك الكامل للموافقة من الإدارة.",
        "say_full_name_now": "من فضلك قل اسمك الكامل الآن.",
        "could_not_understand_type": "عذراً، لم أفهم. من فضلك اكتب اسمك الكامل.",
        "request_sent": "شكراً لك. تم إرسال طلبك إلى الإدارة للموافقة.",
        "cooldown": "يمكن للشخص التالي التقدم.",
        "please_speak_louder": "لم أسمع صوتك جيداً. من فضلك ارفع صوتك وحاول مرة أخرى.",
    },
}

def speak_key(key: str, blocking: bool = False, **kwargs):
    text = ui(key, **kwargs)
    extra = None
    if key == "welcome_granted" and "name" in kwargs and kwargs["name"]:
        extra = str(kwargs["name"])
    print(f"[TTS] key={key} lang={_UI_LANG} enabled={_SPEAK_ENABLED} blocking={blocking}")
    fn = _tts_speak_sync if blocking else _tts_speak
    fn(text, key=key, lang=_UI_LANG, enabled=_SPEAK_ENABLED, extra_after=extra)

def set_speak_enabled(enabled: bool) -> None:
    global _SPEAK_ENABLED
    _SPEAK_ENABLED = bool(enabled)

def set_ui_lang(lang: str) -> None:
    global _UI_LANG
    _UI_LANG = lang if lang in _UI else "en"

def ui(key: str, **kwargs) -> str:
    msg = _UI.get(_UI_LANG, _UI["en"]).get(key, key)
    try:
        return msg.format(**kwargs)
    except Exception:
        return msg

import time
from pathlib import Path
import uuid

import numpy as np
import requests
import cv2

import sounddevice as sd
import soundfile as sf

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    _VOSK_OK = True
except Exception:
    VoskModel = None
    KaldiRecognizer = None
    _VOSK_OK = False

def stt_name_vosk(wav_path: str, model_dir: str) -> str:
    try:
        from vosk import Model, KaldiRecognizer
        import json
        import wave

        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000, 44100):
            pass

        model = Model(model_dir)
        rec = KaldiRecognizer(model, wf.getframerate())

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        final = json.loads(rec.FinalResult())
        text = (final.get("text") or "").strip()
        return " ".join(text.split())
    except Exception:
        return ""

def _clean_name(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[^A-Za-z\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = " ".join([w.capitalize() for w in text.split()])
    return text

def _transcribe_wav_bytes_vosk(wav_bytes: bytes, model_path: str) -> str:
    if not _VOSK_OK:
        return ""
    model_path = str(model_path)
    if not os.path.isdir(model_path):
        return ""
    try:
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
        sr = wf.getframerate()
        if wf.getnchannels() != 1:
            return ""
        if wf.getsampwidth() != 2:
            return ""
        rec = KaldiRecognizer(VoskModel(model_path), sr)
        rec.SetWords(False)
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
        out = json.loads(rec.FinalResult() or "{}")
        return (out.get("text") or "").strip()
    except Exception:
        return ""

def capture_name(args) -> str:
    if not getattr(args, "stt_name", False):
        return ""

    prompt_delay = float(getattr(args, "name_prompt_delay", 0.8))
    retries = int(getattr(args, "stt_retries", 2))
    min_words = int(getattr(args, "stt_min_words", 2))

    for attempt in range(1, retries + 1):
        speak_key("say_full_name_now", blocking=True)
        time.sleep(prompt_delay)

        name_wav, _rms = record_audio_wav_bytes(getattr(args, "name_seconds", 6.0), sr=args.sr)
        text = _transcribe_wav_bytes_vosk(name_wav, getattr(args, "vosk_model", ""))
        text = _clean_name(text)

        if text and len(text.split()) >= min_words:
            print(f"[STT] Name recognized: {text}")
            return text

        print(f"[STT] Attempt {attempt}/{retries} failed (empty/too short).")

    print("[STT] Could not recognize the name after retries.")
    return ""

def b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def record_audio_wav_bytes(seconds: float = 4.0, sr: int = 16000) -> tuple[bytes, float]:
    n_samples = int(seconds * sr)
    print(f"[AUDIO] Recording {seconds:.1f}s @ {sr}Hz (mono)...")
    audio = sd.rec(n_samples, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio, axis=1)

    rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)

    tmp = Path(f"tmp_{uuid.uuid4().hex}.wav")
    sf.write(str(tmp), audio, sr, subtype="PCM_16")
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return data, rms

def encode_frame_jpeg(frame_bgr, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()

def detect_face(frame_bgr, face_cascade, scale=1.1, neighbors=5, min_size=(80, 80)):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return (x, y, w, h)

def gen_teacher_id():
    return "TNEW_%d" % int(time.time())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="AI server base URL")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--width", type=int, default=640, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")

    ap.add_argument("--seconds", type=float, default=6.0, help="Audio record seconds (4-6 recommended)")
    ap.add_argument("--sr", type=int, default=16000, help="Audio sample rate")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    ap.add_argument("--record_start_delay", type=float, default=0.8,
                    help="Delay AFTER the spoken prompt before recording starts (seconds)")
    ap.add_argument("--voice_retries", type=int, default=2,
                    help="Re-record if the audio is too quiet")
    ap.add_argument("--min_audio_rms", type=float, default=0.006,
                    help="Minimum RMS loudness for a recording to be accepted")

    ap.add_argument("--stable_frames", type=int, default=8,
                    help="How many consecutive frames with a face to trigger (stability)")
    ap.add_argument("--cooldown", type=float, default=6.0,
                    help="Seconds to wait after each verification attempt")
    ap.add_argument("--min_face", type=int, default=90,
                    help="Minimum face box size (pixels) to accept detection")

    ap.add_argument("--allow_retrigger_while_face_present", action="store_true",
                    help="If set, can trigger again without the person leaving the frame (not recommended)")
    ap.add_argument("--rearm_no_face_frames", type=int, default=20,
                    help="How many consecutive NO-FACE frames to re-arm after an attempt")

    ap.add_argument("--register_on_reject", action="store_true",
                    help="Call /api/register_teacher on REJECT")
    ap.add_argument("--name", default="", help="Name to use for registration (optional)")

    ap.add_argument("--image_path", default="", help="Use existing image instead of webcam")
    ap.add_argument("--audio_path", default="", help="Use existing wav instead of microphone")

    ap.add_argument("--stt_name", action="store_true", help="Use Speech-to-Text to capture name on reject")
    ap.add_argument("--vosk_model", default="vosk-model-small-en-us-0.15", help="Path to Vosk model folder")
    ap.add_argument("--name_seconds", type=float, default=3.0, help="Seconds to record name for STT")

    ap.add_argument("--name_prompt_delay", type=float, default=0.8, help="Delay before recording name (seconds)")
    ap.add_argument("--stt_retries", type=int, default=2, help="How many STT attempts before typing fallback")
    ap.add_argument("--stt_min_words", type=int, default=2, help="Minimum words required to accept STT result")

    ap.add_argument("--ui_lang", choices=["en", "ar"], default="en",
                    help="Spoken prompt language (for laptop user prompts)")
    ap.add_argument("--mute", action="store_true",
                    help="Disable spoken prompts")

    args = ap.parse_args()

    set_ui_lang(getattr(args, "ui_lang", "en"))
    set_speak_enabled(not getattr(args, "mute", False))

    if args.image_path and args.audio_path:

        img_b = Path(args.image_path).read_bytes()
        aud_b = Path(args.audio_path).read_bytes()

        payload = {
            "image": base64.b64encode(img_b).decode("utf-8"),
            "audio": base64.b64encode(aud_b).decode("utf-8"),
            "top_k": 1
        }

        url = args.server.rstrip("/") + "/api/verify_fusion"
        r = requests.post(url, json=payload, timeout=args.timeout)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))
        raise SystemExit(0)

    verify_url = args.server.rstrip("/") + "/api/verify_fusion"
    register_url = args.server.rstrip("/") + "/api/register_teacher"

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade. Check OpenCV installation.")

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.cam}")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("[INFO] Face-trigger loop started.")
    print(" - When a stable face is detected, it will capture + record + send.\n")

    speak_key("ready")

    stable = 0
    last_sent = 0.0
    face_prompted = False
    waiting_clear = False
    no_face_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            face = detect_face(frame, face_cascade, min_size=(args.min_face, args.min_face))
            if face is not None:
                stable += 1
                if stable == 1 and not face_prompted:
                    speak_key("face_detected")
                    face_prompted = True
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face detected: {stable}/{args.stable_frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                stable = 0
                face_prompted = False
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("PC Client (Face Trigger) - Press q to quit", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            if waiting_clear and not args.allow_retrigger_while_face_present:
                if face is None:
                    no_face_count += 1
                else:
                    no_face_count = 0

                if no_face_count >= args.rearm_no_face_frames:
                    waiting_clear = False
                    no_face_count = 0
                    stable = 0
                    face_prompted = False
                    speak_key("ready")
                continue

            now = time.time()
            if now - last_sent < args.cooldown:
                continue

            if stable >= args.stable_frames:
                stable = 0
                last_sent = now

                print("[TRIGGER] Stable face detected -> capturing + recording + sending...")
                speak_key("starting_verification")

                img_bytes = encode_frame_jpeg(frame)

                speak_key("recording_voice", blocking=True)
                if args.record_start_delay > 0:
                    time.sleep(float(args.record_start_delay))

                aud_bytes = None
                last_rms = 0.0
                for _ in range(max(1, int(args.voice_retries))):
                    aud_bytes, last_rms = record_audio_wav_bytes(args.seconds, sr=args.sr)
                    if last_rms >= float(args.min_audio_rms):
                        break
                    speak_key("please_speak_louder", blocking=True)
                    if args.record_start_delay > 0:
                        time.sleep(float(args.record_start_delay))

                if aud_bytes is None or last_rms < float(args.min_audio_rms):
                    print(f"[AUDIO] Too quiet (rms={last_rms:.6f}). Skipping this attempt.")
                    last_sent = time.time() - float(args.cooldown) - 1.0
                    continue
                speak_key("processing")

                payload = {
                    "image": b64_bytes(img_bytes),
                    "audio": b64_bytes(aud_bytes),
                    "top_k": 1
                }

                try:
                    res = post_json(verify_url, payload, timeout=args.timeout)
                except Exception as e:
                    print(f"[ERROR] verify_fusion failed: {e}")
                    speak_key("verify_failed")
                    continue

                print("\n---- VERIFY RESULT ----")
                print(json.dumps(res, indent=2))

                decision = (res.get("decision") or "").upper()
                if decision == "ACCEPT":
                    final = res.get("final") or {}
                    name = final.get("name") or "teacher"
                    speak_key("welcome_granted", name=name)
                else:
                    speak_key("access_denied")
                    if args.register_on_reject:
                        if args.stt_name:
                            speak_key("not_recognized_say_name")
                            name = (args.name or capture_name(args) or "").strip()
                            if not name:
                                speak_key("could_not_understand_type")
                                name = (args.name or input("\n[REGISTER] Enter full name: ").strip())
                        else:
                            speak_key("not_recognized_type_name")
                            name = (args.name or input("\n[REGISTER] Enter full name: ").strip())
                        name = (name or "UNKNOWN")
                        reg_payload = {
                            "teacher_id": None,
                            "name": name,
                            "image": b64_bytes(img_bytes),
                            "audio": b64_bytes(aud_bytes),
                            "robot_captured": False,
                            "pending_approval": True
                        }
                        try:
                            reg_res = post_json(register_url, reg_payload, timeout=args.timeout)
                            print("\n---- REGISTER RESULT ----")
                            print(json.dumps(reg_res, indent=2))
                            speak_key("request_sent")
                        except Exception as e:
                            print(f"[ERROR] register_teacher failed: {e}")

                waiting_clear = not args.allow_retrigger_while_face_present
                no_face_count = 0

                speak_key("cooldown")
                print("\n[INFO] Cooldown...\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
