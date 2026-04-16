# -*- coding: utf-8 -*-
# nao_client.py (Python 2.7)
import os
import sys
import time
import json
import base64
import argparse
import uuid
import subprocess

try:
    import urllib2
except ImportError:
    import urllib.request as urllib2

sys.path.append(r"C:\Users\DELL\Downloads\pynaoqi-python2.7-2.8.6.23-win64-vs2015-20191127_152649\lib")
from naoqi import ALProxy

try:
    basestring
except NameError:
    basestring = str

def ensure_dir(path):
    if path and (not os.path.exists(path)):
        os.makedirs(path)

def b64_local_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read())

def http_post_json(url, payload, timeout=60):
    data = json.dumps(payload)
    req = urllib2.Request(url, data, {"Content-Type": "application/json"})
    resp = urllib2.urlopen(req, timeout=timeout)
    out = resp.read()
    return json.loads(out)

def speak(tts, text):
    try:
        tts.say(text)
    except:
        pass

def _pscp_download(pscp_path, robot_user, robot_pass, robot_ip, remote_path, local_path):
    cmd = [pscp_path, "-pw", robot_pass, "%s@%s:%s" % (robot_user, robot_ip, remote_path), local_path]
    subprocess.check_call(cmd)

def newest_file_with_prefix(folder, prefix, ext=".jpg"):
    newest = None
    newest_t = -1
    for fn in os.listdir(folder):
        if fn.lower().endswith(ext) and fn.startswith(prefix):
            fp = os.path.join(folder, fn)
            t = os.path.getmtime(fp)
            if t > newest_t:
                newest_t = t
                newest = fp
    return newest

def capture_face_jpeg(robot_ip, robot_port, photo_dir, photo_prefix, resolution=2, fmt="jpg"):
    ensure_dir(photo_dir)
    photo = ALProxy("ALPhotoCapture", robot_ip, robot_port)
    photo.setResolution(resolution)
    photo.setPictureFormat(fmt)

    res = photo.takePicture(photo_dir, photo_prefix)

    remote_path = None
    try:
        if isinstance(res, list) and len(res) >= 2:
            remote_path = os.path.join(str(res[0]), str(res[1]))
        elif isinstance(res, tuple) and len(res) >= 2:
            remote_path = os.path.join(str(res[0]), str(res[1]))
        elif isinstance(res, basestring):
            remote_path = str(res)
    except Exception:
        remote_path = None

    if not remote_path:
        remote_path = os.path.join(photo_dir, photo_prefix + "." + fmt.lower())

    return remote_path

def record_audio_wav(robot_ip, robot_port, out_path, seconds=4.0, sample_rate=16000, channels=None):
    rec = ALProxy("ALAudioRecorder", robot_ip, robot_port)
    try:
        rec.stopMicrophonesRecording()
    except:
        pass

    if channels is None:
        channels = [1, 0, 0, 0]

    rec.startMicrophonesRecording(out_path, "wav", sample_rate, channels)
    time.sleep(seconds)
    rec.stopMicrophonesRecording()
    return out_path

def wait_for_touch(robot_ip, robot_port):
    mem = ALProxy("ALMemory", robot_ip, robot_port)
    keys = ["FrontTactilTouched", "MiddleTactilTouched", "RearTactilTouched"]
    while True:
        for k in keys:
            try:
                v = mem.getData(k)
                if v == 1.0 or v == 1:
                    return True
            except:
                pass
        time.sleep(0.15)

def wait_for_face(robot_ip, robot_port, stable_count=3, interval=0.2):
    face_det = ALProxy("ALFaceDetection", robot_ip, robot_port)
    mem = ALProxy("ALMemory", robot_ip, robot_port)
    try:
        face_det.subscribe("nao_face_trigger", 500, 0.0)
    except:
        pass

    consecutive = 0
    while True:
        try:
            data = mem.getData("FaceDetected")
        except:
            data = None

        if data and isinstance(data, list) and len(data) > 0:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= stable_count:
            return True
        time.sleep(interval)

def run_one_cycle(args, tts):
    ensure_dir(args.local_cache)

    img_remote = capture_face_jpeg(
        args.nao_ip, args.nao_port,
        photo_dir=args.photo_dir,
        photo_prefix=args.photo_prefix,
        resolution=args.photo_resolution,
        fmt=args.photo_format
    )

    verify_remote = record_audio_wav(
        args.nao_ip, args.nao_port,
        out_path=args.verify_audio_path,
        seconds=args.verify_seconds,
        sample_rate=args.sr
    )

    img_local = os.path.join(args.local_cache, "capture_%s.jpg" % uuid.uuid4().hex)
    wav_local = os.path.join(args.local_cache, "verify_%s.wav" % uuid.uuid4().hex)

    _pscp_download(args.pscp_path, args.robot_user, args.robot_pass, args.nao_ip, img_remote, img_local)
    _pscp_download(args.pscp_path, args.robot_user, args.robot_pass, args.nao_ip, verify_remote, wav_local)

    payload = {"image": b64_local_file(img_local), "audio": b64_local_file(wav_local), "top_k": 1}
    res = http_post_json(args.server.rstrip("/") + "/api/verify_fusion", payload, timeout=args.timeout)

    decision = (res.get("decision") or "REJECT").upper()
    final = res.get("final") or {}
    final_name = final.get("name") or ""

    if decision == "ACCEPT" and final_name:
        speak(tts, "Welcome %s. Access granted. Please proceed." % final_name)
        return "ACCEPT"

    speak(tts, "Access denied. I could not recognize you.")
    speak(tts, "Please say your full name for administrator approval.")

    name_remote = record_audio_wav(
        args.nao_ip, args.nao_port,
        out_path=args.name_audio_path,
        seconds=args.name_seconds,
        sample_rate=args.sr
    )

    name_local = os.path.join(args.local_cache, "name_%s.wav" % uuid.uuid4().hex)
    _pscp_download(args.pscp_path, args.robot_user, args.robot_pass, args.nao_ip, name_remote, name_local)

    reg_payload = {
        "teacher_id": None,
        "name": "UNKNOWN",
        "image": b64_local_file(img_local),
        "audio": b64_local_file(wav_local),
        "name_audio": b64_local_file(name_local),
        "robot_captured": True,
        "pending_approval": True
    }

    http_post_json(args.server.rstrip("/") + "/api/register_teacher", reg_payload, timeout=args.timeout)
    speak(tts, "Thank you. Your request has been sent to the administrator for approval.")
    return "REJECT"

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nao_ip", required=True)
    ap.add_argument("--nao_port", type=int, default=9559)

    ap.add_argument("--server", required=True)
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument("--trigger", choices=["face", "touch"], default="face")
    ap.add_argument("--face_stable", type=int, default=3)
    ap.add_argument("--face_interval", type=float, default=0.2)
    ap.add_argument("--cooldown", type=float, default=8.0)

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--verify_seconds", type=float, default=4.0)
    ap.add_argument("--name_seconds", type=float, default=3.0)

    ap.add_argument("--verify_audio_path", default="/home/nao/recordings/verify.wav")
    ap.add_argument("--name_audio_path", default="/home/nao/recordings/name.wav")
    ap.add_argument("--photo_dir", default="/home/nao/recordings")
    ap.add_argument("--photo_prefix", default="capture")
    ap.add_argument("--photo_resolution", type=int, default=2)
    ap.add_argument("--photo_format", default="jpg")

    ap.add_argument("--pscp_path", default=r"C:\Program Files\PuTTY\pscp.exe")
    ap.add_argument("--robot_user", default="nao")
    ap.add_argument("--robot_pass", default="nao")
    ap.add_argument("--local_cache", default="nao_cache")

    args = ap.parse_args()

    tts = ALProxy("ALTextToSpeech", args.nao_ip, args.nao_port)
    speak(tts, "System is ready.")

    while True:
        if args.trigger == "touch":
            speak(tts, "Touch my head to start verification.")
            wait_for_touch(args.nao_ip, args.nao_port)
        else:
            speak(tts, "Please stand in front of me for verification.")
            wait_for_face(args.nao_ip, args.nao_port,
                          stable_count=args.face_stable,
                          interval=args.face_interval)

        try:
            run_one_cycle(args, tts)
        except Exception:
            speak(tts, "An error occurred. I will try again.")

        time.sleep(args.cooldown)

if __name__ == "__main__":
    main()