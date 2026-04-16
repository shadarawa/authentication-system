# main.py
import base64
import os
import tempfile
import uuid
import re
import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import torchaudio

from face_model_insightface import InsightFaceModel
from voice_model import ECAPATDNNModel
from db_utils import load_db, save_db, index_teachers, ensure_teacher, finalize_db
from verify_fusion import cosine_sim as cos_sim, fusion_decision

from server_utils import read_json, write_json_atomic, append_jsonl, FileLock, now_iso, gen_request_id

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = str(APP_DIR / "config.json")

app = FastAPI(title="NAO Robot AI Backend Server", version="1.0")

def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    cfg = read_json(path, default={})
    # minimal defaults
    cfg.setdefault("db_path", "db/teachers.json")
    cfg.setdefault("pending_path", "db/pending.json")
    cfg.setdefault("pending_samples_dir", "pending_samples")
    cfg.setdefault("teachers_csv", "Teachers.csv")

    cfg.setdefault("logs_path", "logs/attempts.jsonl")
    cfg.setdefault("thresholds", {})
    cfg["thresholds"].setdefault("face", 0.46)
    cfg["thresholds"].setdefault("voice", 0.40)
    cfg["thresholds"].setdefault("alpha", 0.6)
    cfg["thresholds"].setdefault("require_both", True)
    cfg["thresholds"].setdefault("high_confidence_margin", 0.03)

    cfg.setdefault("security", {})
    cfg["security"].setdefault("max_audio_seconds", 8.0)
    cfg["security"].setdefault("max_image_bytes", 4_000_000)
    cfg["security"].setdefault("max_audio_bytes", 8_000_000)

    cfg.setdefault("incremental", {})
    cfg["incremental"].setdefault("enabled", True)
    cfg["incremental"].setdefault("min_fused_for_update", 0.90)
    cfg["incremental"].setdefault("min_face_for_update", 0.90)
    cfg["incremental"].setdefault("min_voice_for_update", 0.90)
    cfg["incremental"].setdefault("max_face_embeddings", 30)
    cfg["incremental"].setdefault("max_voice_embeddings", 30)
    cfg["incremental"].setdefault("save_samples", True)
    cfg["incremental"].setdefault("samples_dir", "samples")

    cfg.setdefault("admin", {})
    cfg["admin"].setdefault("allow_pending_verify", False)

    cfg.setdefault("models", {})
    cfg["models"].setdefault("face", {})
    cfg["models"]["face"].setdefault("model_pack", "buffalo_l")
    cfg["models"]["face"].setdefault("use_gpu", False)
    cfg["models"]["face"].setdefault("det_thresh", 0.5)
    cfg["models"]["face"].setdefault("det_size", [640, 640])

    cfg["models"].setdefault("voice", {})
    cfg["models"]["voice"].setdefault("device", "cpu")
    cfg["models"]["voice"].setdefault("ffmpeg", "ffmpeg")

    cfg.setdefault("email", {})
    cfg["email"].setdefault("enabled", False)
    return cfg


CFG = load_config()

DB_PATH = str(APP_DIR / CFG["db_path"])
LOGS_PATH = str(APP_DIR / CFG["logs_path"])
SAMPLES_DIR = str(APP_DIR / CFG["incremental"]["samples_dir"])

print("Loading AI Models...")
face_cfg = CFG["models"]["face"]
voice_cfg = CFG["models"]["voice"]

face_model = InsightFaceModel(
    model_pack=face_cfg.get("model_pack", "buffalo_l"),
    det_thresh=float(face_cfg.get("det_thresh", 0.5)),
    det_size=tuple(face_cfg.get("det_size", [640, 640])),
    use_gpu=bool(face_cfg.get("use_gpu", False)),
)

voice_model = ECAPATDNNModel(device=voice_cfg.get("device", "cpu"), ffmpeg_path=voice_cfg.get("ffmpeg", "ffmpeg"), max_seconds=float(CFG["security"]["max_audio_seconds"]))
print("Models Loaded Successfully.")

class VerifyPayload(BaseModel):
    image: Optional[str] = None
    audio: Optional[str] = None
    top_k: int = 1
    source: str = "pc"

    face_thresh: Optional[float] = None
    voice_thresh: Optional[float] = None
    alpha: Optional[float] = None
    require_both: Optional[bool] = None

class RegisterPayload(BaseModel):
    teacher_id: Optional[str] = None
    name: str = Field(..., min_length=1)
    image: Optional[str] = None
    audio: Optional[str] = None
    robot_captured: bool = True
    pending_approval: bool = True
    name_audio: Optional[str] = None

class ApprovePayload(BaseModel):
    teacher_id: str
    approved: bool = True

_db_cache: Optional[Dict[str, Any]] = None
_db_mtime: float = -1.0

def _load_db_cached() -> Dict[str, Any]:
    global _db_cache, _db_mtime
    p = Path(DB_PATH)
    mtime = p.stat().st_mtime if p.exists() else -1.0
    if _db_cache is None or mtime != _db_mtime:
        _db_cache = load_db(DB_PATH)
        _db_mtime = mtime
    return _db_cache

def _teachers_list(allow_pending: bool = False) -> list:
    db = _load_db_cached()
    teachers = db.get("teachers", [])
    if allow_pending:
        return teachers
    out = []
    for t in teachers:
        if not isinstance(t, dict):
            continue
        pending = bool(t.get("meta", {}).get("pending_approval", False))
        if pending and not CFG["admin"].get("allow_pending_verify", False):
            continue
        out.append(t)
    return out

PENDING_PATH = str(APP_DIR / str(CFG.get("pending_path", "db/pending.json")))
PENDING_SAMPLES_DIR = str(APP_DIR / str(CFG.get("pending_samples_dir", "pending_samples")))
TEACHERS_CSV_PATH = str(APP_DIR / str(CFG.get("teachers_csv", "Teachers.csv")))

_pending_cache = None
_pending_mtime = -1.0

def _load_pending_cached() -> Dict[str, Any]:
    global _pending_cache, _pending_mtime
    p = Path(PENDING_PATH)
    mtime = p.stat().st_mtime if p.exists() else -1.0
    if _pending_cache is None or mtime != _pending_mtime:
        _pending_cache = read_json(PENDING_PATH, default={"version": 1, "pending": []})
        _pending_mtime = mtime
    if not isinstance(_pending_cache, dict):
        _pending_cache = {"version": 1, "pending": []}
    _pending_cache.setdefault("pending", [])
    return _pending_cache

def _write_pending(pending_db: Dict[str, Any]) -> None:
    pending_db.setdefault("version", 1)
    pending_db.setdefault("pending", [])
    write_json_atomic(PENDING_PATH, pending_db)
    # reset cache
    global _pending_cache, _pending_mtime
    _pending_cache = pending_db
    _pending_mtime = Path(PENDING_PATH).stat().st_mtime if Path(PENDING_PATH).exists() else -1.0

def _pending_list() -> list:
    return list(_load_pending_cached().get("pending", []) or [])

def _get_pending(pending_id: str) -> Optional[Dict[str, Any]]:
    for item in _pending_list():
        if isinstance(item, dict) and item.get("pending_id") == pending_id:
            return item
    return None

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _save_pending_files(pending_id: str, img_bytes: Optional[bytes], aud_bytes: Optional[bytes], name_bytes: Optional[bytes] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    _ensure_dir(PENDING_SAMPLES_DIR)
    item_dir = Path(PENDING_SAMPLES_DIR) / pending_id
    item_dir.mkdir(parents=True, exist_ok=True)
    img_path = None
    aud_path = None
    name_path = None
    if img_bytes:
        img_path = str(item_dir / "image.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
    if aud_bytes:
        aud_path = str(item_dir / "audio.wav")
        with open(aud_path, "wb") as f:
            f.write(aud_bytes)
    if name_bytes:
        name_path = str(item_dir / "name.wav")
        with open(name_path, "wb") as f:
            f.write(name_bytes)
    return img_path, aud_path, name_path

def _delete_pending_files(pending_id: str) -> None:
    item_dir = Path(PENDING_SAMPLES_DIR) / pending_id
    if item_dir.exists() and item_dir.is_dir():
        for p in item_dir.rglob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            item_dir.rmdir()
        except Exception:
            pass

def _parse_teacher_num(tid: str) -> Optional[int]:
    m = re.match(r"^T(\d+)$", str(tid).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _max_teacher_num_from_csv(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    mx = 0
    try:
        import csv
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                tid = (row.get("id") or "").strip()
                n = _parse_teacher_num(tid)
                if n is not None:
                    mx = max(mx, n)
    except Exception:
        pass
    return mx

def _max_teacher_num_from_db() -> int:
    mx = 0
    for t in _teachers_list(allow_pending=True):
        n = _parse_teacher_num(t.get("id"))
        if n is not None:
            mx = max(mx, n)
    return mx

def allocate_next_teacher_id() -> str:
    mx = max(_max_teacher_num_from_csv(TEACHERS_CSV_PATH), _max_teacher_num_from_db())
    return f"T{mx+1:04d}"

def append_to_teachers_csv(teacher_id: str, name: str) -> None:
    import csv
    p = Path(TEACHERS_CSV_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    if not exists:
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "name"])
            w.writeheader()
            w.writerow({"id": teacher_id, "name": name})
        return
    rows = []
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    except Exception:
        rows = []
    for row in rows:
        if (row.get("id") or "").strip() == teacher_id:
            return
    with p.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        if not exists:
            w.writeheader()
        w.writerow({"id": teacher_id, "name": name})

def _b64_to_bytes(s: str, max_bytes: int) -> bytes:
    try:
        raw = base64.b64decode(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 payload")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Payload too large")
    return raw

def _write_temp_file(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="tmp_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path

def _trim_wav_inplace(wav_path: str, max_seconds: float = 8.0) -> None:
    try:
        wav, sr = torchaudio.load(wav_path)
        max_len = int(max_seconds * sr)
        if wav.shape[-1] > max_len:
            wav = wav[..., :max_len]
            torchaudio.save(wav_path, wav, sr)
    except Exception:
        pass

def _best_match(query_emb: np.ndarray, teachers: list, key: str) -> Tuple[Optional[str], Optional[str], float]:
    best_id, best_name, best_score = None, None, -1.0
    for t in teachers:
        tid = t.get("id")
        name = t.get("name")
        for e in t.get(key, []) or []:
            score = cos_sim(query_emb, np.asarray(e, dtype=np.float32))
            if score > best_score:
                best_score = score
                best_id = tid
                best_name = name
    return best_id, best_name, float(best_score)

def _save_samples(teacher_id: str, img_bytes: Optional[bytes], aud_bytes: Optional[bytes]) -> None:
    if not CFG["incremental"].get("save_samples", True):
        return
    root = Path(APP_DIR) / SAMPLES_DIR / teacher_id
    if img_bytes:
        (root / "images").mkdir(parents=True, exist_ok=True)
        fn = root / "images" / (now_iso().replace(":", "-") + ".jpg")
        fn.write_bytes(img_bytes)
    if aud_bytes:
        (root / "audio").mkdir(parents=True, exist_ok=True)
        fn = root / "audio" / (now_iso().replace(":", "-") + ".wav")
        fn.write_bytes(aud_bytes)

def _append_attempt_log(obj: Dict[str, Any]) -> None:
    obj.setdefault("ts", now_iso())
    append_jsonl(LOGS_PATH, obj)

def _incremental_update(final_id: str, face_emb: Optional[np.ndarray], voice_emb: Optional[np.ndarray],
                        face_score: float, voice_score: float, fused_score: float,
                        img_bytes: Optional[bytes], aud_bytes: Optional[bytes]) -> None:
    inc = CFG.get("incremental", {})
    if not inc.get("enabled", True):
        return
    if fused_score < float(inc.get("min_fused_for_update", 0.90)):
        return

    do_face = face_emb is not None and face_score >= float(inc.get("min_face_for_update", 0.90))
    do_voice = voice_emb is not None and voice_score >= float(inc.get("min_voice_for_update", 0.90))
    if not (do_face or do_voice):
        return

    with FileLock(DB_PATH, timeout=5.0):
        db = load_db(DB_PATH)
        idx = index_teachers(db)
        t = idx.get(final_id)
        if not t:
            return
        pending = bool(t.get("meta", {}).get("pending_approval", False))
        if pending and not CFG["admin"].get("allow_pending_verify", False):
            return

        if do_face:
            t.setdefault("face_embeddings", [])
            t["face_embeddings"].append(face_emb.astype(np.float32).tolist())
            maxn = int(inc.get("max_face_embeddings", 30))
            if len(t["face_embeddings"]) > maxn:
                t["face_embeddings"] = t["face_embeddings"][-maxn:]

        if do_voice:
            t.setdefault("voice_embeddings", [])
            t["voice_embeddings"].append(voice_emb.astype(np.float32).tolist())
            maxn = int(inc.get("max_voice_embeddings", 30))
            if len(t["voice_embeddings"]) > maxn:
                t["voice_embeddings"] = t["voice_embeddings"][-maxn:]

        t.setdefault("meta", {})
        t["meta"]["num_images_used"] = len(t.get("face_embeddings", []))
        t["meta"]["num_audios_used"] = len(t.get("voice_embeddings", []))
        t["meta"]["last_updated"] = now_iso()

        idx[final_id] = t
        db = finalize_db(db, idx)
        write_json_atomic(DB_PATH, db)

    _save_samples(final_id, img_bytes if do_face else None, aud_bytes if do_voice else None)


def _send_pending_email(pending_id: str, name: str, img_path: Optional[str], aud_path: Optional[str]) -> None:
    cfg = CFG.get("email", {}) or {}
    if not cfg.get("enabled", False):
        return

    to_list = cfg.get("to") or []
    if isinstance(to_list, str):
        to_list = [to_list]
    to_list = [t for t in to_list if str(t).strip()]
    if not to_list:
        return

    smtp_host = cfg.get("smtp_host")
    smtp_port = int(cfg.get("smtp_port", 587))
    smtp_user = cfg.get("smtp_user") or ""
    smtp_pass = cfg.get("smtp_password") or ""
    sender = cfg.get("from") or smtp_user

    if not sender or not smtp_host:
        return

    subj = f"{cfg.get('subject_prefix','[NAO AUTH]')} Pending approval: {name} ({pending_id})"
    body = f"A new teacher registration is pending admin approval.\n\nPending ID: {pending_id}\nName: {name}\n\nOpen the dashboard and approve/reject in Pending Approvals tab."
    msg = EmailMessage()
    msg["Subject"] = subj
    msg["From"] = sender
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    if img_path and os.path.exists(img_path):
        ctype, encoding = mimetypes.guess_type(img_path)
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(img_path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(img_path))

    if cfg.get("attach_audio", True) and aud_path and os.path.exists(aud_path):
        ctype, encoding = mimetypes.guess_type(aud_path)
        if ctype is None:
            ctype = "audio/wav"
        maintype, subtype = ctype.split("/", 1)
        with open(aud_path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(aud_path))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
            if cfg.get("use_tls", True):
                s.starttls()
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    except Exception as e:
        print(f"[EMAIL] Failed to send pending notification: {e}")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "db_path": CFG["db_path"],
        "teachers": len(_load_db_cached().get("teachers", [])),
        "ts": now_iso(),
        "version": app.version,
    }

@app.post("/api/verify_face")
async def verify_face(payload: VerifyPayload, request: Request):
    if not payload.image:
        raise HTTPException(status_code=400, detail="No image provided")

    rid = gen_request_id()
    img_bytes = _b64_to_bytes(payload.image, int(CFG["security"]["max_image_bytes"]))
    tmp_img = _write_temp_file(".jpg", img_bytes)

    try:
        teachers = _teachers_list()
        face_emb = face_model.embed_image(tmp_img)
        best_id, best_name, best_score = _best_match(face_emb, teachers, "face_embeddings")

        face_thresh = float(payload.face_thresh if payload.face_thresh is not None else CFG["thresholds"]["face"])
        recognized = (best_id is not None) and (best_score >= face_thresh)

        res = {
            "request_id": rid,
            "recognized": recognized,
            "best_id": best_id,
            "name": best_name if best_name else "Unknown",
            "score": float(best_score),
            "threshold": face_thresh,
        }

        _append_attempt_log({
            "request_id": rid,
            "type": "face",
            "source": payload.source,
            "ip": request.client.host if request.client else None,
            "recognized": recognized,
            "best_id": best_id,
            "score": float(best_score),
            "threshold": face_thresh,
        })

        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_img)
        except Exception:
            pass

@app.post("/api/verify_voice")
async def verify_voice(payload: VerifyPayload, request: Request):
    if not payload.audio:
        raise HTTPException(status_code=400, detail="No audio provided")

    rid = gen_request_id()
    aud_bytes = _b64_to_bytes(payload.audio, int(CFG["security"]["max_audio_bytes"]))
    tmp_wav = _write_temp_file(".wav", aud_bytes)

    try:
        _trim_wav_inplace(tmp_wav, float(CFG["security"]["max_audio_seconds"]))
        teachers = _teachers_list()
        voice_emb = voice_model.embed_file(tmp_wav)
        best_id, best_name, best_score = _best_match(voice_emb, teachers, "voice_embeddings")

        voice_thresh = float(payload.voice_thresh if payload.voice_thresh is not None else CFG["thresholds"]["voice"])
        recognized = (best_id is not None) and (best_score >= voice_thresh)

        res = {
            "request_id": rid,
            "recognized": recognized,
            "best_id": best_id,
            "name": best_name if best_name else "Unknown",
            "score": float(best_score),
            "threshold": voice_thresh,
        }

        _append_attempt_log({
            "request_id": rid,
            "type": "voice",
            "source": payload.source,
            "ip": request.client.host if request.client else None,
            "recognized": recognized,
            "best_id": best_id,
            "score": float(best_score),
            "threshold": voice_thresh,
        })

        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

@app.post("/api/verify_fusion")
async def verify_fusion(payload: VerifyPayload, request: Request):
    if not payload.image and not payload.audio:
        raise HTTPException(status_code=400, detail="Need image and/or audio")

    rid = gen_request_id()
    img_bytes = None
    aud_bytes = None
    tmp_img = None
    tmp_wav = None

    try:
        if payload.image:
            img_bytes = _b64_to_bytes(payload.image, int(CFG["security"]["max_image_bytes"]))
            tmp_img = _write_temp_file(".jpg", img_bytes)
        if payload.audio:
            aud_bytes = _b64_to_bytes(payload.audio, int(CFG["security"]["max_audio_bytes"]))
            tmp_wav = _write_temp_file(".wav", aud_bytes)
            _trim_wav_inplace(tmp_wav, float(CFG["security"]["max_audio_seconds"]))

        teachers = _teachers_list()

        face_thresh = float(payload.face_thresh if payload.face_thresh is not None else CFG["thresholds"]["face"])
        voice_thresh = float(payload.voice_thresh if payload.voice_thresh is not None else CFG["thresholds"]["voice"])
        alpha = float(payload.alpha if payload.alpha is not None else CFG["thresholds"]["alpha"])
        require_both = bool(payload.require_both if payload.require_both is not None else CFG["thresholds"]["require_both"])

        face_id = face_name = None
        voice_id = voice_name = None
        face_score = 0.0
        voice_score = 0.0
        face_emb = None
        voice_emb = None

        if tmp_img:
            face_emb = face_model.embed_image(tmp_img)
            face_id, face_name, face_score = _best_match(face_emb, teachers, "face_embeddings")

        if tmp_wav:
            voice_emb = voice_model.embed_file(tmp_wav)
            voice_id, voice_name, voice_score = _best_match(voice_emb, teachers, "voice_embeddings")

        accepted, fused_score, reason = fusion_decision(
            face_id, face_score,
            voice_id, voice_score,
            face_thresh=face_thresh,
            voice_thresh=voice_thresh,
            alpha=alpha,
            require_both=require_both
        )

        final_id = None
        final_name = None
        if face_id and voice_id and face_id == voice_id:
            final_id = face_id
            final_name = face_name or voice_name
        elif face_id and (face_score >= voice_score):
            final_id = face_id
            final_name = face_name
        elif voice_id:
            final_id = voice_id
            final_name = voice_name

        decision = "ACCEPT" if accepted else "REJECT"

        res = {
            "request_id": rid,
            "decision": decision,
            "reason": reason,
            "final": {"id": final_id, "name": final_name or "Unknown"},
            "face": {
                "best_id": face_id,
                "name": face_name or "Unknown",
                "score": float(face_score),
                "threshold": face_thresh,
                "recognized": (face_id is not None) and (face_score >= face_thresh),
            },
            "voice": {
                "best_id": voice_id,
                "name": voice_name or "Unknown",
                "score": float(voice_score),
                "threshold": voice_thresh,
                "recognized": (voice_id is not None) and (voice_score >= voice_thresh),
            },
            "fusion": {
                "alpha": alpha,
                "fused_score": float(fused_score),
                "require_both": require_both,
                "same_id": (face_id is not None) and (voice_id is not None) and (face_id == voice_id),
            },
        }

        _append_attempt_log({
            "request_id": rid,
            "type": "fusion",
            "source": payload.source,
            "ip": request.client.host if request.client else None,
            "decision": decision,
            "reason": reason,
            "final_id": final_id,
            "face_id": face_id,
            "voice_id": voice_id,
            "face_score": float(face_score),
            "voice_score": float(voice_score),
            "fused_score": float(fused_score),
            "face_thresh": face_thresh,
            "voice_thresh": voice_thresh,
            "alpha": alpha,
            "require_both": require_both,
        })

        if decision == "ACCEPT" and final_id:
            _incremental_update(
                final_id=final_id,
                face_emb=face_emb if (face_id == final_id) else None,
                voice_emb=voice_emb if (voice_id == final_id) else None,
                face_score=float(face_score),
                voice_score=float(voice_score),
                fused_score=float(fused_score),
                img_bytes=img_bytes,
                aud_bytes=aud_bytes
            )

        return res

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tmp_img, tmp_wav]:
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.post("/api/face_recognition")
async def face_recognition(payload: VerifyPayload, request: Request):
    return await verify_face(payload, request)

@app.post("/api/voice_recognition")
async def voice_recognition(payload: VerifyPayload, request: Request):
    return await verify_voice(payload, request)

@app.post("/api/register_teacher")
async def register_teacher(payload: RegisterPayload, request: Request):
    rid = gen_request_id()

    img_bytes = _b64_to_bytes(payload.image, int(CFG["security"]["max_image_bytes"])) if payload.image else None
    aud_bytes = _b64_to_bytes(payload.audio, int(CFG["security"]["max_audio_bytes"])) if payload.audio else None
    name_bytes = _b64_to_bytes(payload.name_audio,
                               int(CFG["security"]["max_audio_bytes"])) if payload.name_audio else None

    tmp_img = _write_temp_file(".jpg", img_bytes) if img_bytes else None
    tmp_wav = _write_temp_file(".wav", aud_bytes) if aud_bytes else None
    if tmp_wav:
        _trim_wav_inplace(tmp_wav, float(CFG["security"]["max_audio_seconds"]))

    try:
        face_emb = None
        voice_emb = None
        if tmp_img:
            face_emb = face_model.embed_image(tmp_img)
        if tmp_wav:
            voice_emb = voice_model.embed_file(tmp_wav)

        teacher_id = (payload.teacher_id or "").strip() or None
        pending_requested = bool(getattr(payload, "pending_approval", False))

        if teacher_id and (not pending_requested):
            with FileLock(DB_PATH, timeout=5.0):
                db = load_db(DB_PATH)
                idx = index_teachers(db)
                t = idx.get(teacher_id)
                if t is None:
                    raise HTTPException(status_code=404, detail=f"Teacher id '{teacher_id}' not found for direct update. Use pending_approval for new teacher.")
                t = ensure_teacher(t, teacher_id, payload.name)
                if face_emb is not None:
                    t["face_embeddings"].append(face_emb.astype(np.float32).tolist())
                    maxn = int(CFG["incremental"].get("max_face_embeddings", 50))
                    if len(t["face_embeddings"]) > maxn:
                        t["face_embeddings"] = t["face_embeddings"][-maxn:]
                if voice_emb is not None:
                    t["voice_embeddings"].append(voice_emb.astype(np.float32).tolist())
                    maxn = int(CFG["incremental"].get("max_voice_embeddings", 30))
                    if len(t["voice_embeddings"]) > maxn:
                        t["voice_embeddings"] = t["voice_embeddings"][-maxn:]

                t["meta"]["num_images_used"] = len(t["face_embeddings"])
                t["meta"]["num_audios_used"] = len(t["voice_embeddings"])
                t["meta"]["last_updated"] = now_iso()

                idx[teacher_id] = t
                db = finalize_db(db, idx)
                write_json_atomic(DB_PATH, db)

            if img_bytes or aud_bytes:
                _save_samples(teacher_id, img_bytes, aud_bytes)

            _append_attempt_log({
                "request_id": rid,
                "type": "update_existing",
                "source": "nao" if payload.robot_captured else "pc",
                "ip": request.client.host if request.client else None,
                "teacher_id": teacher_id,
                "name": payload.name,
            })

            return {"request_id": rid, "status": "UPDATED", "teacher_id": teacher_id, "name": payload.name}

        pending_id = "P_" + uuid.uuid4().hex[:10]
        name_bytes = _b64_to_bytes(payload.name_audio,
                                   int(CFG["security"]["max_audio_bytes"])) if payload.name_audio else None
        img_path, aud_path, name_path = _save_pending_files(pending_id, img_bytes, aud_bytes, name_bytes)

        with FileLock(PENDING_PATH, timeout=5.0):
            pdb = _load_pending_cached()
            pending_list = list(pdb.get("pending", []) or [])
            pending_list.append({
                "pending_id": pending_id,
                "requested_teacher_id": teacher_id,
                "name": payload.name,
                "robot_captured": bool(payload.robot_captured),
                "source_ip": request.client.host if request.client else None,
                "created_at": now_iso(),
                "img_path": img_path,
                "aud_path": aud_path,
                "name_path": name_path,
                "face_embedding": face_emb.astype(np.float32).tolist() if face_emb is not None else None,
                "voice_embedding": voice_emb.astype(np.float32).tolist() if voice_emb is not None else None,
                "status": "PENDING",
            })
            pdb["pending"] = pending_list
            _write_pending(pdb)

        _append_attempt_log({
            "request_id": rid,
            "type": "register_pending",
            "source": "nao" if payload.robot_captured else "pc",
            "ip": request.client.host if request.client else None,
            "pending_id": pending_id,
            "name": payload.name,
        })

        return {"request_id": rid, "status": "PENDING_APPROVAL", "pending_id": pending_id, "name": payload.name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tmp_img, tmp_wav]:
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.get("/api/teachers")
def list_teachers(pending_only: bool = False):
    db = _load_db_cached()
    teachers = db.get("teachers", [])
    out = []
    for t in teachers:
        if not isinstance(t, dict):
            continue
        pending = bool(t.get("meta", {}).get("pending_approval", False))
        if pending_only and not pending:
            continue
        out.append({
            "id": t.get("id"),
            "name": t.get("name"),
            "pending_approval": pending,
            "num_images": len(t.get("face_embeddings", []) or []),
            "num_audios": len(t.get("voice_embeddings", []) or []),
            "last_updated": t.get("meta", {}).get("last_updated"),
        })
    return {"count": len(out), "teachers": out}


@app.get("/api/admin/pending")
def list_pending():
    return {"count": len(_pending_list()), "pending": _pending_list()}

@app.get("/api/admin/pending/{pending_id}")
def get_pending(pending_id: str):
    item = _get_pending(pending_id)
    if not item:
        raise HTTPException(status_code=404, detail="Pending item not found")
    return item

@app.get("/api/admin/pending/{pending_id}/image")
def get_pending_image(pending_id: str):
    item = _get_pending(pending_id)
    if not item or not item.get("img_path"):
        raise HTTPException(status_code=404, detail="Image not found")
    from fastapi.responses import FileResponse
    return FileResponse(item["img_path"], media_type="image/jpeg")

@app.get("/api/admin/pending/{pending_id}/audio")
def get_pending_audio(pending_id: str):
    item = _get_pending(pending_id)
    if not item or not item.get("aud_path"):
        raise HTTPException(status_code=404, detail="Audio not found")
    from fastapi.responses import FileResponse
    return FileResponse(item["aud_path"], media_type="audio/wav")

class PendingDecisionPayload(BaseModel):
    pending_id: str
    action: str = Field(..., pattern="^(approve|reject)$")
    name: Optional[str] = None


@app.get("/api/admin/pending/{pending_id}/name_audio")
def admin_pending_name_audio(pending_id: str):
    item = _get_pending(pending_id)
    if not item:
        raise HTTPException(status_code=404, detail="Pending request not found.")

    p = item.get("name_path")
    if not p or (not Path(p).exists()):
        raise HTTPException(status_code=404, detail="No name audio for this pending request.")

    from fastapi.responses import FileResponse
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/admin/pending/decision")
def pending_decision(payload: PendingDecisionPayload):
    item = _get_pending(payload.pending_id)
    if not item:
        raise HTTPException(status_code=404, detail="Pending item not found")

    action = payload.action.lower().strip()
    if action == "reject":
        with FileLock(PENDING_PATH, timeout=5.0):
            pdb = _load_pending_cached()
            new_list = [x for x in (pdb.get("pending", []) or []) if x.get("pending_id") != payload.pending_id]
            pdb["pending"] = new_list
            _write_pending(pdb)
        _delete_pending_files(payload.pending_id)
        return {"status": "REJECTED", "pending_id": payload.pending_id}

    name = (payload.name or item.get("name") or "").strip() or "UNKNOWN"
    face_emb = item.get("face_embedding")
    voice_emb = item.get("voice_embedding")

    new_teacher_id = None
    with FileLock(DB_PATH, timeout=5.0):
        db = load_db(DB_PATH)
        idx = index_teachers(db)

        new_teacher_id = allocate_next_teacher_id()

        t = ensure_teacher(idx.get(new_teacher_id) or {}, new_teacher_id, name)

        if face_emb is not None:
            t["face_embeddings"].append(face_emb)
        if voice_emb is not None:
            t["voice_embeddings"].append(voice_emb)

        t["meta"]["created_at"] = t.get("meta", {}).get("created_at") or now_iso()
        t["meta"]["approved_at"] = now_iso()
        t["meta"]["pending_approval"] = False
        t["meta"]["robot_captured"] = bool(item.get("robot_captured", True))
        t["meta"]["num_images_used"] = len(t.get("face_embeddings", []) or [])
        t["meta"]["num_audios_used"] = len(t.get("voice_embeddings", []) or [])
        t["meta"]["last_updated"] = now_iso()

        idx[new_teacher_id] = t
        db = finalize_db(db, idx)
        write_json_atomic(DB_PATH, db)

    append_to_teachers_csv(new_teacher_id, name)

    try:
        img_bytes = None
        aud_bytes = None
        if item.get("img_path") and Path(item["img_path"]).exists():
            img_bytes = Path(item["img_path"]).read_bytes()
        if item.get("aud_path") and Path(item["aud_path"]).exists():
            aud_bytes = Path(item["aud_path"]).read_bytes()
        if img_bytes or aud_bytes:
            _save_samples(new_teacher_id, img_bytes, aud_bytes)
    except Exception:
        pass

    with FileLock(PENDING_PATH, timeout=5.0):
        pdb = _load_pending_cached()
        new_list = [x for x in (pdb.get("pending", []) or []) if x.get("pending_id") != payload.pending_id]
        pdb["pending"] = new_list
        _write_pending(pdb)
    _delete_pending_files(payload.pending_id)

    return {"status": "APPROVED", "teacher_id": new_teacher_id, "name": name, "pending_id": payload.pending_id}

@app.post("/api/admin/approve_teacher")
def approve_teacher(payload: ApprovePayload):
    with FileLock(DB_PATH, timeout=5.0):
        db = load_db(DB_PATH)
        idx = index_teachers(db)
        t = idx.get(payload.teacher_id)
        if not t:
            raise HTTPException(status_code=404, detail="Teacher not found")
        t.setdefault("meta", {})
        if payload.approved:
            t["meta"]["pending_approval"] = False
        else:
            idx.pop(payload.teacher_id, None)

        db = finalize_db(db, idx)
        write_json_atomic(DB_PATH, db)

    return {"status": "ok", "teacher_id": payload.teacher_id, "approved": payload.approved}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

