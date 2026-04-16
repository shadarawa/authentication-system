# db_utils.py
import json
from pathlib import Path
from typing import Dict, Any

def empty_db() -> Dict[str, Any]:
    return {"version": 1, "teachers": []}

def load_db(db_path: str) -> Dict[str, Any]:
    p = Path(db_path)
    if not p.exists():
        return empty_db()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db_path: str, db: Dict[str, Any]) -> None:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def index_teachers(db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for t in db.get("teachers", []):
        if isinstance(t, dict) and "id" in t:
            idx[t["id"]] = t
    return idx

def ensure_teacher(idx: Dict[str, Dict[str, Any]], tid: str, name: str) -> Dict[str, Any]:
    t = idx.get(tid)
    if t is None:
        t = {
            "id": tid,
            "name": name,
            "face_embeddings": [],
            "voice_embeddings": [],
            "meta": {"num_images_used": 0, "num_audios_used": 0},
        }
        idx[tid] = t

    t["name"] = name

    t.setdefault("face_embeddings", [])
    t.setdefault("voice_embeddings", [])
    t.setdefault("meta", {})

    t["meta"]["num_images_used"] = len(t["face_embeddings"])
    t["meta"]["num_audios_used"] = len(t["voice_embeddings"])

    return t

def finalize_db(db: Dict[str, Any], idx: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    db["version"] = 1
    db["teachers"] = list(idx.values())
    return db
