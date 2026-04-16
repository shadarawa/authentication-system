# verify_fusion.py
import argparse
import json
from pathlib import Path
import numpy as np

from face_model_insightface import InsightFaceModel
from voice_model import ECAPATDNNModel

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x)
    return x / (n + eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def load_teachers(db_path: str) -> list:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"DB not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        db = json.load(f)
    return db.get("teachers", [])

def best_match_from_embeddings(query_emb: np.ndarray, teachers: list, key: str):
    best_id, best_name, best_score = None, None, -1.0
    for t in teachers:
        tid = t.get("id")
        name = t.get("name")
        embs = t.get(key, [])
        for e in embs:
            score = cosine_sim(query_emb, np.asarray(e, dtype=np.float32))
            if score > best_score:
                best_score = score
                best_id = tid
                best_name = name
    return best_id, best_name, best_score

def build_index_by_id(teachers: list) -> dict:
    return {t.get("id"): t for t in teachers if isinstance(t, dict) and t.get("id")}

def fusion_decision(
    face_id, face_score,
    voice_id, voice_score,
    face_thresh: float,
    voice_thresh: float,
    alpha: float = 0.5,
    require_both: bool = True
):
    face_ok = (face_id is not None) and (face_score >= face_thresh)
    voice_ok = (voice_id is not None) and (voice_score >= voice_thresh)

    same_id = (face_id is not None) and (voice_id is not None) and (face_id == voice_id)

    fused = alpha * float(face_score) + (1.0 - alpha) * float(voice_score)

    if require_both:
        accepted = face_ok and voice_ok and same_id
        reason = []
        if not face_ok: reason.append("face<thresh")
        if not voice_ok: reason.append("voice<thresh")
        if (face_id and voice_id) and not same_id: reason.append("id_mismatch")
        return accepted, fused, ("OK" if accepted else ",".join(reason) if reason else "REJECT")
    else:
        accepted = (face_ok or voice_ok) and (same_id or (face_id is None) or (voice_id is None))
        reason = "OK" if accepted else "REJECT"
        return accepted, fused, reason

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="db/teachers.json", help="Path to unified teachers DB json")

    ap.add_argument("--img", default="images_test/Omar.jpg", help="Path to test image")
    ap.add_argument("--wav", default="audio_test/DrOmar.wav", help="Path to test audio (wav)")

    ap.add_argument("--model_pack", default="buffalo_l")
    ap.add_argument("--use_gpu_face", action="store_true")
    ap.add_argument("--det_thresh", type=float, default=0.5)

    ap.add_argument("--device_voice", default=None, help="cpu or cuda")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg path (used only if your voice_model needs it)")

    ap.add_argument("--face_thresh", type=float, default=0.464, help="Typical range 0.30-0.45 (tune!)")
    ap.add_argument("--voice_thresh", type=float, default=0.402, help="Typical range 0.20-0.35 (tune!)")
    ap.add_argument("--alpha", type=float, default=0.6, help="Weight for face in fusion (0..1)")
    ap.add_argument(
        "--require_both",
        dest="require_both",
        action="store_true",
        help="Require both modalities AND same ID to accept",
    )
    ap.add_argument(
        "--no_require_both",
        dest="require_both",
        action="store_false",
        help="Disable require_both (relaxed policy)",
    )
    ap.set_defaults(require_both=True)

    args = ap.parse_args()

    teachers = load_teachers(args.db)
    idx = build_index_by_id(teachers)

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")

    face_model = InsightFaceModel(
        model_pack=args.model_pack,
        det_thresh=args.det_thresh,
        use_gpu=args.use_gpu_face
    )
    face_emb = face_model.embed_image(str(img_path))
    face_id, face_name, face_score = best_match_from_embeddings(face_emb, teachers, "face_embeddings")

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio not found: {wav_path.resolve()}")

    voice_model = ECAPATDNNModel(device=args.device_voice, ffmpeg_path=args.ffmpeg)
    voice_emb = voice_model.embed_file(str(wav_path))
    voice_id, voice_name, voice_score = best_match_from_embeddings(voice_emb, teachers, "voice_embeddings")

    accepted, fused_score, reason = fusion_decision(
        face_id, face_score,
        voice_id, voice_score,
        face_thresh=args.face_thresh,
        voice_thresh=args.voice_thresh,
        alpha=args.alpha,
        require_both=args.require_both
    )

    final_id = None
    if accepted:
        if face_id and voice_id and face_id == voice_id:
            final_id = face_id
        else:
            face_margin = face_score - args.face_thresh
            voice_margin = voice_score - args.voice_thresh
            final_id = face_id if face_margin >= voice_margin else voice_id

    final_name = idx.get(final_id, {}).get("name") if final_id else None

    print("---- RESULTS ----")
    print(f"Face  : best_id={face_id} | name={face_name} | score={face_score:.4f} | thresh={args.face_thresh:.2f}")
    print(f"Voice : best_id={voice_id} | name={voice_name} | score={voice_score:.4f} | thresh={args.voice_thresh:.2f}")
    print(f"Fusion: alpha(face)={args.alpha:.2f} | fused_score={fused_score:.4f} | require_both={args.require_both}")
    print(f"Decision: {'ACCEPT' if accepted else 'REJECT'} | reason={reason}")
    if accepted:
        print(f"Final: id={final_id} | name={final_name}")

if __name__ == "__main__":
    main()
