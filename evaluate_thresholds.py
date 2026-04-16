# evaluate_thresholds.py
import argparse
import json
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

from face_model_insightface import InsightFaceModel
from voice_model import ECAPATDNNModel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUD_EXTS = {".wav"}

def list_files(folder: Path, exts: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x) + eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def mean_embedding(embs: list[np.ndarray]) -> np.ndarray:
    m = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(m)

def roc_eer(scores_pos: np.ndarray, scores_neg: np.ndarray):
    scores_pos = np.asarray(scores_pos, dtype=np.float32)
    scores_neg = np.asarray(scores_neg, dtype=np.float32)

    if len(scores_pos) == 0 or len(scores_neg) == 0:
        raise RuntimeError("ROC/EER needs both positive and negative score sets (non-empty).")

    all_scores = np.concatenate([scores_pos, scores_neg])
    thrs = np.unique(all_scores)
    thrs.sort()

    fars = np.array([(scores_neg >= t).mean() for t in thrs], dtype=np.float32)
    frrs = np.array([(scores_pos < t).mean() for t in thrs], dtype=np.float32)

    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    thr_eer = float(thrs[idx])
    return eer, thr_eer

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="dataset", help="dataset root")
    ap.add_argument("--alpha", type=float, default=0.6, help="fusion weight for face")

    ap.add_argument("--out_json", default="logs/eval_thresholds.json", help="Save results summary to JSON")

    ap.add_argument("--enroll_imgs", type=int, default=1)
    ap.add_argument("--enroll_auds", type=int, default=1)

    ap.add_argument("--min_imgs", type=int, default=2, help="need >=2 images (1 enroll + >=1 probe)")
    ap.add_argument("--min_auds_any", type=int, default=1, help="teacher is considered 'has voice' if >= this many audios")
    ap.add_argument("--min_auds_for_voice_eval", type=int, default=2,
                    help="need >= this many audios to be included in VOICE/FUSION evaluation (>= enroll_auds + 1 recommended)")

    ap.add_argument("--impostor_mode", choices=["hardmax", "all"], default="hardmax")

    ap.add_argument("--model_pack", default="buffalo_l")
    ap.add_argument("--use_gpu_face", action="store_true")
    ap.add_argument("--det_thresh", type=float, default=0.5)

    ap.add_argument("--device_voice", default=None)
    ap.add_argument("--ffmpeg", default="ffmpeg")

    args = ap.parse_args()
    ds = Path(args.dataset)

    face_model = InsightFaceModel(
        model_pack=args.model_pack,
        det_thresh=args.det_thresh,
        use_gpu=args.use_gpu_face
    )
    voice_model = ECAPATDNNModel(device=args.device_voice, ffmpeg_path=args.ffmpeg)

    teacher_dirs = sorted([p for p in ds.iterdir() if p.is_dir() and p.name.startswith("T")])
    if not teacher_dirs:
        raise FileNotFoundError(f"No teacher dirs found under {ds.resolve()} (expected dataset/T0001, ...).")

    teachers = []
    for td in teacher_dirs:
        tid = td.name
        imgs = list_files(td / "images", IMG_EXTS)
        auds = list_files(td / "audio", AUD_EXTS)
        if len(imgs) >= args.min_imgs and len(auds) >= args.min_auds_any:
            teachers.append({"id": tid, "imgs": imgs, "auds": auds})

    if len(teachers) < 2:
        raise RuntimeError("Need at least 2 teachers with enough images and at least 1 audio.")

    min_need_for_voice = max(args.min_auds_for_voice_eval, args.enroll_auds + 1)

    teachers_face_eval = [t for t in teachers if len(t["imgs"]) >= args.enroll_imgs + 1]
    teachers_voice_eval = [t for t in teachers if len(t["auds"]) >= min_need_for_voice and len(t["imgs"]) >= args.enroll_imgs + 1]

    print(f"Teachers total (imgs>=min_imgs & auds>=min_auds_any): {len(teachers)}")
    print(f"Teachers usable for FACE eval  : {len(teachers_face_eval)}")
    print(f"Teachers usable for VOICE/FUSION eval (auds>={min_need_for_voice}): {len(teachers_voice_eval)}")

    if len(teachers_face_eval) < 2:
        raise RuntimeError("Need at least 2 teachers with enough images for FACE evaluation.")

    if len(teachers_voice_eval) < 2:
        raise RuntimeError(
            "Need at least 2 teachers with enough audio segments for VOICE/FUSION evaluation.\n"
            f"Currently require audios >= {min_need_for_voice}. "
            "Either add more audio segments for more teachers, or lower --min_auds_for_voice_eval (not recommended below enroll_auds+1)."
        )

    alpha = float(args.alpha)

    face_ids = [t["id"] for t in teachers_face_eval]
    enroll_face = {}

    print("Building FACE enrollment templates...")
    for t in tqdm(teachers_face_eval, desc="FaceEnroll"):
        tid = t["id"]
        embs = []
        for p in t["imgs"][:args.enroll_imgs]:
            try:
                embs.append(face_model.embed_image(str(p)).astype(np.float32))
            except Exception:
                pass
        if len(embs) > 0:
            enroll_face[tid] = mean_embedding(embs)

    voice_ids = [t["id"] for t in teachers_voice_eval]
    enroll_voice = {}

    print("Building VOICE enrollment templates...")
    for t in tqdm(teachers_voice_eval, desc="VoiceEnroll"):
        tid = t["id"]
        embs = []
        for p in t["auds"][:args.enroll_auds]:
            try:
                embs.append(voice_model.embed_file(str(p)).astype(np.float32))
            except Exception:
                pass
        if len(embs) > 0:
            enroll_voice[tid] = mean_embedding(embs)

    face_ids = sorted(set(face_ids) & set(enroll_face.keys()))
    voice_ids = sorted(set(voice_ids) & set(enroll_face.keys()) & set(enroll_voice.keys()))

    if len(face_ids) < 2:
        raise RuntimeError("After FACE enrollment failures, fewer than 2 teachers remain.")

    if len(voice_ids) < 2:
        raise RuntimeError("After VOICE enrollment failures, fewer than 2 teachers remain for voice/fusion.")

    face_pos, face_neg = [], []
    voice_pos, voice_neg = [], []
    fusion_pos, fusion_neg = [], []

    print("Scoring FACE probes...")
    for t in tqdm([x for x in teachers_face_eval if x["id"] in face_ids], desc="FaceProbes"):
        tid = t["id"]
        probes = t["imgs"][args.enroll_imgs:]
        for p in probes:
            try:
                q = face_model.embed_image(str(p)).astype(np.float32)
            except Exception:
                continue

            s_pos = cosine_sim(q, enroll_face[tid])
            face_pos.append(s_pos)

            if args.impostor_mode == "hardmax":
                s_max = -1.0
                for oid in face_ids:
                    if oid == tid:
                        continue
                    s = cosine_sim(q, enroll_face[oid])
                    if s > s_max:
                        s_max = s
                face_neg.append(s_max)
            else:
                for oid in face_ids:
                    if oid == tid:
                        continue
                    face_neg.append(cosine_sim(q, enroll_face[oid]))

    print("Scoring VOICE probes...")
    for t in tqdm([x for x in teachers_voice_eval if x["id"] in voice_ids], desc="VoiceProbes"):
        tid = t["id"]
        probes = t["auds"][args.enroll_auds:]
        for p in probes:
            try:
                q = voice_model.embed_file(str(p)).astype(np.float32)
            except Exception:
                continue

            s_pos = cosine_sim(q, enroll_voice[tid])
            voice_pos.append(s_pos)

            if args.impostor_mode == "hardmax":
                s_max = -1.0
                for oid in voice_ids:
                    if oid == tid:
                        continue
                    s = cosine_sim(q, enroll_voice[oid])
                    if s > s_max:
                        s_max = s
                voice_neg.append(s_max)
            else:
                for oid in voice_ids:
                    if oid == tid:
                        continue
                    voice_neg.append(cosine_sim(q, enroll_voice[oid]))

    print("Scoring FUSION probes...")
    for t in tqdm([x for x in teachers_voice_eval if x["id"] in voice_ids], desc="FusionProbes"):
        tid = t["id"]

        img_probes = t["imgs"][args.enroll_imgs:]
        aud_probes = t["auds"][args.enroll_auds:]
        n_pair = min(len(img_probes), len(aud_probes))
        for i in range(n_pair):
            try:
                fi = face_model.embed_image(str(img_probes[i])).astype(np.float32)
                vi = voice_model.embed_file(str(aud_probes[i])).astype(np.float32)
            except Exception:
                continue

            sF_pos = cosine_sim(fi, enroll_face[tid])
            sV_pos = cosine_sim(vi, enroll_voice[tid])
            s_pos = alpha * sF_pos + (1.0 - alpha) * sV_pos
            fusion_pos.append(s_pos)

            if args.impostor_mode == "hardmax":
                s_max = -1.0
                for oid in voice_ids:
                    if oid == tid:
                        continue
                    sF = cosine_sim(fi, enroll_face[oid])
                    sV = cosine_sim(vi, enroll_voice[oid])
                    s = alpha * sF + (1.0 - alpha) * sV
                    if s > s_max:
                        s_max = s
                fusion_neg.append(s_max)
            else:
                for oid in voice_ids:
                    if oid == tid:
                        continue
                    sF = cosine_sim(fi, enroll_face[oid])
                    sV = cosine_sim(vi, enroll_voice[oid])
                    fusion_neg.append(alpha * sF + (1.0 - alpha) * sV)

    face_pos = np.array(face_pos, dtype=np.float32)
    face_neg = np.array(face_neg, dtype=np.float32)
    voice_pos = np.array(voice_pos, dtype=np.float32)
    voice_neg = np.array(voice_neg, dtype=np.float32)
    fusion_pos = np.array(fusion_pos, dtype=np.float32)
    fusion_neg = np.array(fusion_neg, dtype=np.float32)

    if len(face_pos) == 0 or len(face_neg) == 0:
        raise RuntimeError("FACE evaluation produced no scores. Check images and enroll_imgs/min_imgs.")

    if len(voice_pos) == 0 or len(voice_neg) == 0:
        raise RuntimeError(
            "VOICE evaluation produced no scores. This usually means not enough teachers have >= enroll_auds+1 audio files.\n"
            f"Try checking your audio splits or lowering --min_auds_for_voice_eval (but keep it >= {args.enroll_auds+1})."
        )

    f_eer, f_thr = roc_eer(face_pos, face_neg)
    v_eer, v_thr = roc_eer(voice_pos, voice_neg)

    print("\n==== EVALUATION RESULTS ====")
    print(f"Face : pos={len(face_pos)} neg={len(face_neg)} | EER={f_eer:.4f} | thr@EER={f_thr:.4f}")
    print(f"Voice: pos={len(voice_pos)} neg={len(voice_neg)} | EER={v_eer:.4f} | thr@EER={v_thr:.4f}")

    if len(fusion_pos) > 0 and len(fusion_neg) > 0:
        fu_eer, fu_thr = roc_eer(fusion_pos, fusion_neg)
        print(f"Fusion(alpha={alpha:.2f}): pos={len(fusion_pos)} neg={len(fusion_neg)} | EER={fu_eer:.4f} | thr@EER={fu_thr:.4f}")
    else:
        print("Fusion: not enough paired probes to compute fusion ROC/EER.")

    print("\n---- Suggested thresholds (start point) ----")
    print(f"Suggested face_thresh  ~ {f_thr:.4f}")
    print(f"Suggested voice_thresh ~ {v_thr:.4f}")
    print("Tip: For stricter security, increase thresholds slightly (+0.02..+0.05).")

    try:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "dataset": str(Path(args.dataset).resolve()),
            "alpha": float(args.alpha),
            "params": {
                "enroll_imgs": int(args.enroll_imgs),
                "enroll_auds": int(args.enroll_auds),
                "min_imgs": int(args.min_imgs),
                "min_auds_any": int(args.min_auds_any),
                "min_auds_for_voice_eval": int(args.min_auds_for_voice_eval),
            },
            "counts": {
                "teachers_total": int(len(teachers)),
                "teachers_face_eval": int(len(teachers_face_eval)),
                "teachers_voice_eval": int(len(teachers_voice_eval)),
                "pairs_face": {"pos": int(len(face_pos)), "neg": int(len(face_neg))},
                "pairs_voice": {"pos": int(len(voice_pos)), "neg": int(len(voice_neg))},
                "pairs_fusion": {"pos": int(len(fusion_pos)), "neg": int(len(fusion_neg))},
            },
            "results": {
                "face": {"eer": float(f_eer), "thr_eer": float(f_thr)},
                "voice": {"eer": float(v_eer), "thr_eer": float(v_thr)},
                "fusion": {"eer": float(fu_eer) if (len(fusion_pos) and len(fusion_neg)) else None,
                           "thr_eer": float(fu_thr) if (len(fusion_pos) and len(fusion_neg)) else None},
            },
            "suggested_thresholds": {
                "face": float(f_thr),
                "voice": float(v_thr),
                "fusion": float(fu_thr) if (len(fusion_pos) and len(fusion_neg)) else None,
            },
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("[Saved results summary] %s" % out_path)
    except Exception as e:
        print("[WARN] Failed to save JSON summary: %s" % e)


if __name__ == "__main__":
    main()
