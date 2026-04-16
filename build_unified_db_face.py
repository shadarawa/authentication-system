# build_unified_db_face.py
import argparse
from pathlib import Path
from tqdm import tqdm

from mapping_reader import load_id_name_map
from face_model_insightface import InsightFaceModel
from db_utils import load_db, save_db, index_teachers, ensure_teacher, finalize_db

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(folder: Path):
    if not folder.exists():
        return []
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="dataset root")
    ap.add_argument("--map", default="Teachers.csv", help="Teachers.csv (id,name)")
    ap.add_argument("--out", default="db/teachers.json")
    ap.add_argument("--model_pack", default="buffalo_l")
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--min_imgs", type=int, default=1)
    ap.add_argument("--mode", choices=["replace", "append"], default="replace")
    args = ap.parse_args()

    ds = Path(args.dataset)
    id_name = load_id_name_map(args.map)

    db = load_db(args.out)
    idx = index_teachers(db)

    model = InsightFaceModel(model_pack=args.model_pack, use_gpu=args.use_gpu)

    for tid, name in tqdm(id_name.items(), desc="Teachers"):
        teacher = ensure_teacher(idx, tid, name)

        img_dir = ds / tid / "images"
        images = collect_images(img_dir)

        if len(images) < args.min_imgs:
            continue

        if args.mode == "replace":
            teacher["face_embeddings"] = []

        for img_path in images:
            try:
                emb = model.embed_image(str(img_path))
                teacher["face_embeddings"].append(emb.tolist())
            except Exception:
                pass

        teacher["meta"]["num_images_used"] = len(teacher["face_embeddings"])
        teacher["meta"]["num_audios_used"] = len(teacher["voice_embeddings"])

    db = finalize_db(db, idx)
    save_db(args.out, db)
    print(f"Saved unified DB (face): {args.out} | teachers={len(db['teachers'])}")

if __name__ == "__main__":
    main()
