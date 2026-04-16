# build_unified_db_voice.py
import argparse
from pathlib import Path
from tqdm import tqdm

from mapping_reader import load_id_name_map
from voice_model import ECAPATDNNModel
from db_utils import load_db, save_db, index_teachers, ensure_teacher, finalize_db

AUD_EXTS = {".wav"}

def collect_audio(folder: Path):
    if not folder.exists():
        return []
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUD_EXTS:
            files.append(p)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="dataset root")
    ap.add_argument("--map", default="Teachers.csv", help="Teachers.csv (id,name)")
    ap.add_argument("--out", default="db/teachers.json")
    ap.add_argument("--mode", choices=["replace", "append"], default="replace")
    ap.add_argument("--min_audios", type=int, default=1)
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg path or 'ffmpeg'")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ds = Path(args.dataset)
    id_name = load_id_name_map(args.map)

    db = load_db(args.out)
    idx = index_teachers(db)

    model = ECAPATDNNModel(device=args.device, ffmpeg_path=args.ffmpeg)

    for tid, name in tqdm(id_name.items(), desc="Teachers"):
        teacher = ensure_teacher(idx, tid, name)

        aud_dir = ds / tid / "audio"
        audios = collect_audio(aud_dir)

        if len(audios) < args.min_audios:
            continue

        if args.mode == "replace":
            teacher["voice_embeddings"] = []

        for a in audios:
            try:
                emb = model.embed_file(str(a))
                teacher["voice_embeddings"].append(emb.tolist())
            except Exception:
                pass

        teacher["meta"]["num_audios_used"] = len(teacher["voice_embeddings"])
        teacher["meta"]["num_images_used"] = len(teacher["face_embeddings"])

    db = finalize_db(db, idx)
    save_db(args.out, db)
    print(f"Saved unified DB (voice): {args.out} | teachers={len(db['teachers'])}")

if __name__ == "__main__":
    main()
