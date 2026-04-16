# mapping_reader.py
from pathlib import Path
import csv, json
import argparse

def load_id_name_map(map_path: str) -> dict:
    p = Path(map_path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file not found: {p}")

    ext = p.suffix.lower()

    if ext in [".csv", ".tsv"]:
        delim = "\t" if ext == ".tsv" else ","
        out = {}
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            for row in r:
                tid = (row.get("id") or "").strip()
                name = (row.get("name") or "").strip()
                if tid and name:
                    out[tid] = name
        return out

    if ext == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip() for k, v in data.items()}
        if isinstance(data, list):
            out = {}
            for item in data:
                tid = str(item.get("id","")).strip()
                name = str(item.get("name","")).strip()
                if tid and name:
                    out[tid] = name
            return out

    raise ValueError("Use .csv/.tsv/.json for mapping file")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Teachers.csv", help="Path to Teachers.csv (id,name)")
    ap.add_argument("--n", type=int, default=5, help="Print first N records")
    args = ap.parse_args()

    m = load_id_name_map(args.map)
    print("Total teachers:", len(m))
    for i, (k, v) in enumerate(m.items()):
        if i >= args.n:
            break
        print(k, "->", v)
