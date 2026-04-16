# face_model_insightface.py
import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class InsightFaceModel:
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size=(640, 640),
        det_thresh: float = 0.5,
        use_gpu: bool = False,
    ):
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(name=model_pack, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)

    def embed_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self.app.get(img)
        if not faces:
            raise ValueError(f"No face detected in: {image_path}")

        def area(f):
            x1, y1, x2, y2 = f.bbox
            return float((x2 - x1) * (y2 - y1))

        best = max(faces, key=area)
        emb = np.asarray(best.normed_embedding, dtype=np.float32)
        return emb

    def count_faces(self, image_path: str) -> int:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        faces = self.app.get(img)
        return len(faces)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="dataset/T0001/images/1.jpeg", help="Path to an image file")
    ap.add_argument("--model_pack", default="buffalo_l")
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--det_thresh", type=float, default=0.5)
    args = ap.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")

    model = InsightFaceModel(
        model_pack=args.model_pack,
        det_thresh=args.det_thresh,
        use_gpu=args.use_gpu
    )

    emb = model.embed_image(str(img_path))
    print("Embedding shape:", emb.shape)
    print("Embedding (first 10 vals):", emb[:10])
