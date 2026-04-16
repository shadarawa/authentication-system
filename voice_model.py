# voice_model.py
import argparse
from pathlib import Path
import subprocess
import tempfile
import os
import shutil

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x) + eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def _ffmpeg_reencode_to_temp_wav(path: str, ffmpeg_path: str = "ffmpeg") -> str:
    src = str(Path(path).resolve())
    ff = shutil.which(ffmpeg_path) or ffmpeg_path

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        ff,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", src,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        tmp_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")
    if proc.returncode != 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed to re-encode '{src}'.\n{err}")

    return tmp_path

class ECAPATDNNModel:
    def __init__(self, device: str | None = None, ffmpeg_path: str = "ffmpeg", max_seconds: float = 8.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ffmpeg_path = ffmpeg_path
        self.max_seconds = float(max_seconds)
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def _load_audio_16k_mono(self, path: str) -> torch.Tensor:
        p = str(Path(path).resolve())

        try:
            wav, sr = torchaudio.load(p)
        except Exception:
            tmp_wav = _ffmpeg_reencode_to_temp_wav(p, ffmpeg_path=self.ffmpeg_path)
            try:
                wav, sr = torchaudio.load(tmp_wav)
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        return wav

    @torch.inference_mode()
    def embed_file(self, audio_path: str) -> np.ndarray:
        wav = self._load_audio_16k_mono(audio_path).to(self.device)
        lens = torch.ones(wav.shape[0], device=self.device)
        emb = self.model.encode_batch(wav, lens)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def best_match(self, query_emb: np.ndarray, teachers: list) -> tuple[str | None, float]:
        best_id, best_score = None, -1.0
        for t in teachers:
            for e in t.get("voice_embeddings", []):
                score = cosine_sim(query_emb, np.asarray(e, dtype=np.float32))
                if score > best_score:
                    best_score, best_id = score, t.get("id")
        return best_id, best_score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="dataset/T0001/audio/1.wav", help="Path to WAV file")
    ap.add_argument("--device", default=None, help="cpu or cuda (optional)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg path or 'ffmpeg' if in PATH")
    args = ap.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    model = ECAPATDNNModel(device=args.device, ffmpeg_path=args.ffmpeg)
    emb = model.embed_file(str(wav_path))

    print("Device:", model.device)
    print("Embedding shape:", emb.shape)
    print("Embedding (first 10 vals):", emb[:10])