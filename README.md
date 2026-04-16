# authentication-system
# Authentication System (Face + Voice + Optional STT) — Privacy-Safe Repo

A practical authentication workflow that can combine:

Face verification (InsightFace embeddings)  
Speaker verification (SpeechBrain speaker embeddings)  
Optional offline speech-to-text (STT) for capturing a name using Vosk  

This repository is published in a privacy-safe way: the project structure is complete, but any sensitive datasets/logs/identities were removed or replaced with placeholders.

---

## Demo Run (How we run the full system)

We typically run the complete system using:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang en
```

---

## What these flags mean

- `--stt_name` Enables offline speech-to-text for capturing/confirming a spoken name.  
- `--vosk_model <folder>` Path (or folder name) of the Vosk model directory.  
- `--name_seconds 6` How many seconds to listen for the name.  
- `--cooldown 10` Cooldown time between attempts.  
- `--pc_ui_lang en` UI language.  

---

## Key Features

- Multi-factor verification (Face + Voice)  
- Offline STT option (Vosk)  
- Privacy-safe dataset templates  
- UI prompts (AR/EN)  
- Modular scripts  

---

## Privacy & Data Handling Notes

This repo intentionally excludes any private data.

- dataset/ exists as structure only  
- db files are placeholders  
- logs are empty  

---

## Project Structure

```text
.
├─ run_system.py
├─ main.py
├─ pc_client.py
├─ verify_fusion.py
├─ face_model_insightface.py
├─ voice_model.py
├─ db/
├─ dataset/
├─ logs/
└─ assets/prompts/
```

---

## Requirements

- Python 3.9+
- numpy, requests, tqdm
- opencv-python
- torch, torchaudio
- speechbrain
- insightface
- fastapi, uvicorn
- optional: vosk

---

## Setup

### 1) Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 2) Install dependencies

```bash
pip install -U pip
pip install numpy requests tqdm opencv-python
pip install torch torchaudio
pip install speechbrain insightface
pip install fastapi uvicorn
pip install vosk
```

---

## Vosk Model (Not Included)

Download Vosk model manually and place it inside project.

Example:

```text
Authentication-System/
└─ vosk-model-small-en-us-0.15/
```

---

## Running the System

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang en
```

---

## What you need to provide

- your own dataset  
- your own identities  
- generate embeddings locally  

---

## Troubleshooting

- Model not found → check path  
- Empty dataset → add your data  
- Too many triggers → increase cooldown  

---

## Contact

Open an issue or pull request if needed.
