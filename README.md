# authentication-system
## Authentication System (Face + Voice + Optional STT) — Privacy-Safe Repo

A practical authentication workflow that can combine:

Face verification (InsightFace embeddings)  
Speaker verification (SpeechBrain speaker embeddings)  
Optional offline speech-to-text (STT) for capturing a name using Vosk  

This repository is published in a privacy-safe way: the project structure is complete, but any sensitive datasets/logs/identities were removed or replaced with placeholders.


<img width="1189" height="622" alt="image" src="https://github.com/user-attachments/assets/1409a30d-8f0c-40e3-9c6f-db647beb95bd" />


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
├─ run_system.py              # Main runner (full system)
├─ main.py                    # FastAPI server (if used by your flow)
├─ pc_client.py               # PC-side logic (STT + UI prompts)
├─ verify_fusion.py           # Fusion logic (face + voice)
├─ face_model_insightface.py  # Face embeddings using InsightFace
├─ voice_model.py             # Speaker embeddings using SpeechBrain
├─ db/
│  ├─ teachers.json           # Template (placeholder)
│  └─ pending.json            # Template (placeholder)
├─ dataset/                   # Template only (no private media)
├─ logs/
│  └─ attempts.jsonl          # Empty placeholder
└─ assets/prompts/            # Audio/text prompts (AR/EN)
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

### Why the Vosk model folder is not in this repo

A folder like `vosk-model-small-en-us-0.15` is an external pre-trained model downloaded from a third-party source. To avoid licensing and ownership issues, and to reduce repo size, the model is not committed here.

### How to use Vosk with this project

Download any compatible Vosk model, whether English or another language.

Place the model folder inside the project directory, for example:

```text
Authentication-System/
└─ vosk-model-small-en-us-0.15/
```

### If your model has a different folder name

No problem. Just pass its name or path using `--vosk_model`:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.22 --name_seconds 6 --cooldown 10 --pc_ui_lang en
```

Alternatively, you can change the default in the code where the argument is defined by searching for `--vosk_model` in `run_system.py`.

## Running the Full System

From the project root:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang en
```

### Example variations

Run without STT, if supported by your setup:

```bash
python run_system.py --vosk_model vosk-model-small-en-us-0.15 --cooldown 10 --pc_ui_lang en
```

Switch UI language, if you have prompts for Arabic:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang ar
```

## What you need to provide (because this repo is privacy-safe)

To actually enroll or verify identities, you must provide your own data:

- Add your own images and audio into the expected `dataset/` structure.
- Fill your local teacher or user list, such as CSV or JSON templates, with your own IDs.
- Generate embeddings and databases according to the scripts used in your workflow.

## Troubleshooting

- **The system can’t find the Vosk model:** Make sure the folder exists and the path matches the `--vosk_model` value.
- **Empty dataset or missing identities:** This repo ships without private media by design. Add your own data locally.
- **Repeated triggers or too many attempts:** Increase `--cooldown` to reduce back-to-back attempts.

## Third-Party Models & Licensing

This repository contains project code and placeholder templates only. Third-party models, such as Vosk, are governed by their original licenses and must be obtained separately.

## Contact

If you build on this project, feel free to open an issue or submit a pull request.



