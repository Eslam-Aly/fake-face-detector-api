---
title: Fake Face Detector Api
emoji: 👁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# Fake Face Detection FastAPI

Standalone FastAPI service to expose your fake-face model for a JavaScript client.

## 1) Install

```bash
cd ffdapi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure model source

Choose one option:

### Option A: Local model path

By default, the API looks for:

`../fake-face-detection/fake_face_model`

If your model is elsewhere, set `FFD_MODEL_DIR` to:

- a SavedModel directory, or
- a `.keras` file path

```bash
export FFD_MODEL_DIR="/absolute/path/to/fake_face_model"
```

### Option B: Hugging Face model repo

Set your model repo:

```bash
export HF_MODEL_REPO="username/fake-face-model"
```

Current repo example:

```bash
export HF_MODEL_REPO="eslamaly/fake-face-xception-model"
```

The service will automatically load `xception_face_detector.keras` from that repo if present.

If the repo is private, also set:

```bash
export HF_TOKEN="hf_xxx"
```

## 3) Run API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### CORS configuration (recommended)

For local frontend on Vite:

```bash
export CORS_ALLOW_ORIGINS="http://localhost:5173"
export CORS_ALLOW_CREDENTIALS="false"
```

For production, set `CORS_ALLOW_ORIGINS` to your deployed client URL(s), comma-separated.

## 4) Endpoints

- `GET /health`
- `POST /predict` (multipart form-data with field name `image`)

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@/path/to/face.jpg"
```

Example response:

```json
{
  "label": "REAL",
  "confidence": 99.96,
  "probability_real": 99.96
}
```

`confidence` and `probability_real` are percentages (`0-100`).

Example JS (browser):

```js
const formData = new FormData();
formData.append("image", fileInput.files[0]);

const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const data = await res.json();
console.log(data); // { label, confidence, probability_real } percentages (0-100)
```

## 5) Docker

Build and run locally:

```bash
cd ffdapi
docker build -t ffdapi .
docker run --rm -p 7860:7860 \
  -e HF_MODEL_REPO="eslamaly/fake-face-xception-model" \
  -e HF_TOKEN="hf_xxx" \
  ffdapi
```

## 6) Deploy to Hugging Face Spaces (Docker)

1. Create a new Space on Hugging Face with SDK = `Docker`.
2. Push files from `ffdapi/` to that Space repo (`Dockerfile`, `main.py`, `requirements.txt`, `.dockerignore`).
3. In Space Settings -> Variables and secrets:
   - Variable: `HF_MODEL_REPO` = `eslamaly/fake-face-xception-model`
   - Secret (if private): `HF_TOKEN` = `hf_xxx`
4. Hugging Face will build and run the container automatically on port `7860`.

After deploy, your endpoint will be:

- `POST https://<your-space>.hf.space/predict`

Live example:

- `POST https://eslamaly-fake-face-detector-api.hf.space/predict`
