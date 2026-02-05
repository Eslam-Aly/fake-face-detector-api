from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .preprocessing import load_and_preprocess_image
from .inference import predict_image

app = FastAPI(title="Fake Face Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://fake-face-detector-client.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/detect")
async def detect(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG/PNG/WEBP.")

    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        batch = load_and_preprocess_image(file_bytes, 299)
        return predict_image(batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))