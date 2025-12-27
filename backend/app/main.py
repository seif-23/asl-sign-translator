from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # asl-sign-translator

MODEL_PATH = PROJECT_ROOT / "models" / "asl_finetuned_nomix_best.pth"

from backend.app.inference import ASLSignRecognizer

app = FastAPI(title="ASL Sign Translator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer: ASLSignRecognizer | None = None


@app.on_event("startup")
def load_model_on_startup():
    global recognizer
    print("[INFO] Loading ASL model...")

    try:
        recognizer = ASLSignRecognizer(
            model_path=str(MODEL_PATH).strip(),
            topk=5,
            num_clips=12,              # improved
            use_flip_tta=True,
            temperature=1.0,
            deterministic=True,        # improved
            motion_window_factor=2.5,  # improved
        )
        print("[INFO] Model loaded and ready 🚀")
    except Exception as e:
        recognizer = None
        print("[ERROR] Model failed to load!")
        print(str(e))


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "message": "Backend is running",
        "model_loaded": recognizer is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    global recognizer
    if recognizer is None:
        return {"error": "Model not loaded. Check console logs."}

    suffix = os.path.splitext(file.filename)[-1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        tmp.write(await file.read())

    try:
        label, confidence, top5 = recognizer.predict_video(temp_path)
        return {
            "prediction": label,
            "confidence": confidence,
            "top5": top5
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
