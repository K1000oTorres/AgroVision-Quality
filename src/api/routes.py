import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..api.schemas import PredictionResponse
from ..application.inference_service import InferenceService
from ..infrastructure.yolo_model import YoloModel

router = APIRouter()

MODEL_PATH = "models/best.pt"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

model = YoloModel(MODEL_PATH)
inference_service = InferenceService(model)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.1, le=1.0),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed.",
        )

    file_extension = Path(file.filename or "").suffix or ".jpg"
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return inference_service.predict(
            image_path=str(temp_path),
            confidence=confidence,
        )

    finally:
        await file.close()
        if temp_path.exists():
            temp_path.unlink()
