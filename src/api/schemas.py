from pydantic import BaseModel


class DetectionResponse(BaseModel):
    label: str
    confidence: float
    bbox: list[float]


class PredictionResponse(BaseModel):
    status: str
    total_detections: int
    summary: dict[str, int]
    quality_decision: str
    detections: list[DetectionResponse]