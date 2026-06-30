from ..application.quality_rule_service import QualityRuleService
from ..domain.detection import Detection
from ..infrastructure.yolo_model import YoloModel


class InferenceService:
    def __init__(self, model: YoloModel):
        self.model = model
        self.quality_rules = QualityRuleService()

    def predict(self, image_path: str, confidence: float = 0.25) -> dict:
        result = self.model.predict(image_path=image_path, confidence=confidence)

        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    label=label,
                    confidence=round(conf, 4),
                    bbox=[round(value, 2) for value in bbox],
                )
            )

        summary: dict[str, int] = {}

        for detection in detections:
            summary[detection.label] = summary.get(detection.label, 0) + 1

        quality_decision = self.quality_rules.decide(detections)

        return {
            "status": "success",
            "total_detections": len(detections),
            "summary": summary,
            "quality_decision": quality_decision,
            "detections": [detection.model_dump() for detection in detections],
        }
