from pathlib import Path
from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    def predict(self, image_path: str, confidence: float = 0.25):
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            save=False,
            verbose=False,
        )

        return results[0]