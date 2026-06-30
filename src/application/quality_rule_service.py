from ..domain.detection import Detection


class QualityRuleService:
    def decide(self, detections: list[Detection]) -> str:
        if not detections:
            return "NO_FRUIT_DETECTED"

        total = len(detections)
        defective = [
            detection
            for detection in detections
            if "rotten" in detection.label.lower()
            or "overripe" in detection.label.lower()
        ]

        defective_ratio = len(defective) / total

        if defective_ratio >= 0.2:
            return "REJECT_BATCH"

        return "ACCEPT_BATCH"
