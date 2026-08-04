import numpy as np
import cv2
from bagvision.types import Detection


def _to_numpy(value):
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class BagDetector:
    def __init__(self, weights_path: str | None = None, conf_threshold: float = 0.5, model=None):
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, color_image: np.ndarray) -> list[Detection]:
        results = self.model.predict(color_image, conf=self.conf_threshold, verbose=False)[0]
        if results.masks is None:
            return []

        h, w = color_image.shape[:2]
        boxes = _to_numpy(results.boxes.xyxy)
        classes = _to_numpy(results.boxes.cls)
        confs = _to_numpy(results.boxes.conf)
        masks = _to_numpy(results.masks.data)

        detections = []
        for box, cls, conf, raw_mask in zip(boxes, classes, confs, masks):
            resized = cv2.resize(raw_mask.astype(np.float32), (w, h)) > 0.5
            detections.append(
                Detection(
                    class_id=int(cls),
                    class_name=self.model.names[int(cls)],
                    confidence=float(conf),
                    mask=resized,
                    bbox=tuple(int(v) for v in box),
                )
            )
        return detections
