import numpy as np
import cv2
from bagvision.types import Detection


def _to_numpy(value):
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class BagDetector:
    def __init__(
        self,
        weights_path: str | None = None,
        conf_threshold: float = 0.5,
        model=None,
        imgsz: int | None = None,
        device: str | None = None,
        iou_threshold: float = 0.5,
    ):
        if model is not None:
            self.model = model
        else:
            # matplotlib 기본 백엔드가 TkAgg면 ultralytics import 시점에 빈 Tk 창이
            # 하나 떠서(headless로 안 씀에도) 화면에 뜬다 -> YOLO import 전에
            # 비GUI 백엔드로 고정.
            import matplotlib
            matplotlib.use("Agg")
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        # ultralytics 기본 NMS iou=0.7은 헐거워서 같은 물체에 박스 두 개가
        # 겹친 채로 살아남는 경우가 잦았음 -> 더 빡빡하게 눌러서 하나만 남긴다.
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' 요청했지만 GPU 없음 (torch.cuda.is_available()==False)")
        self.device = device

    def detect(self, color_image: np.ndarray) -> list[Detection]:
        kwargs = {"imgsz": self.imgsz} if self.imgsz else {}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(
            color_image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False, **kwargs
        )[0]
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
