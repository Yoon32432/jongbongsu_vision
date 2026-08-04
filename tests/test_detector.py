import os
import numpy as np
import pytest
from bagvision.detector import BagDetector


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf


class FakeMasks:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class FakeModel:
    names = {0: "bag"}

    def __init__(self, result):
        self._result = result

    def predict(self, image, conf, verbose):
        return [self._result]


def test_detect_parses_single_detection():
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    small_mask = np.zeros((20, 20), dtype=np.float32)
    small_mask[5:15, 5:15] = 1.0  # small mask, will be resized to 100x100

    result = FakeResult(
        boxes=FakeBoxes(
            xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
            cls=np.array([0.0]),
            conf=np.array([0.87]),
        ),
        masks=FakeMasks(data=np.array([small_mask])),
    )
    detector = BagDetector(model=FakeModel(result))

    detections = detector.detect(color_image)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "bag"
    assert det.confidence == 0.87
    assert det.mask.shape == (100, 100)
    assert det.mask.dtype == bool
    assert det.mask.sum() > 0


def test_detect_returns_empty_list_when_no_masks():
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = FakeResult(boxes=None, masks=None)
    detector = BagDetector(model=FakeModel(result))

    assert detector.detect(color_image) == []


WEIGHTS_PATH = "models/bagvision-4/weights/best.pt"


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="학습된 가중치 없음 (Task 9 선행 필요)")
def test_detect_with_real_weights_runs_without_error():
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.3)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(dummy_image)

    assert isinstance(detections, list)
