import cv2
import numpy as np
import pytest
from bagvision.types import CameraIntrinsics, Detection
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline

# color_verification의 HUE_RANGE(29~97) 안에 드는 초록색 (BGR)
GREEN_BGR = tuple(int(c) for c in cv2.cvtColor(
    np.array([[[60, 150, 150]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
)[0, 0])


class FakeCamera:
    def __init__(self, color, depth, intrinsics):
        self.color = color
        self.depth = depth
        self.intrinsics = intrinsics

    def get_frames(self):
        return self.color, self.depth, self.intrinsics


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, color_image):
        return self.detections


def make_fixture(mask_color=GREEN_BGR, bulge=True):
    intr = CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)
    if bulge:
        # 평균은 정확히 500(0.5m)이지만 위/아래 반씩 470/530으로 갈라 60mm(0.06m)
        # 볼록도를 만든다 -> depth 볼록도 검증(MIN_BULGE_M=0.05) 통과
        depth[200:240, 280:360] = 470
        depth[240:280, 280:360] = 530
    else:
        depth[200:280, 280:360] = 500  # 평평함 -> 볼록도 0
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:280, 280:360] = True
    color[mask] = mask_color
    detection = Detection(class_id=0, class_name="bag", confidence=0.9, mask=mask, bbox=(280, 200, 360, 280))
    return FakeCamera(color, depth, intr), FakeDetector([detection])


def test_step_returns_best_detection():
    camera, detector = make_fixture()
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    result = pipeline.step()

    assert result is not None
    assert result.class_name == "bag"


def test_step_returns_none_when_no_detections():
    camera, _ = make_fixture()
    detector = FakeDetector([])
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    assert pipeline.step() is None


def test_capture_stable_target_returns_averaged_result():
    camera, detector = make_fixture()
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    target = pipeline.capture_stable_target(max_frames=10)

    assert target.class_name == "bag"
    assert target.n_frames_averaged == 3
    assert target.position_xyz[2] == pytest.approx(0.5, abs=1e-6)


def test_capture_stable_target_times_out_when_never_ready():
    camera, _ = make_fixture()
    detector = FakeDetector([])  # never produces a detection
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    with pytest.raises(TimeoutError):
        pipeline.capture_stable_target(max_frames=5)


def test_step_rejects_detection_with_wrong_color():
    camera, detector = make_fixture(mask_color=(0, 0, 0))  # 검정 -> 초록 시그니처 아님
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    assert pipeline.step() is None


def test_step_rollback_disables_color_filter():
    camera, detector = make_fixture(mask_color=(0, 0, 0))
    pipeline = RecognitionPipeline(
        detector, camera, TargetStabilizer(window_size=3), color_filter_enabled=False
    )

    result = pipeline.step()

    assert result is not None
    assert result.class_name == "bag"


def test_step_rejects_flat_object_despite_correct_color():
    # 색은 맞지만(초록) 평평함(볼록도 0) -> 딱딱한 물체(쓰레받이 등)로 간주해 거부
    camera, detector = make_fixture(bulge=False)
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    assert pipeline.step() is None


def test_step_rollback_disables_depth_filter():
    camera, detector = make_fixture(bulge=False)
    pipeline = RecognitionPipeline(
        detector, camera, TargetStabilizer(window_size=3), depth_filter_enabled=False
    )

    result = pipeline.step()

    assert result is not None
    assert result.class_name == "bag"


def test_evaluate_all_reports_every_detection_independently():
    intr = CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)

    bag_mask = np.zeros((480, 640), dtype=bool)
    bag_mask[200:280, 280:360] = True
    color[bag_mask] = GREEN_BGR
    depth[200:240, 280:360] = 470
    depth[240:280, 280:360] = 530  # 볼록함 -> 통과

    dustpan_mask = np.zeros((480, 640), dtype=bool)
    dustpan_mask[100:150, 100:150] = True
    color[dustpan_mask] = (0, 0, 0)  # 색상 불일치
    depth[100:150, 100:150] = 500  # 평평함

    bag = Detection(class_id=0, class_name="bag", confidence=0.9, mask=bag_mask, bbox=(280, 200, 360, 280))
    dustpan = Detection(class_id=0, class_name="bag", confidence=0.7, mask=dustpan_mask, bbox=(100, 100, 150, 150))

    camera = FakeCamera(color, depth, intr)
    detector = FakeDetector([bag, dustpan])
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    _, _, _, evaluations = pipeline.evaluate_all()

    assert len(evaluations) == 2
    by_bbox = {e.detection.bbox: e for e in evaluations}
    assert by_bbox[(280, 200, 360, 280)].verified is True
    assert by_bbox[(100, 100, 150, 150)].verified is False
