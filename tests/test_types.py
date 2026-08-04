import numpy as np
from bagvision.types import CameraIntrinsics, Detection, DepthEstimate, StableTarget


def test_camera_intrinsics_fields():
    intr = CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    assert intr.fx == 600.0
    assert intr.depth_scale == 0.001


def test_detection_fields():
    mask = np.zeros((10, 10), dtype=bool)
    det = Detection(class_id=0, class_name="bag", confidence=0.9, mask=mask, bbox=(0, 0, 5, 5))
    assert det.class_name == "bag"
    assert det.mask.shape == (10, 10)


def test_depth_estimate_and_stable_target_fields():
    est = DepthEstimate(position_xyz=(0.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.02), n_points=100)
    assert est.n_points == 100

    target = StableTarget(
        class_name="bag",
        confidence=0.9,
        position_xyz=(0.0, 0.0, 0.5),
        size_estimate=(0.1, 0.1, 0.02),
        frame_id="camera_optical_frame",
        timestamp=123.0,
        n_frames_averaged=15,
    )
    assert target.n_frames_averaged == 15
