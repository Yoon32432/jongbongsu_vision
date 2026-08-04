import numpy as np
import pytest
from bagvision.types import Detection, DepthEstimate
from bagvision.stabilizer import TargetStabilizer


def make_detection(conf=0.9):
    mask = np.zeros((10, 10), dtype=bool)
    return Detection(class_id=0, class_name="bag", confidence=conf, mask=mask, bbox=(0, 0, 5, 5))


def test_not_ready_before_window_filled():
    stab = TargetStabilizer(window_size=3)
    assert stab.is_ready() is False
    with pytest.raises(RuntimeError):
        stab.capture(frame_id="camera_optical_frame")


def test_ready_after_window_filled_and_capture_averages():
    stab = TargetStabilizer(window_size=3)
    positions = [(0.0, 0.0, 0.5), (0.02, 0.0, 0.5), (-0.02, 0.0, 0.5)]
    for pos in positions:
        est = DepthEstimate(position_xyz=pos, size_estimate=(0.1, 0.1, 0.0), n_points=50)
        stab.add_frame(make_detection(), est)

    assert stab.is_ready() is True
    target = stab.capture(frame_id="camera_optical_frame")
    assert target.position_xyz[0] == pytest.approx(0.0, abs=1e-9)
    assert target.n_frames_averaged == 3
    assert target.class_name == "bag"


def test_none_estimates_are_ignored():
    stab = TargetStabilizer(window_size=2)
    stab.add_frame(make_detection(), None)
    stab.add_frame(make_detection(), None)
    assert stab.is_ready() is False


def test_window_slides_and_drops_oldest():
    stab = TargetStabilizer(window_size=2)
    est1 = DepthEstimate(position_xyz=(0.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.0), n_points=10)
    est2 = DepthEstimate(position_xyz=(1.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.0), n_points=10)
    est3 = DepthEstimate(position_xyz=(2.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.0), n_points=10)
    stab.add_frame(make_detection(), est1)
    stab.add_frame(make_detection(), est2)
    stab.add_frame(make_detection(), est3)  # est1 should be dropped

    target = stab.capture(frame_id="camera_optical_frame")
    assert target.position_xyz[0] == pytest.approx(1.5, abs=1e-9)
