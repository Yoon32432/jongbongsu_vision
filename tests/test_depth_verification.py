import numpy as np
import pytest
from bagvision.types import CameraIntrinsics
from bagvision.depth_verification import MIN_BULGE_M, bulge_height, verify_depth_bulge


def make_intrinsics():
    return CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)


def make_mask_region():
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:280, 280:360] = True
    return mask


def test_bulge_height_zero_for_flat_surface():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = make_mask_region()
    depth[mask] = 500  # 평평한 0.5m 표면 (쓰레받이 등 딱딱한 물체)

    assert bulge_height(depth, mask, intr) == pytest.approx(0.0, abs=1e-9)


def test_bulge_height_reflects_filled_bag_shape():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = make_mask_region()
    depth[200:240, 280:360] = 470
    depth[240:280, 280:360] = 530  # 60mm 굴곡

    assert bulge_height(depth, mask, intr) == pytest.approx(0.06, abs=1e-6)


def test_bulge_height_none_when_depth_invalid():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)  # 전부 0 -> valid_range 밖
    mask = make_mask_region()

    assert bulge_height(depth, mask, intr) is None


def test_verify_depth_bulge_rejects_flat_object():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = make_mask_region()
    depth[mask] = 500

    assert verify_depth_bulge(depth, mask, intr) is False


def test_verify_depth_bulge_accepts_bulging_bag():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = make_mask_region()
    depth[200:240, 280:360] = 470
    depth[240:280, 280:360] = 530

    assert verify_depth_bulge(depth, mask, intr) is True


def test_verify_depth_bulge_passes_through_when_depth_invalid():
    # blind zone 등으로 depth 무효 -> 이 단계는 통과, 색상 결과에 맡긴다
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = make_mask_region()

    assert verify_depth_bulge(depth, mask, intr) is True


def test_verify_depth_bulge_disabled_always_passes():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = make_mask_region()
    depth[mask] = 500  # 평평함

    assert verify_depth_bulge(depth, mask, intr, enabled=False) is True


def test_min_bulge_threshold_is_positive():
    assert MIN_BULGE_M > 0.0
