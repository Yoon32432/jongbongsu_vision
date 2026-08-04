import numpy as np
import pytest
from bagvision.types import CameraIntrinsics
from bagvision.depth_fusion import estimate_from_mask


def make_intrinsics():
    return CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)


def test_estimate_from_mask_flat_region():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    # rows 200..279, cols 280..359, depth raw=500 -> 0.5m
    depth[200:280, 280:360] = 500
    mask[200:280, 280:360] = True

    est = estimate_from_mask(depth, mask, intr)

    assert est is not None
    assert est.n_points == 80 * 80
    assert est.position_xyz[2] == pytest.approx(0.5, abs=1e-6)
    assert est.position_xyz[0] == pytest.approx(-0.0004167, abs=1e-3)
    assert est.position_xyz[1] == pytest.approx(-0.0004167, abs=1e-3)
    assert est.size_estimate[0] == pytest.approx(0.06583, abs=1e-3)
    assert est.size_estimate[1] == pytest.approx(0.06583, abs=1e-3)
    assert est.size_estimate[2] == pytest.approx(0.0, abs=1e-9)


def test_estimate_from_mask_empty_mask_returns_none():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)

    assert estimate_from_mask(depth, mask, intr) is None


def test_estimate_from_mask_filters_out_of_range_depth():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    # valid region: 0.5m
    depth[200:220, 280:300] = 500
    mask[200:220, 280:300] = True
    # invalid region: 0.05m (too close, below default valid_range min 0.3m)
    depth[400:420, 280:300] = 50
    mask[400:420, 280:300] = True

    est = estimate_from_mask(depth, mask, intr, valid_range=(0.3, 2.0))

    assert est is not None
    assert est.n_points == 20 * 20
    assert est.position_xyz[2] == pytest.approx(0.5, abs=1e-6)


def test_estimate_from_mask_filters_floor_bleed():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    # bag region: 0.5m
    depth[200:220, 280:300] = 500
    mask[200:220, 280:300] = True
    # floor-bleed region at mask edge: 0.9m, close to floor_z=0.9
    depth[220:225, 280:300] = 900
    mask[220:225, 280:300] = True

    est = estimate_from_mask(depth, mask, intr, valid_range=(0.3, 2.0), floor_z=0.9, floor_margin=0.05)

    assert est is not None
    assert est.n_points == 20 * 20
    assert est.position_xyz[2] == pytest.approx(0.5, abs=1e-6)
