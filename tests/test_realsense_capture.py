from types import SimpleNamespace
from bagvision.realsense_capture import intrinsics_from_realsense


def test_intrinsics_from_realsense_converts_fields():
    rs_intrinsics = SimpleNamespace(fx=615.0, fy=615.0, ppx=319.5, ppy=239.5)

    intr = intrinsics_from_realsense(rs_intrinsics, depth_scale=0.001)

    assert intr.fx == 615.0
    assert intr.fy == 615.0
    assert intr.ppx == 319.5
    assert intr.ppy == 239.5
    assert intr.depth_scale == 0.001
