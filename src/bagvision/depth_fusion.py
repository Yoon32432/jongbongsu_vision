import numpy as np
from bagvision.types import CameraIntrinsics, DepthEstimate


def estimate_from_mask(
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    valid_range: tuple[float, float] = (0.3, 2.0),
    floor_z: float | None = None,
    floor_margin: float = 0.02,
) -> DepthEstimate | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    z = depth_image[ys, xs].astype(np.float64) * intrinsics.depth_scale
    valid = (z >= valid_range[0]) & (z <= valid_range[1])

    if floor_z is not None:
        valid &= z < (floor_z - floor_margin)

    if not np.any(valid):
        return None

    xs, ys, z = xs[valid], ys[valid], z[valid]
    x = (xs - intrinsics.ppx) * z / intrinsics.fx
    y = (ys - intrinsics.ppy) * z / intrinsics.fy

    position = (float(np.mean(x)), float(np.mean(y)), float(np.mean(z)))
    size = (
        float(x.max() - x.min()),
        float(y.max() - y.min()),
        float(z.max() - z.min()),
    )
    return DepthEstimate(position_xyz=position, size_estimate=size, n_points=int(len(z)))
