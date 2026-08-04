from bagvision.depth_fusion import estimate_from_mask
from bagvision.types import CameraIntrinsics

# 색상과 달리 오프라인 데이터셋에 depth 채널이 없어 실측 캘리브레이션이 불가능했다.
# 실제 봉투/오탐 후보(쓰레받이 등)를 라이브로 띄워 bulge_height 값을 비교해보고
# 그 사이 지점으로 조정할 것 (임시값).
MIN_BULGE_M = 0.05


def bulge_height(
    depth_image,
    mask,
    intrinsics: CameraIntrinsics,
    valid_range: tuple[float, float] = (0.3, 2.0),
    floor_z: float | None = None,
) -> float | None:
    estimate = estimate_from_mask(depth_image, mask, intrinsics, valid_range=valid_range, floor_z=floor_z)
    if estimate is None:
        return None
    return estimate.size_estimate[2]


def verify_depth_bulge(
    depth_image,
    mask,
    intrinsics: CameraIntrinsics,
    valid_range: tuple[float, float] = (0.3, 2.0),
    floor_z: float | None = None,
    enabled: bool = True,
) -> bool:
    if not enabled:
        return True
    height = bulge_height(depth_image, mask, intrinsics, valid_range, floor_z)
    if height is None:
        # depth 무효(blind zone 등) -> 색상 검증 결과만으로 판단, 이 단계는 통과 처리
        return True
    return height >= MIN_BULGE_M
