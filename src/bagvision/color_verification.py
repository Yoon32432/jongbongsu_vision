import cv2
import numpy as np

# data/garbage bag.v3i.yolo26/train 라벨 마스크 223장에서 실측 (그림자 V<=40,
# 하이라이트 S<=40 픽셀 제외 후 P5~P95). 세션이 늘어나면 재캘리브레이션 대상.
#
# HUE_RANGE 상한은 이후 실측(axis_pass_rates)에서 97로는 실제 봉투 자체 픽셀의
# 40%가 넘게 잘려나가는 게 확인돼(봉투 평균 H=87, hue 통과율 60%) 110으로
# 완화. 쓰레받이(오탐 후보)는 평균 H=77로 hue만으로는 애초에 구별 안 되므로
# (hue 통과율 100%) 완화해도 위험 증가는 없음.
HUE_RANGE = (29, 110)
SATURATION_MIN = 40
VALUE_RANGE = (40, 250)
MIN_GREEN_RATIO = 0.5

# 임시값: 실내 조명에서 실측(봉투 S=166~170 vs 쓰레받이 S=216~226)의 간격을
# 이용한 상한선. 190으로는 봉투 자신의 픽셀 45%가 잘려 통과율이 42%까지
# 떨어져(기준 50% 미달) 200으로 완화. 쓰레받이는 190 기준 통과율이 이미
# 4~5%로 낮아 여유가 있어 200으로 올려도 크게 새지 않을 것으로 예상.
# 다른 조명(야간/야외)에서 검증 전이라 대회 영상용 임시치 -> 계속 재측정하며 다듬을 것.
SATURATION_MAX = 200


def mean_hsv(color_image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    if not np.any(mask):
        return None
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]
    return float(h.mean()), float(s.mean()), float(v.mean())


def axis_pass_rates(color_image: np.ndarray, mask: np.ndarray) -> dict | None:
    """디버그용: H/S/V 각 조건을 개별로 봤을 때 통과하는 픽셀 비율.

    전체 통과율(green_signature_ratio)이 왜 낮은지(어느 축이 병목인지)를
    파악하기 위한 것 -> 어느 임계값을 조정할지 감이 아니라 데이터로 정하기 위함.
    """
    if not np.any(mask):
        return None
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]

    h_ok = (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1])
    s_ok = (s >= SATURATION_MIN) & (s <= SATURATION_MAX)
    v_ok = (v >= VALUE_RANGE[0]) & (v <= VALUE_RANGE[1])
    return {
        "hue": float(h_ok.mean()),
        "saturation": float(s_ok.mean()),
        "value": float(v_ok.mean()),
        "all": float((h_ok & s_ok & v_ok).mean()),
    }


def green_signature_ratio(color_image: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]

    in_range = (
        (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1])
        & (s >= SATURATION_MIN) & (s <= SATURATION_MAX)
        & (v >= VALUE_RANGE[0]) & (v <= VALUE_RANGE[1])
    )
    return float(in_range.mean())


def verify_color_signature(color_image: np.ndarray, mask: np.ndarray, enabled: bool = True) -> bool:
    if not enabled:
        return True
    return green_signature_ratio(color_image, mask) >= MIN_GREEN_RATIO
