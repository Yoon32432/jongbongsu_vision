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

# 야외/야간 재캘리브레이션 (2026-08-05, logs/hsv_tuning_1785929165.csv 라벨링 로그 기반).
# bag 303 / other 91 detection의 평균 H/S/V로 근사 스윕한 값 -> 표본이 적어(오탐
# 후보 사실상 1~2종) 확정치는 아니고 시작점. SATURATION_MAX만 200->190으로 낮추고
# VALUE_RANGE 하한만 40->28로 낮추면 bag 통과율 0.83->0.89, other 통과율
# 0.23->0.08로 재현율/오탐배제율이 같이 개선됨. HUE_RANGE/SATURATION_MIN은
# 주간 값 그대로 사용.
#
# 전조등 버전 추가 반영 (2026-08-11, logs/hsv_tuning_1786451045.csv, bag 193 /
# other 21). 전조등 직사광 때문에 쓰레받이(other) V가 무광 야간 대비 크게
# 뜀(V 평균 31.8 -> 160.4, bag과 겹침) -> 기존 VALUE_RANGE 상한 250 그대로 두면
# other 통과율이 0.08 -> 0.57까지 치솟음. 상한을 200으로 낮추면 두 세션 모두에서
# bag 통과율(0.89/0.82)은 거의 안 건드리면서 other 통과율을 0.08/0.43으로 낮춤.
# 다만 전조등 조건에서 other 0.43은 여전히 높음 -> 헤드라이트 하에서는 H/S/V
# 색상만으로는 쓰레받이와 구분이 잘 안 됨(H 평균 77.5 vs 80.1로 거의 동일).
# 전조등 데모면 depth 검증(bulge_height) 재활성화나 별도 보완책을 검토할 것.
NIGHT_SATURATION_MAX = 150
NIGHT_VALUE_RANGE = (20, 220)

# S값만 재조정 (2026-08-13, hsv_tuning 로그 4개 합산: bag 619 / other 171, 평균
# saturation 기준). 190에서는 other 통과율이 46.8%(bag 90.5%)까지 새고 있었음 ->
# 문턱을 150~230 범위로 스윕해 bag_pass-other_pass(Youden's J) 최대점을 찾으니
# 175에서 J 최대(bag 78.8% / other 9.9%). bag 재현율은 12%p 낮아지지만 오탐이
# 46.8%->9.9%로 크게 줄어듦. H/V는 이번엔 안 건드림 -> 다음 재라벨링에서 재검증할 것.

# 실사용 라이브 테스트(2026-08-05, live_demo_night.py, 봉투2+쓰레받기1)에서 실측한
# green_signature_ratio(all): 흰색 내용물 봉투 0.683(통과) vs 파란색 내용물 봉투
# 0.345(거부, 병목은 value 44%/saturation 65%) vs 쓰레받기 0.155(정상 거부,
# saturation 30%). MIN_GREEN_RATIO=0.5는 파란 봉투와 쓰레받기 사이가 아니라
# 파란 봉투보다 위에 있어 재현율을 깎고 있었음. 0.345와 0.155 사이 여유를 이용해
# 0.3으로 낮춤 -> 내용물 색이 진한(반투명 비닐 색이 크게 섞이는) 봉투의 재현율을
# 높이면서 쓰레받기는 여전히 확실히 배제. 표본 1개씩이라 계속 재검증 필요.
NIGHT_MIN_GREEN_RATIO = 0.3

# 전조등 밝기 낮춘 뒤 재라벨링 검증 (2026-08-12, logs/hsv_tuning_1786538973.csv +
# 1786539527.csv, bag 123 / other 59). NIGHT_SATURATION_MAX=190/NIGHT_VALUE_RANGE=
# (28,200) 그대로도 bag 통과율 85%, other 통과율 5%로 나옴 (전조등 안 낮췄을 때
# other 43%였던 것 대비 큰 개선) -> 값 변경 불필요, 헤드라이트 직사광 문제는
# 밝기 조절만으로 해소됨. bulge_height도 bag 평균 24.3cm/other 28.1cm로 안 겹쳐서
# depth 보완책은 당장 불필요.
#
# VALUE_RANGE 하한 28->25->20 수동 조정 (2026-08-12). 재측정으로 검증된 값은 아님 ->
# 다음 재라벨링에서 bag/other 통과율 재확인할 것.


# 무더기로 쌓인 봉투는 옆 봉투/그림자와 맞닿은 마스크 가장자리 몇 픽셀이 실측보다
# 채도가 크게 낮게(그림자) 또는 튀게(하이라이트) 나와 판정을 오염시킴 (2026-08-19,
# compare_demo 실측: 프레임당 봉투 1개 통과율 87% vs 7개 이상 70%, 거부 사유의
# 55%가 채도 하한 미달). 판정용 픽셀만 마스크를 안쪽으로 침식시켜 이 오염된
# 경계를 빼고 본다 -> 탐지 박스/마스크 자체(화면 표시, xyz 추정)는 그대로 둔다.
_EROSION_KERNEL = np.ones((5, 5), np.uint8)


def _eroded_mask(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), _EROSION_KERNEL, iterations=1) > 0
    # 마스크가 원래 작아서 침식시키면 통째로 사라지면(가는 봉투 끝자락 등)
    # 판정 불능이 되므로 그럴 때만 원본 마스크로 되돌아간다.
    return eroded if np.any(eroded) else mask


def mean_hsv(color_image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    if not np.any(mask):
        return None
    mask = _eroded_mask(mask)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]
    return float(h.mean()), float(s.mean()), float(v.mean())


def axis_pass_rates(
    color_image: np.ndarray,
    mask: np.ndarray,
    hue_range: tuple[float, float] = HUE_RANGE,
    saturation_range: tuple[float, float] = (SATURATION_MIN, SATURATION_MAX),
    value_range: tuple[float, float] = VALUE_RANGE,
) -> dict | None:
    """디버그용: H/S/V 각 조건을 개별로 봤을 때 통과하는 픽셀 비율.

    전체 통과율(green_signature_ratio)이 왜 낮은지(어느 축이 병목인지)를
    파악하기 위한 것 -> 어느 임계값을 조정할지 감이 아니라 데이터로 정하기 위함.
    """
    if not np.any(mask):
        return None
    mask = _eroded_mask(mask)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]

    h_ok = (h >= hue_range[0]) & (h <= hue_range[1])
    s_ok = (s >= saturation_range[0]) & (s <= saturation_range[1])
    v_ok = (v >= value_range[0]) & (v <= value_range[1])
    return {
        "hue": float(h_ok.mean()),
        "saturation": float(s_ok.mean()),
        "value": float(v_ok.mean()),
        "all": float((h_ok & s_ok & v_ok).mean()),
    }


def green_signature_ratio(
    color_image: np.ndarray,
    mask: np.ndarray,
    hue_range: tuple[float, float] = HUE_RANGE,
    saturation_range: tuple[float, float] = (SATURATION_MIN, SATURATION_MAX),
    value_range: tuple[float, float] = VALUE_RANGE,
) -> float:
    if not np.any(mask):
        return 0.0

    mask = _eroded_mask(mask)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]

    in_range = (
        (h >= hue_range[0]) & (h <= hue_range[1])
        & (s >= saturation_range[0]) & (s <= saturation_range[1])
        & (v >= value_range[0]) & (v <= value_range[1])
    )
    return float(in_range.mean())


def verify_color_signature(
    color_image: np.ndarray,
    mask: np.ndarray,
    enabled: bool = True,
    hue_range: tuple[float, float] = HUE_RANGE,
    saturation_range: tuple[float, float] = (SATURATION_MIN, SATURATION_MAX),
    value_range: tuple[float, float] = VALUE_RANGE,
    min_green_ratio: float = MIN_GREEN_RATIO,
) -> bool:
    if not enabled:
        return True
    return green_signature_ratio(color_image, mask, hue_range, saturation_range, value_range) >= min_green_ratio
