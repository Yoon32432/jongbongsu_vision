import cv2
import numpy as np
import pytest
from bagvision.color_verification import (
    MIN_GREEN_RATIO,
    axis_pass_rates,
    green_signature_ratio,
    mean_hsv,
    verify_color_signature,
)


def make_solid_image(bgr, shape=(50, 50)):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[:] = bgr
    mask = np.ones(shape, dtype=bool)
    return image, mask


def hsv_to_bgr(h, s, v):
    pixel = np.array([[[h, s, v]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def test_green_signature_ratio_is_high_for_calibrated_green():
    image, mask = make_solid_image(hsv_to_bgr(60, 150, 150))

    assert green_signature_ratio(image, mask) == 1.0


def test_green_signature_ratio_is_zero_for_black():
    image, mask = make_solid_image((0, 0, 0))

    assert green_signature_ratio(image, mask) == 0.0


def test_green_signature_ratio_is_zero_for_unrelated_blue_object():
    image, mask = make_solid_image(hsv_to_bgr(120, 150, 150))  # 파랑 계열

    assert green_signature_ratio(image, mask) == 0.0


def test_green_signature_ratio_rejects_high_saturation_opaque_object():
    # 실측: 쓰레받이(불투명, S=216)가 봉투(S=166)보다 채도가 높게 나옴 -> 상한선으로 배제
    image, mask = make_solid_image(hsv_to_bgr(77, 216, 120))

    assert green_signature_ratio(image, mask) == 0.0


def test_green_signature_ratio_empty_mask_returns_zero():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    assert green_signature_ratio(image, mask) == 0.0


def test_verify_color_signature_passes_calibrated_green():
    image, mask = make_solid_image(hsv_to_bgr(60, 150, 150))

    assert verify_color_signature(image, mask) is True


def test_verify_color_signature_rejects_unrelated_object():
    image, mask = make_solid_image((0, 0, 0))

    assert verify_color_signature(image, mask) is False


def test_verify_color_signature_disabled_always_passes():
    image, mask = make_solid_image((0, 0, 0))

    assert verify_color_signature(image, mask, enabled=False) is True


def test_min_green_ratio_threshold_is_reasonable():
    # 완전 배제/완전 통과가 아니라 실제 캘리브레이션 여지를 남기는 값인지 확인
    assert 0.0 < MIN_GREEN_RATIO < 1.0


def test_mean_hsv_returns_expected_values():
    image, mask = make_solid_image(hsv_to_bgr(60, 150, 150))

    h, s, v = mean_hsv(image, mask)

    assert h == pytest.approx(60, abs=1)
    assert s == pytest.approx(150, abs=1)
    assert v == pytest.approx(150, abs=1)


def test_mean_hsv_empty_mask_returns_none():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    assert mean_hsv(image, mask) is None


def test_axis_pass_rates_all_pass_for_calibrated_green():
    image, mask = make_solid_image(hsv_to_bgr(60, 150, 150))

    rates = axis_pass_rates(image, mask)

    assert rates == {"hue": 1.0, "saturation": 1.0, "value": 1.0, "all": 1.0}


def test_axis_pass_rates_identifies_saturation_as_bottleneck():
    # H=89(범위 내), S=216(상한 190 초과) -> saturation만 막힘
    image, mask = make_solid_image(hsv_to_bgr(89, 216, 120))

    rates = axis_pass_rates(image, mask)

    assert rates["hue"] == 1.0
    assert rates["saturation"] == 0.0
    assert rates["all"] == 0.0


def test_axis_pass_rates_empty_mask_returns_none():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    assert axis_pass_rates(image, mask) is None
