import csv
import time
import cv2
import numpy as np
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline
from bagvision.depth_fusion import estimate_from_mask
from live_demo_night import NIGHT_COLOR_THRESHOLDS, FlickerFilter

WEIGHTS_PATH = "models/bagvision-5/weights/best.pt"
WINDOW_BEFORE = "before_verification (q=quit)"
WINDOW_AFTER = "after_verification (q=quit)"
DISPLAY_SCALE = 1.3
# RecognitionPipeline 기본값(valid_range 인자를 안 줬을 때)과 맞춰야
# "너무 멀음/가까움" 판정이 실제 검증 기준과 어긋나지 않는다.
DEPTH_VALID_RANGE = (0.3, 2.0)


def overlay_before(color_image, evaluations):
    vis = color_image.copy()
    cv2.putText(vis, "2차 검증 미적용", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    for evaluation in evaluations:
        det = evaluation.detection
        x1, y1, x2, y2 = det.bbox
        overlay = vis.copy()
        overlay[det.mask] = (0, 255, 0)
        vis[:] = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, f"{det.class_name} {det.confidence * 100:.0f}%",
            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )
    return vis


def _xyz_text(depth_image, mask, intrinsics):
    """검증 통과 여부와 무관하게 xyz(또는 사유)를 항상 문자열로 반환.

    estimate_from_mask는 유효 범위(DEPTH_VALID_RANGE) 안 픽셀이 하나도 없으면
    None을 주는데, 그게 "너무 멀어서/가까워서"인지 "센서 blind zone(z=0)"인지는
    구분 안 해줘서 여기서 raw 평균 z로 따로 판단한다.
    """
    estimate = estimate_from_mask(depth_image, mask, intrinsics, valid_range=DEPTH_VALID_RANGE)
    if estimate is not None:
        x, y, z = estimate.position_xyz
        return f" xyz=({x:.2f},{y:.2f},{z:.2f})m"

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return " xyz=?"
    z = depth_image[ys, xs].astype(np.float64) * intrinsics.depth_scale
    z = z[z > 0]
    if len(z) == 0:
        return " xyz=?(blind zone)"
    mean_z = float(z.mean())
    if mean_z > DEPTH_VALID_RANGE[1]:
        return f" xyz=?(너무 멀음 {mean_z:.2f}m)"
    return f" xyz=?(너무 가까움 {mean_z:.2f}m)"


def overlay_after(color_image, evaluations, depth_image, intrinsics):
    vis = color_image.copy()
    cv2.putText(vis, "2차 검증 적용 (색상+depth)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    for evaluation in evaluations:
        det = evaluation.detection
        x1, y1, x2, y2 = det.bbox
        color = (0, 255, 0) if evaluation.verified else (0, 0, 255)
        if evaluation.verified:
            overlay = vis.copy()
            overlay[det.mask] = color
            vis[:] = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if evaluation.mean_hsv is not None:
            _, s_value, v_value = evaluation.mean_hsv
            sv_text = f", S={s_value:.0f} V={v_value:.0f}"
        else:
            sv_text = ""
        pos_text = _xyz_text(depth_image, det.mask, intrinsics)
        if evaluation.verified:
            label = f"{det.class_name} {det.confidence * 100:.0f}%{pos_text} (통과{sv_text})"
        else:
            label = f"거부{pos_text} (색상 {evaluation.color_ratio * 100:.0f}%{sv_text})"
        cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def main():
    camera = RealSenseCamera()
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.4)
    pipeline = RecognitionPipeline(
        detector, camera, TargetStabilizer(window_size=15),
        color_filter_enabled=True, depth_filter_enabled=True,
        color_thresholds=NIGHT_COLOR_THRESHOLDS,
    )
    flicker_filter = FlickerFilter(miss_tolerance=5, iou_threshold=0.3)

    camera.start()
    cv2.namedWindow(WINDOW_BEFORE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_AFTER, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_BEFORE, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
    cv2.resizeWindow(WINDOW_AFTER, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
    cv2.moveWindow(WINDOW_BEFORE, 0, 0)
    cv2.moveWindow(WINDOW_AFTER, int(640 * DISPLAY_SCALE) + 20, 0)

    log_path = f"logs/compare_demo_{int(time.time())}.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    # ground_truth_label 없음(라벨 전제 없이 그냥 판정 원본만 남김) -> 나중에 영상이랑
    # 대조해서 직접 눈으로 정탐/오탐 판단. bulge_height가 None이면 그 프레임은 거리
    # 무효(통과/거부 상관없이 파이프라인이 이미 계산해주는 값이라 재계산 불필요).
    writer.writerow(["timestamp", "bbox", "verified", "color_ratio", "h", "s", "v", "bulge_height"])
    print(f"q: 종료 (로그: {log_path})")
    try:
        while True:
            color, depth, intrinsics, evaluations = pipeline.evaluate_all()
            display_evaluations = flicker_filter.update(evaluations)

            before = cv2.resize(overlay_before(color, display_evaluations), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            after = cv2.resize(
                overlay_after(color, display_evaluations, depth, intrinsics), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE
            )
            cv2.imshow(WINDOW_BEFORE, before)
            cv2.imshow(WINDOW_AFTER, after)

            now = time.time()
            for evaluation in evaluations:
                h, s, v = evaluation.mean_hsv if evaluation.mean_hsv is not None else (None, None, None)
                writer.writerow([
                    now, evaluation.detection.bbox, evaluation.verified,
                    evaluation.color_ratio, h, s, v, evaluation.bulge_height,
                ])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        log_file.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
