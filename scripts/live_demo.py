import cv2
import numpy as np
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline
from bagvision.depth_fusion import estimate_from_mask
from bagvision.color_verification import MIN_GREEN_RATIO, axis_pass_rates
from bagvision.depth_verification import MIN_BULGE_M

WEIGHTS_PATH = "models/bagvision-5/weights/best.pt"
VAL_MASK_MAP50 = 0.507  # bagvision-5 학습 시 valid셋 기준 mask mAP50 (고정값, 실시간 계산 아님)
WINDOW_NAME = "live_demo (c=capture, q=quit)"
DISPLAY_SCALE = 1.6  # 캡처 해상도(640x480)는 depth 정합용이라 그대로 두고 창만 확대
COLOR_FILTER_ENABLED = True  # False로 두면 2차 색상 검증 없이 이전 동작으로 롤백
DEPTH_FILTER_ENABLED = True  # False로 두면 depth 볼록도 검증 없이 롤백 (MIN_BULGE_M 아직 임시값)


def _bulge_text(bulge_height):
    if bulge_height is None:
        return "볼록도=? (depth 무효)"
    ok = "OK" if bulge_height >= MIN_BULGE_M else "부족"
    return f"볼록도 {bulge_height * 100:.1f}cm ({ok}, 기준 {MIN_BULGE_M * 100:.0f}cm)"


def _draw_evaluation(vis, evaluation, depth_image, intrinsics):
    det = evaluation.detection
    x1, y1, x2, y2 = det.bbox
    color = (0, 255, 0) if evaluation.verified else (0, 0, 255)

    if evaluation.verified:
        overlay = vis.copy()
        overlay[det.mask] = color
        vis[:] = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    hsv_text = ""
    if evaluation.mean_hsv is not None:
        h, s, v = evaluation.mean_hsv
        hsv_text = f" raw H={h:.0f} S={s:.0f} V={v:.0f}"

    if evaluation.verified:
        estimate = estimate_from_mask(depth_image, det.mask, intrinsics)
        pos_text = (
            f" xyz=({estimate.position_xyz[0]:.2f},{estimate.position_xyz[1]:.2f},{estimate.position_xyz[2]:.2f})m"
            if estimate else " xyz=?"
        )
        label = f"{det.class_name} {det.confidence * 100:.0f}%{pos_text}"
    else:
        label = "거부"

    cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(
        vis,
        f"색상 {evaluation.color_ratio * 100:.0f}% / {_bulge_text(evaluation.bulge_height)}{hsv_text}",
        (x1, min(vis.shape[0] - 5, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def overlay_evaluations(color_image, evaluations, depth_image, intrinsics):
    vis = color_image.copy()
    cv2.putText(
        vis,
        f"model valid mask mAP50: {VAL_MASK_MAP50:.3f}  (detections: {len(evaluations)})",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    for evaluation in evaluations:
        _draw_evaluation(vis, evaluation, depth_image, intrinsics)
    return vis


def main():
    camera = RealSenseCamera()
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.4)
    stabilizer = TargetStabilizer(window_size=15)
    pipeline = RecognitionPipeline(
        detector, camera, stabilizer,
        color_filter_enabled=COLOR_FILTER_ENABLED,
        depth_filter_enabled=DEPTH_FILTER_ENABLED,
    )

    camera.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
    print("c: 안정된 타겟 캡처, h: 색상 축별(H/S/V) 통과율 출력, q: 종료")
    try:
        while True:
            color, depth, intrinsics, evaluations = pipeline.evaluate_all()
            vis = overlay_evaluations(color, evaluations, depth, intrinsics)
            vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                try:
                    target = pipeline.capture_stable_target(max_frames=60)
                    print(f"확정된 타겟: pos={target.position_xyz}, size={target.size_estimate}, "
                          f"n_frames={target.n_frames_averaged}")
                except TimeoutError as exc:
                    print(f"캡처 실패: {exc}")
            elif key == ord("h"):
                for i, evaluation in enumerate(evaluations):
                    rates = axis_pass_rates(color, evaluation.detection.mask)
                    print(f"[detection {i}] bbox={evaluation.detection.bbox} "
                          f"mean_hsv={evaluation.mean_hsv} axis_pass_rates={rates}")
            elif key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
