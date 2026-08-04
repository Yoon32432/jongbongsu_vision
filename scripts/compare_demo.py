import csv
import time
import cv2
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline

WEIGHTS_PATH = "models/bagvision-5/weights/best.pt"
WINDOW_BEFORE = "2차 검증 미적용 (q=종료)"
WINDOW_AFTER = "2차 검증 적용 (q=종료)"
DISPLAY_SCALE = 1.6

# 실행할 때마다 라벨(예: "genuine_bag" 또는 "impostor")을 바꿔서, 한 번은 진짜
# 봉투만, 한 번은 오탐 후보만 화면에 두고 따로따로 돌린다. 그러면 이 CSV의
# verified 컬럼 통계가 곧바로 그 클래스의 정탐률/오탐배제율이 된다.
GROUND_TRUTH_LABEL = "impostor"  # 또는 "genuine_bag"
LOG_PATH = f"logs/verification_log_{GROUND_TRUTH_LABEL}_{int(time.time())}.csv"


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


def overlay_after(color_image, evaluations):
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
        s_value = evaluation.mean_hsv[1] if evaluation.mean_hsv is not None else None
        s_text = f", S={s_value:.0f}" if s_value is not None else ""
        if evaluation.verified:
            label = f"{det.class_name} {det.confidence * 100:.0f}% (통과{s_text})"
        else:
            label = f"거부 (색상 {evaluation.color_ratio * 100:.0f}%{s_text})"
        cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def main():
    camera = RealSenseCamera()
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.4)
    pipeline = RecognitionPipeline(
        detector, camera, TargetStabilizer(window_size=15),
        color_filter_enabled=True, depth_filter_enabled=True,
    )

    camera.start()
    cv2.namedWindow(WINDOW_BEFORE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_AFTER, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_BEFORE, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
    cv2.resizeWindow(WINDOW_AFTER, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
    cv2.moveWindow(WINDOW_BEFORE, 0, 0)
    cv2.moveWindow(WINDOW_AFTER, int(640 * DISPLAY_SCALE) + 20, 0)

    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "ground_truth_label", "bbox", "verified", "color_ratio", "h", "s", "v", "bulge_height"])
    print(f"q: 종료 (로그 저장 경로: {LOG_PATH}, ground_truth_label={GROUND_TRUTH_LABEL})")
    try:
        while True:
            color, depth, intrinsics, evaluations = pipeline.evaluate_all()

            before = cv2.resize(overlay_before(color, evaluations), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            after = cv2.resize(overlay_after(color, evaluations), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow(WINDOW_BEFORE, before)
            cv2.imshow(WINDOW_AFTER, after)

            now = time.time()
            for evaluation in evaluations:
                h, s, v = evaluation.mean_hsv if evaluation.mean_hsv is not None else (None, None, None)
                writer.writerow([
                    now, GROUND_TRUTH_LABEL, evaluation.detection.bbox, evaluation.verified,
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
