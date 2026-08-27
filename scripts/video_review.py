"""
녹화된 영상을 프레임 단위로 재생하며 2차 검증(색상+depth) 통과/거부 여부를
눈으로 직접 판단하고, g(=genuine_bag)/x(=impostor) 키로 그때그때 라벨을 매겨
영상 하나 다 보고 나면 정확도(재현율/오탐율)를 바로 출력한다. compare_demo.py
처럼 genuine_bag용/impostor용으로 스크립트를 따로 두 번 실행할 필요 없이,
영상 하나 안에 섞여 있는 장면을 라벨만 바꿔가며 한 번에 훑는다.

영상에는 depth 채널이 없어 항상 무효(z=0) depth를 흘려보낸다 -> bulge_height는
늘 None이 되어 verify_depth_bulge가 자동 통과 처리(depth_verification.py 참고)
하고, 실질적으로 색상 검증만으로 판정된다.

사용법:
    python scripts/video_review.py [영상 경로 (기본값: night_outdoor_rec1.mp4)]
"""
import csv
import sys
import time
import cv2
import numpy as np
from bagvision.detector import BagDetector
from bagvision.pipeline import RecognitionPipeline
from bagvision.stabilizer import TargetStabilizer
from bagvision.types import CameraIntrinsics
from live_demo_night import NIGHT_COLOR_THRESHOLDS, FlickerFilter

WEIGHTS_PATH = "models/bagvision-5/weights/best.pt"
WINDOW_NAME = "video_review (g=genuine_bag x=impostor u=unlabeled space=pause q=quit)"
DISPLAY_SCALE = 1.6
DEFAULT_VIDEO_PATH = "night_outdoor_rec1.mp4"


def stub_depth(color_image: np.ndarray) -> np.ndarray:
    """영상엔 depth 채널이 없어 늘 무효(0)인 depth 프레임을 대신 흘려보낸다."""
    return np.zeros(color_image.shape[:2], dtype=np.uint16)


class VideoFrameSource:
    """녹화 영상을 RecognitionPipeline이 기대하는 camera 인터페이스로 감싼다."""

    def __init__(self, path: str):
        self.path = path
        self._cap = None
        self._intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, ppx=0.0, ppy=0.0, depth_scale=1.0)

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없음: {self.path}")

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frames(self):
        ok, color = self._cap.read()
        if not ok:
            raise StopIteration
        return color, stub_depth(color), self._intrinsics


def overlay_evaluations(color_image, evaluations):
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
        if evaluation.verified:
            text = f"{det.class_name} {det.confidence * 100:.0f}% (통과{sv_text})"
        else:
            text = f"거부 (색상 {evaluation.color_ratio * 100:.0f}%{sv_text})"
        cv2.putText(vis, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def summarize(counts: dict) -> str:
    lines = []
    tp = fn = tn = fp = 0
    for label, c in counts.items():
        total, verified = c["total"], c["verified"]
        if total == 0:
            continue
        rate = verified / total
        if label == "genuine_bag":
            tp, fn = verified, total - verified
            lines.append(f"genuine_bag: 정탐율(recall) {rate * 100:.1f}%  ({verified}/{total})")
        elif label == "impostor":
            fp, tn = verified, total - verified
            lines.append(f"impostor:    오탐율(false accept) {rate * 100:.1f}%  ({verified}/{total})")
    if tp + fn + tn + fp > 0:
        accuracy = (tp + tn) / (tp + fn + tn + fp)
        lines.append(f"종합 정확도(accuracy) {accuracy * 100:.1f}%  (TP={tp} FN={fn} TN={tn} FP={fp})")
    return "\n".join(lines) if lines else "라벨 매긴 프레임 없음 (g/x 키로 라벨링해야 정확도 나옴)"


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    camera = VideoFrameSource(video_path)
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.4)
    pipeline = RecognitionPipeline(
        detector, camera, TargetStabilizer(window_size=15),
        color_filter_enabled=True, depth_filter_enabled=True,
        color_thresholds=NIGHT_COLOR_THRESHOLDS,
    )
    flicker_filter = FlickerFilter(miss_tolerance=5, iou_threshold=0.3)

    camera.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))

    log_path = f"logs/video_review_{int(time.time())}.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "ground_truth_label", "bbox", "verified", "color_ratio", "h", "s", "v", "bulge_height"])
    counts = {"genuine_bag": {"total": 0, "verified": 0}, "impostor": {"total": 0, "verified": 0}}
    print(f"영상: {video_path}")
    print(f"로그: {log_path}")
    print("g: genuine_bag 라벨, x: impostor 라벨, u: unlabeled, space: 일시정지, q: 종료")

    label = "unlabeled"
    paused = False
    vis = None
    try:
        while True:
            if not paused:
                try:
                    color, depth, intrinsics, evaluations = pipeline.evaluate_all()
                except StopIteration:
                    print("영상 끝")
                    break
                display_evaluations = flicker_filter.update(evaluations)
                vis = overlay_evaluations(color, display_evaluations)
                cv2.putText(vis, f"ground_truth={label}", (10, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                now = time.time()
                for evaluation in evaluations:
                    h, s, v = evaluation.mean_hsv if evaluation.mean_hsv is not None else (None, None, None)
                    writer.writerow([
                        now, label, evaluation.detection.bbox, evaluation.verified,
                        evaluation.color_ratio, h, s, v, evaluation.bulge_height,
                    ])
                    if label in counts:
                        counts[label]["total"] += 1
                        if evaluation.verified:
                            counts[label]["verified"] += 1

            cv2.imshow(WINDOW_NAME, cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))

            key = cv2.waitKey(0 if paused else 30) & 0xFF
            if key == ord("g"):
                label = "genuine_bag"
                print("라벨: genuine_bag")
            elif key == ord("x"):
                label = "impostor"
                print("라벨: impostor")
            elif key == ord("u"):
                label = "unlabeled"
                print("라벨: unlabeled")
            elif key == ord(" "):
                paused = not paused
            elif key == ord("q"):
                break
    finally:
        log_file.close()
        camera.stop()
        cv2.destroyAllWindows()
        print(summarize(counts))


def demo() -> None:
    color = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = stub_depth(color)
    assert depth.shape == (4, 6) and depth.dtype == np.uint16 and not depth.any()

    counts = {"genuine_bag": {"total": 10, "verified": 9}, "impostor": {"total": 10, "verified": 1}}
    out = summarize(counts)
    assert "90.0%" in out and "10.0%" in out and "90.0%" in out.split("종합")[1]


if __name__ == "__main__":
    demo()
    main()
