import time
import cv2
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline
from bagvision.depth_fusion import estimate_from_mask
from bagvision.color_verification import (
    HUE_RANGE, SATURATION_MIN,
    NIGHT_SATURATION_MAX, NIGHT_VALUE_RANGE, NIGHT_MIN_GREEN_RATIO, axis_pass_rates,
)

WEIGHTS_PATH = "models/bagvision-5/weights/best.pt"
# cv2 윈도우 제목에 한글 넣으면 Win32 API로 넘어갈 때 인코딩이 호출마다 다르게
# 깨져서(mojibake), OpenCV가 같은 창으로 매칭 못 하고 내용 없는 빈 창을 하나 더
# 만드는 버그가 있었음(2026-08-11, 화면 녹화로 실측). 창 제목은 ASCII만 사용.
WINDOW_NAME = "live_demo_night (c=capture, h=axis pass rates, q=quit)"
DISPLAY_SCALE = 1.6

# 2026-08-05 야외/야간(bag 303/other 91) + 2026-08-11 전조등(bag 193/other 21)
# 라벨링 로그 기반. 전조등 직사광에 other의 V가 튀어 bag과 겹치는 게 확인돼
# NIGHT_VALUE_RANGE 상한을 250->200으로 낮춤(그래도 전조등에서 other 통과율
# 0.43으로 무전조등 대비 여전히 높음, 색상만으로 완전히 못 거름). 표본이 적어
# 확정치는 아니고 시작점 -> color_verification.py의 NIGHT_SATURATION_MAX/
# NIGHT_VALUE_RANGE를 계속 재캘리브레이션할 것.
NIGHT_COLOR_THRESHOLDS = {
    "hue_range": HUE_RANGE,
    "saturation_range": (SATURATION_MIN, NIGHT_SATURATION_MAX),
    "value_range": NIGHT_VALUE_RANGE,
    "min_green_ratio": NIGHT_MIN_GREEN_RATIO,
}

# 야외 직사광선에서 RealSense IR depth가 깨져 볼록도(bulge_height) 신호를 신뢰할
# 수 없음이 실측으로 확인됨 (2026-08-05 로그: other의 bulge가 bag보다 오히려 큼).
# 원인 조사 전까지 depth 검증은 끄고 색상 검증만으로 판정한다.
DEPTH_FILTER_ENABLED = False


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FlickerFilter:
    """conf/색상 문턱 근처서 detection이 매 프레임 켜짐/꺼짐 하는 걸 막는다.
    IoU로 이전 트랙과 매칭해서, miss_tolerance 프레임 안에서는 놓쳐도 마지막
    evaluation을 그대로 유지해서 보여준다 (표시용, stabilizer/step()엔 영향 없음)."""

    def __init__(self, miss_tolerance: int = 10, iou_threshold: float = 0.3, overlap_suppress_iou: float = 0.1):
        self.miss_tolerance = miss_tolerance
        self.iou_threshold = iou_threshold
        # match(iou_threshold=0.3)보다 낮은 문턱 -> 봉투가 안 움직였는데 프레임 하나
        # detection box가 흔들려서 매칭 실패 -> stale 트랙 + 새 트랙이 같은 봉투에
        # 동시에 겹쳐 그려지는 걸 잡는다. 파일 쌓인 봉투끼리는 보통 이만큼도 안
        # 겹치니 오검출 억제 아님.
        self.overlap_suppress_iou = overlap_suppress_iou
        self._tracks = []  # list of {"evaluation": DetectionEvaluation, "misses": int}

    def update(self, evaluations):
        unmatched = list(evaluations)
        for track in self._tracks:
            best_iou, best_idx = 0.0, -1
            for i, ev in enumerate(unmatched):
                iou = _iou(track["evaluation"].detection.bbox, ev.detection.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                track["evaluation"] = unmatched.pop(best_idx)
                track["misses"] = 0
            else:
                track["misses"] += 1

        self._tracks = [t for t in self._tracks if t["misses"] <= self.miss_tolerance]
        self._tracks.extend({"evaluation": ev, "misses": 0} for ev in unmatched)
        self._tracks = self._suppress_overlaps(self._tracks)

        return [t["evaluation"] for t in self._tracks]

    def _suppress_overlaps(self, tracks):
        kept = []
        for track in sorted(tracks, key=lambda t: t["misses"]):
            box = track["evaluation"].detection.bbox
            if any(_iou(box, k["evaluation"].detection.bbox) >= self.overlap_suppress_iou for k in kept):
                continue
            kept.append(track)
        return kept


def _draw_evaluation(vis, evaluation, depth_image, intrinsics):
    """returns depth_ok: True/False if xyz was attempted (verified target), else None"""
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

    depth_ok = None
    if evaluation.verified:
        estimate = estimate_from_mask(depth_image, det.mask, intrinsics)
        depth_ok = estimate is not None
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
        f"색상 {evaluation.color_ratio * 100:.0f}%{hsv_text}",
        (x1, min(vis.shape[0] - 5, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )
    return depth_ok


def overlay_evaluations(color_image, evaluations, depth_image, intrinsics, fps, depth_valid_rate):
    vis = color_image.copy()
    cv2.putText(
        vis,
        f"NIGHT mode (S<={NIGHT_SATURATION_MAX}, V={NIGHT_VALUE_RANGE[0]}-{NIGHT_VALUE_RANGE[1]}, ratio>={NIGHT_MIN_GREEN_RATIO}, depth 검증 꺼짐)  detections: {len(evaluations)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    depth_rate_text = "n/a" if depth_valid_rate is None else f"{depth_valid_rate * 100:.0f}%"
    cv2.putText(
        vis,
        f"fps={fps:.1f}  xyz 유효율(최근 2초)={depth_rate_text}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    depth_oks = [_draw_evaluation(vis, evaluation, depth_image, intrinsics) for evaluation in evaluations]
    return vis, [ok for ok in depth_oks if ok is not None]


def main():
    camera = RealSenseCamera()
    # 야간 데모는 무조건 GPU로 추론(device="cuda"). GPU 없으면 조용히 CPU로
    # 안 떨어지고 바로 에러 (BagDetector.__init__에서 체크).
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.5, imgsz=480, device="cuda")
    stabilizer = TargetStabilizer(window_size=15)
    flicker_filter = FlickerFilter(miss_tolerance=5, iou_threshold=0.3)
    pipeline = RecognitionPipeline(
        detector, camera, stabilizer,
        color_filter_enabled=True,
        depth_filter_enabled=DEPTH_FILTER_ENABLED,
        color_thresholds=NIGHT_COLOR_THRESHOLDS,
    )

    camera.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
    print("c: 안정된 타겟 캡처, h: 색상 축별(H/S/V) 통과율 출력, q: 종료")

    fps = 0.0
    last_ts = time.time()
    DEPTH_WINDOW_SEC = 2.0
    depth_history = []  # list of (timestamp, ok: bool)

    try:
        while True:
            color, depth, intrinsics, evaluations = pipeline.evaluate_all()
            evaluations = flicker_filter.update(evaluations)

            now = time.time()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1  # EMA, 실제 루프 처리 속도(카메라 요청 fps와 다를 수 있음)

            # 직전 프레임까지의 window로 표시(이번 프레임 결과는 이번 draw가 끝난 뒤 반영)
            depth_history = [(t, ok) for t, ok in depth_history if now - t <= DEPTH_WINDOW_SEC]
            depth_valid_rate = (
                sum(ok for _, ok in depth_history) / len(depth_history) if depth_history else None
            )

            vis, depth_oks = overlay_evaluations(color, evaluations, depth, intrinsics, fps, depth_valid_rate)
            depth_history.extend((now, ok) for ok in depth_oks)

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
                    rates = axis_pass_rates(
                        color, evaluation.detection.mask,
                        saturation_range=(SATURATION_MIN, NIGHT_SATURATION_MAX),
                        value_range=NIGHT_VALUE_RANGE,
                    )
                    print(f"[detection {i}] bbox={evaluation.detection.bbox} "
                          f"mean_hsv={evaluation.mean_hsv} axis_pass_rates={rates}")
            elif key == ord("q"):
                break
    finally:
        # Windows에서는 destroyAllWindows 직후 waitKey를 몇 번 더 펌핑해야 창이
        # 실제로 닫힘 -> 순서를 camera.stop()보다 앞으로 옮겨서, RealSense
        # 파이프라인 종료(다소 걸림)로 창이 "응답 없음"으로 떠 있는 동안 걸리는
        # 유령 창을 없앤다.
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)
        camera.stop()


if __name__ == "__main__":
    main()
