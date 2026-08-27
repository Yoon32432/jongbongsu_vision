"""
야간/야외 등 새 환경에서 color_verification.py / depth_verification.py의 임계값을
재캘리브레이션하기 위한 실시간 로깅 스크립트.

live_demo.py와 달리 현재 임계값으로 걸러내지 않고, 탐지된 모든 detection의
raw HSV/볼록도 값을 주기적으로 CSV에 계속 기록한다. 화면을 보며 g/x로 라벨을
붙이면, 실행 중에도 로그 파일을 tail해서 봉투(bag) vs 오탐 후보(other) 값의
분포 차이를 바로 확인할 수 있다.
"""
import csv
import os
import time
import cv2
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.color_verification import axis_pass_rates, green_signature_ratio, mean_hsv
from bagvision.depth_verification import bulge_height

WEIGHTS_PATH = "models/bagvision-5/weights/best.pt"
LOG_DIR = "logs"
LOG_INTERVAL_SEC = 0.5  # 매 프레임 다 쓰면 파일이 과도하게 커져서 스로틀링


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/hsv_tuning_{int(time.time())}.csv"

    camera = RealSenseCamera()
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.3)
    camera.start()

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "timestamp", "label", "bbox", "h", "s", "v",
        "hue_pass", "saturation_pass", "value_pass",
        "green_ratio", "bulge_height_cm",
    ])
    log_file.flush()

    label = "unlabeled"
    last_log = 0.0
    print(f"로그 파일: {log_path}")
    print("g: 라벨=bag, x: 라벨=other(오탐 후보), u: 라벨=unlabeled, q: 종료")

    try:
        while True:
            color, depth, intrinsics = camera.get_frames()
            detections = detector.detect(color)
            now = time.time()
            should_log = (now - last_log) >= LOG_INTERVAL_SEC

            vis = color.copy()
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                hsv = mean_hsv(color, det.mask)
                ratio = green_signature_ratio(color, det.mask)
                rates = axis_pass_rates(color, det.mask)
                bulge = bulge_height(depth, det.mask, intrinsics)
                bulge_cm = bulge * 100 if bulge is not None else None

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if hsv is not None:
                    h, s, v = hsv
                    cv2.putText(
                        vis, f"H={h:.0f} S={s:.0f} V={v:.0f} ratio={ratio * 100:.0f}%",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )

                if should_log and hsv is not None:
                    h, s, v = hsv
                    writer.writerow([
                        now, label, det.bbox, h, s, v,
                        rates["hue"], rates["saturation"], rates["value"],
                        ratio, bulge_cm,
                    ])

            if should_log:
                log_file.flush()
                last_log = now

            cv2.putText(vis, f"label={label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("tune_thresholds (g=bag x=other u=unlabeled q=quit)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("g"):
                label = "bag"
                print("라벨: bag")
            elif key == ord("x"):
                label = "other"
                print("라벨: other")
            elif key == ord("u"):
                label = "unlabeled"
                print("라벨: unlabeled")
            elif key == ord("q"):
                break
    finally:
        log_file.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
