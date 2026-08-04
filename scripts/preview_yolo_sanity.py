import cv2
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.depth_fusion import estimate_from_mask

# COCO 사전학습 가중치로 배선만 확인하는 용도. 종량제봉투 클래스는 없음.
WEIGHTS = "yolo26n-seg.pt"
# 일반 물체 배선 확인용이라 실제 파이프라인의 0.3~2.0m보다 넓게 잡는다.
SANITY_VALID_RANGE = (0.1, 5.0)


def overlay(color_image, detections, depth_image, intrinsics):
    vis = color_image.copy()
    for det in detections:
        overlay_layer = vis.copy()
        overlay_layer[det.mask] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.6, overlay_layer, 0.4, 0)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        estimate = estimate_from_mask(depth_image, det.mask, intrinsics, valid_range=SANITY_VALID_RANGE)
        depth_text = f" z={estimate.position_xyz[2]:.2f}m" if estimate is not None else " z=?"

        cv2.putText(
            vis,
            f"{det.class_name} {det.confidence:.2f}{depth_text}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return vis


def main():
    camera = RealSenseCamera()
    detector = BagDetector(weights_path=WEIGHTS, conf_threshold=0.4)

    camera.start()
    print("q: 종료")
    try:
        while True:
            color, depth, intrinsics = camera.get_frames()
            detections = detector.detect(color)
            vis = overlay(color, detections, depth, intrinsics)
            cv2.imshow("preview_yolo_sanity (q=quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
