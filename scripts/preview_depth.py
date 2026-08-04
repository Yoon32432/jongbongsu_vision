import cv2
from bagvision.realsense_capture import RealSenseCamera

VALID_RANGE = (0.3, 2.0)


def main():
    camera = RealSenseCamera()
    camera.start()
    print("q: 종료")
    try:
        while True:
            color, depth, intr = camera.get_frames()
            cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
            center_depth_m = depth[cy, cx] * intr.depth_scale
            in_range = VALID_RANGE[0] <= center_depth_m <= VALID_RANGE[1]
            color_bgr = (0, 255, 0) if in_range else (0, 0, 255)

            vis = color.copy()
            cv2.drawMarker(vis, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                vis,
                f"{center_depth_m:.3f} m",
                (cx + 15, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color_bgr,
                2,
            )
            cv2.imshow("preview_depth (q=quit)", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
