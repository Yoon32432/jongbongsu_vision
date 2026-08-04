import time
import cv2
from bagvision.realsense_capture import RealSenseCamera

OUTPUT_DIR = "data/raw"


def main():
    camera = RealSenseCamera()
    camera.start()
    count = 0
    print("스페이스바: 캡처, q: 종료")
    try:
        while True:
            color, depth, _ = camera.get_frames()
            cv2.imshow("preview (space=capture, q=quit)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                ts = int(time.time() * 1000)
                cv2.imwrite(f"{OUTPUT_DIR}/{ts}.jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 95])
                count += 1
                print(f"저장됨: {ts}.jpg (누적 {count}장)")
            elif key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
