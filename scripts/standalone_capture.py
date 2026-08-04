"""
다른 PC/노트북에 iCIR_cle 프로젝트를 통째로 옮기지 않고, RealSense로 사진만
찍고 싶을 때 쓰는 독립 스크립트. bagvision 패키지에 의존하지 않는다.

필요한 설치: RealSense SDK(드라이버) + `pip install pyrealsense2 opencv-python`
"""
import time
import cv2
import numpy as np
import pyrealsense2 as rs

OUTPUT_DIR = "raw"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 워밍업: 첫 프레임 타임아웃 회피
    for _ in range(5):
        try:
            pipeline.wait_for_frames(timeout_ms=10000)
        except RuntimeError:
            continue

    count = 0
    print("스페이스바: 캡처, q: 종료")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color = np.asanyarray(color_frame.get_data())

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
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
