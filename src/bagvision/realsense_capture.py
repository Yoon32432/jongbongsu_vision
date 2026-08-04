import numpy as np
import pyrealsense2 as rs
from bagvision.types import CameraIntrinsics


def intrinsics_from_realsense(rs_intrinsics, depth_scale: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=rs_intrinsics.fx,
        fy=rs_intrinsics.fy,
        ppx=rs_intrinsics.ppx,
        ppy=rs_intrinsics.ppy,
        depth_scale=depth_scale,
    )


class RealSenseCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._align = None
        self._intrinsics = None

    def start(self) -> None:
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = self._pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = intrinsics_from_realsense(color_stream.get_intrinsics(), depth_scale)
        self._align = rs.align(rs.stream.color)

        # D435i 계열은 스트림 시작 직후 첫 프레임이 워밍업 지연으로 간헐적
        # 타임아웃 나는 경우가 있어, 실제 사용 전에 몇 프레임 흘려보낸다.
        for _ in range(5):
            try:
                self._pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                continue

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def get_frames(self):
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image, self._intrinsics
