from bagvision.color_verification import green_signature_ratio, mean_hsv, verify_color_signature
from bagvision.depth_fusion import estimate_from_mask
from bagvision.depth_verification import bulge_height, verify_depth_bulge
from bagvision.types import Detection, DetectionEvaluation, StableTarget


class RecognitionPipeline:
    def __init__(
        self,
        detector,
        camera,
        stabilizer,
        valid_range=(0.3, 2.0),
        floor_z=None,
        color_filter_enabled: bool = True,
        depth_filter_enabled: bool = True,
    ):
        self.detector = detector
        self.camera = camera
        self.stabilizer = stabilizer
        self.valid_range = valid_range
        self.floor_z = floor_z
        self.color_filter_enabled = color_filter_enabled
        self.depth_filter_enabled = depth_filter_enabled
        # 디버그/시각화용 진단 정보. step()의 반환값(검증 통과분만)과 별개로,
        # 검증에서 거부된 raw detection도 확인할 수 있도록 남겨둔다.
        self.last_raw_detection: Detection | None = None
        self.last_color_ratio: float | None = None
        self.last_bulge_height: float | None = None
        self.last_mean_hsv: tuple[float, float, float] | None = None

    def step(self) -> Detection | None:
        color, depth, intrinsics = self.camera.get_frames()
        detections = self.detector.detect(color)
        if not detections:
            self.last_raw_detection = None
            self.last_color_ratio = None
            self.last_bulge_height = None
            self.last_mean_hsv = None
            return None

        best = max(detections, key=lambda d: d.confidence)
        self.last_raw_detection = best
        self.last_color_ratio = green_signature_ratio(color, best.mask)
        self.last_mean_hsv = mean_hsv(color, best.mask)
        self.last_bulge_height = bulge_height(
            depth, best.mask, intrinsics, valid_range=self.valid_range, floor_z=self.floor_z
        )
        if not verify_color_signature(color, best.mask, enabled=self.color_filter_enabled):
            return None
        if not verify_depth_bulge(
            depth, best.mask, intrinsics,
            valid_range=self.valid_range, floor_z=self.floor_z, enabled=self.depth_filter_enabled,
        ):
            return None

        estimate = estimate_from_mask(
            depth, best.mask, intrinsics, valid_range=self.valid_range, floor_z=self.floor_z
        )
        self.stabilizer.add_frame(best, estimate)
        return best

    def evaluate_all(self):
        """한 프레임 안의 모든 raw detection을 검증 결과와 함께 반환한다 (시각화/디버그용).

        step()은 로봇 파이프라인이 쓸 단일 최선의 타겟만 반환/누적하는 반면,
        이 메서드는 한 화면에 여러 후보(예: 봉투+쓰레받이)가 같이 잡혔을 때
        각각을 독립적으로 평가해 전부 보여주기 위한 것으로, stabilizer 상태에는
        영향을 주지 않는다.
        """
        color, depth, intrinsics = self.camera.get_frames()
        detections = self.detector.detect(color)

        evaluations = []
        for det in detections:
            color_ratio = green_signature_ratio(color, det.mask)
            hsv = mean_hsv(color, det.mask)
            height = bulge_height(
                depth, det.mask, intrinsics, valid_range=self.valid_range, floor_z=self.floor_z
            )
            verified = (
                verify_color_signature(color, det.mask, enabled=self.color_filter_enabled)
                and verify_depth_bulge(
                    depth, det.mask, intrinsics,
                    valid_range=self.valid_range, floor_z=self.floor_z,
                    enabled=self.depth_filter_enabled,
                )
            )
            evaluations.append(DetectionEvaluation(
                detection=det, verified=verified, color_ratio=color_ratio,
                mean_hsv=hsv, bulge_height=height,
            ))
        return color, depth, intrinsics, evaluations

    def capture_stable_target(self, max_frames: int = 60) -> StableTarget:
        self.stabilizer.reset()
        for _ in range(max_frames):
            self.step()
            if self.stabilizer.is_ready():
                return self.stabilizer.capture(frame_id="camera_optical_frame")
        raise TimeoutError("failed to gather enough stable frames within max_frames")
