import time
import numpy as np
from bagvision.types import Detection, DepthEstimate, StableTarget


class TargetStabilizer:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self._buffer: list[tuple[Detection, DepthEstimate]] = []

    def add_frame(self, detection: Detection, estimate: DepthEstimate | None) -> None:
        if estimate is None:
            return
        self._buffer.append((detection, estimate))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def is_ready(self) -> bool:
        return len(self._buffer) >= self.window_size

    def capture(self, frame_id: str) -> StableTarget:
        if not self.is_ready():
            raise RuntimeError("not enough frames buffered yet")

        positions = np.array([e.position_xyz for _, e in self._buffer])
        sizes = np.array([e.size_estimate for _, e in self._buffer])
        confidences = [d.confidence for d, _ in self._buffer]
        best_det = max((d for d, _ in self._buffer), key=lambda d: d.confidence)

        return StableTarget(
            class_name=best_det.class_name,
            confidence=float(np.mean(confidences)),
            position_xyz=tuple(positions.mean(axis=0)),
            size_estimate=tuple(sizes.mean(axis=0)),
            frame_id=frame_id,
            timestamp=time.time(),
            n_frames_averaged=len(self._buffer),
        )

    def reset(self) -> None:
        self._buffer.clear()
