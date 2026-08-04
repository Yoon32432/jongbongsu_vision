# 종량제봉투 인식 시스템 (YOLO + RealSense) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 바닥에 놓인 종량제봉투를 YOLOv8-seg로 탐지하고 RealSense depth와 융합해,
로봇팔(그리퍼)에 장착된 카메라의 근접 blind zone 문제를 피하도록 "안전거리에서
한 번 안정된 값을 확정해 내보내는" 인식 파이프라인을 만든다.

**Architecture:** RealSense RGB-D 캡처 → YOLOv8-seg 마스크 추론 → 마스크 영역의
depth만 역투영해 3D 위치/크기 추정 → 여러 프레임을 평균해 안정된 단일 값으로
확정(`capture_stable_target`)하는 4단 파이프라인. 각 단은 독립적으로 테스트
가능하도록 의존성을 주입받는 순수 함수/클래스로 분리한다.

**Tech Stack:** Python 3.10+, `pyrealsense2`, `ultralytics` (YOLOv8-seg),
`numpy`, `opencv-python`, `pytest`.

## Global Constraints

- 독립 프로젝트 (`~/iCIR_cle`). git 저장소 아님, ROS2/기존 워크스페이스와 무관.
- 카메라는 그리퍼 부착(eye-in-hand). depth 유효 거리 기본값 0.3m~2.0m를
  depth 융합 전 구간에서 강제한다.
- YOLO 클래스는 단일 클래스 `"bag"`으로 시작한다 (봉투 종류/용량 세분류 없음).
- 출력 인터페이스는 연속 스트림 API와 더불어 **단일 캡처(lock-in) API**
  `capture_stable_target()`을 반드시 제공한다.
- 로봇팔 제어, hand-eye calibration, 좌표계 변환 실행 로직은 이 저장소
  범위 밖이다 — 구현하지 않는다.

---

## File Structure

```
iCIR_cle/
  pyproject.toml
  src/bagvision/
    __init__.py
    types.py             # Detection, CameraIntrinsics, DepthEstimate, StableTarget
    depth_fusion.py       # estimate_from_mask()
    stabilizer.py         # TargetStabilizer
    detector.py           # BagDetector (YOLOv8-seg wrapper)
    realsense_capture.py  # RealSenseCamera, intrinsics_from_realsense()
    pipeline.py            # RecognitionPipeline
  tests/
    test_depth_fusion.py
    test_stabilizer.py
    test_detector.py
    test_realsense_capture.py
    test_pipeline.py
  scripts/
    check_camera.py    # 하드웨어 연결 확인 (수동 실행)
    collect_data.py    # 학습용 이미지 수집 (수동 실행)
    train.py           # YOLOv8-seg 학습 CLI 래퍼 (수동 실행)
    live_demo.py       # 실시간 시각화 + 캡처 데모 (수동 실행)
  data/
    raw/               # collect_data.py 출력 (gitignore 대상이지만 git 미사용)
    dataset.yaml        # ultralytics 학습용 데이터 설정
  models/
    .gitkeep
```

---

### Task 1: 프로젝트 스캐폴딩

**Files:**
- Create: `pyproject.toml`
- Create: `src/bagvision/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: `bagvision` 패키지 import 경로, `pytest` 실행 환경

- [ ] **Step 1: 디렉터리 및 pyproject.toml 작성**

```bash
mkdir -p ~/iCIR_cle/src/bagvision ~/iCIR_cle/tests ~/iCIR_cle/scripts ~/iCIR_cle/data/raw ~/iCIR_cle/models
touch ~/iCIR_cle/src/bagvision/__init__.py ~/iCIR_cle/tests/__init__.py
```

`pyproject.toml`:

```toml
[project]
name = "bagvision"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "opencv-python",
    "ultralytics",
    "pyrealsense2",
]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 2: 가상환경 생성 및 설치**

```bash
cd ~/iCIR_cle
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: `ultralytics`, `pyrealsense2`, `numpy`, `opencv-python`, `pytest`가
에러 없이 설치됨. (`pyrealsense2`는 플랫폼에 따라 wheel이 없을 수 있음 —
설치 실패 시 Intel RealSense SDK 공식 설치 가이드를 따로 확인해야 함을
기록해두고 다음 태스크로 진행)

- [ ] **Step 3: 스모크 테스트로 pytest 동작 확인**

`tests/test_smoke.py`:

```python
def test_pytest_works():
    assert 1 + 1 == 2
```

Run: `pytest -q`
Expected: `1 passed`

- [ ] **Step 4: Step 3의 스모크 테스트 파일 삭제**

```bash
rm ~/iCIR_cle/tests/test_smoke.py
```

(pytest 동작 확인용 임시 파일이므로 실제 커밋 대상 아님)

---

### Task 2: 공통 타입 정의 (`types.py`)

**Files:**
- Create: `src/bagvision/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Produces:
  - `CameraIntrinsics(fx, fy, ppx, ppy, depth_scale)`
  - `Detection(class_id, class_name, confidence, mask, bbox)`
  - `DepthEstimate(position_xyz, size_estimate, n_points)`
  - `StableTarget(class_name, confidence, position_xyz, size_estimate, frame_id, timestamp, n_frames_averaged)`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_types.py`:

```python
import numpy as np
from bagvision.types import CameraIntrinsics, Detection, DepthEstimate, StableTarget


def test_camera_intrinsics_fields():
    intr = CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    assert intr.fx == 600.0
    assert intr.depth_scale == 0.001


def test_detection_fields():
    mask = np.zeros((10, 10), dtype=bool)
    det = Detection(class_id=0, class_name="bag", confidence=0.9, mask=mask, bbox=(0, 0, 5, 5))
    assert det.class_name == "bag"
    assert det.mask.shape == (10, 10)


def test_depth_estimate_and_stable_target_fields():
    est = DepthEstimate(position_xyz=(0.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.02), n_points=100)
    assert est.n_points == 100

    target = StableTarget(
        class_name="bag",
        confidence=0.9,
        position_xyz=(0.0, 0.0, 0.5),
        size_estimate=(0.1, 0.1, 0.02),
        frame_id="camera_optical_frame",
        timestamp=123.0,
        n_frames_averaged=15,
    )
    assert target.n_frames_averaged == 15
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bagvision.types'` (또는 import 대상 없음)

- [ ] **Step 3: 최소 구현 작성**

`src/bagvision/types.py`:

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class DepthEstimate:
    position_xyz: tuple[float, float, float]
    size_estimate: tuple[float, float, float]
    n_points: int


@dataclass
class StableTarget:
    class_name: str
    confidence: float
    position_xyz: tuple[float, float, float]
    size_estimate: tuple[float, float, float]
    frame_id: str
    timestamp: float
    n_frames_averaged: int
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_types.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit 대신 상태 기록**

git 저장소를 쓰지 않으므로 커밋 단계는 생략한다. 대신 각 태스크 완료 후
`pytest -q`로 전체 스위트가 여전히 통과하는지 확인하고 다음 태스크로 넘어간다.

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과

---

### Task 3: Depth 융합 — `estimate_from_mask()`

**Files:**
- Create: `src/bagvision/depth_fusion.py`
- Test: `tests/test_depth_fusion.py`

**Interfaces:**
- Consumes: `CameraIntrinsics`, `DepthEstimate` (Task 2)
- Produces: `estimate_from_mask(depth_image: np.ndarray, mask: np.ndarray, intrinsics: CameraIntrinsics, valid_range: tuple[float, float] = (0.3, 2.0), floor_z: float | None = None, floor_margin: float = 0.02) -> DepthEstimate | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_depth_fusion.py`:

```python
import numpy as np
import pytest
from bagvision.types import CameraIntrinsics
from bagvision.depth_fusion import estimate_from_mask


def make_intrinsics():
    return CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)


def test_estimate_from_mask_flat_region():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    # rows 200..279, cols 280..359, depth raw=500 -> 0.5m
    depth[200:280, 280:360] = 500
    mask[200:280, 280:360] = True

    est = estimate_from_mask(depth, mask, intr)

    assert est is not None
    assert est.n_points == 80 * 80
    assert est.position_xyz[2] == pytest.approx(0.5, abs=1e-6)
    assert est.position_xyz[0] == pytest.approx(-0.0004167, abs=1e-3)
    assert est.position_xyz[1] == pytest.approx(-0.0004167, abs=1e-3)
    assert est.size_estimate[0] == pytest.approx(0.06583, abs=1e-3)
    assert est.size_estimate[1] == pytest.approx(0.06583, abs=1e-3)
    assert est.size_estimate[2] == pytest.approx(0.0, abs=1e-9)


def test_estimate_from_mask_empty_mask_returns_none():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)

    assert estimate_from_mask(depth, mask, intr) is None


def test_estimate_from_mask_filters_out_of_range_depth():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    # valid region: 0.5m
    depth[200:220, 280:300] = 500
    mask[200:220, 280:300] = True
    # invalid region: 0.05m (too close, below default valid_range min 0.3m)
    depth[400:420, 280:300] = 50
    mask[400:420, 280:300] = True

    est = estimate_from_mask(depth, mask, intr, valid_range=(0.3, 2.0))

    assert est is not None
    assert est.n_points == 20 * 20
    assert est.position_xyz[2] == pytest.approx(0.5, abs=1e-6)


def test_estimate_from_mask_filters_floor_bleed():
    intr = make_intrinsics()
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    # bag region: 0.5m
    depth[200:220, 280:300] = 500
    mask[200:220, 280:300] = True
    # floor-bleed region at mask edge: 0.9m, close to floor_z=0.9
    depth[220:225, 280:300] = 900
    mask[220:225, 280:300] = True

    est = estimate_from_mask(depth, mask, intr, valid_range=(0.3, 2.0), floor_z=0.9, floor_margin=0.05)

    assert est is not None
    assert est.n_points == 20 * 20
    assert est.position_xyz[2] == pytest.approx(0.5, abs=1e-6)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_depth_fusion.py -v`
Expected: FAIL (`estimate_from_mask` 없음)

- [ ] **Step 3: 최소 구현 작성**

`src/bagvision/depth_fusion.py`:

```python
import numpy as np
from bagvision.types import CameraIntrinsics, DepthEstimate


def estimate_from_mask(
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    valid_range: tuple[float, float] = (0.3, 2.0),
    floor_z: float | None = None,
    floor_margin: float = 0.02,
) -> DepthEstimate | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    z = depth_image[ys, xs].astype(np.float64) * intrinsics.depth_scale
    valid = (z >= valid_range[0]) & (z <= valid_range[1])

    if floor_z is not None:
        valid &= z < (floor_z - floor_margin)

    if not np.any(valid):
        return None

    xs, ys, z = xs[valid], ys[valid], z[valid]
    x = (xs - intrinsics.ppx) * z / intrinsics.fx
    y = (ys - intrinsics.ppy) * z / intrinsics.fy

    position = (float(np.mean(x)), float(np.mean(y)), float(np.mean(z)))
    size = (
        float(x.max() - x.min()),
        float(y.max() - y.min()),
        float(z.max() - z.min()),
    )
    return DepthEstimate(position_xyz=position, size_estimate=size, n_points=int(len(z)))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_depth_fusion.py -v`
Expected: `4 passed`

- [ ] **Step 5: 전체 스위트 재확인**

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과

---

### Task 4: 안정화 모듈 — `TargetStabilizer`

**Files:**
- Create: `src/bagvision/stabilizer.py`
- Test: `tests/test_stabilizer.py`

**Interfaces:**
- Consumes: `Detection`, `DepthEstimate`, `StableTarget` (Task 2)
- Produces:
  - `TargetStabilizer(window_size: int = 15)`
  - `.add_frame(detection: Detection, estimate: DepthEstimate | None) -> None`
  - `.is_ready() -> bool`
  - `.capture(frame_id: str) -> StableTarget` (raises `RuntimeError` if not ready)
  - `.reset() -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stabilizer.py`:

```python
import numpy as np
import pytest
from bagvision.types import Detection, DepthEstimate
from bagvision.stabilizer import TargetStabilizer


def make_detection(conf=0.9):
    mask = np.zeros((10, 10), dtype=bool)
    return Detection(class_id=0, class_name="bag", confidence=conf, mask=mask, bbox=(0, 0, 5, 5))


def test_not_ready_before_window_filled():
    stab = TargetStabilizer(window_size=3)
    assert stab.is_ready() is False
    with pytest.raises(RuntimeError):
        stab.capture(frame_id="camera_optical_frame")


def test_ready_after_window_filled_and_capture_averages():
    stab = TargetStabilizer(window_size=3)
    positions = [(0.0, 0.0, 0.5), (0.02, 0.0, 0.5), (-0.02, 0.0, 0.5)]
    for pos in positions:
        est = DepthEstimate(position_xyz=pos, size_estimate=(0.1, 0.1, 0.0), n_points=50)
        stab.add_frame(make_detection(), est)

    assert stab.is_ready() is True
    target = stab.capture(frame_id="camera_optical_frame")
    assert target.position_xyz[0] == pytest.approx(0.0, abs=1e-9)
    assert target.n_frames_averaged == 3
    assert target.class_name == "bag"


def test_none_estimates_are_ignored():
    stab = TargetStabilizer(window_size=2)
    stab.add_frame(make_detection(), None)
    stab.add_frame(make_detection(), None)
    assert stab.is_ready() is False


def test_window_slides_and_drops_oldest():
    stab = TargetStabilizer(window_size=2)
    est1 = DepthEstimate(position_xyz=(0.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.0), n_points=10)
    est2 = DepthEstimate(position_xyz=(1.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.0), n_points=10)
    est3 = DepthEstimate(position_xyz=(2.0, 0.0, 0.5), size_estimate=(0.1, 0.1, 0.0), n_points=10)
    stab.add_frame(make_detection(), est1)
    stab.add_frame(make_detection(), est2)
    stab.add_frame(make_detection(), est3)  # est1 should be dropped

    target = stab.capture(frame_id="camera_optical_frame")
    assert target.position_xyz[0] == pytest.approx(1.5, abs=1e-9)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_stabilizer.py -v`
Expected: FAIL (`TargetStabilizer` 없음)

- [ ] **Step 3: 최소 구현 작성**

`src/bagvision/stabilizer.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_stabilizer.py -v`
Expected: `4 passed`

- [ ] **Step 5: 전체 스위트 재확인**

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과

---

### Task 5: YOLOv8-seg 래퍼 — `BagDetector`

**Files:**
- Create: `src/bagvision/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `Detection` (Task 2)
- Produces:
  - `BagDetector(weights_path: str | None = None, conf_threshold: float = 0.5, model=None)`
  - `.detect(color_image: np.ndarray) -> list[Detection]`

이 태스크는 실제 학습된 가중치 없이도 파싱/후처리 로직을 검증할 수 있도록
`model` 주입 지점을 둔다. 실제 가중치 연동 검증은 Task 10에서 수행한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_detector.py`:

```python
import numpy as np
from bagvision.detector import BagDetector


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf


class FakeMasks:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class FakeModel:
    names = {0: "bag"}

    def __init__(self, result):
        self._result = result

    def predict(self, image, conf, verbose):
        return [self._result]


def test_detect_parses_single_detection():
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    small_mask = np.zeros((20, 20), dtype=np.float32)
    small_mask[5:15, 5:15] = 1.0  # small mask, will be resized to 100x100

    result = FakeResult(
        boxes=FakeBoxes(
            xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
            cls=np.array([0.0]),
            conf=np.array([0.87]),
        ),
        masks=FakeMasks(data=np.array([small_mask])),
    )
    detector = BagDetector(model=FakeModel(result))

    detections = detector.detect(color_image)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "bag"
    assert det.confidence == 0.87
    assert det.mask.shape == (100, 100)
    assert det.mask.dtype == bool
    assert det.mask.sum() > 0


def test_detect_returns_empty_list_when_no_masks():
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = FakeResult(boxes=None, masks=None)
    detector = BagDetector(model=FakeModel(result))

    assert detector.detect(color_image) == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL (`BagDetector` 없음)

- [ ] **Step 3: 최소 구현 작성**

`src/bagvision/detector.py`:

```python
import numpy as np
import cv2
from bagvision.types import Detection


def _to_numpy(value):
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class BagDetector:
    def __init__(self, weights_path: str | None = None, conf_threshold: float = 0.5, model=None):
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, color_image: np.ndarray) -> list[Detection]:
        results = self.model.predict(color_image, conf=self.conf_threshold, verbose=False)[0]
        if results.masks is None:
            return []

        h, w = color_image.shape[:2]
        boxes = _to_numpy(results.boxes.xyxy)
        classes = _to_numpy(results.boxes.cls)
        confs = _to_numpy(results.boxes.conf)
        masks = _to_numpy(results.masks.data)

        detections = []
        for box, cls, conf, raw_mask in zip(boxes, classes, confs, masks):
            resized = cv2.resize(raw_mask.astype(np.float32), (w, h)) > 0.5
            detections.append(
                Detection(
                    class_id=int(cls),
                    class_name=self.model.names[int(cls)],
                    confidence=float(conf),
                    mask=resized,
                    bbox=tuple(int(v) for v in box),
                )
            )
        return detections
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_detector.py -v`
Expected: `2 passed`

- [ ] **Step 5: 전체 스위트 재확인**

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과

---

### Task 6: RealSense 캡처 래퍼

**Files:**
- Create: `src/bagvision/realsense_capture.py`
- Test: `tests/test_realsense_capture.py`
- Create: `scripts/check_camera.py`

**Interfaces:**
- Consumes: `CameraIntrinsics` (Task 2)
- Produces:
  - `intrinsics_from_realsense(rs_intrinsics, depth_scale: float) -> CameraIntrinsics`
  - `RealSenseCamera(width=640, height=480, fps=30)` with `.start()`, `.stop()`,
    `.get_frames() -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]` (color, depth, intrinsics)

`intrinsics_from_realsense`는 순수 함수라 하드웨어 없이 단위 테스트한다.
`RealSenseCamera` 자체는 실제 장치가 있어야 검증 가능하므로, 이 태스크에서는
클래스 구조만 만들고 하드웨어 검증은 수동 스크립트(`scripts/check_camera.py`)로
따로 확인한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_realsense_capture.py`:

```python
from types import SimpleNamespace
from bagvision.realsense_capture import intrinsics_from_realsense


def test_intrinsics_from_realsense_converts_fields():
    rs_intrinsics = SimpleNamespace(fx=615.0, fy=615.0, ppx=319.5, ppy=239.5)

    intr = intrinsics_from_realsense(rs_intrinsics, depth_scale=0.001)

    assert intr.fx == 615.0
    assert intr.fy == 615.0
    assert intr.ppx == 319.5
    assert intr.ppy == 239.5
    assert intr.depth_scale == 0.001
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_realsense_capture.py -v`
Expected: FAIL (`intrinsics_from_realsense` 없음)

- [ ] **Step 3: 최소 구현 작성**

`src/bagvision/realsense_capture.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_realsense_capture.py -v`
Expected: `1 passed`

(참고: `import pyrealsense2 as rs`가 설치 환경에 따라 실패할 수 있음. 이 경우
Task 1의 Step 2에서 기록해둔 설치 이슈를 먼저 해결해야 함)

- [ ] **Step 5: 하드웨어 확인용 수동 스크립트 작성**

`scripts/check_camera.py`:

```python
from bagvision.realsense_capture import RealSenseCamera


def main():
    camera = RealSenseCamera()
    camera.start()
    try:
        color, depth, intr = camera.get_frames()
        print(f"color shape: {color.shape}, depth shape: {depth.shape}")
        print(f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} ppx={intr.ppx:.1f} ppy={intr.ppy:.1f} depth_scale={intr.depth_scale}")
        cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
        center_depth_m = depth[cy, cx] * intr.depth_scale
        print(f"center pixel depth: {center_depth_m:.3f} m")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
```

**수동 검증 (하드웨어 필요):** RealSense를 USB에 연결한 뒤 실행.

Run: `python scripts/check_camera.py`
Expected: color/depth shape가 `(480, 640, 3)` / `(480, 640)`으로 출력되고,
카메라 앞 물체까지의 대략적 실측 거리와 `center pixel depth` 값이 합리적
범위(예: 0.3~2.0m)로 일치함

- [ ] **Step 6: 전체 스위트 재확인**

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과 (하드웨어 스크립트는 pytest 대상 아님)

---

### Task 7: 파이프라인 통합 — `RecognitionPipeline`

**Files:**
- Create: `src/bagvision/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `BagDetector.detect()` (Task 5), `RealSenseCamera.get_frames()` (Task 6),
  `estimate_from_mask()` (Task 3), `TargetStabilizer` (Task 4)
- Produces:
  - `RecognitionPipeline(detector, camera, stabilizer, valid_range=(0.3, 2.0), floor_z=None)`
  - `.step() -> Detection | None`
  - `.capture_stable_target(max_frames: int = 60) -> StableTarget` (raises `TimeoutError`)

파이프라인 테스트에서는 실제 `estimate_from_mask`와 `TargetStabilizer`를
그대로 사용하고, 하드웨어/모델이 필요한 `camera`와 `detector`만 가짜 객체로
주입한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_pipeline.py`:

```python
import numpy as np
import pytest
from bagvision.types import CameraIntrinsics, Detection
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline


class FakeCamera:
    def __init__(self, color, depth, intrinsics):
        self.color = color
        self.depth = depth
        self.intrinsics = intrinsics

    def get_frames(self):
        return self.color, self.depth, self.intrinsics


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, color_image):
        return self.detections


def make_fixture():
    intr = CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[200:280, 280:360] = 500  # 0.5m
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:280, 280:360] = True
    detection = Detection(class_id=0, class_name="bag", confidence=0.9, mask=mask, bbox=(280, 200, 360, 280))
    return FakeCamera(color, depth, intr), FakeDetector([detection])


def test_step_returns_best_detection():
    camera, detector = make_fixture()
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    result = pipeline.step()

    assert result is not None
    assert result.class_name == "bag"


def test_step_returns_none_when_no_detections():
    camera, _ = make_fixture()
    detector = FakeDetector([])
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    assert pipeline.step() is None


def test_capture_stable_target_returns_averaged_result():
    camera, detector = make_fixture()
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    target = pipeline.capture_stable_target(max_frames=10)

    assert target.class_name == "bag"
    assert target.n_frames_averaged == 3
    assert target.position_xyz[2] == pytest.approx(0.5, abs=1e-6)


def test_capture_stable_target_times_out_when_never_ready():
    camera, _ = make_fixture()
    detector = FakeDetector([])  # never produces a detection
    pipeline = RecognitionPipeline(detector, camera, TargetStabilizer(window_size=3))

    with pytest.raises(TimeoutError):
        pipeline.capture_stable_target(max_frames=5)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (`RecognitionPipeline` 없음)

- [ ] **Step 3: 최소 구현 작성**

`src/bagvision/pipeline.py`:

```python
from bagvision.depth_fusion import estimate_from_mask
from bagvision.types import Detection, StableTarget


class RecognitionPipeline:
    def __init__(self, detector, camera, stabilizer, valid_range=(0.3, 2.0), floor_z=None):
        self.detector = detector
        self.camera = camera
        self.stabilizer = stabilizer
        self.valid_range = valid_range
        self.floor_z = floor_z

    def step(self) -> Detection | None:
        color, depth, intrinsics = self.camera.get_frames()
        detections = self.detector.detect(color)
        if not detections:
            return None

        best = max(detections, key=lambda d: d.confidence)
        estimate = estimate_from_mask(
            depth, best.mask, intrinsics, valid_range=self.valid_range, floor_z=self.floor_z
        )
        self.stabilizer.add_frame(best, estimate)
        return best

    def capture_stable_target(self, max_frames: int = 60) -> StableTarget:
        self.stabilizer.reset()
        for _ in range(max_frames):
            self.step()
            if self.stabilizer.is_ready():
                return self.stabilizer.capture(frame_id="camera_optical_frame")
        raise TimeoutError("failed to gather enough stable frames within max_frames")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: `4 passed`

- [ ] **Step 5: 전체 스위트 재확인**

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과

---

### Task 8: 데이터 수집 스크립트

**Files:**
- Create: `scripts/collect_data.py`

이 태스크는 실제 라벨링 대상 이미지를 모으는 수동 실행 스크립트다. 자동
테스트 대상이 아니다 (하드웨어 필요, 사람이 셔터를 누르는 워크플로우).

- [ ] **Step 1: 수집 스크립트 작성**

`scripts/collect_data.py`:

```python
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
                cv2.imwrite(f"{OUTPUT_DIR}/{ts}.png", color)
                count += 1
                print(f"저장됨: {ts}.png (누적 {count}장)")
            elif key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 수동 검증 (하드웨어 필요)**

Run: `python scripts/collect_data.py`
Expected: 미리보기 창이 뜨고, 스페이스바를 누를 때마다 `data/raw/`에 PNG
파일이 저장되며 콘솔에 저장 로그가 출력됨. 최소 200~500장 목표로 다양한
거리/각도/조명에서 수집

- [ ] **Step 3: 라벨링 안내 메모 작성**

`data/README.md`:

```markdown
# 데이터 라벨링 가이드

1. AI Hub(aihub.or.kr)에서 "생활폐기물", "폐기물 이미지" 등으로 검색해
   활용 가능한 공개 데이터셋이 있는지 먼저 확인한다.
2. 부족하면 `scripts/collect_data.py`로 직접 촬영한 `data/raw/` 이미지를
   Roboflow 또는 CVAT에 업로드해 폴리곤(세그멘테이션) 라벨을 단다.
   클래스명은 `bag` 하나만 사용한다.
3. YOLO-seg 포맷(polygon txt)으로 export 후 `data/dataset.yaml`에 경로를
   맞춰 넣는다:

   ```yaml
   path: ../data
   train: images/train
   val: images/val
   names:
     0: bag
   ```
```

---

### Task 9: 학습 스크립트

**Files:**
- Create: `scripts/train.py`

**Interfaces:**
- Consumes: `data/dataset.yaml` (Task 8에서 라벨링 완료 후 존재)
- Produces: `models/` 아래 학습된 가중치 파일 (`best.pt`)

- [ ] **Step 1: 학습 CLI 래퍼 작성**

`scripts/train.py`:

```python
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--project", default="models")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, project=args.project, name="bagvision")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 수동 검증 (라벨링된 데이터 필요)**

Run: `python scripts/train.py --epochs 50`
Expected: `models/bagvision/weights/best.pt`가 생성되고, 학습 로그에 mask
mAP 지표가 출력됨. 데이터가 적으면 epoch을 늘리거나 augmentation을
조정하며 반복

---

### Task 10: 실 가중치로 `BagDetector` 통합 확인

**Files:**
- Modify: `tests/test_detector.py` (통합 테스트 추가, 조건부 skip)

**Interfaces:**
- Consumes: `models/bagvision/weights/best.pt` (Task 9 산출물)

- [ ] **Step 1: 조건부 통합 테스트 추가**

`tests/test_detector.py`에 추가:

```python
import os
import numpy as np
import pytest
from bagvision.detector import BagDetector

WEIGHTS_PATH = "models/bagvision/weights/best.pt"


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="학습된 가중치 없음 (Task 9 선행 필요)")
def test_detect_with_real_weights_runs_without_error():
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.3)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(dummy_image)

    assert isinstance(detections, list)
```

- [ ] **Step 2: 실행**

Run: `pytest tests/test_detector.py -v`
Expected: 가중치 파일이 없으면 `SKIPPED`, 있으면 `PASSED` (빈 이미지라 탐지
결과는 0개일 수 있으나 에러 없이 리스트 반환하면 통과)

- [ ] **Step 3: 전체 스위트 재확인**

Run: `pytest -q`
Expected: 지금까지의 모든 테스트 통과 (또는 통합 테스트 skip)

---

### Task 11: 라이브 데모

**Files:**
- Create: `scripts/live_demo.py`

**Interfaces:**
- Consumes: `RealSenseCamera`, `BagDetector`, `TargetStabilizer`, `RecognitionPipeline` (Task 3–7, 9)

- [ ] **Step 1: 데모 스크립트 작성**

`scripts/live_demo.py`:

```python
import cv2
import numpy as np
from bagvision.realsense_capture import RealSenseCamera
from bagvision.detector import BagDetector
from bagvision.stabilizer import TargetStabilizer
from bagvision.pipeline import RecognitionPipeline

WEIGHTS_PATH = "models/bagvision/weights/best.pt"


def overlay_mask(color_image, detection):
    if detection is None:
        return color_image
    overlay = color_image.copy()
    overlay[detection.mask] = (0, 255, 0)
    return cv2.addWeighted(color_image, 0.6, overlay, 0.4, 0)


def main():
    camera = RealSenseCamera()
    detector = BagDetector(weights_path=WEIGHTS_PATH, conf_threshold=0.4)
    stabilizer = TargetStabilizer(window_size=15)
    pipeline = RecognitionPipeline(detector, camera, stabilizer)

    camera.start()
    print("c: 안정된 타겟 캡처, q: 종료")
    try:
        while True:
            detection = pipeline.step()
            color, _, _ = camera.get_frames()
            vis = overlay_mask(color, detection)
            cv2.imshow("live_demo (c=capture, q=quit)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                try:
                    target = pipeline.capture_stable_target(max_frames=60)
                    print(f"확정된 타겟: pos={target.position_xyz}, size={target.size_estimate}, "
                          f"n_frames={target.n_frames_averaged}")
                except TimeoutError as exc:
                    print(f"캡처 실패: {exc}")
            elif key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 수동 검증 (하드웨어 + 학습된 가중치 필요)**

Run: `python scripts/live_demo.py`
Expected: 실시간 화면에 봉투 마스크가 초록색으로 오버레이되고, `c`를 눌렀을 때
0.3~2.0m 범위에서 확정된 3D 위치/크기 값이 콘솔에 출력됨. 여러 거리에서
반복 캡처하며 값의 일관성(반복 정밀도)을 눈으로 확인

---

## Self-Review Notes

- **스펙 커버리지:** 데이터 파이프라인(Task 8), 모델(Task 9),
  뎁스 융합 모듈(Task 3), 출력 인터페이스의 연속/단일 캡처 API(Task 7),
  eye-in-hand 근접 제약 반영(Task 3의 `valid_range`/`floor_z`, Task 7의
  `capture_stable_target`)까지 스펙의 모든 섹션이 태스크로 매핑됨.
- **플레이스홀더 스캔:** "TODO"/"나중에 구현" 등 표현 없음. 하드웨어/데이터
  의존 단계는 "수동 검증"으로 명시하고 정확한 실행 명령과 기대 결과를 기술함.
- **타입 일관성:** `Detection`/`DepthEstimate`/`StableTarget`/`CameraIntrinsics`
  필드명이 Task 2 정의 이후 모든 태스크에서 동일하게 사용됨
  (`position_xyz`, `size_estimate`, `mask`, `class_name` 등).
