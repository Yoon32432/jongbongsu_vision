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


@dataclass
class DetectionEvaluation:
    detection: Detection
    verified: bool
    color_ratio: float
    mean_hsv: tuple[float, float, float] | None
    bulge_height: float | None
