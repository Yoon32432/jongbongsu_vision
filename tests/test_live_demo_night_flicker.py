import importlib.util
from pathlib import Path

import numpy as np

from bagvision.types import Detection, DetectionEvaluation

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "live_demo_night.py"


def _load_flicker_filter():
    spec = importlib.util.spec_from_file_location("live_demo_night", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlickerFilter


def _make_eval(bbox):
    det = Detection(class_id=0, class_name="bag", confidence=0.9, mask=np.zeros((1, 1), dtype=bool), bbox=bbox)
    return DetectionEvaluation(detection=det, verified=True, color_ratio=1.0, mean_hsv=None, bulge_height=None)


def test_flicker_filter_survives_short_miss():
    FlickerFilter = _load_flicker_filter()
    f = FlickerFilter(miss_tolerance=3, iou_threshold=0.3)

    f.update([_make_eval((0, 0, 10, 10))])
    f.update([])
    tracked = f.update([])
    assert len(tracked) == 1
    assert tracked[0].detection.bbox == (0, 0, 10, 10)


def test_flicker_filter_drops_after_miss_tolerance_exceeded():
    FlickerFilter = _load_flicker_filter()
    f = FlickerFilter(miss_tolerance=2, iou_threshold=0.3)

    f.update([_make_eval((0, 0, 10, 10))])
    f.update([])
    f.update([])
    tracked = f.update([])
    assert tracked == []


def test_flicker_filter_suppresses_overlapping_duplicate():
    FlickerFilter = _load_flicker_filter()
    f = FlickerFilter(miss_tolerance=5, iou_threshold=0.3, overlap_suppress_iou=0.1)

    f.update([_make_eval((0, 0, 10, 10))])
    # box shifted just enough to miss the 0.3 match threshold, but still
    # overlaps the stale track's box - should replace it, not duplicate it
    tracked = f.update([_make_eval((6, 0, 16, 10))])
    assert len(tracked) == 1
    assert tracked[0].detection.bbox == (6, 0, 16, 10)


def test_flicker_filter_keeps_distinct_nonoverlapping_tracks():
    FlickerFilter = _load_flicker_filter()
    f = FlickerFilter(miss_tolerance=5, iou_threshold=0.3, overlap_suppress_iou=0.1)

    tracked = f.update([_make_eval((0, 0, 10, 10)), _make_eval((100, 0, 110, 10))])
    assert len(tracked) == 2
