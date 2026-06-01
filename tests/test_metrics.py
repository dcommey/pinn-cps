import numpy as np

from pinncps.eval.metrics import compute_detection_metrics, detection_delay
from pinncps.models.detector import _smooth


def test_perfect_separation_gives_f1_one():
    scores = np.array([[0.0, 0.0, 5.0, 5.0, 5.0]])
    labels = np.array([[0, 0, 1, 1, 1]])
    m = compute_detection_metrics(scores, labels, threshold=1.0)
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0


def test_detection_delay_zero_when_threshold_met_immediately():
    scores = np.array([[0.0, 0.0, 5.0, 5.0, 5.0]])
    labels = np.array([[0, 0, 1, 1, 1]])
    d = detection_delay(scores, labels, threshold=1.0)
    assert d["mean_delay"] == 0.0
    assert d["miss_rate"] == 0.0


def test_detection_delay_records_misses():
    scores = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    labels = np.array([[0, 0, 1, 1, 1]])
    d = detection_delay(scores, labels, threshold=1.0)
    assert d["miss_rate"] == 1.0


def test_smoothing_is_causal_trailing_average():
    scores = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    smoothed = _smooth(scores, window=3)
    np.testing.assert_allclose(smoothed, [0.0, 0.0, 10.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0])
