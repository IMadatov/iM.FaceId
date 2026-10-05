import numpy as np
import pytest

from app.errors import LivenessFailedError
from app.pipeline.fake_liveness import FakeLivenessChecker


def test_fake_liveness_passes_by_default():
    checker = FakeLivenessChecker()
    img = np.full((80, 80, 3), 120, dtype=np.uint8)
    score = checker.ensure_live(img, threshold=0.5)
    assert score >= 0.5
    assert checker.ready()


def test_fake_liveness_force_spoof_raises():
    checker = FakeLivenessChecker(force_spoof=True)
    img = np.full((80, 80, 3), 120, dtype=np.uint8)
    with pytest.raises(LivenessFailedError):
        checker.ensure_live(img, threshold=0.5)
