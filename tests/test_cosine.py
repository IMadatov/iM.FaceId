import numpy as np
import pytest
from app.matching.cosine import cosine_similarity, is_match


def test_identical_vectors_score_one():
    v = np.ones(512, dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vectors_score_near_zero():
    a = np.zeros(512, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(512, dtype=np.float32)
    b[1] = 1.0
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


def test_is_match_boundary():
    assert is_match(0.40, 0.40) is True
    assert is_match(0.3999, 0.40) is False
