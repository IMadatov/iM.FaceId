import numpy as np
import pytest
from app.matching.cosine import cosine_similarity, is_match, cosine_distance


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


def test_cosine_distance_identical_is_zero():
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert cosine_distance(v, v) == 0.0


def test_cosine_distance_orthogonal_is_one():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_distance(a, b) - 1.0) < 1e-6
