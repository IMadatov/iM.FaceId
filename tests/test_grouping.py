import numpy as np
import pytest

from app.matching.grouping import group_similar_faces


def _unit(i: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_two_similar_one_distant_form_one_group():
    # a and b are identical → cosine 1.0; c orthogonal to both
    items = [
        ("a", _unit(0)),
        ("b", _unit(0)),
        ("c", _unit(1)),
    ]
    groups = group_similar_faces(items, min_score=0.40, min_size=2)
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b"}


def test_min_size_filters_singletons():
    items = [
        ("a", _unit(0)),
        ("b", _unit(1)),
    ]
    groups = group_similar_faces(items, min_score=0.40, min_size=2)
    assert groups == []


def test_empty_input():
    assert group_similar_faces([], min_score=0.40, min_size=2) == []


def test_chain_connects_transitive_group():
    # a~b and b~c above threshold ⇒ one component {a,b,c}
    a = _unit(0)
    b = a.copy()
    b[1] = 0.5
    b /= np.linalg.norm(b)
    c = b.copy()
    c[2] = 0.5
    c /= np.linalg.norm(c)
    items = [("a", a), ("b", b), ("c", c)]
    # ensure pairwise edges exist at a moderate threshold
    from app.matching.cosine import cosine_similarity

    assert cosine_similarity(a, b) >= 0.85
    assert cosine_similarity(b, c) >= 0.85
    groups = group_similar_faces(items, min_score=0.85, min_size=2)
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b", "c"}
