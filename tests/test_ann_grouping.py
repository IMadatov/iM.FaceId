import numpy as np

from app.matching.ann_grouping import build_ann_groups


def _unit(vec: list[float]) -> np.ndarray:
    a = np.asarray(vec, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_ann_two_similar_one_distant():
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([0.95, 0.05, 0.0])
    c = _unit([0.0, 1.0, 0.0])
    items = [("a", a), ("b", b), ("c", c)]
    catalog = dict(items)

    def search_fn(vector, limit, min_score):
        scored = [
            (fid, float(np.dot(vector, vec)))
            for fid, vec in catalog.items()
        ]
        scored = [x for x in scored if x[1] >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    groups = build_ann_groups(
        items, search_fn, min_score=0.90, top_k=2, min_size=2
    )
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b"}


def test_ann_chain_connects_transitive():
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([0.92, 0.08, 0.0])
    c = _unit([0.84, 0.16, 0.0])
    items = [("a", a), ("b", b), ("c", c)]
    catalog = dict(items)

    def search_fn(vector, limit, min_score):
        scored = [
            (fid, float(np.dot(vector, vec)))
            for fid, vec in catalog.items()
        ]
        scored = [x for x in scored if x[1] >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    groups = build_ann_groups(
        items, search_fn, min_score=0.85, top_k=1, min_size=2
    )
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b", "c"}


def test_ann_empty_and_singleton():
    assert build_ann_groups([], lambda *_: [], min_score=0.4, top_k=5) == []
    v = _unit([1.0, 0.0])
    groups = build_ann_groups(
        [("only", v)],
        lambda vector, limit, min_score: [("only", 1.0)],
        min_score=0.4,
        top_k=5,
        min_size=2,
    )
    assert groups == []
