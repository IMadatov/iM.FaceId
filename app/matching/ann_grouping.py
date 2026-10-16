"""ANN-graph grouping: top-K neighbor edges + connected components."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.matching.grouping import UnionFind, components_from_union_find

SearchFn = Callable[[np.ndarray, int, float], list[tuple[str, float]]]


def build_ann_groups(
    items: list[tuple[str, np.ndarray]],
    search_fn: SearchFn,
    *,
    min_score: float,
    top_k: int,
    min_size: int = 2,
) -> list[list[str]]:
    """Group face_ids by ANN neighbor graph (cosine score >= min_score).

    ``search_fn(vector, limit, min_score)`` must return ``(face_id, score)`` hits
    ordered by descending score. Self-hits are ignored.
    """
    if not items or min_size < 1 or top_k < 1:
        return []

    ids = [face_id for face_id, _ in items]
    id_to_idx = {face_id: i for i, face_id in enumerate(ids)}
    uf = UnionFind(len(ids))
    limit = top_k + 1  # room for the query point itself when present in results

    for face_id, vector in items:
        i = id_to_idx[face_id]
        hits = search_fn(vector, limit, min_score)
        linked = 0
        for other_id, score in hits:
            if other_id == face_id or score < min_score:
                continue
            j = id_to_idx.get(other_id)
            if j is None:
                continue
            uf.union(i, j)
            linked += 1
            if linked >= top_k:
                break

    return components_from_union_find(ids, uf, min_size=min_size)
