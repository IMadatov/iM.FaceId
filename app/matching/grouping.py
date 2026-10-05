"""On-demand grouping of similar face embeddings via connected components."""

from __future__ import annotations

import numpy as np

from app.matching.cosine import cosine_similarity


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            self.parent[ri] = rj
        elif self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
        else:
            self.parent[rj] = ri
            self.rank[ri] += 1


def group_similar_faces(
    items: list[tuple[str, np.ndarray]],
    *,
    min_score: float,
    min_size: int = 2,
) -> list[list[str]]:
    """Return groups of face_ids linked by cosine similarity >= min_score.

    Singletons (and groups smaller than min_size) are omitted.
    Group member order is stable by first-seen face_id order within the component.
    """
    if not items or min_size < 1:
        return []

    ids = [face_id for face_id, _ in items]
    vectors = [vec.astype(np.float32, copy=False).reshape(-1) for _, vec in items]
    n = len(ids)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(vectors[i], vectors[j]) >= min_score:
                uf.union(i, j)

    buckets: dict[int, list[str]] = {}
    for i, face_id in enumerate(ids):
        root = uf.find(i)
        buckets.setdefault(root, []).append(face_id)

    groups = [members for members in buckets.values() if len(members) >= min_size]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups
