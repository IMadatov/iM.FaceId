import numpy as np
import pytest
from app.store.memory import MemoryFaceStore
from app.matching.cosine import cosine_similarity


def test_upsert_get_delete():
    store = MemoryFaceStore()
    v = np.random.randn(512).astype(np.float32)
    store.upsert("id-1", v)
    got = store.get("id-1")
    assert got is not None
    assert cosine_similarity(got, v) == pytest.approx(1.0, abs=1e-5)
    assert store.delete("id-1") is True
    assert store.get("id-1") is None
    assert store.delete("id-1") is False


def test_search_returns_top_by_score():
    store = MemoryFaceStore()
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    near = base.copy()
    near[1] = 0.1
    far = np.zeros(512, dtype=np.float32)
    far[2] = 1.0
    store.upsert("near", near)
    store.upsert("far", far)
    hits = store.search(base, limit=5, min_score=0.0)
    assert hits[0][0] == "near"


def test_list_all_returns_copies():
    store = MemoryFaceStore()
    v = np.ones(512, dtype=np.float32)
    store.upsert("id-1", v)
    items = store.list_all()
    assert len(items) == 1
    assert items[0][0] == "id-1"
    items[0][1][0] = 0.0
    assert store.get("id-1")[0] == pytest.approx(1.0)
