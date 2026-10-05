from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import DependencyUnavailableError
from app.store.qdrant import QdrantFaceStore

FID = "11111111-1111-1111-1111-111111111111"


def make_store(client: MagicMock | None = None) -> tuple[QdrantFaceStore, MagicMock]:
    client = client or MagicMock()
    return QdrantFaceStore(client=client, collection="faces", dim=512), client


def test_init_creates_collection_when_missing():
    client = MagicMock()
    client.collection_exists.return_value = False
    make_store(client)
    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "faces"
    assert kwargs["vectors_config"].size == 512
    assert kwargs["vectors_config"].distance.value == "Cosine"


def test_init_skips_create_when_exists():
    client = MagicMock()
    client.collection_exists.return_value = True
    make_store(client)
    client.create_collection.assert_not_called()


def test_init_connection_error_wrapped():
    client = MagicMock()
    client.collection_exists.side_effect = ConnectionError("down")
    with pytest.raises(DependencyUnavailableError):
        make_store(client)


def test_upsert_calls_client():
    store, client = make_store()
    v = np.ones(512, dtype=np.float32)
    store.upsert(FID, v)
    assert client.upsert.called
    point = client.upsert.call_args.kwargs["points"][0]
    assert point.id == FID
    assert len(point.vector) == 512


def test_upsert_error_wrapped():
    store, client = make_store()
    client.upsert.side_effect = ConnectionError("down")
    with pytest.raises(DependencyUnavailableError):
        store.upsert(FID, np.ones(512, dtype=np.float32))


def test_get_missing_returns_none():
    store, client = make_store()
    client.retrieve.return_value = []
    assert store.get(FID) is None


def test_get_returns_vector():
    store, client = make_store()
    rec = MagicMock()
    rec.vector = [0.5] * 512
    client.retrieve.return_value = [rec]
    out = store.get(FID)
    assert out is not None
    assert out.dtype == np.float32
    assert out.shape == (512,)


def test_delete_false_when_missing():
    store, client = make_store()
    client.retrieve.return_value = []
    assert store.delete(FID) is False
    client.delete.assert_not_called()


def test_delete_true_when_present():
    store, client = make_store()
    client.retrieve.return_value = [MagicMock()]
    assert store.delete(FID) is True
    client.delete.assert_called_once()


def test_search_maps_points_to_tuples():
    store, client = make_store()
    p1, p2 = MagicMock(), MagicMock()
    p1.id, p1.score = "a", 0.9
    p2.id, p2.score = "b", 0.7
    client.query_points.return_value = MagicMock(points=[p1, p2])
    out = store.search(np.ones(512, dtype=np.float32), limit=5, min_score=0.6)
    assert out == [("a", pytest.approx(0.9)), ("b", pytest.approx(0.7))]
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["limit"] == 5
    assert kwargs["score_threshold"] == 0.6


def test_search_error_wrapped():
    store, client = make_store()
    client.query_points.side_effect = ConnectionError("down")
    with pytest.raises(DependencyUnavailableError):
        store.search(np.ones(512, dtype=np.float32), limit=1, min_score=0.0)


def test_ping_true_and_false():
    store, client = make_store()
    assert store.ping() is True
    client.get_collections.side_effect = ConnectionError("down")
    assert store.ping() is False
