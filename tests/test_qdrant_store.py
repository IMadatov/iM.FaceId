from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

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


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "", "123"])
def test_get_invalid_id_returns_none_without_calling_qdrant(bad_id):
    store, client = make_store()
    assert store.get(bad_id) is None
    client.retrieve.assert_not_called()


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "", "123"])
def test_delete_invalid_id_returns_false_without_calling_qdrant(bad_id):
    store, client = make_store()
    assert store.delete(bad_id) is False
    client.retrieve.assert_not_called()
    client.delete.assert_not_called()


def test_upsert_invalid_id_raises_value_error():
    store, client = make_store()
    with pytest.raises(ValueError):
        store.upsert("not-a-uuid", np.ones(512, dtype=np.float32))
    client.upsert.assert_not_called()


def test_non_connection_errors_not_wrapped():
    store, client = make_store()
    client.retrieve.side_effect = ValueError("bad request")
    with pytest.raises(ValueError):
        store.get(FID)


def test_qdrant_api_errors_wrapped():
    store, client = make_store()
    client.retrieve.side_effect = ResponseHandlingException(Exception("conn"))
    with pytest.raises(DependencyUnavailableError):
        store.get(FID)


def test_ping_true_and_false():
    store, client = make_store()
    assert store.ping() is True
    client.get_collections.side_effect = ConnectionError("down")
    assert store.ping() is False


def test_list_all_scrolls_points():
    store, client = make_store()
    rec = MagicMock()
    rec.id = FID
    rec.vector = np.ones(512, dtype=np.float32).tolist()
    client.scroll.return_value = ([rec], None)
    items = store.list_all()
    assert len(items) == 1
    assert items[0][0] == FID
    assert items[0][1].shape == (512,)
    client.scroll.assert_called_once()
    assert client.scroll.call_args.kwargs["with_vectors"] is True


def test_clear_deletes_and_recreates_collection():
    store, client = make_store()
    client.count.return_value = MagicMock(count=3)
    # wipe sees collection present; ensure_collection sees it missing after delete
    client.collection_exists.side_effect = [True, False]
    assert store.clear() == 3
    client.count.assert_called_once()
    client.delete_collection.assert_called_once_with(collection_name="faces")
    client.create_collection.assert_called()


def test_count_uses_qdrant_count():
    store, client = make_store()
    client.count.return_value = MagicMock(count=42)
    assert store.count() == 42


def test_list_ids_scrolls_and_pages():
    store, client = make_store()
    client.count.return_value = MagicMock(count=3)
    r1, r2, r3 = MagicMock(), MagicMock(), MagicMock()
    r1.id, r2.id, r3.id = "a", "b", "c"
    # first scroll page returns all three then stop
    client.scroll.return_value = ([r1, r2, r3], None)
    assert store.list_ids(limit=2, offset=1) == ["b", "c"]
    assert client.scroll.call_args.kwargs["with_vectors"] is False
