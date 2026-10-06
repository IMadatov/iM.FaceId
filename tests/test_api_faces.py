import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.main import create_app
from app.pipeline.fake import FakeFacePipeline
from app.store.memory import MemoryFaceStore

def _client():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    return TestClient(app)

def _img_bytes(val: int = 120) -> bytes:
    img = np.full((80, 80, 3), val, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()

def test_enroll_update_verify_search_delete_flow():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    assert r.status_code == 201
    face_id = r.json()["face_id"]

    r2 = c.put(f"/v1/faces/{face_id}", files={"image": ("b.png", _img_bytes(121), "image/png")})
    assert r2.status_code == 200
    assert r2.json()["updated"] is True

    r3 = c.post(
        f"/v1/faces/{face_id}/verify",
        files={"image": ("c.png", _img_bytes(121), "image/png")},
    )
    assert r3.status_code == 200
    body = r3.json()
    assert body["face_id"] == face_id
    assert "score" in body and "match" in body

    # second enroll then search
    r4 = c.post("/v1/faces", files={"image": ("d.png", _img_bytes(121), "image/png")})
    assert r4.status_code == 201
    r5 = c.post(
        "/v1/faces/search",
        files={"image": ("e.png", _img_bytes(121), "image/png")},
        params={"limit": 5},
    )
    assert r5.status_code == 200
    assert "results" in r5.json()

    r6 = c.delete(f"/v1/faces/{face_id}")
    assert r6.status_code == 200
    assert r6.json()["deleted"] is True
    r7 = c.delete(f"/v1/faces/{face_id}")
    assert r7.status_code == 404


def test_clear_all_faces():
    c = _client()
    r1 = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    r2 = c.post("/v1/faces", files={"image": ("b.png", _img_bytes(130), "image/png")})
    assert r1.status_code == r2.status_code == 201
    id_a, id_b = r1.json()["face_id"], r2.json()["face_id"]

    r = c.delete("/v1/faces")
    assert r.status_code == 200
    body = r.json()
    assert body["cleared"] is True
    assert body["deleted_count"] == 2

    assert c.get("/v1/faces/groups").json()["groups"] == []
    assert c.delete(f"/v1/faces/{id_a}").status_code == 404
    assert c.delete(f"/v1/faces/{id_b}").status_code == 404

    r_empty = c.delete("/v1/faces")
    assert r_empty.status_code == 200
    assert r_empty.json()["deleted_count"] == 0


def test_count_list_get_stats():
    c = _client()
    assert c.get("/v1/faces/count").json() == {"count": 0}
    assert c.get("/v1/stats").json() == {
        "faces_count": 0,
        "store_ok": True,
        "pipeline_ok": True,
    }

    r1 = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    r2 = c.post("/v1/faces", files={"image": ("b.png", _img_bytes(140), "image/png")})
    assert r1.status_code == r2.status_code == 201
    id_a, id_b = r1.json()["face_id"], r2.json()["face_id"]

    assert c.get("/v1/faces/count").json()["count"] == 2
    listed = c.get("/v1/faces", params={"limit": 1, "offset": 0})
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert len(body["face_ids"]) == 1
    assert body["face_ids"][0] in {id_a, id_b}

    got = c.get(f"/v1/faces/{id_a}")
    assert got.status_code == 200
    assert got.json() == {"face_id": id_a, "exists": True}

    missing = c.get("/v1/faces/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404

    stats = c.get("/v1/stats").json()
    assert stats["faces_count"] == 2
    assert stats["store_ok"] is True
    assert stats["pipeline_ok"] is True

def test_verify_unknown_404():
    c = _client()
    r = c.post(
        "/v1/faces/00000000-0000-0000-0000-000000000000/verify",
        files={"image": ("a.png", _img_bytes(), "image/png")},
    )
    assert r.status_code == 404

def test_empty_image_400():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", b"", "image/png")})
    assert r.status_code == 400

def test_no_face_422():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(0), "image/png")})
    assert r.status_code == 422

def test_health_503_when_store_down():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    store = MemoryFaceStore()
    store.ping = lambda: False  # type: ignore
    app.state.store = store
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 503

def test_corrupt_image_400():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", b"not-an-image", "image/png")})
    assert r.status_code == 400

def test_multiple_faces_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline(faces_count=2)
    app.state.store = MemoryFaceStore()
    c = TestClient(app)
    r = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    assert r.status_code == 422

def test_search_limit_clamped_to_max():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    store = MemoryFaceStore()
    received: dict[str, int | float] = {}
    original_search = store.search

    def spy_search(vector, *, limit, min_score):
        received["limit"] = limit
        received["min_score"] = min_score
        return original_search(vector, limit=limit, min_score=min_score)

    store.search = spy_search  # type: ignore[method-assign]
    app.state.store = store
    c = TestClient(app)
    c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    r = c.post(
        "/v1/faces/search",
        files={"image": ("b.png", _img_bytes(120), "image/png")},
        params={"limit": 999},
    )
    assert r.status_code == 200
    assert received["limit"] == 20

def test_verify_503_when_store_raises_dependency():
    from app.errors import DependencyUnavailableError
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    store = MemoryFaceStore()
    def boom(*args, **kwargs):
        raise DependencyUnavailableError("qdrant down")
    store.get = boom  # type: ignore
    app.state.store = store
    c = TestClient(app)
    r = c.post(
        "/v1/faces/11111111-1111-1111-1111-111111111111/verify",
        files={"image": ("a.png", _img_bytes(), "image/png")},
    )
    assert r.status_code == 503


def test_groups_clusters_similar_enrollments():
    c = _client()
    r1 = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    r2 = c.post("/v1/faces", files={"image": ("b.png", _img_bytes(120), "image/png")})
    r3 = c.post("/v1/faces", files={"image": ("c.png", _img_bytes(200), "image/png")})
    assert r1.status_code == r2.status_code == r3.status_code == 201
    id_a, id_b, id_c = r1.json()["face_id"], r2.json()["face_id"], r3.json()["face_id"]

    r = c.get("/v1/faces/groups", params={"min_score": 0.99})
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.99
    # identical fake embeddings for same mean image → one pair group; distant singleton omitted
    assert len(body["groups"]) == 1
    assert set(body["groups"][0]["face_ids"]) == {id_a, id_b}
    assert body["groups"][0]["size"] == 2
    assert id_c not in body["groups"][0]["face_ids"]


def test_groups_rejects_min_size_below_two():
    c = _client()
    r = c.get("/v1/faces/groups", params={"min_size": 1})
    assert r.status_code == 400


def test_enroll_liveness_fail_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app.state.liveness = FakeLivenessChecker(force_spoof=True)
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        files={"image": ("a.png", _img_bytes(120), "image/png")},
        data={"liveness": "true"},
    )
    assert r.status_code == 422
    assert "liveness" in r.json()["detail"].lower()


def test_enroll_liveness_pass_returns_score():
    c = _client()
    r = c.post(
        "/v1/faces",
        files={"image": ("a.png", _img_bytes(120), "image/png")},
        data={"liveness": "true"},
    )
    assert r.status_code == 201
    assert r.json()["liveness_score"] is not None
    assert r.json()["liveness_score"] >= 0.5


def test_enroll_liveness_unavailable_503():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    from app.pipeline.fake_liveness import UnavailableLivenessChecker

    app.state.liveness = UnavailableLivenessChecker()
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        files={"image": ("a.png", _img_bytes(120), "image/png")},
        data={"liveness": "true"},
    )
    assert r.status_code == 503


def test_groups_rejects_when_over_max_faces():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    store = MemoryFaceStore()
    store.list_all = lambda: [("x", np.ones(512, dtype=np.float32))] * 3  # type: ignore
    app.state.store = store
    # override settings via env is heavy; monkeypatch settings on dependency by
    # replacing get path — use a tiny max via wrapping list_face_groups config:
    from app.config import Settings

    original = Settings.model_fields["face_groups_max_faces"].default
    try:
        # Build client that uses settings with max 2
        class TinySettings(Settings):
            face_groups_max_faces: int = 2

        from app import deps

        app.dependency_overrides[deps.settings_dep] = lambda: TinySettings()
        c = TestClient(app)
        r = c.get("/v1/faces/groups")
        assert r.status_code == 400
        assert "too many faces" in r.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        _ = original
