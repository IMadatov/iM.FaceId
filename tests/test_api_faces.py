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
    c = _client()
    c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    r = c.post(
        "/v1/faces/search",
        files={"image": ("b.png", _img_bytes(120), "image/png")},
        params={"limit": 999},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 20

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
