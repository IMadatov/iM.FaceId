import base64

import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.config import Settings
from app.deps import settings_dep
from app.main import create_app
from app.pipeline.fake import FakeFacePipeline
from app.store.memory import MemoryFaceStore
from tests.synthetic_face import flat_replay, turning_head


def _client():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    return TestClient(app)


def _override_settings(app, **kwargs) -> Settings:
    data = Settings().model_dump()
    data.update(kwargs)
    settings = Settings(**data)
    app.dependency_overrides[settings_dep] = lambda: settings
    return settings


def _img_bytes(val: int = 120, size: int = 80) -> bytes:
    img = np.full((size, size, 3), val, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _textured_img_bytes(val: int = 120, size: int = 200) -> bytes:
    img = np.full((size, size, 3), val, dtype=np.uint8)
    img[::4, :] = np.clip(val + 80, 0, 255)
    img[:, ::4] = np.clip(val + 80, 0, 255)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _blurry_img_bytes(val: int = 120, size: int = 200) -> bytes:
    img = np.full((size, size, 3), val, dtype=np.uint8)
    img[::4, :] = np.clip(val + 80, 0, 255)
    img[:, ::4] = np.clip(val + 80, 0, 255)
    img = cv2.GaussianBlur(img, (31, 31), 0)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _face_json(img: bytes, **kwargs):
    body = {"image_base64": base64.b64encode(img).decode("ascii")}
    body.update(kwargs)
    return body


def test_enroll_update_verify_search_delete_flow():
    c = _client()
    r = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    assert r.status_code == 201
    face_id = r.json()["face_id"]

    r2 = c.put(f"/v1/faces/{face_id}", json=_face_json(_img_bytes(121)))
    assert r2.status_code == 200
    assert r2.json()["updated"] is True

    r3 = c.post(
        f"/v1/faces/{face_id}/verify",
        json=_face_json(_img_bytes(121)),
    )
    assert r3.status_code == 200
    body = r3.json()
    assert body["face_id"] == face_id
    assert "score" in body and "match" in body

    r4 = c.post("/v1/faces", json=_face_json(_img_bytes(121)))
    assert r4.status_code == 201
    r5 = c.post(
        "/v1/faces/search",
        json=_face_json(_img_bytes(121), limit=5),
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
    r1 = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    r2 = c.post("/v1/faces", json=_face_json(_img_bytes(130)))
    assert r1.status_code == r2.status_code == 201
    id_a, id_b = r1.json()["face_id"], r2.json()["face_id"]

    r = c.delete("/v1/faces")
    assert r.status_code == 200
    body = r.json()
    assert body["cleared"] is True
    assert body["deleted_count"] == 2

    assert c.post("/v1/faces/groups/list", json={}).json()["groups"] == []
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

    r1 = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    r2 = c.post("/v1/faces", json=_face_json(_img_bytes(140)))
    assert r1.status_code == r2.status_code == 201
    id_a, id_b = r1.json()["face_id"], r2.json()["face_id"]

    assert c.get("/v1/faces/count").json()["count"] == 2
    listed = c.post("/v1/faces/list", json={"limit": 1, "offset": 0})
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
        json=_face_json(_img_bytes()),
    )
    assert r.status_code == 404


def test_empty_image_400():
    c = _client()
    r = c.post("/v1/faces", json={"image_base64": ""})
    assert r.status_code == 400


def test_no_face_422():
    c = _client()
    r = c.post("/v1/faces", json=_face_json(_img_bytes(0)))
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
    r = c.post("/v1/faces", json=_face_json(b"not-an-image"))
    assert r.status_code == 400


def test_multiple_faces_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline(faces_count=2)
    app.state.store = MemoryFaceStore()
    c = TestClient(app)
    r = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
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
    c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    r = c.post(
        "/v1/faces/search",
        json=_face_json(_img_bytes(120), limit=999),
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
        json=_face_json(_img_bytes()),
    )
    assert r.status_code == 503


def test_groups_clusters_similar_enrollments():
    c = _client()
    r1 = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    r2 = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    r3 = c.post("/v1/faces", json=_face_json(_img_bytes(200)))
    assert r1.status_code == r2.status_code == r3.status_code == 201
    id_a, id_b, id_c = r1.json()["face_id"], r2.json()["face_id"], r3.json()["face_id"]

    assert c.post("/v1/faces/groups/list", json={}).json()["groups"] == []

    started = c.post(
        "/v1/faces/groups/rebuild", json={"min_score": 0.99}
    )
    assert started.status_code == 202
    run_id = started.json()["run_id"]

    status = c.get(f"/v1/faces/groups/rebuild/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    r = c.post("/v1/faces/groups/list", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.99
    assert body["run_id"] == run_id
    assert body["built_at"] is not None
    assert len(body["groups"]) == 1
    assert set(body["groups"][0]["face_ids"]) == {id_a, id_b}
    assert body["groups"][0]["size"] == 2
    assert id_c not in body["groups"][0]["face_ids"]


def test_groups_rebuild_list_sorted_by_started_at_desc():
    c = _client()
    assert c.get("/v1/faces/groups/rebuild").json() == {"runs": []}

    first = c.post(
        "/v1/faces/groups/rebuild", json={"min_score": 0.99}
    )
    assert first.status_code == 202
    first_id = first.json()["run_id"]
    second = c.post(
        "/v1/faces/groups/rebuild", json={"min_score": 0.98}
    )
    assert second.status_code == 202
    second_id = second.json()["run_id"]

    listed = c.get("/v1/faces/groups/rebuild")
    assert listed.status_code == 200
    runs = listed.json()["runs"]
    assert [r["run_id"] for r in runs] == [second_id, first_id]
    assert all(r["status"] == "completed" for r in runs)
    assert runs[0]["started_at"] is not None
    assert runs[0]["finished_at"] is not None
    assert runs[0]["threshold"] == 0.98


def test_groups_rejects_min_size_below_two():
    c = _client()
    r = c.post("/v1/faces/groups/list", json={"min_size": 1})
    assert r.status_code == 400
    r2 = c.post("/v1/faces/groups/rebuild", json={"min_size": 1})
    assert r2.status_code == 400


def test_groups_rebuild_conflict_when_locked():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    service = app.state.group_rebuild
    assert service._lock.acquire(blocking=False)
    try:
        c = TestClient(app)
        r = c.post("/v1/faces/groups/rebuild", json={})
        assert r.status_code == 409
    finally:
        service._lock.release()


def test_enroll_liveness_fail_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app.state.liveness = FakeLivenessChecker(force_spoof=True)
    c = TestClient(app)
    r = c.post("/v1/faces", json=_face_json(_img_bytes(120), liveness=True))
    assert r.status_code == 422
    assert "liveness" in r.json()["detail"].lower()


def test_enroll_liveness_pass_returns_score():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    c = TestClient(app)
    r = c.post("/v1/faces", json=_face_json(_img_bytes(120), liveness=True))
    assert r.status_code == 201
    assert r.json()["liveness_score"] is not None
    assert r.json()["liveness_score"] >= 0.5


def test_enroll_liveness_unavailable_503():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    from app.pipeline.fake_liveness import UnavailableLivenessChecker

    app.state.liveness = UnavailableLivenessChecker()
    c = TestClient(app)
    r = c.post("/v1/faces", json=_face_json(_img_bytes(120), liveness=True))
    assert r.status_code == 503


def test_enroll_blurry_with_liveness_quality_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=50.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=80.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        json=_face_json(_blurry_img_bytes(120), liveness=True),
    )
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "quality" in detail or "blur" in detail
    assert "liveness" not in detail


def test_enroll_blurry_without_liveness_still_ok():
    c = _client()
    r = c.post("/v1/faces", json=_face_json(_blurry_img_bytes(120)))
    assert r.status_code == 201


def test_enroll_sharp_with_liveness_passes_quality():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=50.0,
        face_brightness_min=40.0,
        face_brightness_max=220.0,
        face_min_side_px=80.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        json=_face_json(_textured_img_bytes(120), liveness=True),
    )
    assert r.status_code == 201
    assert r.json()["liveness_score"] is not None


def test_enroll_client_blur_override_stricter_422():
    """Server blur disabled; client JSON requires blur → quality 422."""
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        json=_face_json(
            _blurry_img_bytes(120),
            liveness=True,
            blur_min_variance=50,
        ),
    )
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "quality" in detail or "blur" in detail


def test_verify_client_match_threshold():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(app, face_match_threshold=0.99)
    c = TestClient(app)
    enroll = c.post("/v1/faces", json=_face_json(_img_bytes(120)))
    face_id = enroll.json()["face_id"]
    r = c.post(
        f"/v1/faces/{face_id}/verify",
        json=_face_json(_img_bytes(120), match_threshold=0.0),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.0
    assert body["match"] is True


def test_enroll_client_liveness_threshold_stricter_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
        liveness_threshold=0.50,
    )
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app.state.liveness = FakeLivenessChecker(score=0.60)
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        json=_face_json(
            _img_bytes(120),
            liveness=True,
            liveness_threshold=0.90,
        ),
    )
    assert r.status_code == 422
    assert "liveness" in r.json()["detail"].lower()


def test_enroll_unknown_json_field_422():
    c = _client()
    r = c.post(
        "/v1/faces",
        json=_face_json(_img_bytes(120), not_a_field=1),
    )
    assert r.status_code == 422


def test_groups_rebuild_status_not_found():
    c = _client()
    r = c.get("/v1/faces/groups/rebuild/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_check_face_ok_does_not_store():
    c = _client()
    assert c.get("/v1/faces/count").json()["count"] == 0
    r = c.post("/v1/faces/check", json=_face_json(_img_bytes(120)))
    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "liveness_score": None}
    assert "face_id" not in body
    assert "embedding" not in body
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_check_face_liveness_pass_returns_score():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces/check",
        json=_face_json(_img_bytes(120), liveness=True),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["liveness_score"] is not None
    assert body["liveness_score"] >= 0.5
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_check_face_blurry_with_liveness_quality_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=50.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces/check",
        json=_face_json(_blurry_img_bytes(120), liveness=True),
    )
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "quality" in detail or "blur" in detail
    assert "liveness" not in detail
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_check_face_liveness_fail_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app.state.liveness = FakeLivenessChecker(force_spoof=True)
    c = TestClient(app)
    r = c.post(
        "/v1/faces/check",
        json=_face_json(_img_bytes(120), liveness=True),
    )
    assert r.status_code == 422
    assert "liveness" in r.json()["detail"].lower()
    assert c.get("/v1/faces/count").json()["count"] == 0


def _liveness_json(images: list[bytes], **kwargs):
    body = {
        "images_base64": [base64.b64encode(i).decode("ascii") for i in images]
    }
    body.update(kwargs)
    return body


def _liveness_app(**settings_kwargs):
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
        **settings_kwargs,
    )
    return app


def test_liveness_distinct_frames_pass():
    app = _liveness_app()
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121), _img_bytes(122)]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["frame_count"] == 3
    assert body["liveness_score"] >= 0.5
    assert body["inter_frame_distance"] > 0.05
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_liveness_identical_frames_rejected_as_too_similar():
    app = _liveness_app()
    c = TestClient(app)
    frame = _img_bytes(120)
    r = c.post("/v1/faces/liveness", json=_liveness_json([frame, frame]))
    assert r.status_code == 422
    assert "similar" in r.json()["detail"].lower()
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_liveness_low_mean_score_rejected():
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app = _liveness_app()
    app.state.liveness = FakeLivenessChecker(force_spoof=True)
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121)]),
    )
    assert r.status_code == 422
    assert "score" in r.json()["detail"].lower()


def test_liveness_single_frame_rejected_by_validation():
    app = _liveness_app()
    c = TestClient(app)
    r = c.post("/v1/faces/liveness", json=_liveness_json([_img_bytes(120)]))
    assert r.status_code == 422


def test_liveness_too_many_frames_rejected():
    app = _liveness_app()  # liveness_max_frames default = 5
    c = TestClient(app)
    frames = [_img_bytes(120 + i) for i in range(6)]
    r = c.post("/v1/faces/liveness", json=_liveness_json(frames))
    assert r.status_code == 422
    assert "many" in r.json()["detail"].lower()


def test_liveness_exactly_max_frames_pass():
    app = _liveness_app()  # max = 5
    c = TestClient(app)
    frames = [_img_bytes(120 + i) for i in range(5)]
    r = c.post("/v1/faces/liveness", json=_liveness_json(frames))
    assert r.status_code == 200
    assert r.json()["frame_count"] == 5


def test_liveness_unknown_field_rejected():
    app = _liveness_app()
    c = TestClient(app)
    body = _liveness_json([_img_bytes(120), _img_bytes(121)])
    body["liveness"] = True
    r = c.post("/v1/faces/liveness", json=body)
    assert r.status_code == 422


def test_liveness_bad_frame_names_index():
    app = _liveness_app()
    c = TestClient(app)
    # second frame is all-black -> FakeFacePipeline raises "no face"
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(0)]),
    )
    assert r.status_code == 422
    assert "frame[1]" in r.json()["detail"]


def test_liveness_data_uri_prefix_decodes():
    app = _liveness_app()
    c = TestClient(app)
    b64 = base64.b64encode(_img_bytes(120)).decode("ascii")
    body = {
        "images_base64": [
            f"data:image/png;base64,{b64}",
            base64.b64encode(_img_bytes(121)).decode("ascii"),
        ]
    }
    r = c.post("/v1/faces/liveness", json=body)
    assert r.status_code == 200
    assert r.json()["frame_count"] == 2


def test_liveness_model_unavailable_503():
    from app.pipeline.fake_liveness import UnavailableLivenessChecker

    app = _liveness_app()
    app.state.liveness = UnavailableLivenessChecker()
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121)]),
    )
    assert r.status_code == 503


def test_liveness_rejects_a_flat_frame_sequence():
    """A tilted photo or replayed screen is planar, so the nose shows no parallax."""
    app = _liveness_app()
    app.state.pipeline = FakeFacePipeline(kps_frames=flat_replay([-25.0, 0.0, 25.0]))
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121), _img_bytes(122)]),
    )
    assert r.status_code == 422
    assert "flat" in r.json()["detail"].lower()


def test_liveness_rejects_insufficient_head_movement():
    """Without rotation a real face shows no parallax either, so the score is meaningless."""
    app = _liveness_app()
    app.state.pipeline = FakeFacePipeline(kps_frames=turning_head([0.0, 0.5]))
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121)]),
    )
    assert r.status_code == 422
    assert "movement" in r.json()["detail"].lower()


def test_liveness_reports_planarity_residual_for_a_real_turn():
    app = _liveness_app()
    app.state.pipeline = FakeFacePipeline(kps_frames=turning_head([-20.0, 0.0, 20.0]))
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121), _img_bytes(122)]),
    )
    assert r.status_code == 200
    assert r.json()["planarity_residual"] > 0.06


def test_liveness_without_keypoints_is_unavailable():
    """Fail closed: a pack with no keypoint head cannot run the planarity check."""
    app = _liveness_app()
    app.state.pipeline = FakeFacePipeline(kps_frames=[None, None])
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121)]),
    )
    assert r.status_code == 503
    assert "keypoint" in r.json()["detail"].lower()


def test_liveness_planarity_check_disabled_at_zero_threshold():
    """Zero relaxes the check, matching how the face-quality thresholds disable."""
    app = _liveness_app(liveness_min_planarity=0.0, liveness_min_view_change=0.0)
    app.state.pipeline = FakeFacePipeline(kps_frames=flat_replay([-25.0, 0.0, 25.0]))
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121), _img_bytes(122)]),
    )
    assert r.status_code == 200


def test_liveness_client_can_tighten_planarity_threshold():
    app = _liveness_app()
    app.state.pipeline = FakeFacePipeline(kps_frames=turning_head([-20.0, 0.0, 20.0]))
    c = TestClient(app)
    body = _liveness_json([_img_bytes(120), _img_bytes(121), _img_bytes(122)])
    body["liveness_min_planarity"] = 0.95
    r = c.post("/v1/faces/liveness", json=body)
    assert r.status_code == 422
    assert "flat" in r.json()["detail"].lower()
