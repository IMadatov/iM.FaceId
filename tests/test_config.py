from app.config import Settings


def test_multiframe_liveness_defaults():
    s = Settings()
    assert s.liveness_max_frames == 5
    assert s.liveness_min_distance == 0.05
