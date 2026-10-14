from app.pipeline.onnx_insightface import InsightFacePipeline


def test_insightface_passes_providers_and_ctx(monkeypatch):
    captured = {}

    class FakeFA:
        def __init__(self, name, root, providers):
            captured["providers"] = providers

        def prepare(self, ctx_id, det_size):
            captured["ctx_id"] = ctx_id
            captured["det_size"] = det_size

    monkeypatch.setattr(
        "insightface.app.FaceAnalysis",
        FakeFA,
        raising=False,
    )
    import app.pipeline.onnx_insightface as mod

    monkeypatch.setattr(mod, "FaceAnalysis", FakeFA, raising=False)

    import insightface.app as ia

    monkeypatch.setattr(ia, "FaceAnalysis", FakeFA)

    InsightFacePipeline(
        "/tmp",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ctx_id=0,
    )
    assert captured["providers"][0] == "CUDAExecutionProvider"
    assert captured["ctx_id"] == 0


def test_insightface_defaults_to_cpu_providers_when_omitted(monkeypatch):
    captured = {}

    class FakeFA:
        def __init__(self, name, root, providers):
            captured["providers"] = providers

        def prepare(self, ctx_id, det_size):
            captured["ctx_id"] = ctx_id

    import insightface.app as ia

    monkeypatch.setattr(ia, "FaceAnalysis", FakeFA)

    InsightFacePipeline("/tmp")
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["ctx_id"] == -1
