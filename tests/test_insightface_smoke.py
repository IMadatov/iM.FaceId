import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INSIGHTFACE_SMOKE") != "1",
    reason="set RUN_INSIGHTFACE_SMOKE=1 with models downloaded",
)


def test_pipeline_ready():
    from app.pipeline.onnx_insightface import InsightFacePipeline

    pipe = InsightFacePipeline(model_dir="./models")
    assert pipe.ready()
