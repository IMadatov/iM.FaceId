from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.pipeline import FaceDetector, FaceEmbedder, FacePipeline, PipelineProbe
from app.domain.ports.store import (
    FaceCatalog,
    FaceReader,
    FaceSearcher,
    FaceWriter,
    HealthProbe,
)
from app.pipeline.fake import FakeFacePipeline
from app.pipeline.fake_liveness import FakeLivenessChecker, UnavailableLivenessChecker
from app.store.memory import MemoryFaceStore


def test_fake_store_satisfies_every_store_role():
    store = MemoryFaceStore()
    for port in (FaceReader, FaceWriter, FaceSearcher, FaceCatalog, HealthProbe):
        assert isinstance(store, port), f"MemoryFaceStore does not satisfy {port.__name__}"


def test_fake_pipeline_satisfies_pipeline_roles():
    pipeline = FakeFacePipeline()
    for port in (FaceDetector, FaceEmbedder, PipelineProbe, FacePipeline):
        assert isinstance(pipeline, port), f"FakeFacePipeline does not satisfy {port.__name__}"


def test_both_liveness_checkers_satisfy_the_scorer_port():
    assert isinstance(FakeLivenessChecker(), LivenessScorer)
    assert isinstance(UnavailableLivenessChecker(), LivenessScorer)
