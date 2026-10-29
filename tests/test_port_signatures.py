"""Signature conformance between each port and its real adapters.

`runtime_checkable` Protocols only check that a method *name* exists —
`isinstance(store, FaceSearcher)` passes even if `search` takes entirely
different arguments. That blindness is the expensive kind: Tasks 3-10 write
adapters and use cases against these signatures, so a port that drifts from
its adapter surfaces as a `TypeError` several commits downstream instead of
here. `tests/test_ports_satisfied.py` covers presence; this file covers shape.

Annotations are deliberately ignored. A port annotates with `Any` where the
domain may not name a SQLAlchemy row, so the adapter's richer annotation is
expected to differ. Parameter names, kinds and defaults are what callers
depend on, and those must match exactly.
"""

import inspect

import pytest

from app.db.repository import GroupsRepository
from app.domain.ports.groups import GroupsReader, GroupsWriter
from app.domain.ports.identity import IdentityFactory
from app.domain.ports.imaging import ImageDecoder
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.metrics import FaceImageMetrics
from app.domain.ports.pipeline import FaceDetector, FaceEmbedder, PipelineProbe
from app.domain.ports.rebuild_lock import RebuildLock
from app.domain.ports.store import (
    FaceCatalog,
    FaceReader,
    FaceSearcher,
    FaceWriter,
    HealthProbe,
)
from app.domain.ports.unit_of_work import UnitOfWork
from app.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.identity import Uuid4IdentityFactory
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.locks.thread_lock import ThreadRebuildLock
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics
from app.pipeline.fake import FakeFacePipeline
from app.pipeline.fake_liveness import FakeLivenessChecker, UnavailableLivenessChecker
from app.pipeline.onnx_insightface import InsightFacePipeline
from app.pipeline.onnx_liveness import OnnxLivenessChecker
from app.store.memory import MemoryFaceStore
from app.store.qdrant import QdrantFaceStore

STORE_ROLES = (FaceReader, FaceWriter, FaceSearcher, FaceCatalog, HealthProbe)
PIPELINE_ROLES = (FaceDetector, FaceEmbedder, PipelineProbe)

# Every (port, adapter) pair the refactor relies on. RebuildLock itself wraps
# threading.Lock, a C builtin with no introspectable signature — but
# ThreadRebuildLock is a real Python class with real methods, so it is
# introspectable and is listed below.
PAIRS = [
    (port, adapter)
    for adapter in (MemoryFaceStore, QdrantFaceStore)
    for port in STORE_ROLES
] + [
    (port, adapter)
    for adapter in (FakeFacePipeline, InsightFacePipeline)
    for port in PIPELINE_ROLES
] + [
    (LivenessScorer, adapter)
    for adapter in (FakeLivenessChecker, UnavailableLivenessChecker, OnnxLivenessChecker)
] + [
    (port, GroupsRepository) for port in (GroupsReader, GroupsWriter)
] + [
    (ImageDecoder, OpenCvImageDecoder),
    (IdentityFactory, Uuid4IdentityFactory),
    (UnitOfWork, SqlAlchemyUnitOfWork),
    (RebuildLock, ThreadRebuildLock),
    (FaceImageMetrics, OpenCvFaceImageMetrics),
]


def _shape(func) -> list[tuple[str, str, object]]:
    """Parameter names, kinds and defaults — the part a caller depends on."""
    return [
        (p.name, str(p.kind), p.default)
        for p in inspect.signature(func).parameters.values()
    ]


def _port_methods(port) -> list[str]:
    return sorted(name for name in vars(port) if not name.startswith("_"))


@pytest.mark.parametrize(
    ("port", "adapter"), PAIRS, ids=lambda x: getattr(x, "__name__", str(x))
)
def test_adapter_signatures_match_their_port(port, adapter):
    methods = _port_methods(port)
    assert methods, f"{port.__name__} declares no methods — the port is empty"
    for name in methods:
        impl = getattr(adapter, name, None)
        assert impl is not None, f"{adapter.__name__} is missing {port.__name__}.{name}"
        expected = _shape(getattr(port, name))
        actual = _shape(impl)
        assert actual == expected, (
            f"{adapter.__name__}.{name} does not match {port.__name__}.{name}:\n"
            f"  port:    {expected}\n"
            f"  adapter: {actual}"
        )


def test_every_pair_is_covered():
    """Guard against a port or adapter being silently dropped from PAIRS."""
    assert len(PAIRS) == 26, f"expected 26 port/adapter pairs, found {len(PAIRS)}"
    ports_covered = {port.__name__ for port, _ in PAIRS}
    assert ports_covered == {
        "FaceReader",
        "FaceWriter",
        "FaceSearcher",
        "FaceCatalog",
        "HealthProbe",
        "FaceDetector",
        "FaceEmbedder",
        "PipelineProbe",
        "LivenessScorer",
        "GroupsReader",
        "GroupsWriter",
        "ImageDecoder",
        "IdentityFactory",
        "UnitOfWork",
        "RebuildLock",
        "FaceImageMetrics",
    }
