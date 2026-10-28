import numpy as np
import pytest

from app.domain.errors import InvalidImageError
from app.domain.ports.identity import IdentityFactory
from app.domain.ports.imaging import ImageDecoder
from app.domain.ports.rebuild_lock import RebuildLock
from app.infrastructure.identity import Uuid4IdentityFactory
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.locks.thread_lock import ThreadRebuildLock


def test_decoder_satisfies_port_and_rejects_empty_bytes():
    decoder = OpenCvImageDecoder()
    assert isinstance(decoder, ImageDecoder)
    with pytest.raises(InvalidImageError):
        decoder.decode(b"", max_side=640, max_bytes=5_000_000)


def test_identity_factory_returns_distinct_uuid_strings():
    factory = Uuid4IdentityFactory()
    assert isinstance(factory, IdentityFactory)
    ids = {factory.new_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 36 for i in ids)


def test_thread_lock_is_single_flight():
    lock = ThreadRebuildLock()
    assert isinstance(lock, RebuildLock)
    assert lock.acquire(blocking=False) is True
    assert lock.acquire(blocking=False) is False
    assert lock.locked() is True
    lock.release()
    assert lock.locked() is False
    assert lock.acquire(blocking=False) is True
    lock.release()
