import threading


class ThreadRebuildLock:
    """Process-local RebuildLock.

    Single-flight holds within one worker process only. Running uvicorn with
    --workers > 1 lets one rebuild start per worker; see the spec's "Known
    limitation". Swapping in a distributed adapter is a config change here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True) -> bool:
        return self._lock.acquire(blocking)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()
