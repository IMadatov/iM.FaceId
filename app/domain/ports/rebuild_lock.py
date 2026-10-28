from typing import Protocol, runtime_checkable


@runtime_checkable
class RebuildLock(Protocol):
    """Single-flight guard for group rebuilds.

    The shipped adapter is process-local (threading.Lock), so single-flight
    holds only within one worker process. See the spec's "Known limitation".
    `locked()` exists because `GroupRebuildService.execute()` calls it in its
    `finally` branch; a future distributed adapter will have to emulate it.
    """

    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...
    def locked(self) -> bool: ...
