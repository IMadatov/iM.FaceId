from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from app.domain.ports.groups import GroupsReader, GroupsWriter


@runtime_checkable
class GroupsRepositoryPort(GroupsReader, GroupsWriter, Protocol): ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Opens a transaction and yields the groups repository bound to it.

    Commits on clean exit, rolls back on exception, always closes — the
    semantics of the existing app/db/session.py::session_scope, which the
    adapter in Task 3 delegates to rather than reimplementing.
    """

    def groups(self) -> AbstractContextManager[GroupsRepositoryPort]: ...
