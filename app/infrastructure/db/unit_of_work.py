from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.db.repository import GroupsRepository
from app.infrastructure.db.session import session_scope


class SqlAlchemyUnitOfWork:
    """UnitOfWork adapter built on the existing session_scope contextmanager.

    session_scope already commits on clean exit, rolls back on exception and
    always closes; this class only binds a repository to that session.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def groups(self) -> Iterator[GroupsRepository]:
        with session_scope(self._session_factory) as session:
            yield GroupsRepository(session)
