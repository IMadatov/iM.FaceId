"""Single-flight ANN group rebuild job."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.repository import GroupsRepository
from app.matching.ann_grouping import build_ann_groups

logger = logging.getLogger(__name__)


class GroupRebuildConflict(Exception):
    """Another rebuild is already running."""


class GroupRebuildService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._lock = threading.Lock()

    def start(
        self,
        *,
        store,
        settings: Settings,
        min_score: float | None = None,
        min_size: int | None = None,
        top_k: int | None = None,
    ) -> str:
        threshold = settings.face_search_min_score if min_score is None else min_score
        size = settings.face_groups_min_size if min_size is None else min_size
        k = settings.face_groups_ann_top_k if top_k is None else top_k
        if size < 2:
            raise ValueError("min_size must be >= 2")
        if k < 1:
            raise ValueError("top_k must be >= 1")

        if not self._lock.acquire(blocking=False):
            raise GroupRebuildConflict("group rebuild already running")

        session = self._session_factory()
        try:
            repo = GroupsRepository(session)
            if repo.has_running():
                raise GroupRebuildConflict("group rebuild already running")
            run = repo.create_run(threshold=threshold, top_k=k, min_size=size)
            session.commit()
            run_id = run.id
        except Exception:
            session.rollback()
            self._lock.release()
            raise
        finally:
            session.close()

        return run_id

    def execute(self, run_id: str, *, store, settings: Settings) -> None:
        """Run ANN grouping and persist results. Always releases the start lock."""
        try:
            self._run(run_id, store=store, settings=settings)
        finally:
            if self._lock.locked():
                self._lock.release()

    def _run(self, run_id: str, *, store, settings: Settings) -> None:
        session = self._session_factory()
        try:
            repo = GroupsRepository(session)
            run = repo.get_run(run_id)
            if run is None:
                logger.error("group rebuild: run %s missing", run_id)
                return

            items = store.list_all()
            face_count = len(items)
            if face_count > settings.face_groups_max_faces:
                logger.warning(
                    "group rebuild: large face_count=%s (legacy max was %s)",
                    face_count,
                    settings.face_groups_max_faces,
                )

            def search_fn(vector, limit: int, min_score: float):
                return store.search(vector, limit=limit, min_score=min_score)

            groups = build_ann_groups(
                items,
                search_fn,
                min_score=run.threshold,
                top_k=run.top_k,
                min_size=run.min_size,
            )
            repo.save_completed(
                run_id,
                face_count=face_count,
                groups=groups,
                keep_runs=settings.face_groups_keep_runs,
            )
            session.commit()
            logger.info(
                "group rebuild %s completed: faces=%s groups=%s",
                run_id,
                face_count,
                len(groups),
            )
        except Exception as exc:
            session.rollback()
            logger.exception("group rebuild %s failed: %s", run_id, exc)
            fail_session = self._session_factory()
            try:
                GroupsRepository(fail_session).mark_failed(run_id, str(exc))
                fail_session.commit()
            except Exception:
                fail_session.rollback()
                logger.exception("group rebuild %s failed to persist error", run_id)
            finally:
                fail_session.close()
        finally:
            session.close()


def bind_execute(
    service: GroupRebuildService, run_id: str, *, store, settings: Settings
) -> Callable[[], None]:
    def _task() -> None:
        service.execute(run_id, store=store, settings=settings)

    return _task
