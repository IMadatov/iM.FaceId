from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.infrastructure.db.models import FaceGroupRow, GroupMember, GroupRun, GroupRunStatus


@dataclass(frozen=True)
class StoredFaceGroup:
    face_ids: list[str]
    size: int


@dataclass(frozen=True)
class StoredGroupsSnapshot:
    run_id: str
    threshold: float
    built_at: datetime | None
    groups: list[StoredFaceGroup]


class GroupsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(
        self, *, threshold: float, top_k: int, min_size: int
    ) -> GroupRun:
        run = GroupRun(
            status=GroupRunStatus.running,
            threshold=threshold,
            top_k=top_k,
            min_size=min_size,
            started_at=datetime.now(timezone.utc),
        )
        self._session.add(run)
        self._session.flush()
        return run

    def get_run(self, run_id: str) -> GroupRun | None:
        return self._session.get(GroupRun, run_id)

    def list_runs(self) -> list[GroupRun]:
        stmt = select(GroupRun).order_by(
            GroupRun.started_at.desc().nulls_last(),
            GroupRun.id.desc(),
        )
        return list(self._session.scalars(stmt))

    def has_running(self) -> bool:
        stmt = select(GroupRun.id).where(
            GroupRun.status.in_((GroupRunStatus.pending, GroupRunStatus.running))
        ).limit(1)
        return self._session.scalar(stmt) is not None

    def mark_failed(self, run_id: str, error: str) -> None:
        run = self.get_run(run_id)
        if run is None:
            return
        run.status = GroupRunStatus.failed
        run.error = error[:4000]
        run.finished_at = datetime.now(timezone.utc)

    def save_completed(
        self,
        run_id: str,
        *,
        face_count: int,
        groups: list[list[str]],
        keep_runs: int = 2,
    ) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"group run {run_id} not found")

        for members in groups:
            row = FaceGroupRow(run_id=run_id, size=len(members))
            self._session.add(row)
            self._session.flush()
            for face_id in members:
                self._session.add(GroupMember(group_id=row.id, face_id=face_id))

        run.status = GroupRunStatus.completed
        run.face_count = face_count
        run.group_count = len(groups)
        run.finished_at = datetime.now(timezone.utc)
        run.error = None
        self._session.flush()
        self._prune_old_runs(keep=keep_runs)

    def _prune_old_runs(self, *, keep: int) -> None:
        if keep < 1:
            keep = 1
        stmt = (
            select(GroupRun)
            .where(GroupRun.status == GroupRunStatus.completed)
            .order_by(GroupRun.finished_at.desc().nulls_last(), GroupRun.id.desc())
        )
        completed = list(self._session.scalars(stmt))
        for stale in completed[keep:]:
            self._session.delete(stale)

        # Drop failed/pending leftovers beyond a small cap
        other = list(
            self._session.scalars(
                select(GroupRun)
                .where(GroupRun.status != GroupRunStatus.completed)
                .order_by(GroupRun.started_at.desc().nulls_last())
            )
        )
        for stale in other[keep:]:
            self._session.delete(stale)

    def latest_completed(self) -> StoredGroupsSnapshot | None:
        stmt = (
            select(GroupRun)
            .where(GroupRun.status == GroupRunStatus.completed)
            .options(
                selectinload(GroupRun.groups).selectinload(FaceGroupRow.members)
            )
            .order_by(GroupRun.finished_at.desc().nulls_last(), GroupRun.id.desc())
            .limit(1)
        )
        run = self._session.scalars(stmt).first()
        if run is None:
            return None
        groups: list[StoredFaceGroup] = []
        for g in run.groups:
            face_ids = [m.face_id for m in g.members]
            face_ids.sort()
            groups.append(StoredFaceGroup(face_ids=face_ids, size=len(face_ids)))
        groups.sort(key=lambda g: (-g.size, g.face_ids[0] if g.face_ids else ""))
        return StoredGroupsSnapshot(
            run_id=run.id,
            threshold=run.threshold,
            built_at=run.finished_at,
            groups=groups,
        )
