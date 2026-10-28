from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class GroupRunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


def _uuid() -> str:
    return str(uuid.uuid4())


class GroupRun(Base):
    __tablename__ = "group_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    status: Mapped[GroupRunStatus] = mapped_column(
        Enum(GroupRunStatus, native_enum=False, length=32),
        nullable=False,
        default=GroupRunStatus.pending,
    )
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    min_size: Mapped[int] = mapped_column(Integer, nullable=False)
    face_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    group_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    groups: Mapped[list[FaceGroupRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class FaceGroupRow(Base):
    __tablename__ = "groups"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("group_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False)

    run: Mapped[GroupRun] = relationship(back_populates="groups")
    members: Mapped[list[GroupMember]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )


class GroupMember(Base):
    __tablename__ = "group_members"
    __table_args__ = (UniqueConstraint("group_id", "face_id", name="uq_group_member"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("groups.id", ondelete="CASCADE"), nullable=False, index=True
    )
    face_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    group: Mapped[FaceGroupRow] = relationship(back_populates="members")
