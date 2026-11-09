"""Отображение таблицы очереди заданий `eval_jobs`.

Одно durable-выполнение задания анализа: статус + лиз + JSON-вход/результат/ошибка.
Лиз-токен хранится только хэшем (`lease_token_hash`); сырой токен живёт в памяти
процесса (см. `src/durable/` и threat-model Т2/Т5).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import CjxBase


class EvalJob(CjxBase):
    """Один durable-проход оценки: лиз + JSON вход/результат/ошибка."""

    __tablename__ = "eval_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    queue_order: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="queued"
    )
    progress: Mapped[str] = mapped_column(
        String(500), nullable=False, default=""
    )
    phase: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    phase_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    phase_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    input_schema_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    input_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    input_artifact_owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    result_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    worker_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    lease_token_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    claim_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancel_requested_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )