"""Отображения полиморфной оценки: `evaluations` и `run_evaluations`.

`run_evaluations` — наследник `evaluations` (`evaluation_type='run'`); src пишет
ровно одну оценку на прогон. Уникальный `run_id` — инвариант схемы: повторная
запись того же прогона конфликтует, и двойного результата не бывает (НФ-9).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import CjxBase, pg_enum


class Evaluation(CjxBase):
    """Базовая запись полиморфной оценки (joined inheritance)."""

    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    evaluation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    evaluator_type: Mapped[str] = mapped_column(
        pg_enum("evaluatortype", "ai", "manual", "aggregated"), nullable=False
    )
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    critical_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RunEvaluation(CjxBase):
    """Оценка прогона (таблица `run_evaluations`) — ровно одна на `run_id`."""

    __tablename__ = "run_evaluations"

    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluations.id"),
        primary_key=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id"),
        nullable=False,
        unique=True,
    )
    model_used: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    criteria_blocks: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    eval_model_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_models.id"), nullable=True
    )
    eval_model_version: Mapped[Optional[str]] = mapped_column(
        String(40), nullable=True
    )
    base_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    friction_penalty: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    friction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    stop_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)