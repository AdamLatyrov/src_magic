"""Отображение нормализованной проблемы прогона: таблица `run_issues`.

Одна строка — одна консолидированная проблема пути (единица штрафа). Связывается
с прогоном и оценкой, несёт зону, критерий, улики и пин, а также привязку к
версионированной критериальной модели (`eval_model_*`).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import CjxBase


class RunIssue(CjxBase):
    """Нормализованная консолидированная проблема прогона."""

    __tablename__ = "run_issues"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    evaluation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluations.id"), nullable=True
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    severity: Mapped[str] = mapped_column(
        String(10), nullable=False, default="mid"
    )
    criterion: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    rec: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    steps: Mapped[Optional[list]] = mapped_column(ARRAY(Integer), nullable=True)
    pins: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    evidence: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    eval_model_version: Mapped[Optional[str]] = mapped_column(
        String(40), nullable=True
    )
    applied_profiles: Mapped[Optional[list]] = mapped_column(
        ARRAY(String), nullable=True
    )
    eval_model_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_models.id"), nullable=True
    )
    model_check_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_model_checks.id"), nullable=True
    )
    model_defect_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_model_defects.id"), nullable=True
    )
    defect_ref: Mapped[Optional[str]] = mapped_column(String(180), nullable=True)
    defect_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    check_key: Mapped[Optional[str]] = mapped_column(
        String(120), nullable=True
    )
    zone: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    source: Mapped[str] = mapped_column(
        String(10), nullable=False, default="agent"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )