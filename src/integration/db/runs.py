"""Отображения прогона и его кадров: `runs` и `run_screenshots`.

Прогон привязывается по `journey_id`/`app_id` из задания — каталога не читает.
Статус валидации `validation_status` и прогоновые поля (`status`, `model`,
`token_usage`) переиспользуются как у вендора.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import CjxBase, pg_enum


class Run(CjxBase):
    """Запись прогона анализа (таблица `runs`, не FK на очередь)."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    journey_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("journeys.id"), nullable=False
    )
    app_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("applications.id"), nullable=False
    )
    platform: Mapped[str] = mapped_column(
        pg_enum("platformtype", "ios", "android", "web"), nullable=False
    )
    method: Mapped[str] = mapped_column(
        pg_enum("runmethod", "auto", "manual", "mockup", "video"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        pg_enum("runstatus", "pending", "running", "done", "failed"),
        nullable=False,
        default="pending",
    )
    validation_status: Mapped[str] = mapped_column(
        pg_enum(
            "runvalidationstatus",
            "draft",
            "in_review_cx",
            "in_review_product",
            "finalized",
        ),
        nullable=False,
        default="draft",
    )
    model: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    package_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    app_version: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    token_usage: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)


class RunScreenshot(CjxBase):
    """Кадр прогона: ключ в хранилище, шаг, без байтов кадров в БД."""

    __tablename__ = "run_screenshots"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "step_index", name="uq_run_screenshots_run_step"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    s3_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    package_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    app_version: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    masked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    mask_meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )