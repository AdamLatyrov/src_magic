"""Отображения версионированной критериальной модели: таблицы `eval_models*`.

Иерархия: `EvalModel` → `EvalModelBlock` → `EvalModelCheck` → `EvalModelDefect`.
Версии неизменяемы после публикации; активная (`is_active`) — не более одной на
`key`. Снимок активной версии для прогона читается в `src/catalog/repo.py`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import CjxBase


class EvalModel(CjxBase):
    """Версия критериальной модели (реестр в БД)."""

    __tablename__ = "eval_models"
    __table_args__ = (
        UniqueConstraint("key", "version", name="uq_eval_models_key_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key: Mapped[str] = mapped_column(
        String(60), nullable=False, default="cx-general"
    )
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(
        String(12), nullable=False, default="draft"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class EvalModelBlock(CjxBase):
    """Блок критериальной модели."""

    __tablename__ = "eval_model_blocks"
    __table_args__ = (
        UniqueConstraint(
            "model_id", "key", name="uq_eval_model_blocks_model_key"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_models.id"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(60), nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EvalModelCheck(CjxBase):
    """Критерий проверки внутри блока."""

    __tablename__ = "eval_model_checks"
    __table_args__ = (
        UniqueConstraint(
            "model_id", "key", name="uq_eval_model_checks_model_key"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_models.id"), nullable=False
    )
    block_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_model_blocks.id"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    zone: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)


class EvalModelDefect(CjxBase):
    """Дефект каталога, привязанный к критерию (и к блоку)."""

    __tablename__ = "eval_model_defects"
    __table_args__ = (
        UniqueConstraint(
            "model_id", "defect_ref", name="uq_eval_model_defects_model_ref"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_models.id"), nullable=False
    )
    block_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_model_blocks.id"), nullable=False
    )
    check_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_model_checks.id"), nullable=False
    )
    defect_ref: Mapped[str] = mapped_column(String(180), nullable=False)
    display_code: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    manifestation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    zone: Mapped[str] = mapped_column(String(10), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)