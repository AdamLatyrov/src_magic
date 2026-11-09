"""Публичные сериализуемые контракты запуска анализа.

Эти модели предназначены интеграции и ноутбуку. Они не раскрывают состояние
LangGraph и не выбирают механизм хранения прогона.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from src.pipeline.schemas.core import CoreSchema
from src.pipeline.schemas.nodes import CXFrame
from src.pipeline.schemas.result import AnalyzeResult, Issue


class PipelineRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class PipelineStageSnapshot(CoreSchema):
    name: str
    state: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    note: str = ""
    counters: dict[str, Any] = Field(default_factory=dict)


class PipelineStateSnapshot(CoreSchema):
    run_id: str
    status: PipelineRunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    current_stage: str | None = None
    stages: list[PipelineStageSnapshot] = Field(default_factory=list)
    findings_count: int = 0
    error_code: str | None = None
    error_message: str | None = None


class PipelineProgressSnapshot(PipelineStateSnapshot):
    """Расширенный текущий статус для опроса по run_id.

    Отдельный DTO сохраняет прежний контракт PipelineRunResult.state.
    """

    revision: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_status: str | None = None
    is_degraded: bool | None = None
    degradation_reasons: list[str] = Field(default_factory=list)
    failed_blocks: list[str] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)


class ExpertDecision(str, Enum):
    """Единые статусы решения эксперта для прошлой находки."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PreviousFinding(CoreSchema):
    """Находка прошлого прогона и отдельное решение эксперта по ней."""

    issue: Issue
    expert_status: ExpertDecision


class PipelineRequest(CoreSchema):
    """Вход одного прогона; модель и каталог задаются при создании фасада."""

    run_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    product_name: str = Field(min_length=1)
    product_type: str = Field(min_length=1)
    frames: list[CXFrame] = Field(min_length=1)
    screenshot_keys: list[str] = Field(default_factory=list)
    previous: list[PreviousFinding] = Field(default_factory=list)
    material: str = ""

    @field_validator("run_id", "product_name", "product_type")
    @classmethod
    def non_blank_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("идентификатор и поля продукта не должны быть пустыми")
        return value

    @field_validator("frames")
    @classmethod
    def frames_are_ordered(cls, frames: list[CXFrame]) -> list[CXFrame]:
        steps = [frame.step for frame in frames]
        if any(step < 1 for step in steps):
            raise ValueError("номер шага должен быть положительным")
        if any(left >= right for left, right in zip(steps, steps[1:])):
            raise ValueError("кадры должны идти по возрастанию номера шага")
        return frames

    @model_validator(mode="after")
    def screenshot_keys_match_frames(self) -> "PipelineRequest":
        if self.screenshot_keys and len(self.screenshot_keys) != len(self.frames):
            raise ValueError("для каждого кадра нужен один screenshot_key")
        return self


class PipelineRunResult(CoreSchema):
    """Ответ фасада с исходным анализом, сверенными находками и состоянием."""

    run_id: str
    product_name: str
    product_type: str
    analysis: AnalyzeResult
    findings: list[Issue] = Field(default_factory=list)
    state: PipelineStateSnapshot
