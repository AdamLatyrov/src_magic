"""Отображения владельца и каталога путей: `users`, `applications`, `journeys`.

Таблицы заводит не этот сервис (схемой владеет S01), и работник их только
узнаёт: `eval_jobs.owner_id` и `runs.journey_id`/`app_id` ссылаются на них
внешними ключами, а SQLAlchemy разбирает внешний ключ по имени таблицы в
отображениях, а не в живой базе. Без пары строк здесь первый же `flush` очереди
падает на сортировке таблиц ещё до `UPDATE`, и задание не снимается никогда —
ровно так очередь стояла на стенде.

Поля описаны целиком по схеме `cjx`: строка читается и для проверки, что
владелец задания и хозяин пути на месте.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import CjxBase


class User(CjxBase):
    """Владелец материалов, заданий и прогонов (таблица `users`)."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    login: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )


class Application(CjxBase):
    """Приложение, к которому относится путь (таблица `applications`)."""

    __tablename__ = "applications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )


class Journey(CjxBase):
    """Путь, к которому привязан прогон (таблица `journeys`)."""

    __tablename__ = "journeys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    app_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
