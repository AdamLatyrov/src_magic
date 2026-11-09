"""Postgres как ресурс процесса: пул, фабрика сессий, проверка связи.

Здесь нет ни схем, ни запросов к `eval_jobs`: очередь — фаза 1, а каркасу от
базы нужен ровно один ответ — «соединение ставится». Проверка делает `SELECT 1`,
как это делает ядро (`runtime_health.py`), чтобы `/health/ready` нового работника
означал для ingress ровно то же, что означал раньше.

Пустая строка подключения — отказ собрать ресурс, а не отказ импортировать
модуль: тестам живости удобно подставить поддельную фабрику сессий, а контейнер
без базы обязан подняться и ответить на `/health`, отдав `503` только на
`/health/ready`.
"""

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.integration.settings.api.service import DatabaseSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Database:
    """Движок и фабрика сессий — то, что процесс отдаёт проверкам и работе."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    async def ping(self, timeout_s: float) -> bool:
        """`SELECT 1` в пределах срока.

        Недоступная база даёт `False`, а не исключение: проверка готовности
        обязана отличать «базы нет» от «маршрут упал», иначе недоступная база
        выглядела бы как погибший процесс.

        :param timeout_s: срок установки соединения и ответа
        :return: отвечает ли база
        """
        try:
            async with self.session_factory() as session:
                await asyncio.wait_for(
                    session.execute(text("SELECT 1")), timeout=timeout_s
                )
        except Exception:
            logger.warning("database is not available", exc_info=True)
            return False
        return True

    async def dispose(self) -> None:
        """Закрывает соединения пула: процесс уходит, не оставив открытых сокетов."""
        await self.engine.dispose()


def build_database(settings: DatabaseSettings) -> Database:
    """Пул и фабрику сессий — по настройкам слоя.

    `pool_size` и `max_overflow` держатся маленькими: работник берёт задания
    одним циклом, и большой пул означает лишь то, что база отдаст соединения
    соседям позже. `connect_args.timeout` — срок рукопожатия asyncpg, он же
    потолок ожидания свободного соединения при пустом пуле.

    :param settings: настройки базы этого процесса
    :raises ValueError: строка подключения не задана
    :return: собранный ресурс
    """
    if not settings.url.strip():
        raise ValueError("DATABASE_URL не задан")
    engine = create_async_engine(
        settings.url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=int(settings.pool_timeout_s),
        connect_args={
            "timeout": settings.connect_timeout_s,
            "server_settings": {"search_path": settings.db_schema},
        },
    )
    return Database(
        engine=engine,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
    )
