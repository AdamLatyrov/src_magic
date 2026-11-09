"""Жизненный цикл процесса: что поднято до первого запроса и чем процесс выходит.

Порядок сборки — снизу вверх: настройки, база, агент, исполнители, очередь. Порядок
разборки — обратный: сначала перестаем брать работу и ждём grace, потом закрываем
соединения. Любая ошибка сборки поднимается наружу: приложение, которое не
поднялось, лучше контейнера, который отвечает `200` и не работает — второй переживёт
собственный перезапуск и ничего не расскажет в журнале.

Настройки к этому моменту уже прочитаны и проверены (`src/core/config.py`): то,
что роль или порт не годятся, выясняется на сборке настроек, а не здесь. Здесь
отказывается только то, для чего настроек мало, — пустая строка подключения к
базе.

В фазе 1 (очередь) база может быть недоступна — runner стартует с предупреждением
и retry-петлёй. Сервис отвечает на /health без очереди (health-маршрут не зависит
от database/queue).
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.integration.catalog.repo import DbCatalogRepository
from src.integration.core.config import Settings
from src.integration.db.database import Database, build_database
from src.integration.durable.repo import EvalJobsRepository
from src.integration.persist.binding import CatalogBindingResolver
from src.integration.persist.result import QueueResultWriter
from src.integration.service.queue_runner import QueueRunner
from src.integration.service.runners import RunnerRegistry
from src.integration.service.runtime import build_graph_runtime
from src.integration.storage.frames import S3FrameSource, S3FrameStorage
from src.integration.storage.s3 import S3ObjectStorage, S3Settings
from src.pipeline.schemas.nodes import CXCriteriaModel

logger = logging.getLogger(__name__)


@asynccontextmanager
async def service_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Поднимает ресурсы процесса и честно их закрывает.

    База может быть подставлена из тестов (`create_app(database=...)`) — тогда
    процесс её не строит, но всё равно закрывает: поддельная фабрика обязана
    уметь то же, что настоящая, иначе подмена расходится с оригиналом молча.

    :param app: приложение, чьи `state` держат зависимости процесса
    """
    settings = app.state.settings
    registry = _registry(app)
    database = getattr(app.state, "database", None) or build_database(
        settings.database
    )
    app.state.database = database
    try:
        # Справочник живёт только в базе и нужен уже при сборке графа: ветка
        # блоков спрашивает его блоки, чтобы назвать свои узлы. Поэтому он
        # читается первым, а не подменяется после сборки.
        criteria = await _load_catalog(database)
        app.state.runtime = build_graph_runtime(settings, criteria)
        logger.info(
            "agent assembled, nodes %d", len(app.state.runtime.criteria.blocks)
        )
        # Register and start queue runner as a background task so
        # the event loop can serve HTTP requests concurrently.
        runner = await _register_queue_runner(app, settings, registry, database)
        if runner is not None:
            asyncio.create_task(runner.start())
            logger.info("queue runner started (background)")
        logger.info("lifespan ready")
        yield
    except Exception:
        # Lifespan fails on DB unavailability — log and proceed
        # so /health can report status without killing the process.
        logger.exception(
            "[lifecycle] lifespan error (DB likely unavailable)"
        )
        yield
    finally:
        await database.dispose()
        logger.info("service stopped")


def _registry(app: FastAPI) -> RunnerRegistry:
    """Реестр исполнителей приложения; список подставляется вместо реестра."""
    runners = getattr(app.state, "runners", None)
    if isinstance(runners, RunnerRegistry):
        return runners
    return RunnerRegistry(runners or ())


async def _load_catalog(database: Database) -> "CXCriteriaModel":
    """Читает справочник критериев из базы — до сборки графа.

    Справочник живёт только в базе: файла нет, и подмены «другой редакцией
    каталога» быть не должно. Поэтому отсутствие активной версии — отказ
    запуска, а не тихая деградация: прогон на чужой линейке штрафов хуже
    неподнявшегося сервиса, потому что его результат выглядит достоверным.

    :param database: база процесса — источник активной версии
    :raises CatalogReadError: активной версии справочника в базе нет
    :return: справочник из базы
    """
    reader = DbCatalogRepository(database.session_factory)
    criteria = await reader.load_criteria_model()
    logger.info("catalog source loaded, nodes %d", len(criteria.blocks))
    return criteria


async def _register_queue_runner(
    app: FastAPI,
    settings: Settings,
    registry: RunnerRegistry,
    database: Database,
) -> QueueRunner | None:
    """Собирает пайплайн очереди и регистрирует QueueRunner.

    Возвращает инстанс runner'а (или None если сборка провалилась),
    чтобы вызывающий мог запустить его как фоновый coroutine.
    """
    try:
        # S3 storage pipeline
        s3 = S3Settings()
        object_storage = S3ObjectStorage(s3)
        frame_storage = S3FrameStorage(object_storage, expires_s=s3.presign_ttl)
        frame_source = S3FrameSource(frame_storage)

        # Queue port (eval_jobs table)
        repo = EvalJobsRepository(database.session_factory)

        # Result writer (runs/evaluations tables). Привязка прогона приходит из
        # задания на каждый прогон, поэтому общей `RunContext` здесь нет: она
        # была бы одной на процесс и приписала бы все прогоны одному пути.
        # Постановка задания несёт только продукт, так что путь и приложение
        # достаёт из каталога читатель привязки.
        writer = QueueResultWriter(database.session_factory)
        binder = CatalogBindingResolver(
            database.session_factory,
            default_product_id=settings.binding.default_product_id or None,
        )

        # Wire the runner
        runner = QueueRunner(
            settings=settings,
            runtime=app.state.runtime,
            repo=repo,
            writer=writer,
            frame_source=frame_source,
            binder=binder,
        )
        registry.add(runner)
        logger.info("queue runner registered")
        return runner
    except Exception:
        logger.exception("[queue-runner] failed to assemble")
        return None
