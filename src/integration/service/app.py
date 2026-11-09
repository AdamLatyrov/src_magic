"""ASGI-приложение работника: живость, готовность и зависимости процесса.

HTTP-поверхность работника намеренно узкая: заданий он не принимает — их ставит в
очередь S02, — а значит наружу у него нет ни пользовательской схемы, ни
документации (`docs_url=None`, `openapi_url=None`, как в ядре). Остаются две
пробы живости и внутренний снимок исполнения, который S02 снимает перед
постановкой задания; ingress не должен публиковать `/internal/*`.

Зависимости лежат в `app.state` (`settings`, `database`, `runners`), а маршруты
читают их при запросе: тесты подставляют поддельные, а lifespan заменяет базу на
настоящую, если её не поднесли.
"""

import logging
from collections.abc import Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from src.integration.core.config import Settings
from src.integration.db.database import Database
from src.integration.service.errors import (
    ServiceError,
    detail_of,
    failure_handler,
)
from src.integration.service.execution_snapshot import (
    SnapshotRefused,
    build_snapshot,
    fingerprint,
)
from src.integration.service.health import Check, liveness_payload, readiness
from src.integration.service.lifecycle import service_lifespan
from src.integration.service.runners import Runner, RunnerRegistry
from src.integration.service.version import version

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    runners: Sequence[Runner] = (),
) -> FastAPI:
    """Приложение этого процесса.

    :param settings: настройки процесса; по умолчанию — как у процесса целиком
    :param database: готовая база вместо собираемой lifespan'ом (тесты живости)
    :param runners: исполнители, которых процесс запустит при старте
    :return: приложение, готовое к `uvicorn.run`
    """
    settings = settings or Settings()
    application = FastAPI(
        title="CJXplorer Analysis Worker",
        version=version(),
        lifespan=service_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.settings = settings
    application.state.database = database
    application.state.runners = RunnerRegistry(runners)
    application.add_exception_handler(ServiceError, failure_handler)

    @application.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Жив ли процесс; про базу и про исполнителей здесь ничего нет."""
        return liveness_payload(version(), settings.service.role)

    @application.get("/health/ready", include_in_schema=False)
    async def health_ready(request: Request) -> JSONResponse:
        """Можно ли давать процессу работу: база и живость исполнителей."""
        state = request.app.state
        timeout_s = state.settings.database.check_timeout_s
        checks: dict[str, Check] = {
            "database": _database_check(state, timeout_s),
            "service": _runners_check(state),
        }
        ready, results = await readiness(checks, timeout_s=timeout_s)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "checks": results,
                "version": version(),
                "role": state.settings.service.role,
            },
        )

    @application.post(
        "/internal/analysis/execution-snapshot", include_in_schema=False
    )
    async def execution_snapshot(request: Request) -> dict[str, object]:
        """Чем будет считаться анализ — снимок на момент постановки задания.

        Маршрут внутренний: у него нет пользовательской проверки, и ingress не
        должен публиковать `/internal/*`. Считает его сам работник, а не S02:
        прогон исполняет этот процесс, и копия логики в Java разошлась бы с ним.

        :param request: тело запроса с необязательной моделью
        :raises ServiceError: 503 — настройки не описывают снимок
        :return: модель и снимок исполнения
        """
        body = await _optional_json(request)
        settings = request.app.state.settings
        criteria = request.app.state.runtime.criteria
        model = str(body.get("model") or "").strip()
        try:
            snapshot = build_snapshot(settings, criteria)
        except SnapshotRefused as exc:
            raise ServiceError(
                "execution_snapshot_unavailable",
                "Настройки анализа не собираются",
                detail=detail_of(exc),
            ) from exc
        model = model or str(snapshot["models"]["base"])
        logger.info(
            "execution snapshot phase model=%s snapshot=%s",
            model,
            fingerprint(snapshot),
        )
        return {"model": model, "analysis_execution": snapshot}

    return application


async def _optional_json(request: Request) -> dict[str, object]:
    """Тело запроса словарём; пустое тело и не-объект читаются как `{}`.

    Запрос снимка приходит от S02 без гарантий по телу: единственное его поле —
    необязательная модель. Отказывать 422 на пустом теле значило бы ломать
    постановку задания ради формы запроса, к которой контракт не обязывает.
    """
    try:
        payload = await request.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def create_application() -> FastAPI:
    """Фабрика приложения для `uvicorn` — вызывается без аргументов."""
    return create_app()


def _database_check(state: State, timeout_s: float) -> Check:
    """Проверка базы: соединение ставится — значит работа возможна.

    Не собранная база — это `unavailable`, а не падение маршрута: процесс, у
    которого lifespan не дошёл до базы, обязан честно сказать «не готов».
    """

    async def check() -> bool:
        database = getattr(state, "database", None)
        if database is None:
            return False
        return await database.ping(timeout_s)

    return check


def _runners_check(state: State) -> Check:
    """Проверка исполнителей: молча упавший цикл очереди виден как `unavailable`."""

    async def check() -> bool:
        return state.runners.ready()

    return check
