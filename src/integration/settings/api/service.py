"""Настройки сервисного слоя работника: роль, HTTP-поверхность, база.

Слой живёт на переменных ядра (`CJX_RUNTIME_PROFILE`, `CJX_HTTP_PORT`,
`EVAL_SHUTDOWN_GRACE_SEC`, `DATABASE_URL`, `DB_POOL_*`), а не на своих именах: на
стенде один файл секмана питает оба процесса, и разночтение имён означало бы две
колонки на одну настройку — с разным значением у каждой.

Адреса здесь только порт: хост лаунчер поднимает жёстко `0.0.0.0`, как в образе
на ядре, и переменной он не задаётся нигде — переменная адреса означала бы, что
работник может слушать наружу на другом интерфейсе молча от развёртывания.

Роль описана отдельным перечислением из одного значения не для красоты: опечатка
в `CJX_RUNTIME_PROFILE` (`api`, `combined`) должна давать понятный отказ старта
с именем переменной, а не молчаливый запуск работника под чужим именем, как это
делает ядро, где отсутствующая роль означает `combined`.
"""

from enum import Enum

from pydantic import Field

from src.pipeline.settings.core import CoreSettings


class ServiceRole(str, Enum):
    """Единственная роль, в которой этот процесс поднимается."""

    WORKER = "worker"


class ServiceSettings(CoreSettings):
    """Роль процесса, порт HTTP-поверхности и срок остановки по grace.

    `shutdown_grace_s` — сколько процесса ждёт дочитывания задания перед отменой;
    в образе ядра это же число лежит в `EVAL_SHUTDOWN_GRACE_SEC` и равно 900,
    потому что прогон пути с десятками кадров стоит минут, а не секунд.
    """

    role: ServiceRole = Field(
        default=ServiceRole.WORKER, alias="CJX_RUNTIME_PROFILE"
    )
    port: int = Field(default=8105, alias="CJX_HTTP_PORT", ge=1, le=65535)
    shutdown_grace_s: float = Field(
        default=900.0, alias="EVAL_SHUTDOWN_GRACE_SEC", gt=0
    )


class DatabaseSettings(CoreSettings):
    """Пул соединений к Postgres и сроки проверки готовности.

    Пустой `url` — не отказ на импорте модуля, а отказ на старте процесса:
    тестам живости удобно подставлять поддельную фабрику сессий, а контейнер без
    базы обязан подняться и ответить на `/health`, отдав `503` только на
    `/health/ready`.
    """

    url: str = Field(default="", alias="DATABASE_URL")
    db_schema: str = Field(default="cjx", alias="DB_SCHEMA")
    pool_size: int = Field(default=5, alias="DB_POOL_SIZE", ge=1)
    max_overflow: int = Field(default=5, alias="DB_MAX_OVERFLOW", ge=0)
    pool_timeout_s: float = Field(
        default=30.0, alias="DB_POOL_TIMEOUT_SEC", gt=0
    )
    connect_timeout_s: float = Field(
        default=3.0, alias="DB_CONNECT_TIMEOUT_SEC", gt=0
    )
    check_timeout_s: float = Field(
        default=3.0, alias="HEALTHCHECK_TIMEOUT_SEC", gt=0
    )
