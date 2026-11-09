"""Что процесс сообщает о себе до старта сервера: сводка и причины отказа.

Живёт в пакете, а не в лаунчере, по той же причине, по какой в ядре живёт
`overlay/src/gigachat_only.py`: лаунчер — единственная команда образа, и каждая
его строчка — это способ поднять процесс, который отвечает живости и не делает
работы. Здесь же перечисление этапов: оно следует за настройками графа, а не
за способом запуска.

Секретов в сводке нет намеренно: строка запуска попадает в журнал контейнера,
который виден всему стенду. Отказ сборки настроек называется именем переменной и
сообщением валидатора — значения полей в текст не попадают, потому что в файле
настроек лежат секреты.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ValidationError

from src.integration.core.config import Settings
from src.integration.service.version import version


def summary(settings: Settings) -> list[str]:
    """Сводка запуска: роль, порт, сборка, база и модель каждого этапа.

    :param settings: собранные настройки процесса
    :return: строки журнала, готовые к печати
    """
    lines = [
        f"starting role={settings.service.role} port={settings.service.port}"
        f" version={version()}",
        f"database={'задан' if settings.database.url.strip() else 'не задан'}"
        f" pool={settings.database.pool_size}+{settings.database.max_overflow}",
    ]
    stages = ", ".join(
        f"{name}={model}" for name, model in stage_models(settings).items()
    )
    lines.append(f"models {stages}")
    return lines


def stage_models(settings: Settings) -> dict[str, str]:
    """Модель каждого этапа графа — поимённо, как её видит прогон.

    Обход рекурсивный: этап живёт и в секции графа, и внутри неё (скан путей
    сидит в `journey_exploration`, проверки проблем пути — списком в
    `journey_frictions`), а перечислять этапы здесь значило бы вести их второй
    список помимо настроек.

    :param settings: собранные настройки процесса
    :return: имя этапа → модель этого этапа
    """
    models: dict[str, str] = {}
    _collect(settings.graph, "", models)
    return models


def refusals(exc: ValidationError) -> list[str]:
    """Почему настройки не собраны — имена переменных, без значений.

    :param exc: отказ сборки настроек
    :return: по строке на каждое поле; значений полей в них нет
    """
    return [
        f"{exc.title}: {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(
            include_url=False, include_input=False, include_context=False
        )
    ]


def _collect(node: object, prefix: str, into: Mapping[str, str]) -> None:
    """Собирает пары «этап → модель» из секции настроек."""
    if not isinstance(node, BaseModel):
        return
    model = getattr(node, "model", None)
    if isinstance(model, str):
        into[prefix.rstrip(".")] = model
    for name, field in type(node).model_fields.items():
        child = getattr(node, name, None)
        target = f"{prefix}{name}"
        if isinstance(child, list):
            for index, item in enumerate(child):
                key = str(getattr(item, "check_key", index))
                child_model = getattr(item, "model", None)
                if isinstance(child_model, str):
                    into[key] = child_model
        elif isinstance(child, BaseModel):
            _collect(child, f"{target}.", into)
