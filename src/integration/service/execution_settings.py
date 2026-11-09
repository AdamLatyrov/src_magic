"""Настройки исполнения, зафиксированные заданием, поверх настроек процесса.

Снимок исполнения (`src/service/execution_snapshot.py`) рождается при постановке
задания и едет в нём: модели этапов, мерки кадров, режим. Пока задание ждёт в
очереди, процесс успевает перезапуститься на другом файле настроек, и прогон,
взятый из очереди, посчитался бы не тем, чем его ставили. Поэтому значения
берутся из снимка, а настройки процесса остаются тем, что было бы без снимка:
их читают сборка графа, пробы старта и значения полей по умолчанию.

Снимок живёт в `ContextVar`, а не в атрибуте процесса: задания идут
параллельно (`max_concurrent` у цикла очереди), и общий словарь на процесс
смешал бы модели соседних заданий. Контекстная переменная видит только свой
прогон, вложенный снимок её перекрывает, а выход из блока возвращает прежнее
значение — даже если прогон упал.

Чем именно здесь отвечают — не решает слой: он берёт имена, которые называет
настройка этапа (`field.alias`), и отдаёт значение того же типа. Значения из
снимка уже проверены при сборке, здесь сверяется только тип: снимок едет через
JSON задания, и число в нём может оказаться строкой.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

ExecutionSettings = Mapping[str, Any]

_current: ContextVar[ExecutionSettings | None] = ContextVar(
    "execution_settings", default=None
)


@contextmanager
def executed_as(settings: ExecutionSettings | None) -> Iterator[None]:
    """Прогон, считающийся настройками из снимка.

    :param settings: настройки исполнения из задания; `None` — считать как процесс
    :return: контекст, в котором `execution_setting` видит снимок
    """
    token: Token = _current.set(settings)
    try:
        yield
    finally:
        _current.reset(token)


def current_settings() -> ExecutionSettings | None:
    """Снимок этого прогона; `None`, если прогон идёт на настройках процесса."""
    return _current.get()


def execution_setting(name: str, default: Any) -> Any:
    """Значение настройки из снимка, иначе — из процесса.

    Имя — это переменная окружения этапа (`LLM_CHECK_CX_CRITERIA`,
    `LLM_MAX_IMAGES_PER_REQUEST` и т. п.); её берут из описания поля, а не из
    отдельного списка, — список пришлось бы держать в согласии с настройками.

    :param name: имя настройки — `alias` поля
    :param default: значение процесса, если снимок этого имени не назвал
    :return: значение из снимка либо значение процесса
    """
    settings = _current.get()
    if not settings or name not in settings:
        return default
    value = settings[name]
    if value is None:
        return default
    if isinstance(default, bool) != isinstance(value, bool):
        logger.warning(
            "execution setting %s: тип из снимка не совпал с настройкой,"
            " остаётся значение процесса",
            name,
        )
        return default
    if isinstance(default, int) and not isinstance(default, bool):
        if not isinstance(value, int):
            logger.warning(
                "execution setting %s: в снимке не целое число, остаётся"
                " значение процесса",
                name,
            )
            return default
    return value


__all__ = [
    "ExecutionSettings",
    "current_settings",
    "execution_setting",
    "executed_as",
]
