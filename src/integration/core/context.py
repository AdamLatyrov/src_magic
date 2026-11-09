"""Контекст записей журнала: идентификаторы прогона у каждой строки.

Прогон пути — это параллельные задачи: ветки блоков, проверок пути и поэкранного
разбора идут рядом, а скан кадров растянут во времени. Идентификаторы поэтому
лежат в `contextvars`, а не в атрибутах логгера: общая на процесс переменная
смешала бы `request_id` двух заданий в одной строке, и журнал перестал бы
отвечать на вопрос «это строка какого прогона».

Имена полей называет `LoggerSettings.context_vars` (`src/core/config.py`), а
формат записи требует, чтобы каждое из них было у каждой строки: пустое значение
дешевле отсутствующего атрибута, иначе `%(request_id)s` роняет запись, которую и
писать-то не из-за чего.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any


class LogParamsContext:
    """Идентификаторы текущей задачи: словарь поименованных полей журнала."""

    def __init__(
        self, keys: Sequence[str] = ("request_id", "user_id", "call_id")
    ) -> None:
        """
        :param keys: имена полей, которые обязана нести каждая запись журнала
        """
        self._keys = tuple(keys)
        self._values: ContextVar[dict[str, Any]] = ContextVar(
            "log_params", default={}
        )

    @property
    def keys(self) -> tuple[str, ...]:
        """Имена полей этого контекста."""
        return self._keys

    @property
    def context(self) -> dict[str, Any]:
        """Поля текущей задачи: все объявленные имена, пустые если их нет."""
        current = self._values.get()
        return {key: current.get(key, "") for key in self._keys}

    def bind(self, **values: Any) -> Token:
        """Добавляет поля к текущей задаче.

        Значения не подменяются на месте: словарь общего `contextvar` видел бы
        обновлённый `request_id` каждый, кто его читает, а не только та задача,
        которая его задала.

        :param values: поля записи; `None` снимает поле до состояния «пусто»
        :return: токен, которым вернуть прежнее содержимое
        """
        merged = dict(self._values.get())
        for key, value in values.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return self._values.set(merged)

    def reset(self, token: Token) -> None:
        """Возвращает поля задачи к состоянию до последнего `bind`."""
        self._values.reset(token)


context_var = LogParamsContext()
"""Контекст журнала процесса: его читает фильтр из `src/core/logger.py`."""


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Идентификаторы прогона на время задачи.

    Оборачивают всё, что должно остаться в журнале связным: задание целиком,
    отдельную ветку, отдельный вызов модели. Выход из блока снимает поля — иначе
    следующая задача в этом же потоке continuation донесла бы чужой `request_id`.

    :param values: поля записи (`request_id`, `call_id`, `user_id`)
    """
    token = context_var.bind(**values)
    try:
        yield
    finally:
        context_var.reset(token)
