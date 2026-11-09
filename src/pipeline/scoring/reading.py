"""Как балл читает поле: находка встречается и формой, и словарём.

Разбор ответа контура передаёт балу то объект схемы, то его дамп, поэтому
чтение одно на оба случая. Модуль переживёт переезд недолго: как только вход
балла станет типизированным, ветка «или словарь» станет не нужна вместе с ним.
"""

from collections.abc import Mapping
from typing import Any


def field_of(value: Any, name: str) -> Any:
    """Поле из dict или из модели — находка читается до превращения в DTO."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
