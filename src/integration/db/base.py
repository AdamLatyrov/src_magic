"""Общая база отображений схемы `cjx`.

Каждый модуль замены ядра объявляет свои лёгкие ORM-классы (по темам в
`src/db/`), наследуясь от этой базы. Отображения не создают таблицы
(миграциями схемы владеет другой компонент) — только потребляют существующие.
"""

from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.orm import DeclarativeBase


class CjxBase(DeclarativeBase):
    """Общая база для отображений схемы `cjx` модулей замены ядра."""


def pg_enum(name: str, *labels: str) -> PgEnum:
    """Существующий в схеме enum — без права его создавать.

    Схемой владеет S01, и столбцы вроде `runs.platform` или
    `evaluations.evaluator_type` там именно enum, а не текст: строка, описанная
    как `String`, уезжает в базу параметром `$n::varchar`, и вставка падает на
    несоответствии типов (`столбец "..." имеет тип ..., а выражение - character
    varying`). Тип в схеме уже есть — отображению остаётся назвать его.
    """
    return PgEnum(*labels, name=name, create_type=False)
