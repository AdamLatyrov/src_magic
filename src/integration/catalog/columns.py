"""Как колонка выгрузки становится значением.

Числа в выгрузке написаны строками, флаг действующей версии приходит и `t`, и
`yes`, а порядок задаёт колонка `sort_order`, а не место строки в файле. Здесь
канон этих трёх колонок: отбор редакций и правила целостности читают значения,
а не сырые колонки.
"""

from typing import Any


def code_of(raw: str) -> str:
    """Значение колонки канонично: пробелы и регистр смысла не меняют."""
    return str(raw or "").strip().lower()


def flag_of(raw: str) -> bool | None:
    """Истинен ли флаг выгрузки; None — значение вне канона колонки."""

    db_true_flags = {"t", "true", "1", "y", "yes"}
    db_false_flags = {"f", "false", "0", "n", "no"}

    code = code_of(raw)

    if code in db_true_flags:
        return True
    if code in db_false_flags:
        return False
    return None


def ordered(rows: list[Any]) -> list[Any]:
    """Строки в порядке справочника: порядок задаёт колонка, а не место.

    Проверки в файле перемешаны между блоками (сортировка своя у каждой
    таблицы), поэтому обход списка по порядку файла отдал бы блоку чужую
    проверку, а равные `sort_order` разбирались бы по случаю выгрузки — тут
    спасает устойчивость сортировки: ничья разрешается порядком в файле.
    """

    def place(pair: tuple[int, Any]) -> tuple[int, int]:
        index, row = pair
        try:
            order = int(code_of(row.sort_order))
        except ValueError:
            order = 0
        return (order, index)

    return [row for _, row in sorted(enumerate(rows), key=place)]
