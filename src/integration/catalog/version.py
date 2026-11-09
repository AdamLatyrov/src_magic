"""Какая строка выгрузки вообще попадает в прогон.

В файле рядом едут черновики, снятые редакции и действующая версия; справочник
прогона — одна опубликованная действующая строка, и вокруг неё уже отбираются
строки блоков, проверок и дефектов.
"""

from typing import Any

from src.integration.catalog.columns import flag_of
from src.pipeline.schemas.criteria_export import ModelRow
from src.pipeline.settings.params.nodes import CXModelStatus


def _title(row: ModelRow) -> str:
    return f"{row.key}@{row.version}"


def pick_version(
    rows: list[ModelRow], errors: list[str]
) -> ModelRow | None:
    """Опубликованная действующая версия — та, которой измеряют прогон.

    Выбор идёт по признакам строки, а не по позиции в файле и не по последней
    версии: версии после публикации неизменяемы, и их в выгрузке несколько,
    когда каталог перекалибровывают, не снимая прежней линейки баллов.
    """
    picked: list[ModelRow] = []
    for row in rows:
        if not row.id:
            errors.append("версия справочника без идентификатора")
            continue
        if CXModelStatus.parse(row.status) is None:
            errors.append(
                f"версия {_title(row)}: статус {row.status!r} вне канона"
            )
        active = flag_of(row.is_active)
        if active is None:
            errors.append(
                f"версия {_title(row)}: признак действующей "
                f"{row.is_active!r} вне канона"
            )
        if CXModelStatus.parse(row.status) == CXModelStatus.PUBLISHED and (
            active is True
        ):
            picked.append(row)
    if len(picked) == 1:
        return picked[0]
    titles = ", ".join(_title(row) for row in rows)
    if not picked:
        errors.append(
            "в выгрузке нет опубликованной действующей версии "
            f"справочника; версии: {titles or 'нет'}"
        )
    else:
        names = ", ".join(_title(row) for row in picked)
        errors.append(
            f"опубликованных действующих версий несколько ({names}): "
            "в оценку может уйти любая"
        )
    return None


def of_version(rows: list[Any], model_id: str) -> list[Any]:
    """Строки одной версии: рядом едут черновики и снятые редакции."""
    return [row for row in rows if row.model_id == model_id]
