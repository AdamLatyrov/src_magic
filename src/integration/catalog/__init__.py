"""Справочник критериев: из плоской выгрузки в нормальную форму.

Справочник живёт в таблицах (`model/blocks/checks/defects`), и оценка обязана
читать его оттуда, а не из дерева, вырезанного под раскладку одного файла.
Форма выгрузки — строки со ссылками по идентификаторам: блок не знает своих
проверок, проверка не знает своих дефектов, порядок задаёт колонка
`sort_order`, а действующая версия выбирается признаками `status` и
`is_active`. Ровно так же справочник выбирает контур, который им измеряет:
опубликованных действующих версий обязана быть одна.

Содержимое задаёт файл, пакет задаёт правила: как читать колонку, какая строка
вообще попадает в прогон, нарушение чего меняло бы оценку, в каком порядке всё
это собирается и сколько снимков каталога живёт в процессе.

Формы выгрузки (`CriteriaExport` и строки таблиц) остаются в
`src/schemas/criteria_export.py` — здесь только то, что из них собирается.
"""

from src.integration.catalog.build import build_criteria_model
from src.integration.catalog.failure import CatalogValidationError
from src.integration.catalog.reading import (
    load_criteria_export,
    parse_criteria_export,
)
from src.integration.catalog.snapshot import load_criteria_model

__all__ = [
    "CatalogValidationError",
    "build_criteria_model",
    "load_criteria_export",
    "load_criteria_model",
    "parse_criteria_export",
]
