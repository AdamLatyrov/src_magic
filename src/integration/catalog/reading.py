"""Как выгрузка попадает в процесс: текст и файл — в формы строк.

Здесь только разбор: форма строк чужого экспорта проверяется формами из
`src/schemas/criteria_export`, а битая форма становится тем же списком причин,
что и брак содержимого, — чинят файл по одному сообщению.
"""

from pathlib import Path

from pydantic import ValidationError

from src.integration.catalog.failure import CatalogValidationError
from src.pipeline.schemas.criteria_export import CriteriaExport


def parse_criteria_export(text: str) -> CriteriaExport:
    """Выгрузка из текста файла; битая форма строк — тот же список причин."""
    try:
        return CriteriaExport.model_validate_json(text)
    except ValidationError as exc:
        raise CatalogValidationError(
            [
                f"{'.'.join(str(part) for part in item['loc'])}: "
                f"{item['msg']}"
                for item in exc.errors()
            ]
        ) from exc


def load_criteria_export(path: str | Path) -> CriteriaExport:
    """Выгрузка из файла плоских таблиц — без сборки и без снимка."""
    return parse_criteria_export(Path(path).read_text(encoding="utf-8"))
