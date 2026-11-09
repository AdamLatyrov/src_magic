"""Настройки записи прогона: чем привязывать, когда задание продукт не назвал.

Постановщик задания называет продукт по контракту необязательным полем
(`SelectionAnalysis.AnalysisRequest`: «null when not given»), и интерфейс постановки
этого поля не имеет — на таком пути привязать прогон не к чему и запись отказывает.
Значение ниже включает стендовый обход: продукт берётся один на процесс.

Обход включён именно настройкой, а не подставлен в код: значение по умолчанию пусто
и отказ остаётся отказом везде, где его не просили явно. Переменная названа так,
чтобы в журнале развёртывания было видно — прогоны пишутся на заранее названный путь.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, field_validator

from src.pipeline.settings.core import CoreSettings


class RunBindingSettings(CoreSettings):
    """Продукт, по которому привязываются задания, не назвавшие свой."""

    default_product_id: str = Field(
        default="", alias="ANALYSIS_DEFAULT_PRODUCT_ID"
    )

    @field_validator("default_product_id")
    @classmethod
    def _is_uuid(cls, value: str) -> str:
        """Пусто или опознаватель; кривая строка — отказ старта с именем переменной.

        Продукт уходит в запрос как `CAST(:p AS uuid)`: неопознаватель база поймёт
        только посреди прогона, и выглядело бы это как сбой анализа, а не как
        опечатка в настройке.
        """
        value = value.strip()
        if not value:
            return ""
        try:
            UUID(value)
        except ValueError as error:
            raise ValueError(
                "ANALYSIS_DEFAULT_PRODUCT_ID должен быть пустым или опознавателем"
                f" продукта, получено {value!r}"
            ) from error
        return value

    @property
    def enabled(self) -> bool:
        """Подставной продукт настроен."""
        return bool(self.default_product_id)
