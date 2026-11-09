"""Формы плоской выгрузки справочника.

Здесь только контракт файла: таблицы `model/blocks/checks/defects` строками со
ссылками по идентификаторам — блок не знает своих проверок, проверка не знает
своих дефектов. Ни правил сборки, ни выбора версии, ни проверки порядка: они в
пакете `src/catalog`, а тут то, что диктует поставщик выгрузки.

Колонка `weight` не читается: вес зоны задаёт шкала рантайма
(`CXZone.weight`), и у поставщика выгрузки она пуста.
`applies_to_journey_types` не читается сознательно: фильтр блоков по типу пути
снят, блок судится по любому пути (issue #474).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExportRow(BaseModel):
    """Строка выгрузки: колонки чужого экспорта, числа написаны строками."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    id: str = Field(default="", description="Идентификатор строки")


class ModelRow(ExportRow):
    key: str = Field(default="", description="Ключ модели")
    version: str = Field(default="", description="Версия модели")
    title: str = Field(default="", description="Название версии")
    status: str = Field(default="", description="Статус версии")
    is_active: str = Field(default="", description="Версия действующая")
    source_s3_key: str = Field(default="", description="Исходник выгрузки")
    source_sha256: str = Field(default="", description="Отпечаток исходника")
    created_at: str = Field(default="", description="Создана")
    published_at: str = Field(default="", description="Опубликована")


class BlockRow(ExportRow):
    model_id: str = Field(default="", description="Идентификатор версии")
    key: str = Field(default="", description="Ключ блока")
    name: str = Field(default="", description="Наименование блока")
    description: str = Field(default="", description="Сущность блока")
    sort_order: str = Field(default="", description="Порядок блока")


class CheckRow(ExportRow):
    model_id: str = Field(default="", description="Идентификатор версии")
    block_id: str = Field(default="", description="Идентификатор блока")
    key: str = Field(default="", description="Ключ проверки")
    name: str = Field(default="", description="Текст критерия")
    description: str = Field(default="", description="Сущность критерия")
    sort_order: str = Field(default="", description="Порядок проверки")
    zone: str = Field(default="", description="Зона подкритерия")


class DefectRow(ExportRow):
    model_id: str = Field(default="", description="Идентификатор версии")
    block_id: str = Field(default="", description="Идентификатор блока")
    check_id: str = Field(default="", description="Идентификатор проверки")
    defect_ref: str = Field(
        default="", description="Адрес дефекта: ключ проверки и код"
    )
    display_code: str = Field(default="", description="Код дефекта")
    name: str = Field(default="", description="Описание дефекта")
    manifestation: str = Field(default="", description="Симптом дефекта")
    zone: str = Field(default="", description="Зона дефекта")
    sort_order: str = Field(default="", description="Порядок дефекта")


class CriteriaExport(BaseModel):
    """Плоская выгрузка справочника — таблицы под именами таблиц БД."""

    model: list[ModelRow] = Field(
        default_factory=list, description="Версии справочника"
    )
    blocks: list[BlockRow] = Field(
        default_factory=list, description="Строки блоков"
    )
    checks: list[CheckRow] = Field(
        default_factory=list, description="Строки проверок"
    )
    defects: list[DefectRow] = Field(
        default_factory=list, description="Строки дефектов"
    )
    counts: dict[str, Any] = Field(
        default_factory=dict,
        alias="_counts",
        description="Счётчик строк, которым выгрузку подписал экспортёр",
    )
