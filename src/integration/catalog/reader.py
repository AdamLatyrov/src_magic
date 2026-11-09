"""Порт читателя критериальной модели и дефектов из базы.

Порт ничего не знает про SQL: что прочитать и в каком src-формате (CXCriteriaModel)
описывает контракт, а реализация (`DbCatalogRepository`) выбирает источник —
активную версию `EvalModel` из БД либо legacy-справочник.
"""

from __future__ import annotations

from typing import Protocol

from src.pipeline.schemas.nodes import CXCriteriaModel


class CatalogReader(Protocol):
    """Читает критериальную модель и дефекты в src-формате.

    Таблицы каталога (`Journey`/`Application`/`Product`) сюда намеренно НЕ
    входят: для прогона они не читаются — привязка идёт по
    `journey_id`/`application_id` из задания.
    """

    async def load_criteria_model(self) -> CXCriteriaModel:
        """Возвращает критериальную модель, пригодную для графа/отчёта.

        Источник (активная версия БД или legacy) скрыт за портом; обе формы
        отдаются единым `CXCriteriaModel`, включающим блоки, подкритерии и
        дефекты с зонами.
        """
        ...