"""Правила выгрузки, нарушение которых меняло бы оценку.

Здесь то, после чего прогон считает не тот справочник: сдвинутый порядок, зона
вне шкалы, потерянная строка, ключ или адрес занятый дважды, счётчик экспортёра
против строк версии. Каждое правило только дописывает в общий список причин —
собирает их и показывает целиком `failure.CatalogValidationError`.
"""

import logging
from typing import Any

from src.integration.catalog.columns import code_of
from src.integration.catalog.failure import listing
from src.pipeline.catalog.address import ref_of
from src.pipeline.schemas.criteria_export import (
    CheckRow,
    CriteriaExport,
    DefectRow,
)
from src.pipeline.settings.params.nodes import CXZone

logger = logging.getLogger(__name__)


def sort_orders(rows: list[Any], what: str, errors: list[str]) -> None:
    """`sort_order` обязан читаться числом: молча сдвинутая строка переставила
    бы подкритерии, а вместе с ними и порядок вердиктов в ответе блока."""
    for row in rows:
        try:
            int(code_of(row.sort_order))
        except ValueError:
            named = getattr(row, "key", "") or row.id
            errors.append(
                f"{what} {named}: порядок {row.sort_order!r} "
                "не читается числом"
            )


def known_zones(rows: list[Any], what: str, errors: list[str]) -> None:
    """Зоны выгрузки против шкалы рантайма — до сборки, а не в момент оценки.

    Шкала замкнута нарочно: её именем держится и JSON-схема ответа, и легенда
    промпта, и вес штрафа. Зона, которой в шкале нет, не имеет веса, и дефект с
    такой зоной выпал бы из счёта незаметно для рейтинга, поэтому список
    неизвестных названий собирается разом — править файл по одному имени
    дороже.
    """
    unknown: list[str] = []
    for row in rows:
        zone = code_of(getattr(row, "zone", ""))
        if zone and CXZone.parse(zone) is None:
            unknown.append(zone)
    if unknown:
        errors.append(
            f"{what}: зоны вне шкалы рантайма — "
            f"{listing(sorted(set(unknown)))} в {len(unknown)} строках"
        )


def unique_keys(rows: list[Any], what: str, errors: list[str]) -> None:
    """Ключ строки один на всю выгрузку.

    По ключу блока настройки ветки выбирают блок, по ключу проверки модель
    отвечает вердиктом, и повтор делает адрес неоднозначным: молча взятая
    первая строка отдала бы вердикты одного блока другому.
    """
    seen: set[str] = set()
    for row in rows:
        key = str(row.key or "").strip()
        if not key:
            errors.append(f"{what} {row.id}: ключ не назван")
        elif key in seen:
            errors.append(f"{what} {key}: ключ повторяется в выгрузке")
        else:
            seen.add(key)


def declared_counts(
    export: CriteriaExport, counts: dict[str, int], errors: list[str]
) -> None:
    """Счётчик экспортёра против строк выбранной версии.

    Экспортёр подписывает числом строк то, что он же и выгрузил; расхождение
    значит, что под подписью лежит другая выборка — версия выбрана не та либо
    строки потерялись при выгрузке.
    """
    for name, declared in export.counts.items():
        if name not in counts:
            continue
        if code_of(declared) != str(counts[name]):
            errors.append(
                f"_counts.{name}={declared}, а строк версии {counts[name]}"
            )


def address_of(
    defect: DefectRow, check: CheckRow | None, errors: list[str]
) -> str:
    """Адрес дефекта выгрузки против того адреса, что подписан в строке.

    Формулу задаёт `pipeline.catalog.address`, и берётся она оттуда же, откуда её
    разбирают: разошлись адрес экспортёра и адрес проверки с кодом — значит
    находка ответит не той проверке, по которой считается штраф.
    """
    if check is None:
        return defect.defect_ref or defect.display_code
    address = ref_of(check.key, defect.display_code)
    if not defect.defect_ref:
        return address
    if defect.defect_ref != address:
        errors.append(
            f"дефект {defect.defect_ref}: адрес не сходится с проверкой "
            f"{check.key} и кодом {defect.display_code}"
        )
    return defect.defect_ref


def shared_codes(codes: dict[str, set[str]]) -> None:
    """Код дефекта, вставший между двумя проверками, — не брак выгрузки.

    У поставщика каталога код короткая метка для человека (`CL-11`), и она
    обязана различаться только внутри проверки: уникален адрес целиком. Такое
    имя нельзя читать как адрес, и тот, кто его использует, возьмёт первую
    совпавшую проверку, — поэтому факт остаётся в журнале.
    """
    shared = sorted(
        code for code, addresses in codes.items() if len(addresses) > 1
    )
    if shared:
        logger.warning(
            "код дефекта не уникален между проверками: %d из %d (%s)",
            len(shared),
            len(codes),
            ", ".join(shared[:5]),
        )
