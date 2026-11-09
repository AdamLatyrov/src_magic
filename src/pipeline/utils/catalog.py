"""Справочник критериальной модели: блоки, подкритерии и адреса дефектов.

Дефект различается адресом — `ключ проверки:код` (`CXCriteriaDefect.ref`);
короткая метка (`CL-11`) различается только внутри своей проверки и по всему
справочнику обязана повторяться, как только каталог растёт. Отвечать на вопрос
«где этот код» первым совпадением значит молча выбрать штраф одного из двух
дефектов, поэтому здесь адрес РАЗБИРАЕТСЯ и возвращает все совпадения: ноль —
адреса нет, одно — адрес, больше — различить нечем.

Формулу адреса задаёт `pipeline.catalog.address`: та же пара «проверка + код»
собирается при сборке справочника, и здесь она разбирается тем же правилом,
иначе найденный дефект отвечал бы чужой проверке.

Ответ модели приносит то адрес целиком, то голый код: разметчики спрашивают код
(`prompts/**`), а свести его с проверкой обязан код, а не догадка. Регистр и
пробелы — оформление ответа, они снимаются до сверки; значения наружу берутся из
справочника, а не из строки модели.
"""

from typing import Iterable, NamedTuple, Optional, Sequence

from src.pipeline.catalog.address import parts_of, ref_of
from src.pipeline.schemas.nodes import (
    CXBlock,
    CXCriteria,
    CXCriteriaDefect,
    CXCriteriaModel,
)


class DefectAt(NamedTuple):
    """Дефект вместе с местом, где он живёт: блок → проверка → дефект."""

    block: CXBlock
    check: CXCriteria
    defect: CXCriteriaDefect

    @property
    def ref(self) -> str:
        """Адрес дефекта так, как его задал справочник."""
        return self.defect.ref or ref_of(self.check.key, self.defect.code)


def locate_in(
    blocks: Iterable[CXBlock], raw: object
) -> list[DefectAt]:
    """Где в перечне блоков лежит названный адрес — все совпадения.

    :param blocks: блоки, среди которых ищется адрес (обычно весь справочник или
        один блок той ветки, что отвечает за находку)
    :param raw: адрес из ответа — `ключ проверки:код` либо одна метка
    :return: совпадения в порядке справочника; пусто, если такого адреса нет
    """
    check_key, code = parts_of(raw)
    if not code:
        return []
    return [
        DefectAt(block, check, defect)
        for block in blocks
        for check in block.criteria
        for defect in check.defects
        if defect.code.lower() == code.lower()
        and (not check_key or check.key.lower() == check_key.lower())
    ]


def locate(model: CXCriteriaModel, raw: object) -> list[DefectAt]:
    """Где в справочнике лежит названный адрес — все совпадения."""
    return locate_in(model.blocks, raw)


def unique(found: Sequence[DefectAt]) -> Optional[DefectAt]:
    """Адрес там, где совпадение одно; там, где их несколько, адреса нет.

    Нужен там, где у вызывающего нет счётчика потерь и он готов остаться без
    подписи дефекта (строка сырья, заголовок находки). Решение «не выбирать
    молча» здесь то же, что в `pipeline.utils.deviation_mapping`: несколько совпадений
    означают, что различить дефекты нечем.
    """
    return found[0] if len(found) == 1 else None


def block_of(model: CXCriteriaModel, key: str) -> Optional[CXBlock]:
    """Блок модели по ключу; чужой ключ считается отсутствующим блоком."""
    for block in model.blocks:
        if block.key == key:
            return block
    return None


def criteria_of(block: CXBlock, key: str) -> Optional[CXCriteria]:
    """Подкритерий блока по ключу."""
    for criteria in block.criteria:
        if criteria.key == key:
            return criteria
    return None


def owner_block_of(
    model: CXCriteriaModel, check_key: str
) -> Optional[CXBlock]:
    """Блок-владелец подкритерия: ключ один на всю модель."""
    for block in model.blocks:
        if criteria_of(block, check_key) is not None:
            return block
    return None
