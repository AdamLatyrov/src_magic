"""Методика балла: доля блока, которую оставляет находка, и база прогона.

Формула здесь одна и та же, что у ядра (`cjxplorer_scoring`) для всякого нового
прогона (`SCORING_VERSION_ISSUES`): каждый блок умножает свою оценку на долю,
которую оставляет ему каждая его находка, — долю берут из зоны и уверенности
(`CXZone.issue_multiplier`, `CXConfidence.issue_weight`); база прогона — среднее
геометрическое оценок блоков на шкале 1..100.

Порядок, в котором эти числа применяются к прогону, задаёт `run.score_run`; здесь
только сами вычисления, по одному числу на функцию.
"""

from collections.abc import Iterable
from math import exp, log
from typing import Any, Optional

from src.pipeline.schemas.scoring import ScoringView
from src.pipeline.scoring.reading import field_of
from src.pipeline.settings.params.nodes import (
    ISSUE_SCALE_CEIL,
    ISSUE_SCALE_FLOOR,
    CXConfidence,
    CXZone,
)


def issue_factor(zone: Any, confidence: Any = None) -> float:
    """Доля оценки блока, которую оставляет ему одна находка.

    Незнакомая зона и неназванная уверенность идут единицей: блок не имеет права
    терять оценку из-за того, что разбор чего-то не знает, — ровно как аддитивный
    штраф не имеет права падать из-за пустого поля.
    """
    parsed_zone = CXZone.parse(zone)
    multiplier = parsed_zone.issue_multiplier if parsed_zone else 1.0
    parsed_conf = CXConfidence.parse(confidence)
    weight = parsed_conf.issue_weight if parsed_conf else 1.0
    return 1 - weight * (1 - multiplier)


def criterion_score_from_issues(views: Iterable[ScoringView]) -> float:
    """Оценка блока 1..100 по его находкам; без находок блок чист."""
    score = ISSUE_SCALE_CEIL
    for view in views or []:
        score *= issue_factor(view.zone, view.confidence)
    return max(ISSUE_SCALE_FLOOR, min(ISSUE_SCALE_CEIL, score))


def geometric_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Среднее геометрическое оценок блоков; пустой вход — отсутствие оценки.

    Арифметическое среднее прячало разгромленный блок за счётными: блок,
    опущенный до единиц, стоил прогону ровно столько же, сколько блок,
    опущенный вдвое. Произведение долей делает плохой блок плохим и в итоге.
    """
    bounded = [
        max(ISSUE_SCALE_FLOOR, value) for value in values if value is not None
    ]
    if not bounded:
        return None
    mean = exp(sum(log(value) for value in bounded) / len(bounded))
    return max(ISSUE_SCALE_FLOOR, min(ISSUE_SCALE_CEIL, mean))


def issues_by_criterion(
    criteria: Iterable[Any], views: Iterable[ScoringView]
) -> dict[str, list[ScoringView]]:
    """Раскладывает находки по блокам: через подкритерий, а без него — по своему блоку.

    Привязка к подкритерию есть не у всякой находки: ветка проблем пути называет
    блок, а проверку справочника — нет. Ядро такую находку всё равно относит к
    названному блоку, и мы за ним: снять её значило бы не оштрафовать блок за то,
    что на него указал другой проход, то есть наказать блок за то, что проблему
    нашли раньше него.
    """
    blocks = list(criteria)
    block_of_check: dict[str, str] = {}
    for block in blocks:
        for item in field_of(block, "items") or []:
            key = field_of(item, "key")
            if key:
                block_of_check[str(key)] = str(field_of(block, "key"))
    known = {str(field_of(block, "key")) for block in blocks}

    grouped: dict[str, list[ScoringView]] = {}
    for view in views or []:
        name = block_of_check.get(str(view.check_key)) if view.check_key else None
        if name is None and view.criterion in known:
            name = view.criterion
        if name:
            grouped.setdefault(name, []).append(view)
    return grouped
