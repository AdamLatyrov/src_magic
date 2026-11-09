"""Аддитивная методика: одной находке — одно списание.

Оставлена для сверки и объяснений: по ней ядро считало сохранённые прогоны
предыдущей версии, и ею удобно показывать конкретной находке её число. В итог
прогона она не входит — иначе наш ответ разошёлся бы с тем, что насчитает ядро
тем же прогоном (действующая методика — `method`).
"""

from collections.abc import Iterable
from typing import Any, Optional

from src.pipeline.scoring.reading import field_of
from src.pipeline.scoring.scale import block_score_from_items
from src.pipeline.settings.params.nodes import CXConfidence, CXZone


def penalty_weight(issue: Any) -> float:
    """Вес находки в аддитивной методике: её зона, а без зоны — её тяжесть.

    Только для сверки и объяснений: действующая методика списывает долю блока
    (`method.issue_factor`), а не фиксированную сумму.
    """
    zone = CXZone.parse(field_of(issue, "zone"))
    if zone is not None:
        return zone.weight
    legacy = CXZone.of_severity(field_of(issue, "severity"))
    return legacy.weight if legacy is not None else 0.0


def confidence_mult(confidence: Any) -> float:
    """Множитель аддитивного штрафа; неизвестный уровень получает полный вес."""
    return CXConfidence.multiplier_of(confidence, default=1.0)


def issue_penalty(issues: Iterable[Any] | None) -> float:
    """Штраф аддитивной методики: зона × confidence, одна проблема — одно списание.

    По этой методике ядро считало прогоны предыдущей версии; здесь она нужна,
    чтобы показать конкретной находке её цену и свериться с архивным прогоном.
    Отклонённые находки сюда не входят — их срез даёт `finding.scoring_view`.
    """
    penalty = 0.0
    for issue in issues or []:
        penalty += penalty_weight(issue) * confidence_mult(
            field_of(issue, "confidence")
        )
    return round(penalty, 2)


def adjusted_score(base: Optional[float], penalty: float) -> Optional[float]:
    """База 0..2 минус аддитивный штраф с клампом 0..2."""
    if base is None:
        return None
    return round(max(0.0, min(2.0, base - penalty)), 2)


def base_score_from_blocks(criteria: Iterable[Any] | None) -> Optional[float]:
    """База аддитивной методики — среднее арифметическое баллов оценённых блоков."""
    scored = [
        score
        for score in (
            block_score_from_items(field_of(criterion, "items"))
            for criterion in criteria or []
        )
        if score is not None
    ]
    if not scored:
        return None
    return round(sum(scored) / len(scored), 2)
