"""Зона находки в том виде, в каком её читают отчёт и разбор.

Ни одна из этих функций балл не считает: первая переводит зону в тяжесть для
формы ответа, вторая выбирает худшую зону из перечисленных. Они живут рядом с
баллом постольку, поскольку берут шкалу из того же места.
"""

from collections.abc import Iterable
from typing import Any, Optional

from src.pipeline.settings.params.nodes import CXSeverity, CXZone


def zone_to_severity(zone: Any) -> Optional[CXSeverity]:
    """Зона → производная трёхзначная severity для API/витрины."""
    parsed = CXZone.parse(zone)
    return parsed.severity if parsed is not None else None


def worst_zone(zones: Iterable[Any]) -> Optional[CXZone]:
    """Худшая зона из перечисленных; None возвращается, если зон нет."""
    ranked = [zone for zone in (CXZone.parse(item) for item in zones) if zone]
    if not ranked:
        return None
    return max(ranked, key=lambda zone: zone.rank)
