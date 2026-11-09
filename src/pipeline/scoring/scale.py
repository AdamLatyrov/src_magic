"""Переводы между шкалами балла: 0..2 внутрь, 1..100 наружу.

Здесь нет методики — только клампы и округление, которыми одна и та же оценка
показывается внутренним числом прогона и рейтингом для печати. Порядок действий
и то, что именно превращается в оценку, живёт в `method`, `marking` и `run`.
"""

from collections.abc import Iterable
from typing import Any, Optional

from src.pipeline.scoring.reading import field_of
from src.pipeline.settings.params.nodes import CXItemStatus


def rating_from_score(score: Optional[float]) -> Optional[int]:
    """Балл 0..2 → рейтинг 1..100 (None, если балла нет)."""
    if score is None:
        return None
    return max(1, min(100, round(score / 2 * 100)))


def score_from_rating(rating: Optional[float]) -> Optional[float]:
    """Оценка 1..100 → внутренний балл 0..2.

    Четыре знака, а не два: блоки перемножаются, и закруглять каждую ступень до
    сотых сложило бы ошибку по числу блоков — итог перестал бы сходиться с ядром
    на прогоне, где блоков много.
    """
    if rating is None:
        return None
    return round(max(0.0, min(2.0, rating / 100 * 2)), 4)


def block_rating_from_items(items: Iterable[Any] | None) -> Optional[int]:
    """Доля `ok` среди оценённых подкритериев; `na` не входит в знаменатель."""
    ok = fail = 0
    for item in items or []:
        status = field_of(item, "status")
        if status == CXItemStatus.OK:
            ok += 1
        elif status == CXItemStatus.FAIL:
            fail += 1
    if ok + fail == 0:
        return None
    return max(1, min(100, round(100 * ok / (ok + fail))))


def block_score_from_items(items: Iterable[Any] | None) -> Optional[float]:
    """Балл блока 0..2 из той же округлённой доли, что и отображаемый рейтинг."""
    rating = block_rating_from_items(items)
    return None if rating is None else round(rating / 100 * 2, 2)


def overall_rating(scores: Iterable[Optional[float]]) -> Optional[int]:
    """Средний балл оценённых блоков → рейтинг 1..100."""
    values = [score for score in scores if score is not None]
    if not values:
        return None
    return rating_from_score(sum(values) / len(values))
