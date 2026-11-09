"""Что балл делает с формой ответа: перещёлкивает статусы и проставляет оценки.

Три функции пишут в блоки оценки, а не возвращают числа: разметку подкритериев,
рейтинг блока и оценку блока по его находкам. Сами формулы — в `method` и
`scale`; здесь только порядок их применения к форме прогона и те случаи, где
оценку приходится не считать, а снимать.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.pipeline.schemas.result import CriterionOut, Issue

from src.pipeline.schemas.scoring import ScoringView
from src.pipeline.scoring.finding import issue_check_key
from src.pipeline.scoring.method import (
    criterion_score_from_issues,
    geometric_mean,
    issues_by_criterion,
)
from src.pipeline.scoring.reading import field_of
from src.pipeline.scoring.scale import (
    block_rating_from_items,
    block_score_from_items,
    score_from_rating,
)
from src.pipeline.settings.params.nodes import CXItemStatus


def apply_issue_scores(
    criteria: Iterable["CriterionOut"], views: Iterable[ScoringView]
) -> Optional[float]:
    """Оценки блоков из их находок и база прогона.

    Блок без оценки остаётся без оценки и здесь: перемножение долей вернуло бы
    сорванному блоку чистый балл только потому, что штрафовать оказалось не за
    что, — а прогон печатался бы благополучным ровно там, где контур молчал.
    """
    blocks = list(criteria)
    grouped = issues_by_criterion(blocks, views)
    scores: list[float] = []
    for block in blocks:
        if field_of(block, "rating") is None:
            continue
        value = criterion_score_from_issues(
            grouped.get(str(field_of(block, "key")), [])
        )
        scores.append(value)
        block.rating = max(1, min(100, round(value)))
        block.score = score_from_rating(value)
    return score_from_rating(geometric_mean(scores))


def apply_ratings(criteria: Iterable["CriterionOut"]) -> None:
    """Рейтинг и балл каждого блока — по его уже размеченным статусам."""
    for block in criteria:
        items = field_of(block, "items")
        block.rating = block_rating_from_items(items)
        block.score = block_score_from_items(items)


def apply_issue_statuses(
    criteria: Iterable["CriterionOut"],
    issues: Iterable["Issue"] | None,
    excluded_blocks: frozenset[str] = frozenset(),
) -> None:
    """Статусы подкритериев из сведённых находок — так же, как их выводит ядро.

    Привязанная активная находка делает подкритерий `fail`, её отсутствие гасит
    `fail` в `ok`: проблема учтена один раз, и разметка не может жить быстрее
    сведённого списка — снятая дедупом или признанная шумом находка не имеет права
    оставлять блок оштрафованным. `na` не приобретает и не теряет никто, кроме
    находки, которая под него легла: «не проверено» не может стать ни защитным
    баллом, ни штрафом само по себе. Блок, где контур сбился, размечается `na`
    целиком и из оценки выпадает.

    Отклонённая находка участвует в развязке, но не штрафуется: она снимает `fail`
    со своего подкритерия, и именно поэтому карта привязок строится по всем
    находкам, а не по оставленным — иначе прогон с единственной отклонённой
    находкой сохранил бы разметку модели вместо «повторно проверено, чисто».

    Единственный случай, когда статусы остаются как у модели, — находки есть, но ни
    одна не привязана ни к какому подкритерию: перещёлкивать по несуществующей
    привязке нельзя, это обнулило бы всю разметку прогона.

    :param criteria: блоки оценки в форме `CriterionOut` — меняются на месте
    :param issues: сведённые находки прогона — и источник `fail`, и снятие `fail`
    :param excluded_blocks: ключи блоков вне оценки, они становятся `na`
    """
    blocks = list(criteria)
    for block in blocks:
        if field_of(block, "key") in excluded_blocks:
            for item in field_of(block, "items") or []:
                item.status = CXItemStatus.NA

    found = list(issues or [])
    linked: dict[str, list["Issue"]] = {}
    for issue in found:
        key = issue_check_key(issue)
        if key:
            linked.setdefault(str(key), []).append(issue)
    if found and not linked:
        return

    for block in blocks:
        if field_of(block, "key") in excluded_blocks:
            continue
        for item in field_of(block, "items") or []:
            active = [
                issue
                for issue in linked.get(str(field_of(item, "key")), [])
                if str(field_of(issue, "review_status") or "") != "rejected"
            ]
            if active:
                item.status = CXItemStatus.FAIL
            elif item.status != CXItemStatus.NA:
                item.status = CXItemStatus.OK
