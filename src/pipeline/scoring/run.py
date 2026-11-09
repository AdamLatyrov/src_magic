"""Итог прогона: в каком порядке балл применяется к форме ответа.

Порядок расчёта задаёт `score_run`, и он не переставляется: статусы подкритериев
выводятся из сведённых находок до того, как посчитан хоть один балл, потому что
оценка блока строится из доли `ok` уже после перещёлкивания. Блок, где нечего
оценивать (все подкритерии `na`) или где контур молчал, выпадает и из
перемножения, и из базы: отсутствие проверки обязано отличаться от нуля.

Отдельного штрафа здесь нет: списание каждой находки уже перемножено внутри её
блока, поэтому `penalty` отдаётся нулём, и `friction_penalty` ответа прогона —
это честный ноль, а не пропущенное число.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.schemas.result import CriterionOut, Issue

from src.pipeline.schemas.scoring import RunScore
from src.pipeline.scoring.finding import scoring_view
from src.pipeline.scoring.marking import (
    apply_issue_scores,
    apply_issue_statuses,
    apply_ratings,
)
from src.pipeline.scoring.scale import rating_from_score
from src.pipeline.settings.params.nodes import CXZone


def score_run(
    criteria: Iterable["CriterionOut"],
    issues: Iterable["Issue"] | None = None,
    *,
    excluded_blocks: frozenset[str] = frozenset(),
) -> RunScore:
    """Итог прогона: статусы из находок → оценки блоков → база без отдельного штрафа.

    Порядок жёсткий: до перещёлкивания статусов считать блок бессмысленно, потому
    что его оценка строится из доли `ok` после перещёлкивания, а находки затем
    перемножают именно её. База берётся средним геометрическим по блокам,
    получившим оценку, поэтому прогон с сорванными блоками теряет эти блоки из
    знаменателя, а не получает за них ноль.

    Итоговый балл равен базе: списание каждой находки уже внутри её блока, и
    вычесть его ещё раз значило бы наказать прогон дважды за одну и ту же
    находку. `penalty` остаётся нулём не как «штрафа нет», а как «штраф учтён» —
    поле ответа живёт, потому что у прогонов старой методики оно ненулевое.

    :param criteria: блоки оценки в форме `CriterionOut` — переводятся на месте
    :param issues: сведённые находки прогона — и оценка блоков, и `fail` статусов
    :param excluded_blocks: ключи сорванных блоков: они вне базы
    :return: база, нулевой штраф, итог, рейтинг, число критичных и стоп-флаг
    """
    blocks = list(criteria)
    found = list(issues or [])
    active = [
        view
        for view in (scoring_view(issue) for issue in found)
        if view is not None
    ]

    apply_issue_statuses(blocks, found, excluded_blocks=excluded_blocks)
    apply_ratings(blocks)
    base = apply_issue_scores(blocks, active)

    zones = [view.zone for view in active if view.zone is not None]
    return RunScore(
        base_score=base,
        penalty=0.0,
        score=base,
        rating=rating_from_score(base),
        critical_count=sum(
            1 for zone in zones if zone.at_least(CXZone.RED)
        ),
        stop_flag=any(zone.at_least(CXZone.BLACK) for zone in zones),
        active=active,
    )
