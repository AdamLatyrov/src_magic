"""Формы балла прогона: срез находки и итог прогона.

Здесь только две формы, которыми балл разговаривается с разбором ответа и с
отчётом: `ScoringView` входит в балл, `RunScore` из него выходит. Вычисления —
в пакете `src/scoring`; ни одна из этих форм форму не знает и пересчитать оценку
не может, поэтому расходиться то, что прогону насчитали, и то, что прогон
печатает, могут только там, а не здесь.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from src.pipeline.settings.params.nodes import CXConfidence, CXSeverity, CXZone


class ScoringView(NamedTuple):
    """Находка в том виде, в каком она входит в балл.

    Это не вторая форма ответа, а срез нужных полей: чужая находка становится
    нашей по тем же признакам, по которым её штрафуют, — зоне, тяжести,
    уверенности и привязке. Хранить весь объект находки значило бы держать при
    бале и то, на что баллу смотреть незачем, а вместе с ним и соблазн прочитать
    сырую зону вместо эффективной.
    """

    zone: Optional[CXZone]
    severity: Optional[CXSeverity]
    confidence: Optional[CXConfidence]
    criterion: str
    check_key: Optional[str]


@dataclass(frozen=True)
class RunScore:
    """Все производные балла прогона одним результатом.

    `active` — те находки, которые действительно вошли в балл (с эффективными
    зонами и уверенностью). У методики без отдельного штрафа это единственный
    способ показать печати, за что именно блок потерял оценку: по нулевому
    `penalty` такого не восстановить.
    """

    base_score: float | None
    penalty: float
    score: float | None
    rating: int | None
    critical_count: int
    stop_flag: bool
    active: list[ScoringView] = field(default_factory=list)
