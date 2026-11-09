"""Балл прогона.

Модуль собирает то, что раньше лежало в `src/schemas/scoring.py` рядом с
формами ответа: методики здесь, формы (`ScoringView`, `RunScore`) — там. Фасад
отдаёт публичный набор целиком, чтобы потребитель не выбирал модуль по догадке;
раздельно функции живут по ответам:

- `run` — в каком порядке балл применяется к прогону;
- `method` — методика ядра: доля блока, которую оставляет находка;
- `marking` — что балл пишет в форму ответа (статусы и оценки блоков);
- `scale` — переводы между шкалой 0..2 и рейтингом 1..100;
- `finding` — эффективные поля находки и её привязка;
- `additive` — прежняя аддитивика, только для сверки;
- `zones` — производные зоны для формы ответа и отчёта;
- `reading` — как балл читает поле, пока вход не типизирован.
"""

from src.pipeline.scoring.additive import (
    adjusted_score,
    base_score_from_blocks,
    confidence_mult,
    issue_penalty,
    penalty_weight,
)
from src.pipeline.scoring.finding import issue_check_key, scoring_view
from src.pipeline.scoring.marking import (
    apply_issue_scores,
    apply_issue_statuses,
    apply_ratings,
)
from src.pipeline.scoring.method import (
    criterion_score_from_issues,
    geometric_mean,
    issue_factor,
    issues_by_criterion,
)
from src.pipeline.scoring.reading import field_of
from src.pipeline.scoring.run import score_run
from src.pipeline.scoring.scale import (
    block_rating_from_items,
    block_score_from_items,
    overall_rating,
    rating_from_score,
    score_from_rating,
)
from src.pipeline.scoring.zones import worst_zone, zone_to_severity

__all__ = [
    "adjusted_score",
    "apply_issue_scores",
    "apply_issue_statuses",
    "apply_ratings",
    "base_score_from_blocks",
    "block_rating_from_items",
    "block_score_from_items",
    "confidence_mult",
    "criterion_score_from_issues",
    "field_of",
    "geometric_mean",
    "issue_check_key",
    "issue_factor",
    "issue_penalty",
    "issues_by_criterion",
    "overall_rating",
    "penalty_weight",
    "rating_from_score",
    "score_from_rating",
    "score_run",
    "scoring_view",
    "worst_zone",
    "zone_to_severity",
]
