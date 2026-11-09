"""Точка входа отображений схемы `cjx` — реэкспорт классов по темам.

Классы определены по темам в `src/db/` (см. `base`, `eval_jobs`, `runs`,
`evaluations`, `issues`, `eval_models`), а здесь собраны одним импортом, чтобы
модули не зависели от внутреннего расклада файлов:

    from src.integration.db.models import EvalJob, Run, RunIssue

Модули замены ядра (durable/catalog/storage/persist) используют эти
отображения, чтобы читать/писать существующие таблицы схемы `cjx`, не импортируя
вендорный пакет `cjxplorer_db.models`.
"""

from __future__ import annotations

from .base import CjxBase
from .directory import Application, Journey, User
from .eval_jobs import EvalJob
from .eval_models import (
    EvalModel,
    EvalModelBlock,
    EvalModelCheck,
    EvalModelDefect,
)
from .evaluations import Evaluation, RunEvaluation
from .issues import RunIssue
from .runs import Run, RunScreenshot

__all__ = [
    "CjxBase",
    "Application",
    "Journey",
    "User",
    "EvalJob",
    "EvalModel",
    "EvalModelBlock",
    "EvalModelCheck",
    "EvalModelDefect",
    "Evaluation",
    "RunEvaluation",
    "RunIssue",
    "Run",
    "RunScreenshot",
]