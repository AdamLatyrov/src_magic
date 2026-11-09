"""Перечни находок для промпта сверки: по строке на находку с номером.

Номера здесь — единственный способ, которым модель ссылается на находку в
ответе. Поэтому нумерация идёт с единицы и отдельно по каждому перечню: своё
`П1` у прошлых, своё `Т1` у текущих. Порядок строк — порядок входа, чтобы
счётчики в промпте и разбор ответа говорили об одних и тех же списках.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.pipeline.schemas.result import Issue


def previous_listing(issues: Sequence[Issue]) -> str:
    """Прошлые находки построчно, с номерами `П`.

    :param issues: находки прошлого прогона в их исходном порядке
    :return: перечень для промпта; пусто — заглушка «нет находок»
    """
    return _listing(issues, "П")


def current_listing(issues: Sequence[Issue]) -> str:
    """Находки текущего прогона построчно, с номерами `Т`.

    :param issues: находки текущего прогона в их исходном порядке
    :return: перечень для промпта; пусто — заглушка «нет находок»
    """
    return _listing(issues, "Т")


def _listing(issues: Sequence[Issue], prefix: str) -> str:
    """Строки вида `П2 | экран 5 | SP-12 | red | текст находки`.

    Экран и код печатаются как есть, без пересчёта: модель не должна
    пересобирать адрес находки по своему вкусу, а цитата и зона из справочника
    дают ей то, чем можно сверить смысл.
    """
    if not issues:
        # Заглушка нужна: пустой блок в промпте читается моделью как «перечень
        # забыли», а не как «находок нет».
        return "  (нет находок)"
    lines = []
    for number, issue in enumerate(issues, start=1):
        screens = ", ".join(str(step) for step in issue.steps) or "—"
        code = issue.defect_code or issue.defect_ref or "uncatalogued"
        zone = getattr(issue.zone, "value", issue.zone) or "—"
        lines.append(
            f"  {prefix}{number} | экран {screens} | {code} | {zone} | "
            f"{_text(issue)}"
        )
    return "\n".join(lines)


def _text(issue: Issue) -> str:
    """Текст находки одной строкой: заголовок и описание без переводов строк.

    Перевод строки внутри находки разорвал бы строку перечня, и модель приняла
    бы её хвост за новую находку.
    """
    parts = [part for part in (issue.title, issue.text) if part]
    text = " — ".join(parts) if parts else ""
    return " ".join(text.split()) or "(без описания)"
