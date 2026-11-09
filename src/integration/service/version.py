"""Строка сборки — её печатают ответы живости.

Опознание сборки нужно приёмке и дежурному: по нему видно, какой именно код
крутится в контейнере. Наружу он отдаётся без версии библиотеки и без пути
файла настроек — то и другое анониму не нужно.
"""

import os
import tomllib
from pathlib import Path


def version() -> str:
    """Что процесс говорит о себе.

    Приоритет такой: `BUILD_REV` (его называет сборка контура — только она знает
    ревизию дерева, из которого собрана пачка), затем версия из `pyproject.toml`
    этого дерева (прогон с машины разработчика остаётся опознаваемым), затем
    `dev` — прогон без сборки вообще.

    :return: строка сборки, пустой не бывает
    """
    revision = os.environ.get("BUILD_REV", "").strip()
    if revision:
        return revision
    return _project_version() or "dev"


def _project_version() -> str:
    """Версия проекта из `pyproject.toml` этого дерева; пусто, если он не читается.

    Файл читается от модуля, а не от текущего каталога: процесс запускают и из
    другого каталога, а ответ его живости обязан быть одинаковым в обоих случаях.
    """
    pyproject = Path(__file__).resolve().parents[2].joinpath("pyproject.toml")
    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get(
            "project"
        )
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    if not isinstance(project, dict):
        return ""
    return str(project.get("version", "")).strip()
