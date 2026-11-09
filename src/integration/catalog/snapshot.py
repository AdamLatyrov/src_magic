"""Снимок справочника на процесс.

Настроек, которым нужен справочник, в прогоне несколько (ветка блоков, оба
разметчика, сведение), и каждая держит его у себя: без общего снимка файл
читался бы четырежды, а подменённый на середине прогона каталог разметил бы
половину находок старой редакцией, половину новой.

Ответственность обратная — не спрятать новую публикацию: две редакции в одном
процессе названы в журнале, даже когда одна из них законная.
"""

import hashlib
import logging
from pathlib import Path

from src.integration.catalog.build import build_criteria_model
from src.integration.catalog.reading import parse_criteria_export
from src.pipeline.schemas.nodes import CXCriteriaModel

logger = logging.getLogger(__name__)

_snapshots: dict[str, tuple[str, CXCriteriaModel]] = {}


def _revision(model: CXCriteriaModel) -> str:
    """Редакция каталога: версия вместе с идентификатором выпуска.

    Одного имени версии мало: под `v3` может лежать переписанный поверх
    опубликованного выпуск, и тогда два прогона, называющие одну версию, меряли
    разными линейками штрафов.
    """
    info = model.info
    return f"{info.key}@{info.version} ({info.id or 'без идентификатора'})"


def _warn_revisions(models: list[CXCriteriaModel]) -> None:
    """Две редакции каталога в одном процессе — событие, а не рабочий фон.

    Так бывает по-разному законно и незаконно: разные прогоны подряд измеряют
    разные выпуски (законно, баллы просто не сравнимы), а две секции одного
    конфига читают разные файлы (незаконно: блоки и коды находок тогда заданы
    одним каталогом, а штраф — другим). Различить их по одному факту сборки
    нельзя, поэтому молчать нельзя ни там, ни там.
    """
    revisions = sorted({_revision(model) for model in models})
    if len(revisions) > 1:
        logger.warning(
            "в процессе собрано несколько снимков справочника (%s): баллы "
            "прогонов по разные стороны этой границы не сравнимы",
            ", ".join(revisions),
        )


def load_criteria_model(path: str | Path) -> CXCriteriaModel:
    """Справочник прогона из файла плоской выгрузки — одним снимком.

    Один и тот же файл отдаёт один и тот же объект: настроек, которым нужен
    справочник, в прогоне несколько (ветка блоков, оба разметчика, сведение), и
    без общего снимка каждая держала бы свою копию каталога целиком — а файл,
    подменённый на середине прогона, разметил бы половину находок старой
    редакцией, половину новой.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    key = str(source.resolve())
    stamp = hashlib.sha256(text.encode("utf-8")).hexdigest()
    stored = _snapshots.get(key)
    if stored is not None and stored[0] == stamp:
        return stored[1]
    model = build_criteria_model(parse_criteria_export(text))
    _warn_revisions([item for _, item in _snapshots.values()] + [model])
    _snapshots[key] = (stamp, model)
    logger.info(
        "критериальная модель %s (%s): блоков %d, проверок %d, дефектов %d",
        model.info.key,
        model.info.version,
        len(model.blocks),
        sum(len(block.criteria) for block in model.blocks),
        sum(
            len(criteria.defects)
            for block in model.blocks
            for criteria in block.criteria
        ),
    )
    return model
