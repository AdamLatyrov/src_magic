"""Реализация читателя критериальной модели на AsyncSession и таблицах `eval_model*`.

Берёт активную опубликованную версию (`EvalModel`), её блоки/критерии/дефекты и
отдаёт src-форматом `CXCriteriaModel`. Справочник живёт только в базе: файла на
диске нет, и прогон без активной версии — отказ, а не тихая подмена другой
редакцией каталога.

Использует собственные отображения (`src/db/models.py`), без vendor-пакета.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.integration.db.models import (
    EvalModel,
    EvalModelBlock,
    EvalModelCheck,
    EvalModelDefect,
)
from src.pipeline.schemas.nodes import (
    CXBlock,
    CXCriteria,
    CXCriteriaDefect,
    CXCriteriaModel,
    CXModelInfo,
)
from src.pipeline.settings.params.nodes import CXZone

logger = logging.getLogger(__name__)


class CatalogReadError(RuntimeError):
    """Активной опубликованной версии справочника в базе нет."""


class DbCatalogRepository:
    """Читатель критериальной модели из базы.

    Реализует `CatalogReader`. Кеширует уже загруженную версию на время
    процесса: версии неизменяемы после публикации, перечитывать их не за чем.

    ВНИМАНИЕ по сроку жизни кеша. Инвалидации нет ни по TTL, ни по версии в
    базе: однажды прочитанная модель живёт до перезапуска процесса. Значит
    новая публикация справочника начнёт действовать только после рестарта всех
    реплик, а не сразу. Разбор — `docs/project-spec/09-maintenance/
    improvements.md`, пункт I-013; `docs/project-spec/06-data/
    caches-and-storage.md`.

    ВНИМАНИЕ по неоднозначности. `_load_active` берёт `limit(1)` с сортировкой
    по `published_at`: если активных опубликованных версий окажется две, ошибки
    не будет — молча выберется самая свежая, по `published_at`, а не по
    `version`. Две реплики при этом могут считать по разным версиям, и в
    результате это не видно. Пункт I-014.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sessions = session_factory
        self._cached: CXCriteriaModel | None = None

    async def load_criteria_model(self) -> CXCriteriaModel:
        """Активная опубликованная версия из базы.

        :raises CatalogReadError: активной версии нет или база не ответила —
            прогон без справочника измерить нечего
        """
        if self._cached is not None:
            return self._cached
        try:
            model = await self._load_active()
        except CatalogReadError:
            raise
        except Exception as exc:
            logger.warning(
                "[catalog] активная модель не прочиталась", exc_info=True
            )
            raise CatalogReadError(
                "активную версию справочника не прочли"
            ) from exc
        if model is None:
            raise CatalogReadError(
                "в базе нет активной опубликованной версии справочника"
            )
        self._cached = model
        return model

    async def _load_active(self) -> CXCriteriaModel | None:
        """Строит src-модель из активной опубликованной версии БД, либо `None`.

        `limit(1)` здесь — не защита от неоднозначности, а её молчание: при двух
        активных версиях вернётся более свежая по `published_at` без единой
        строки в журнале. См. предупреждение в докстринге класса (I-014).
        """
        async with self._sessions() as db:
            version = await db.scalar(
                select(EvalModel)
                .where(
                    EvalModel.status == "published",
                    EvalModel.is_active.is_(True),
                )
                .order_by(EvalModel.published_at.desc())
                .limit(1)
            )
            if version is None:
                return None

            blocks = await self._load_blocks(db, version.id)
            return self._compose(version, blocks)

    async def _load_blocks(
        self, db: AsyncSession, model_id: object
    ) -> list[CXBlock]:
        """Читает блоки/критерии/дефекты версии и связывает их в блоки src."""
        block_rows = {
            b.id: b
            for b in (
                await db.scalars(
                    select(EvalModelBlock)
                    .where(EvalModelBlock.model_id == model_id)
                    .order_by(EvalModelBlock.sort_order)
                )
            ).all()
        }
        checks = (
            await db.scalars(
                select(EvalModelCheck)
                .where(EvalModelCheck.model_id == model_id)
                .order_by(EvalModelCheck.sort_order)
            )
        ).all()
        defects = (
            await db.scalars(
                select(EvalModelDefect)
                .where(EvalModelDefect.model_id == model_id)
                .order_by(EvalModelDefect.sort_order)
            )
        ).all()

        # Критерии, сгруппированные по блоку, в порядке их sort_order.
        criteria_by_block: dict[object, list[CXCriteria]] = defaultdict(list)
        defects_by_check: dict[object, list[CXCriteriaDefect]] = defaultdict(list)
        for check in checks:
            defects_by_check[check.id] = []
        for defect in defects:
            defects_by_check[defect.check_id].append(
                CXCriteriaDefect(
                    code=defect.display_code,
                    defect=defect.name,
                    symptom=defect.manifestation or "",
                    zone=CXZone(defect.zone.strip().lower()),
                )
            )
        for check in checks:
            criteria_by_block[check.block_id].append(
                CXCriteria(
                    key=check.key,
                    text=check.name,
                    essence=check.description or "",
                    defects=defects_by_check.get(check.id, []),
                )
            )

        return [
            CXBlock(
                key=row.key,
                name=row.name,
                essence=row.description or "",
                criteria=criteria_by_block.get(row.id, []),
            )
            for row in block_rows.values()
        ]

    @staticmethod
    def _compose(version: EvalModel, blocks: list[CXBlock]) -> CXCriteriaModel:
        """Обёртка версии в `CXCriteriaModel`: подпись версии и блоки.

        Счётчиков зон здесь нет намеренно: их задаёт содержимое блоков, и
        отдельные поля пережили бы первую же перекалибровку каталога. Прогон
        измеряется подписью версии, а не числами, которые с ней разъедутся.
        """
        info = CXModelInfo(
            id=str(version.id),
            key=version.key,
            version=version.version,
            title=version.title or "",
            status=version.status or "",
            published_at=(
                version.published_at.isoformat() if version.published_at else ""
            ),
        )
        return CXCriteriaModel(info=info, blocks=blocks)