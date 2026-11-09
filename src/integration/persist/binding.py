"""Привязка прогона из задания: путь и приложение по продукту постановки.

`runs.journey_id`/`app_id` обязательны, а постановщик задания (владелец материа́ла)
кладёт в него только `params.product_id`: путь и приложение он выбирает сам и в
задание их не пишет. Распознавание по кадрам называет имена, но не идентификаторы
каталога, поэтому привязку приходится доставать из каталога по продукту — иначе
первый же прогон падает на записи («писать прогон некуда»), честно посчитав всё.

Каталог здесь читается ровно в одном направлении — продукт → путь/приложение.
Оценочные таблицы этого читателя не касаются.

Если продукт в задании всё же не назван (так ставит задание интерфейс, где поля
продукта нет), берётся `ANALYSIS_DEFAULT_PRODUCT_ID` — стендовая мера, включаемая
явно и громко помечаемая в журнале. Без неё отказ остаётся отказом.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.integration.db.eval_jobs import EvalJob
from src.integration.persist.result import RunContext

logger = logging.getLogger(__name__)

#: Задание ставится по кадрам пути мобильного/веб-клиента; различать источники
#: постановки работник не умеет — платформа и метод остаются за постановщиком, а
#: до тех пор они у прогона одни на все задания.
_PLATFORM = "web"
_METHOD = "video"


class BindingUnavailable(RuntimeError):
    """Привязку прогона не собрать: каталог её не даёт однозначно."""


class BindingPort(Protocol):
    """Собирает `RunContext` по заданию."""

    async def resolve(self, *, job_id: Any, input_json: dict[str, Any]) -> RunContext:
        """Возвращает привязку прогона для задания.

        :param job_id: задание, чей результат пишется
        :param input_json: вход задания — носитель продукта постановки
        :raises BindingUnavailable: продукта нет или каталог неоднозначен
        """
        ...


class CatalogBindingResolver:
    """Привязка по каталогу: продукт задания → путь и приложение.

    Единственный путь только тогда, когда он один: два пути одного продукта — это
    два разных прогона, и угадать нужный работник не вправе. Молчаливый выбор
    приписал бы находки чужому сценарию.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        default_product_id: str | None = None,
    ) -> None:
        """`default_product_id` — продукт подстановки для заданий без продукта."""
        self._sessions = session_factory
        self._default_product_id = default_product_id or None

    async def resolve(self, *, job_id: Any, input_json: dict[str, Any]) -> RunContext:
        product_id = _product_of(input_json, self._default_product_id)
        async with self._sessions() as db:
            journeys = await _ids(
                db,
                "SELECT id FROM journeys WHERE product_id = CAST(:p AS uuid)"
                " ORDER BY id",
                product_id,
            )
            apps = await _ids(
                db,
                "SELECT app_id FROM product_in_app WHERE product_id = CAST(:p AS uuid)"
                " ORDER BY app_id",
                product_id,
            )
            owner_id = await db.scalar(
                select(EvalJob.owner_id).where(EvalJob.id == job_id)
            )
        return RunContext(
            journey_id=_single(journeys, "путь", product_id),
            app_id=_single(apps, "приложение", product_id),
            platform=_PLATFORM,
            method=_METHOD,
            user_id=owner_id,
        )


def _product_of(
    input_json: dict[str, Any], default_product_id: str | None
) -> str:
    """Продукт, по которому ищется привязка: свой из задания или подставной.

    Подстановка громко пишется в журнал: прогон, вписанный в заранее названный
    путь, неотличим от названного постановщиком — без записи в журнале его потом
    не опознать.
    """
    product_id = _product_id(input_json)
    if product_id:
        return product_id
    if not default_product_id:
        raise BindingUnavailable(
            "в задании нет продукта — путь и приложение неизвестны"
        )
    logger.warning(
        "продукт в задании не назван — привязка взята из ANALYSIS_DEFAULT_PRODUCT_ID"
        " (%s): прогон приписан этому продукту, хотя постановщик его не выбирал",
        default_product_id,
    )
    return default_product_id


def _product_id(input_json: dict[str, Any]) -> str | None:
    """Продукт постановки из входа задания."""
    params = input_json.get("params") or {}
    value = params.get("product_id") or input_json.get("product_id")
    return str(value) if value else None


async def _ids(db: AsyncSession, statement: str, product_id: str) -> list[uuid.UUID]:
    """Идентификаторы каталога по продукту, в стабильном порядке запроса."""
    rows = await db.execute(text(statement), {"p": str(product_id)})
    return [row[0] for row in rows]


def _single(ids: list[uuid.UUID], what: str, product_id: str) -> uuid.UUID:
    """Ровно один идентификатор; пусто или их несколько — привязки нет."""
    if len(ids) == 1:
        return uuid.UUID(str(ids[0]))
    raise BindingUnavailable(
        f"каталог не называет единственный {what} продукта {product_id}: "
        f"найдено {len(ids)}"
    )
