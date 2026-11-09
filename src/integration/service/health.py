"""Форма ответов живости и готовности.

Живость и готовность отвечают на разные вопросы, поэтому у них разные ответы.
`/health` отвечает процесс, который просто жив: ingress и оркестратор контейнера
должны отличать «процесс упал» от «база отпала» — иначе недоступная база
выглядит как погибший работник, и контейнер перезапускают без всякого толка.
`/health/ready` отвечает на вопрос «можно ли ему давать работу» — вот там
зависимости проверяются все, и каждая в свой срок.

Форма ответа повторяет ядро (`runtime_health.py:41-80`): `status` плюс словарь
`checks` и код `200`/`503`. Проверки идут параллельно и каждая ограничена
`timeout_s`: проверка, зависшая в ожидании ответа базы, не имеет права превратить
ответ живости в зависший запрос — ingress сам решает, что процесс мертв, по
истечению своего таймаута, и решает хуже, чем сама база в статусе проверки.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)

Check = Callable[[], Awaitable[bool]]
"""Проверка готовности: асинхронный вопрос с ответом «да/нет»."""


def liveness_payload(version: str, role: str) -> dict[str, str]:
    """Тело `/health`: сборка и роль, ни одного обращения к зависимости.

    :param version: строка сборки (`src/service/version.py`)
    :param role: роль процесса — приёмка по ней различает новый образ
    :return: тело ответа живости
    """
    return {"status": "ok", "version": version, "role": role}


async def readiness(
    checks: Mapping[str, Check], *, timeout_s: float
) -> tuple[bool, dict[str, str]]:
    """Готовность процесса по набору проверок.

    Каждая проверка — в свой срок; истечение срока или исключение дают
    `unavailable`. Исключение не поднимается наружу намеренно: ответ о
    недоступности зависимости полезнее ошибки маршрута, из которой приёмка не
    поймёт, какая именно зависимость отвалилась.

    :param checks: проверки поимённо — имя ключа попадает в ответ
    :param timeout_s: срок одной проверки
    :return: готовы ли все зависимости и состояние каждой
    """
    outcomes = await asyncio.gather(
        *(_outcome(name, check, timeout_s) for name, check in checks.items())
    )
    states = dict(outcomes)
    return all(state == "ok" for state in states.values()), states


async def _outcome(
    name: str, check: Check, timeout_s: float
) -> tuple[str, str]:
    """Состояние одной проверки в пределах срока."""
    try:
        ok = await asyncio.wait_for(check(), timeout=timeout_s)
    except Exception:
        logger.warning("readiness check %s failed", name, exc_info=True)
        return name, "unavailable"
    if not ok:
        logger.warning("readiness check %s is not ready", name)
    return name, "ok" if ok else "unavailable"
