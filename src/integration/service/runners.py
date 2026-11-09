"""Исполнители процесса — единственный шов, которым в каркас встаёт очередь.

Каркас не знает ни про очередь, ни про задания: ему нужно уметь запускать и
останавливать то, что в нём крутится, и отличать «живо» от «молча умерло». В
фазе 1 сюда встанет `QueueRunner`, реализующий тот же протокол, и больше в
`src/service/` не меняется ничего.

Остановка по grace повторяет поведение ядра (`durable_eval_jobs/queue.py:1547-1573`):
сначала ждать, пока исполнитель дочитает начатое, потом отменять. Ждать дольше
grace процесс не имеет права — контейнеру оркестратор даёт свой срок, и после
него процесс снимают принудительно, вместе с незаписанным результатом задания.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

logger = logging.getLogger(__name__)


class Runner(Protocol):
    """То, что процесс делает, кроме ответов на HTTP.

    Attributes:
        name: имя для журнала и для ключа проверки готовности.
    """

    name: str

    async def start(self) -> None:
        """Запускает исполнителя; отказ поднимается наружу."""
        ...

    async def stop(self, grace_s: float) -> None:
        """Останавливается, успев дочитать начатое в пределах `grace_s`."""
        ...

    def ready(self) -> bool:
        """Жив ли исполнитель сейчас — по нему судит `/health/ready`."""
        ...


class RunnerRegistry:
    """Запускает и останавливает исполнителей как одно целое."""

    def __init__(self, runners: Sequence[Runner] = ()) -> None:
        """
        :param runners: исполнители этого процесса, в порядке их запуска
        """
        self._runners: list[Runner] = list(runners)
        self._started: list[Runner] = []

    def add(self, runner: Runner) -> None:
        """Добавляет исполнителя — до `start_all`, иначе он не запустится."""
        self._runners.append(runner)

    def ready(self) -> bool:
        """Готовы ли все исполнители.

        Пустой реестр готов: в фазе 0 процесс отвечает на запросы живости и без
        исполнителей, а с фазы 1 именно здесь виден молча упавший цикл очереди.
        """
        return all(runner.ready() for runner in self._runners)

    async def start_all(self, *, rollback_grace_s: float = 5.0) -> None:
        """Запускает исполнителей по порядку.

        Полузапущенного процесса не бывает: не смог стартовать второй — первый
        остановлен, ошибка поднята наружу. Иначе приложение, которое отвечает
        `200` и ничего не делает, пережило бы собственный перезапуск. Окно отката
        короткое и своё: начатого задания ещё нет, а ждать общий grace при
        старте — значит прятать отказ сборки за десять минут ничего не делающего
        процесса.

        :param rollback_grace_s: сколько ждать остановки уже запущенных
        :raises Exception: любой из исполнителей не запустился
        """
        for runner in self._runners:
            try:
                await runner.start()
            except Exception:
                logger.error(
                    "runner %s failed to start", runner.name, exc_info=True
                )
                await self._stop_started(rollback_grace_s)
                raise
            self._started.append(runner)
            logger.info("runner %s started", runner.name)

    async def stop_all(self, grace_s: float) -> None:
        """Останавливает запущенных исполнителей, каждому — свой срок.

        Срок общий по смыслу и у каждого свой по механизму: ожидания складывать
        незачем, процесс уходит целиком. Кто не уложился — отменён; ошибку
        остановки наружу этот метод не поднимает, иначе один зависший исполнитель
        помешал бы закрыть базу и выйти остальным.

        :param grace_s: сколько ждать, прежде чем отменять
        """
        await self._stop_started(grace_s)

    async def _stop_started(self, grace_s: float) -> None:
        """Останавливает всё запущенное и забывает реестр."""
        runners, self._started = self._started, []
        if not runners:
            return
        await asyncio.gather(
            *(self._stop_one(runner, grace_s) for runner in runners)
        )

    async def _stop_one(self, runner: Runner, grace_s: float) -> None:
        """Останавливает одного исполнителя в пределах срока и отменяет его."""
        try:
            await asyncio.wait_for(runner.stop(grace_s), timeout=grace_s)
        except Exception:
            logger.warning(
                "runner %s stopped by cancellation after %.1fs",
                runner.name,
                grace_s,
                exc_info=True,
            )
        else:
            logger.info("runner %s stopped", runner.name)
