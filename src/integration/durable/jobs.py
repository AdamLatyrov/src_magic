"""Порт очереди работников: что процессу нужно от очереди, без SQL.

Порт ничего не знает про БД и S3: реализация (`EvalJobsRepository`) читает
таблицу через AsyncSession, а процесс видит только очередь-по-контракту —
взял задание, продлил занятость, завершил или уронил.
"""

from __future__ import annotations

from typing import Protocol

from src.integration.schemas.durable import (
    HeartbeatResult,
    JobLease,
    JobTerminalStatus,
    ResultJson,
    SafeJobError,
)


class QueuePort(Protocol):
    """Очередь работников глазами процесса.

    Реализация гарантирует, что одно задание снимает ровно один процесс, а
    операции владения (`heartbeat`/`complete`/`fail`) принимаются только от
    текущего владельца лиза (см. threat-model Т2/Т5).

    Methods follow the vendr lifecycle: claim → (heartbeat)* → complete | fail.
    """

    async def claim_next(self, worker_id: str) -> JobLease | None:
        """Снимает одно свободное задание FIFO-порядка; `None`, если нет.

        :param worker_id: имя этого процесса в очереди
        :return: снятое задание с действующим лизом либо `None`, когда нечего
            взять (или достигнут лимит параллельных)
        """
        ...

    async def heartbeat(
        self,
        lease: JobLease,
        *,
        progress: str | None = None,
        phase: str | None = None,
    ) -> HeartbeatResult:
        """Продлевает лизу; сбой владения поднимается наружу.

        :param lease: лиз, выданный `claim_next`
        :param progress: продвижение (до 500 символов)
        :param phase: текущая фаза прогона (до 100 символов)
        :return: новый срок лиза и признак запрошенной отмены
        """
        ...

    async def complete(
        self,
        lease: JobLease,
        result: ResultJson,
    ) -> JobTerminalStatus:
        """Завершает задание результатом; результат валидируется.

        Недопустимый/пустой/с секретами результат не принимается. Если в
        ожидании завершения задание переведено в `cancelling`, возвращает
        `cancelled`, а результат не сохраняется.

        :param lease: лиз, выданный `claim_next`
        :param result: допускающий JSON-сериализацию результат задания
        :return: терминальный статус (`done` или `cancelled`)
        """
        ...

    async def fail(
        self,
        lease: JobLease,
        error: SafeJobError,
    ) -> JobTerminalStatus:
        """Помечает задание сбоем: повторяемым или терминальным.

        Временный сбой (429 контура) — `retryable=True` → `failed_retryable`;
        необратимый — `failed_terminal`. `SafeJobError` не несёт секретов.

        :param lease: лиз, выданный `claim_next`
        :param error: безопасная ошибка задания
        :return: терминальный статус (`cancelled`/`failed_retryable`/`failed_terminal`)
        """
        ...