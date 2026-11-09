"""Реализация очереди на AsyncSession и таблице `eval_jobs`.

Повторяет проверенную на проде семантику вендора (`durable_eval_jobs/repository.py`),
но на собственных отображениях (`src/db/models.py`), без импорта vendor-пакета.
Короткие транзакции: каждый метод владеет одной bounded-транзакцией и возвращает
отвязанное состояние — работник обязан отпустить сессию до тяжёлой работы
(S3/модель).

Класс структурно реализует `QueuePort` (`src/durable/jobs.py`): те же
`claim_next`/`heartbeat`/`complete`/`fail` с типами порта.

Безопасность лиза (threat-model Т2/Т4/Т5): в БД хранится только SHA-256 токена,
поколение строго возрастает, операции владения сверяют токен+поколение+worker_id
перед любым изменением — старый лиз после восстановления более не валиден.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.integration.db.models import EvalJob
from src.integration.schemas.durable import (
    HeartbeatResult,
    JobLease,
    JobStatus,
    ResultJson,
    SafeJobError,
    validate_result,
)

logger = logging.getLogger(__name__)

# Активные (занятые) статусы: задание в работе и не может быть снято повторно.
_ACTIVE = (JobStatus.RUNNING.value, JobStatus.CANCELLING.value)

# Внутренняя константа для pg_advisory_xact_lock при снятии (см. вендор).
_ADMISSION_LOCK_ID = 18_943_193_358_967_116


class DurableJobError(RuntimeError):
    """База ошибок очереди."""


class JobNotFound(DurableJobError):
    """Задание не найдено."""


class StaleJobLease(DurableJobError):
    """Операция пришла от не-владельца или с устаревшим лизом."""


class JobStateConflict(DurableJobError):
    """Недопустимый переход состояния задания."""


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class EvalJobsRepository:
    """Очередь работников: снять, продлить, завершить, уронить.

    Реализует `QueuePort`. Метод владеет одной транзакцией и возвращает
    отвязанные данные: вызывающий не должен держать сессию на время S3/модели,
    иначе занялся бы слот пула БД.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        lease_seconds: int = 90,
        max_parallel: int = 4,
        max_attempts: int = 3,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds должен быть >= 1")
        if max_parallel < 1:
            raise ValueError("max_parallel должен быть >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts должен быть >= 1")
        self._sessions = session_factory
        self.lease_seconds = lease_seconds
        self.max_parallel = max_parallel
        self.max_attempts = max_attempts

    async def _db_now(self, db: AsyncSession) -> datetime:
        """Время базы (`clock_timestamp`) — единый эталон для лизы."""
        now = await db.scalar(select(func.clock_timestamp()))
        if now is None:
            raise RuntimeError("PostgreSQL не вернул clock_timestamp()")
        return now

    @staticmethod
    async def _admission_lock(db: AsyncSession) -> None:
        """Сериализует снятие и возврат просроченных на всех репликах."""
        await db.execute(
            select(func.pg_advisory_xact_lock(_ADMISSION_LOCK_ID))
        )

    async def _fenced_job(self, db: AsyncSession, lease: JobLease) -> EvalJob:
        """Задание, чей статус/лиза принадлежат этому лизу, иначе отказ.

        Ряд берётся под `FOR UPDATE` внутри уже начатой транзакции: сверяем
        статус активен, worker_id и хэш токена совпадают, поколение не откатилось.
        Любое расхождение — `StaleJobLease`: строку не меняем (threat-model Т2).
        """
        job = await db.scalar(
            select(EvalJob)
            .where(EvalJob.id == lease.job_id)
            .with_for_update()
        )
        if job is None:
            raise JobNotFound("Задание не найдено")
        if job.status not in _ACTIVE:
            raise StaleJobLease("Задание не в активном статусе для этого лиза")
        if (
            job.worker_id != lease.worker_id
            or job.lease_token_hash != _token_hash(lease.lease_token)
            or job.claim_generation != lease.claim_generation
        ):
            raise StaleJobLease(
                "Лиза не принадлежит этому владельцу или устарела"
            )
        return job

    async def _recover_expired(
        self, db: AsyncSession, *, now: datetime
    ) -> None:
        """Возвращает в `queued` задания с истёкшей лизой (падение процесса).

        Работа выполняется под advisory-локом снятия, чтобы не конфликтовать с
        конкурентным claim. Лиза сброшена: статус → `queued`, лиз-поля обнулены.

        `attempt` растёт на каждом возврате: без этого задание, роняющее процесс,
        возвращалось бы в очередь вечно и жгло платные обращения к контуру на
        каждом круге (НФ-11). Исчерпавший попытки прогон закрывается терминально
        с причиной — «процесс падал N раз» это факт задания, а не тишина.
        """
        await db.execute(
            EvalJob.__table__.update()
            .where(
                EvalJob.status.in_(_ACTIVE)
                & (EvalJob.lease_expires_at < now)
                & (EvalJob.attempt < self.max_attempts)
            )
            .values(
                status=JobStatus.QUEUED.value,
                attempt=EvalJob.attempt + 1,
                worker_id=None,
                lease_token_hash=None,
                heartbeat_at=None,
                lease_expires_at=None,
                updated_at=now,
            )
        )
        await db.execute(
            EvalJob.__table__.update()
            .where(
                EvalJob.status.in_(_ACTIVE)
                & (EvalJob.lease_expires_at < now)
                & (EvalJob.attempt >= self.max_attempts)
            )
            .values(
                status=JobStatus.FAILED_TERMINAL.value,
                error_json=SafeJobError(
                    code="lease_expired",
                    message="Задание брошено процессом и исчерпало попытки",
                    retryable=False,
                    phase="lease",
                    detail="LeaseExpired",
                ).as_dict(),
                worker_id=None,
                lease_token_hash=None,
                heartbeat_at=None,
                lease_expires_at=None,
                finished_at=now,
                updated_at=now,
            )
        )

    async def claim_next(self, worker_id: str) -> JobLease | None:
        """Снимает одно свободное задание FIFO-порядка.

        Шаги: вернуть просроченные → учесть лимит параллельных → взять первый
        `queued` под `FOR UPDATE ... SKIP LOCKED` → выдать новый лиз с токеном.
        Два процесса не снимут одно задание: ряд блокируется, второй пропускает.
        """
        if not worker_id or len(worker_id) > 200:
            raise ValueError("worker_id должен быть длиной 1..200")
        async with self._sessions() as db, db.begin():
            await self._admission_lock(db)
            now = await self._db_now(db)
            await self._recover_expired(db, now=now)

            active = await db.scalar(
                select(func.count())
                .select_from(EvalJob)
                .where(EvalJob.status.in_(_ACTIVE))
            )
            if int(active or 0) >= self.max_parallel:
                return None

            job = await db.scalar(
                select(EvalJob)
                .where(EvalJob.status == JobStatus.QUEUED.value)
                .order_by(EvalJob.queue_order)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                return None

            if job.status != JobStatus.QUEUED.value:
                raise JobStateConflict(
                    f"Недопустимый переход: {job.status!r} -> running"
                )
            raw_token = secrets.token_urlsafe(32)
            generation = job.claim_generation + 1
            lease_expires_at = now + timedelta(seconds=self.lease_seconds)
            job.status = JobStatus.RUNNING.value
            job.worker_id = worker_id
            job.lease_token_hash = _token_hash(raw_token)
            job.claim_generation = generation
            job.heartbeat_at = now
            job.lease_expires_at = lease_expires_at
            job.started_at = now
            job.phase = "claimed"
            job.phase_started_at = now
            job.phase_updated_at = now
            job.updated_at = now
            await db.flush()

            return JobLease(
                job_id=job.id,
                run_id=job.run_id,
                kind=job.kind,
                input_json=dict(job.input_json or {}),
                worker_id=worker_id,
                lease_token=raw_token,
                claim_generation=generation,
                attempt=job.attempt,
                phase=job.phase or "claimed",
            )

    async def heartbeat(
        self,
        lease: JobLease,
        *,
        progress: str | None = None,
        phase: str | None = None,
    ) -> HeartbeatResult:
        """Продлевает лизу; отказ владения поднимает `StaleJobLease`."""
        if progress is not None and len(progress) > 500:
            raise ValueError("progress длиннее 500 символов")
        if phase is not None and len(phase) > 100:
            raise ValueError("phase длиннее 100 символов")
        async with self._sessions() as db, db.begin():
            job = await self._fenced_job(db, lease)
            now = await self._db_now(db)
            if progress is not None:
                job.progress = progress
            if phase is not None:
                if phase != job.phase:
                    job.phase_started_at = now
                job.phase = phase
                job.phase_updated_at = now
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(
                seconds=self.lease_seconds
            )
            job.updated_at = now
            await db.flush()
            return HeartbeatResult(
                lease_expires_at=job.lease_expires_at,
                cancel_requested=(job.status == JobStatus.CANCELLING.value),
            )

    async def complete(
        self, lease: JobLease, result: ResultJson
    ) -> JobStatus:
        """Завершает задание результатом.

        Результат валидируется до записи (непустой JSON без секретов). Если
        задание уже в `cancelling` — переводим в `cancelled`, результат не пишем.
        """
        normalized = validate_result(result)
        async with self._sessions() as db, db.begin():
            job = await self._fenced_job(db, lease)
            now = await self._db_now(db)
            if job.status == JobStatus.CANCELLING.value:
                self._finish_cancelled(job, now)
                await db.flush()
                return JobStatus.CANCELLED
            if job.status != JobStatus.RUNNING.value:
                raise JobStateConflict(
                    f"Недопустимый переход: {job.status!r} -> done"
                )
            job.status = JobStatus.DONE.value
            job.result_json = normalized
            job.error_json = None
            self._finish_common(job, now)
            await db.flush()
            return JobStatus.DONE

    async def fail(
        self, lease: JobLease, error: SafeJobError
    ) -> JobStatus:
        """Помечает задание сбоем: `failed_retryable` или `failed_terminal`."""
        async with self._sessions() as db, db.begin():
            job = await self._fenced_job(db, lease)
            now = await self._db_now(db)
            if job.status == JobStatus.CANCELLING.value:
                self._finish_cancelled(job, now)
                await db.flush()
                return JobStatus.CANCELLED
            target = (
                JobStatus.FAILED_RETRYABLE
                if error.retryable
                else JobStatus.FAILED_TERMINAL
            )
            if job.status != JobStatus.RUNNING.value:
                raise JobStateConflict(
                    f"Недопустимый переход: {job.status!r} -> {target.value}"
                )
            job.status = target.value
            job.result_json = None
            job.error_json = error.as_dict()
            self._finish_common(job, now)
            await db.flush()
            return target

    @staticmethod
    def _finish_cancelled(job: EvalJob, now: datetime) -> None:
        """Терминальный статус отмены: результат/ошибка не пишутся."""
        job.status = JobStatus.CANCELLED.value
        EvalJobsRepository._finish_common(job, now)

    @staticmethod
    def _finish_common(job: EvalJob, now: datetime) -> None:
        """Общие поля завершения: лиз снят, задание больше никому не принадлежит.

        Поля лиза обнуляются вместе со статусом, а не остаются до следующего
        снятия: ограничение `ck_eval_jobs_active_lease` связывает активный лиз со
        статусом `running`/`cancelling`, и запись исхода с нетронутым лизом
        отвергается базой — задание навсегда оставалось бы в работе.
        """
        job.lease_token_hash = None
        job.lease_expires_at = None
        job.heartbeat_at = None
        job.finished_at = now
        job.updated_at = now

