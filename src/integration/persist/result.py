"""Персист результата прогона: порт и адаптер на AsyncSession и таблицах схемы `cjx`.

Сохраняет `AnalyzeResult` src в те же таблицы, что пишет ядро:
`runs` + `evaluations`/`run_evaluations` (полиморф.) + `run_issues` + `run_screenshots`.
Формат результата совпадает с вендорным, чтобы S02 читал его одинаково (НФ-6);
повторная запись того же `run_id` не создаёт второй прогон (НФ-9, threat-model Т7).

Контекст прогона (`journey_id`/`app_id`/`platform`/`method`) берётся не из клиента,
а из того, кто применяет модуль (`RunContext`): прогон привязывается по данным
задания, а не по произвольному вводу (threat-model Т7).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.integration.db.models import (
    Evaluation,
    Run,
    RunEvaluation,
    RunIssue,
    RunScreenshot,
)

# Поле `Issue` не несёт zone как enum; зона передаётся строкой как есть.
ResultIssue = object  # pipeline.schemas.result.Issue
_SOURCE_DEFAULT = "agent"


class ResultWriteError(RuntimeError):
    """Ошибка сохранения результата прогона."""


class RunContextConflict(ResultWriteError):
    """`run_id` результата уже занят другим прогоном (двойной результат)."""


@dataclass(frozen=True)
class PersistedIds:
    """Идентификаторы записанных строк: прогон и его оценка."""

    run_id: str
    evaluation_id: str | None


@dataclass(frozen=True)
class RunContext:
    """Привязка прогона, пришедшая из задания (не из произвольного ввода).

    `journey_id`/`app_id`/`platform`/`method` обязательны у `runs`; `user_id` —
    владелец, если известен.
    """

    journey_id: uuid.UUID
    app_id: uuid.UUID
    platform: str
    method: str
    user_id: uuid.UUID | None = None


class ResultWriter:
    """Порт: сохраняет результат прогона и связанные данные.

    Реализация подставляется; верхний слой знает только
    `persist(AnalyzeResult, RunContext)`.
    """

    async def persist(self, result: object, context: RunContext) -> PersistedIds:
        """Сохраняет результат и возвращает идентификаторы прогона и оценки."""
        # pragma: no cover - порт


class QueueResultWriter:
    """Реализация `ResultWriter` для очереди: привязку даёт вызывающий.

    Прогоны одного процесса принадлежат разным путям и приложениям, поэтому
    общий на процесс `RunContext` приписал бы их все одной привязке (или, что
    было раньше, — случайной). Привязку называет задание: у результата её нет,
    распознавание по кадрам даёт только имена (`AnalyzeResult.detected`).
    """

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self._sessions = session_factory

    async def persist(self, result, context: RunContext) -> PersistedIds:
        """Сохраняет результат в привязке задания.

        :param result: `AnalyzeResult` прогона
        :param context: привязка прогона — путь, приложение, владелец
        :return: идентификаторы прогона и оценки
        """
        adaptor = DBResultAdaptor(self._sessions, run_context=context)
        return await adaptor.persist(result)


def context_from_result(result) -> RunContext | None:
    """Привязка, которую назвал сам результат, — если он её назвал.

    Распознавание по кадрам обычно знает только имена, поэтому отсутствие
    идентификаторов — не ошибка, а сигнал искать привязку в задании. Ошибкой
    это становится позже, когда ни результат, ни задание привязки не дают.

    :param result: `AnalyzeResult` прогона
    :return: контекст прогона или `None`, если идентификаторов нет
    """
    detected = getattr(result, "detected", None)
    raw_journey = getattr(detected, "journey_id", None)
    raw_app = getattr(detected, "app_id", None)
    if not raw_journey or not raw_app:
        return None
    return RunContext(
        journey_id=uuid.UUID(str(raw_journey)),
        app_id=uuid.UUID(str(raw_app)),
        platform="web",
        method="video",
    )


class DBResultAdaptor:
    """Реализация `ResultWriter` поверх `AsyncSession` и таблиц схемы `cjx`.

    Одна транзакция на весь персист: либо сохранился целый результат, либо ничего
    (не полузаписанный прогон). `RunContext` инжектируется применяющим.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        run_context: RunContext,
    ) -> None:
        self._sessions = session_factory
        self._run_context = run_context

    async def persist(self, result) -> PersistedIds:
        """Сохраняет `AnalyzeResult` в `runs`/оценку/находки/экраны.

        :raises RunContextConflict: `run_id` уже занят другим прогоном
        """
        async with self._sessions() as db, db.begin():
            now = datetime.now(UTC)
            run = await self._materialize_run(db, result, now)
            evaluation = await self._materialize_evaluation(
                db, run, result
            )
            await db.flush()
            self._materialize_issues(db, run, evaluation, result)
            self._materialize_screenshots(db, run, result, now)
            await db.flush()
            return PersistedIds(
                run_id=str(run.id),
                evaluation_id=str(evaluation.evaluation_id),
            )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _run_status(result) -> str:
        """Статус прогона по тому, чем кончился анализ.

        `done` ставится только полному прогону: деградированный (часть блоков не
        посчитана, модель ответила отказом) обязан читаться как `failed`, иначе
        прогон с половиной проверок выглядит на витрине как исправный — та же
        ошибка, что и молчаливый `done` на пустом пути.

        :param result: результат прогона
        :return: значение `runstatus`
        """
        return "failed" if getattr(result, "is_degraded", False) else "done"

    async def _materialize_run(self, db, result, now) -> Run:
        """Находит прогон по `run_id` или создаёт новый; чужой не трогаем.

        Если прогон с этим `run_id` уже существует — обновляем только
        analysis-owned поля (статус/модель/время); новый `Run` с другим id не
        создаём. Это и есть идемпотентность по `run_id` (НФ-9).
        """
        run_id = uuid.UUID(str(result.run_id))
        run = await db.scalar(select(Run).where(Run.id == run_id))
        if run is None:
            ctx = self._run_context
            run = Run(
                id=run_id,
                journey_id=ctx.journey_id,
                app_id=ctx.app_id,
                platform=ctx.platform,
                method=ctx.method,
                status="pending",
                validation_status="draft",
                user_id=ctx.user_id,
            )
            db.add(run)
        run.status = self._run_status(result)
        run.model = result.model
        run.package_name = result.package_name
        run.app_version = result.app_version
        run.finished_at = now
        run.token_usage = self._token_usage(result)
        return run

    async def _materialize_evaluation(
        self, db, run: Run, result
    ) -> RunEvaluation:
        """Создаёт оценку прогона (полиморфная: evaluations + run_evaluations).

        `run_id` уникален в `run_evaluations` — повторная запись того же прогона
        конфликтует на уровне БД, и двойной оценки не бывает.
        """
        base = Evaluation(
            evaluation_type="run",
            evaluator_type="ai",
            score=result.score,
            critical_count=result.critical_count or 0,
            created_by=self._run_context.user_id,
        )
        db.add(base)
        await db.flush()
        evaluation = RunEvaluation(
            evaluation_id=base.id,
            run_id=run.id,
            model_used=result.model,
            eval_model_id=(
                uuid.UUID(str(result.eval_model_id))
                if result.eval_model_id
                else None
            ),
            eval_model_version=result.eval_model_version,
            base_score=result.base_score,
            friction_penalty=result.friction_penalty,
            rating=result.rating,
            friction_count=len(result.frictions),
            stop_flag=bool(result.stop_flag),
            criteria_blocks=self._criteria_blocks(result),
        )
        db.add(evaluation)
        return evaluation

    def _materialize_issues(
        self, db, run: Run, evaluation: RunEvaluation, result
    ) -> None:
        """Пишет по одной нормализованной `RunIssue` на проблему прогона."""
        for idx, issue in enumerate(result.issues):
            db.add(
                RunIssue(
                    run_id=run.id,
                    evaluation_id=evaluation.evaluation_id,
                    idx=idx,
                    severity=issue.severity or "mid",
                    criterion=issue.criterion or None,
                    title=issue.title or None,
                    text=issue.text or "",
                    rec=issue.rec or None,
                    steps=list(issue.steps or []),
                    pins=[pin.model_dump(mode="json") for pin in issue.pins],
                    confidence=issue.confidence,
                    evidence=[
                        ev.model_dump(mode="json") for ev in issue.evidence
                    ],
                    eval_model_version=result.eval_model_version,
                    applied_profiles=list(result.applied_profiles or []),
                    eval_model_id=(
                        uuid.UUID(str(result.eval_model_id))
                        if result.eval_model_id
                        else None
                    ),
                    defect_ref=issue.defect_ref or None,
                    defect_code=issue.defect_code or None,
                    check_key=issue.check_key or None,
                    zone=issue.zone or None,
                    source=getattr(issue, "source", None) or _SOURCE_DEFAULT,
                )
            )

    def _materialize_screenshots(
        self, db, run: Run, result, now: datetime
    ) -> None:
        """Пишет экраны кадров по ключам результата — без байтов (Т3).

        `s3_url` = ключ хранилища, `step_index` = позиция. Маски кадров берём из
        метаданных результата (`frame_mask_meta`), если они пришли.
        """
        masks = list(result.frame_mask_meta or [])
        apps = list(result.screenshot_apps or [])
        for index, key in enumerate(result.screenshot_keys):
            mask_meta = masks[index] if index < len(masks) else None
            app = apps[index] if index < len(apps) else None
            masked_at = (
                now
                if mask_meta and not mask_meta.get("skipped")
                else None
            )
            db.add(
                RunScreenshot(
                    run_id=run.id,
                    s3_url=key,
                    step_index=index,
                    package_name=app.package_name if app else None,
                    app_version=app.app_version if app else None,
                    masked_at=masked_at,
                    mask_meta=mask_meta,
                )
            )

    @staticmethod
    def _token_usage(result) -> dict | None:
        """Токены прогона, если результат их принёс."""
        if hasattr(result, "token_usage") and result.token_usage is not None:
            return result.token_usage.model_dump(mode="json")
        return None

    @staticmethod
    def _criteria_blocks(result) -> dict:
        """Снапшот оценки для витрин (`criteria_blocks` JSONB).

        Формат повторяет назначение вендорного снапшота: блоки/экраны/трения/
        находки/сводка — то, что S02 показывал у ядра. Сырьё результата, не
        производное.
        """
        return {
            "criteria": [
                c.model_dump(mode="json") for c in result.criteria
            ],
            "screens": [s.model_dump(mode="json") for s in result.screens],
            "frictions": [
                f.model_dump(mode="json") for f in result.frictions
            ],
            "issues": [i.model_dump(mode="json") for i in result.issues],
            "base_score": result.base_score,
            "friction_penalty": result.friction_penalty,
            "summary": result.summary,
            "feedback": result.feedback,
        }