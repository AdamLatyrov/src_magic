"""Публичный вход в анализ пользовательского пути.

Создайте `AnalysisPipeline` с настройками графа, критериальным каталогом и
провайдером модели. На каждый запуск передавайте `PipelineRequest`.

`start` возвращает задачу. `get_state_json(run_id)` отдаёт текущий снимок,
а `await task` возвращает полный итог. Снимки существуют только в памяти
процесса. Интеграция сохраняет их и результат в своём хранилище.

Контракт и ограничения описаны в `pipeline/documentation/contract.md`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from inspect import iscoroutinefunction
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.pipeline.aef.models import (
        ChatOptions,
        ChatRequest as ModelRequest,
        ChatResponse,
    )
from src.pipeline.agent.graph import build_graph
from src.pipeline.dedup import DedupResult, dedup
from src.pipeline.dedup.reconcile import same_reference
from src.pipeline.dto.agent.result import AgentResult
from src.pipeline.schemas.nodes import CXCriteriaModel, CXFrame
from src.pipeline.schemas.api import (
    ExpertDecision,
    PipelineRequest,
    PipelineRunResult,
    PipelineRunStatus,
    PipelineProgressSnapshot,
    PipelineStageSnapshot,
    PipelineStateSnapshot,
)
from src.pipeline.schemas.result import AnalyzeResult, Issue
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import GraphSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import CXNodeStatus
from src.pipeline.scoring import issue_check_key, score_run, worst_zone
from src.pipeline.status import RunTracker, StageName, StageState
from src.pipeline.utils.journey import verdict_of

logger = logging.getLogger(__name__)


class PipelineRefused(RuntimeError):
    """Прогон не начат: вход непригоден, платить за него нечего."""


class PipelineProvider(Protocol):
    """Асинхронный API провайдера, которого ждут узлы графа."""

    async def chat(
        self,
        request_options: ModelRequest,
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse: ...

    async def chat_structured_output(
        self,
        request_options: ModelRequest,
        structured_output: type[BaseModel],
        method: str = "format_instructions",
    ) -> BaseModel: ...

    async def chat_with_structure_output_with_frames(
        self,
        request_options: ModelRequest,
        structured_output: type[BaseModel],
        method: str = "format_instructions",
    ) -> BaseModel: ...

    async def upload_frame(
        self,
        options: ChatOptions | None,
        model: str,
        data_url: str,
    ) -> str: ...

    async def delete_frames(
        self,
        options: ChatOptions | None,
        model: str,
        file_ids: Sequence[str],
    ) -> None: ...


def _recursion_limit(frames: int) -> int:
    """Рассчитать предел супершагов LangGraph для одного прогона.

    LangGraph по умолчанию разрешает 25 супершагов. Поскольку сканирование
    проходит по одному кадру за супершаг, а затем выполняются ветки проверок и
    сведение, длинный путь может превысить этот лимит. Тогда граф завершится
    ошибкой превышения лимита до выдачи полного результата.

    Публичный фасад передаёт собственный предел в `ainvoke`, поэтому значение
    LangGraph по умолчанию здесь не действует. Формула — `max(50, frames * 4 + 50)`.
    Например, для 30 кадров предел равен 170 супершагам. Это лимит на выполнение
    графа, а не количество запросов к модели. Сейчас формула задана в коде и не
    настраивается через переменную окружения.

    При прямом запуске собранного графа в обход `AnalysisPipeline` вызывающий
    код должен сам передать подходящий `recursion_limit`.
    """
    return max(50, frames * 4 + 50)


class AnalysisPipeline:
    """Объект, который запускает анализ и выдаёт его состояние.

    Передайте готовые настройки этапов, каталог нужной версии и провайдера.
    Граф будет собран при создании объекта. Для каждого прогона создайте
    `PipelineRequest` и вызовите `start`.

    `start` возвращает `asyncio.Task`, а `get_state_json` можно вызывать,
    пока задача работает. Когда она закончится, `await task` вернёт
    `PipelineRunResult`. Задачи и снимки не сохраняются на диск и пропадают
    после перезапуска процесса.
    """

    def __init__(
        self,
        settings: GraphSettings,
        catalog: CXCriteriaModel,
        provider: PipelineProvider,
    ) -> None:
        """
        :param settings: настройки графа и этапов, предоставленные вызывающей стороной
        :param catalog: справочник блоков; без него граф не собирается
        :param provider: провайдер контура, собранный вызывающей стороной
        :raises PipelineRefused: справочник не того типа или пуст
        """
        self._settings = settings
        self._graph = self._build_graph(catalog)
        self._catalog = catalog
        self._provider = self._validated_provider(provider)
        self._tasks: dict[str, asyncio.Task[PipelineRunResult]] = {}
        self._run_trackers: dict[str, RunTracker] = {}
        self._run_states: dict[str, PipelineProgressSnapshot] = {}

    # ── Сборка ─────────────────────────────────────────────────────────

    def _build_graph(self, catalog: CXCriteriaModel) -> CompiledStateGraph:
        """Граф по справочнику: состав веток зависит от числа блоков.

        Справочник раздаётся настройкам здесь, а не позже: узлы спрашивают
        блоки уже при сборке, и без подкладки сборка отказывает. Вызывать эту
        сборку дважды подряд безопасно — настройки держат один и тот же объект.
        """
        if not isinstance(catalog, CXCriteriaModel):
            raise PipelineRefused(
                "справочник должен быть CXCriteriaModel, а пришло "
                f"{type(catalog).__name__}"
            )
        if not catalog.blocks:
            raise PipelineRefused(
                "в справочнике нет ни одного блока — мерить не по чему"
            )
        self._settings.bind_criteria_model(catalog)
        graph = build_graph(self._settings)
        logger.info(
            "%s: справочник %s@%s, блоков %d",
            LogLabel.CHECK_CX_CRITERIA_FRAMES,
            catalog.info.key,
            catalog.info.version,
            len(catalog.blocks),
        )
        return graph

    # ── Настройки, которые можно поменять у работающего объекта ─────────

    @staticmethod
    def _validated_provider(provider: PipelineProvider) -> PipelineProvider:
        """Ранняя проверка имён API, но не подтверждение совместимости форматов.

        Узлы ждут ответы ChatRequest/ChatResponse и Pydantic, а не просто
        одноимённые методы. Подлинную совместимость проверяет интеграционный прогон.
        """
        required = (
            "chat", "chat_structured_output",
            "chat_with_structure_output_with_frames", "upload_frame", "delete_frames",
        )
        missing = [name for name in required if not callable(getattr(provider, name, None))]
        if missing:
            raise PipelineRefused(
                "провайдер не поддерживает API pipeline: " + ", ".join(missing)
            )
        return provider

    @property
    def settings(self) -> GraphSettings:
        """Настройки графа, которыми собран этот объект."""
        return self._settings

    @property
    def graph_settings(self) -> GraphSettings:
        """Настройки графа: то, что читают этапы на каждом прогоне."""
        return self._settings

    @property
    def catalog(self) -> CXCriteriaModel:
        """Справочник, по которому собран граф."""
        return self._catalog

    @property
    def provider(self) -> PipelineProvider:
        """Провайдер контура, которым отвечен прогон."""
        return self._provider

    def _reject_hot_swap(self) -> None:
        if any(state.status in (PipelineRunStatus.QUEUED, PipelineRunStatus.RUNNING)
               for state in self._run_states.values()):
            raise PipelineRefused("нельзя менять модель или провайдера во время прогона")

    def with_provider(self, provider: PipelineProvider) -> AnalysisPipeline:
        """Тот же прогон, но в контур другого провайдера.

        Граф пересобирать не нужно: провайдера он получает в конфигурации
        вызова, а не держит у себя. Меняется только то, кому уходят вопросы.

        :param provider: новый провайдер
        :return: этот же объект — для цепочки вызовов
        """
        self._reject_hot_swap()
        self._provider = self._validated_provider(provider)
        return self

    def with_catalog(self, catalog: CXCriteriaModel) -> AnalysisPipeline:
        """Тот же объект на другом справочнике — с ПЕРЕСБОРКОЙ графа.

        Пересборка здесь обязательна, а не перестраховка. Имена узлов проверки
        блоков собираются из ключей блоков при `build_graph`; если подменить
        справочник и не пересобрать граф, узлы будут спрашивать блоки, которых
        в новом справочнике нет, и прогон вернёт пустой список находок —
        снаружи это неотличимо от чистого интерфейса.

        :param catalog: новый справочник
        :return: этот же объект — для цепочки вызовов
        """
        self._reject_hot_swap()
        self._graph = self._build_graph(catalog)
        self._catalog = catalog
        return self

    # ── Вход прогона ───────────────────────────────────────────────────

    @staticmethod
    def build_state(
        frames: list[CXFrame],
        catalog: CXCriteriaModel,
        material: str = "",
        screenshot_keys: list[str] | None = None,
    ) -> dict:
        """Исходное состояние прогона из готовых кадров.

        Словари находок начинаются пустыми словарями, а не списками: сведение
        полей в графе — `operator.or_`, и список в начальном значении роняет
        прогон на первом же супершаге, до обращения к контуру.

        :param frames: кадры пути в порядке следования
        :param catalog: справочник прогона — кладётся в состояние, чтобы все
            узлы читали один и тот же
        :param material: заметка эксперта; единственный текст, который видит модель
        :param screenshot_keys: ключи кадров, если вызывающая сторона их ведёт
        :raises PipelineRefused: кадров нет
        :return: начальное состояние для `ainvoke`
        """
        if not frames:
            raise PipelineRefused(
                "кадров нет — на пустом пути прогон завершится мгновенно и "
                "вернёт пустой результат, неотличимый от чистого интерфейса"
            )
        return {
            "frames": frames,
            "screenshot_keys": list(screenshot_keys or []),
            "material": material,
            "criteria_model": catalog,
            "frame_readouts": [],
            "cx_findings": {},
            "journey_frictions": {},
            "autofill_findings": None,
            "screen_findings": None,
        }

    def run_config(
        self, frames: int, tracker: RunTracker | None = None
    ) -> dict:
        """Конфигурация вызова графа: провайдер, настройки и предел рекурсии.

        Зависимости узлов уходят через `configurable`, а не замыканием сборки:
        обёртка `@node` берёт их оттуда по именам аргументов функции узла.
        Запуск без `configurable` роняет прогон на первом узле.

        Журнал этапов кладётся туда же: пока прогон идёт, его читает снаружи
        работник очереди или ноутбук, и по нему видно, на каком этапе прогон
        стоит сейчас.

        :param frames: сколько кадров у пути — по нему считается предел рекурсии
        :param tracker: журнал этапов; не задан — узел отметку не пишет
        :return: конфигурация для `ainvoke`
        """
        return {
            "recursion_limit": _recursion_limit(frames),
            "configurable": {
                "gigachat_provider": self._provider,
                "settings": self._settings,
                "tracker": tracker,
            },
        }

    # ── Прогон ─────────────────────────────────────────────────────────

    async def run_graph(
        self,
        frames: list[CXFrame],
        material: str = "",
        tracker: RunTracker | None = None,
        *,
        run_id: str | None = None,
        screenshot_keys: list[str] | None = None,
    ) -> AnalyzeResult:
        """Граф по кадрам и ответ прогона — без сверки с прошлым.

        Разделено с `analyze` намеренно: сверка стоит отдельного обращения к
        модели, и её видно счётчиком. Здесь же — только то, за что платит сам
        прогон.

        :param frames: кадры пути в порядке следования
        :param material: заметка эксперта
        :param tracker: журнал этапов; не задан — на этот вызов свой
        :param run_id: идентификатор корреляции; прокидывается в состояние графа
        :param screenshot_keys: ключи внешнего хранения в порядке кадров
        :raises PipelineRefused: кадров нет
        :return: ответ прогона целиком
        """
        run = tracker or RunTracker()
        state = self.build_state(frames, self._catalog, material, screenshot_keys)
        if run_id is not None:
            state["run_id"] = run_id
        config = self.run_config(len(frames), run)

        run.begin(StageName.SCAN_JOURNEY_FRAME, note=f"кадров {len(frames)}")
        final_state = await self._graph.ainvoke(state, config)
        checked = AgentStateSchema.model_validate(final_state)
        self._record_stages(run, checked)

        result = AgentResult(
            state=checked,
            criteria=self._catalog,
            analyzed_by=self._settings.consolidate_findings.model,
        ).get_result()
        # Не стираем DEGRADED/SKIPPED, зафиксированные по состоянию графа.
        if run.state_of(StageName.CONSOLIDATE_FINDINGS) not in (
            StageState.DEGRADED, StageState.FAILED, StageState.SKIPPED,
        ):
            run.finish(
                StageName.CONSOLIDATE_FINDINGS,
                StageState.DONE if result.issues else StageState.EMPTY,
                note=f"находок {len(result.issues)}",
                counters={"issues": len(result.issues)},
            )
        logger.info(
            "%s: прогон собран, находок %d",
            LogLabel.CONSOLIDATE_FINDINGS,
            len(result.issues),
        )
        if run_id is not None:
            result = result.model_copy(update={"run_id": run_id})
        return result

    @staticmethod
    def _record_stages(run: RunTracker, state: AgentStateSchema) -> None:
        """Записать в журнал этапы, о которых знает итоговое состояние.

        Граф наружу отдаёт состояние, а не историю шагов: этап с деградацией
        виден по полям состояния (`degraded`, `status`, пустой результат), и
        другого способа отличить сбой от честной пустоты у вызывающей стороны
        нет. Поэтому запись собирается здесь, по одному месту на этап, а не
        размазывается по узлам.

        :param run: журнал этапов прогона
        :param state: итоговое состояние графа
        """
        readouts = state.frame_readouts
        answered = sum(1 for readout in readouts if readout.answered)
        flawed = sum(1 for readout in readouts if readout.degraded)
        run.finish(
            StageName.SCAN_JOURNEY_FRAME,
            (StageState.DONE if not flawed and (
                answered == len(state.frames) or
                (verdict_of(state.journey_details) is False and answered)
            ) else StageState.DEGRADED),
            note=f"с ответом {answered} из {len(state.frames)}, замечаний {flawed}",
            counters={"frames": len(readouts), "answered": answered, "flawed": flawed},
        )

        details = state.journey_details
        if details is None:
            run.finish(
                StageName.SUMMARIZE_JOURNEY,
                StageState.SKIPPED,
                note="сведение описаний кадров не запускалось",
            )
        elif verdict_of(details) is False:
            run.finish(
                StageName.SUMMARIZE_JOURNEY,
                StageState.DONE,
                note="путь не подтверждён; ветки не запускались",
            )
        else:
            run.finish(
                StageName.SUMMARIZE_JOURNEY,
                (
                    StageState.DEGRADED
                    if details.degraded or str(details.status) != CXNodeStatus.SUCCESS.value
                    else StageState.DONE
                ),
                note="атрибуция получена" if not details.degraded else "атрибуция неполная",
            )

        for name, findings in (
            (StageName.CRITERIA_MODEL_FRAMES, state.criteria_model_findings),
            (StageName.SCREEN, state.screens),
            (StageName.JOURNEY_FRICTIONS, state.journey_findings),
            (StageName.AUTOFILL, state.autofill),
        ):
            stage_name, stage_state, reason = _findings_stage(name, findings)
            if not findings and verdict_of(details) is not False:
                stage_state = StageState.DEGRADED
                reason = "ветка не вернула состояние"
            run.finish(stage_name, stage_state, note=reason)

        if state.cx_consolidation is None:
            run.finish(
                StageName.CONSOLIDATE_FINDINGS,
                StageState.SKIPPED if verdict_of(details) is False else StageState.DEGRADED,
                note=("ветки пропущены: не пользовательский путь"
                      if verdict_of(details) is False else
                      "сведение не состоялось — находки собраны по веткам"),
            )

    async def reconcile(
        self,
        previous: list[Issue],
        current: list[Issue],
        tracker: RunTracker | None = None,
    ) -> DedupResult:
        """Сверка находок с прошлым прогоном — отдельным обращением к модели.

        Пустой `previous` — первый прогон по продукту: сверка не платится и
        находки возвращаются как есть.

        :param previous: находки прошлого прогона в их порядке
        :param current: находки текущего прогона в их порядке
        :param tracker: журнал этапов; не задан — на этот вызов свой
        :return: итоговый список находок и счётчики сверки
        """
        run = tracker or RunTracker()
        if not previous:
            run.finish(
                StageName.DEDUP_FINDINGS,
                StageState.SKIPPED,
                note="прошлых находок нет — сверки не было",
            )
            return await dedup(
                previous,
                current,
                settings=self._settings.dedup_findings,
                provider=self._provider,
            )

        run.begin(
            StageName.DEDUP_FINDINGS,
            note=f"прошлых {len(previous)}, новых {len(current)}",
        )
        result = await dedup(
            previous,
            current,
            settings=self._settings.dedup_findings,
            provider=self._provider,
        )
        run.finish(
            StageName.DEDUP_FINDINGS,
            StageState.DEGRADED if result.degraded else StageState.DONE,
            note=(
                "сверка без модели — прошлое поднято по коду"
                if result.degraded
                else f"слито {result.merged}, поднято {result.restored}"
            ),
            counters={
                "restored": result.restored,
                "merged": result.merged,
                "added": result.added,
                "model_calls": result.model_calls,
            },
        )
        return result

    async def analyze(
        self,
        frames: list[CXFrame],
        previous: list[Issue] | None = None,
        material: str = "",
        tracker: RunTracker | None = None,
    ) -> list[Issue]:
        """Главный метод: кадры и прошлые находки на входе, проблемы на выходе.

        Возвращает только список найденных проблем. Балла, привязки к пути,
        кодов прогона здесь нет: их считает и пишет вызывающая сторона.

        :param frames: кадры пути в порядке следования
        :param previous: находки прошлого прогона того же продукта; пусто — сверки нет
        :param material: заметка эксперта к материалам пути, пустая допустима
        :param tracker: журнал этапов; не задан — на этот вызов свой
        :raises PipelineRefused: кадров нет
        :return: найденные проблемы, уже сведённые с прошлым прогоном
        """
        run = tracker or RunTracker()
        run.begin_run()
        try:
            result = await self.run_graph(frames, material, run)
            reconciled = await self.reconcile(
                previous or [], result.issues, run
            )
        except Exception:
            run.finish_run()
            raise
        run.finish_run()
        logger.info(
            "%s: итог сверки — находок %d (поднято %d, слито %d, новых %d)",
            LogLabel.DEDUP_FINDINGS,
            len(reconciled.issues),
            reconciled.restored,
            reconciled.merged,
            reconciled.added,
        )
        return reconciled.issues

    async def run(
        self,
        request: PipelineRequest,
        *,
        on_event: Callable[[PipelineStateSnapshot], None] | None = None,
    ) -> PipelineRunResult:
        """Выполнить один прогон; снимки живут только в этом объекте и процессе.

        `on_event` — синхронная функция. Исключения callback журналируются,
        но не изменяют результат анализа. Для асинхронной записи используйте
        `asyncio.Queue.put_nowait` и отдельного consumer-а интеграции.
        """
        self._check_callback(on_event)
        if request.run_id in self._run_trackers or (
            request.run_id in self._run_states
            and self._tasks.get(request.run_id) is not asyncio.current_task()
        ):
            raise PipelineRefused(f"run_id уже используется: {request.run_id}")

        tracker = RunTracker(
            on_change=lambda: self._publish_state(request.run_id, tracker, on_event)
        )
        self._run_trackers[request.run_id] = tracker
        previous_state = self._run_states.get(request.run_id)
        self._run_states[request.run_id] = PipelineProgressSnapshot(
            run_id=request.run_id,
            status=PipelineRunStatus.RUNNING,
            revision=previous_state.revision + 1 if previous_state else 1,
        )
        tracker.begin_run()
        try:
            analysis = await self.run_graph(
                request.frames, request.material, tracker,
                run_id=request.run_id, screenshot_keys=request.screenshot_keys,
            )
            # A confirmed non-journey must not resurrect older problems.
            if not analysis.is_journey:
                reconciled = DedupResult()
                tracker.finish(
                    StageName.DEDUP_FINDINGS, StageState.SKIPPED,
                    note="не пользовательский путь — сверка не выполняется",
                )
            else:
                rejected = [
                    item.issue for item in request.previous
                    if item.expert_status == ExpertDecision.REJECTED
                ]
                previous = [
                    item.issue.model_copy(update={
                        "verified_verdict": None,
                        "review_status": str(item.expert_status),
                    }) for item in request.previous
                    if item.expert_status != ExpertDecision.REJECTED
                ]
                # Exact addresses of expert-rejected findings are excluded even
                # if the model rediscovers them in this run. Unaddressed semantic
                # duplicates cannot be reliably recognized without expert input.
                current = [
                    issue for issue in analysis.issues
                    if not any(same_reference(issue, old) for old in rejected)
                ]
                reconciled = await self.reconcile(previous, current, tracker)

            criteria = [item.model_copy(deep=True) for item in analysis.criteria]
            # The original per-check narrative/zone was built from the raw
            # issues. Synchronize display fields with the reconciled list too.
            for block in criteria:
                for item in block.items:
                    matches = [
                        issue for issue in reconciled.issues
                        if issue.criterion == block.key
                        and issue_check_key(issue) == item.key
                    ]
                    item.improvement = matches[0].rec if matches else ""
                    item.zone = worst_zone(issue.zone for issue in matches)
                block.zone = worst_zone(item.zone for item in block.items)
            score = None if not analysis.is_journey else score_run(
                criteria, reconciled.issues,
                excluded_blocks=frozenset(analysis.failed_blocks),
            )
            degradation_reasons = list(analysis.degradation_reasons)
            if reconciled.degraded:
                degradation_reasons.append("deduplication_degraded")
            incomplete_stages = any(
                stage.state in (StageState.DEGRADED, StageState.FAILED, StageState.RUNNING)
                for stage in tracker.stages()
            )
            if incomplete_stages and not analysis.is_degraded:
                degradation_reasons.append("stage_status_degraded")
            is_degraded = analysis.is_degraded or reconciled.degraded or incomplete_stages
            changes = {
                "run_id": request.run_id,
                "product_name": request.product_name,
                "product_type": request.product_type,
                "screenshot_keys": list(request.screenshot_keys),
                "issues": reconciled.issues,
                "criteria": criteria,
                "analysis_status": (
                    "partial" if is_degraded else analysis.analysis_status
                ),
                "is_degraded": is_degraded,
                "degradation_reasons": degradation_reasons,
            }
            if score is not None:
                changes.update({
                    "score": score.score, "base_score": score.base_score,
                    "friction_penalty": score.penalty, "rating": score.rating,
                    "critical_count": score.critical_count,
                    "stop_flag": score.stop_flag,
                })
            final_analysis = analysis.model_copy(update=changes)
            tracker.finish_run()
            snapshot = self._snapshot(
                request.run_id, tracker, len(reconciled.issues),
                partial=is_degraded,
                analysis_status=final_analysis.analysis_status,
                is_degraded=final_analysis.is_degraded,
                degradation_reasons=final_analysis.degradation_reasons,
                failed_blocks=final_analysis.failed_blocks,
                fallbacks_used=final_analysis.fallbacks_used,
            )
            snapshot = self._save_terminal_state(snapshot, on_event)
            return PipelineRunResult(
                run_id=request.run_id,
                product_name=request.product_name,
                product_type=request.product_type,
                analysis=final_analysis,
                findings=reconciled.issues,
                state=self._result_state(snapshot),
            )
        except asyncio.CancelledError:
            tracker.finish_run(failed=True)
            self._save_terminal_state(self._snapshot(
                request.run_id, tracker, 0, status=PipelineRunStatus.CANCELLED,
                analysis_status="cancelled",
                is_degraded=True,
                degradation_reasons=("pipeline_cancelled",),
            ), on_event)
            raise
        except Exception:
            logger.exception("pipeline run failed for %s", request.run_id)
            tracker.finish_run(failed=True)
            state = self._snapshot(
                request.run_id, tracker, 0, status=PipelineRunStatus.FAILED,
                analysis_status="failed",
                is_degraded=True,
                degradation_reasons=("pipeline_execution_error",),
            )
            state.error_code = "pipeline_execution_error"
            state.error_message = "Анализ прерван; подробности в журнале сервиса"
            self._save_terminal_state(state, on_event)
            raise

    @staticmethod
    def _check_callback(
        on_event: Callable[[PipelineStateSnapshot], None] | None,
    ) -> None:
        if on_event is not None and iscoroutinefunction(on_event):
            raise TypeError("on_event должен быть синхронным callback")

    def start(
        self,
        request: PipelineRequest,
        *,
        on_event: Callable[[PipelineStateSnapshot], None] | None = None,
    ) -> asyncio.Task[PipelineRunResult]:
        """Начать прогон и вернуть задачу для получения полного итога.

        Метод вызывается внутри работающего цикла asyncio. Интеграция может
        читать прогресс через `get_state_json`, а после завершения сохранить
        результат из `await task`. Идентификатор нельзя повторно использовать
        в этом экземпляре фасада.
        """
        self._check_callback(on_event)
        if request.run_id in self._run_states:
            raise PipelineRefused(f"run_id уже используется: {request.run_id}")
        self._run_states[request.run_id] = PipelineProgressSnapshot(
            run_id=request.run_id, status=PipelineRunStatus.QUEUED,
        )
        try:
            task = asyncio.create_task(
                self.run(request, on_event=on_event),
                name=f"pipeline-{request.run_id}",
            )
        except BaseException:
            self._run_states.pop(request.run_id, None)
            raise
        self._tasks[request.run_id] = task

        def finish_before_start(done: asyncio.Task[PipelineRunResult]) -> None:
            # Не держим весь AnalyzeResult в завершённой Task бесконечно;
            # зарезервированный id остаётся в _run_states.
            self._tasks.pop(request.run_id, None)
            state = self._run_states[request.run_id]
            if done.cancelled() and state.status == PipelineRunStatus.QUEUED:
                self._save_terminal_state(PipelineProgressSnapshot(
                    run_id=request.run_id,
                    status=PipelineRunStatus.CANCELLED,
                    finished_at=datetime.now(timezone.utc),
                    analysis_status="cancelled",
                    is_degraded=True,
                    degradation_reasons=["pipeline_cancelled_before_start"],
                ), on_event)

        task.add_done_callback(finish_before_start)
        return task

    def get_state(self, run_id: str) -> PipelineProgressSnapshot:
        """Полный JSON-ready снимок прогресса по run_id из памяти процесса."""
        stored = self._run_states.get(run_id)
        if stored is None:
            raise KeyError(f"неизвестный run_id {run_id}")
        if stored.status != PipelineRunStatus.RUNNING:
            return stored.model_copy(deep=True)
        tracker = self._run_trackers.get(run_id)
        if tracker is None:
            return stored.model_copy(deep=True)
        return self._snapshot(
            run_id,
            tracker,
            stored.findings_count,
            revision=stored.revision,
            updated_at=stored.updated_at,
            analysis_status=stored.analysis_status,
            is_degraded=stored.is_degraded,
            degradation_reasons=stored.degradation_reasons,
            failed_blocks=stored.failed_blocks,
            fallbacks_used=stored.fallbacks_used,
        )

    def get_state_json(self, run_id: str) -> dict[str, Any]:
        """Вернуть накопленный снимок в виде обычного JSON-совместимого словаря.

        HTTP-обработчик интеграции может напрямую вернуть этот словарь.
        Данные хранятся в памяти и доступны, пока живёт этот процесс.
        """
        return self.get_state(run_id).model_dump(mode="json")

    @staticmethod
    def _result_state(snapshot: PipelineProgressSnapshot) -> PipelineStateSnapshot:
        """Сохранить прежнюю схему state внутри PipelineRunResult и callback."""
        base_fields = set(PipelineStateSnapshot.model_fields)
        return PipelineStateSnapshot.model_validate(
            snapshot.model_dump(include=base_fields)
        )

    def _save_terminal_state(
        self,
        snapshot: PipelineProgressSnapshot,
        on_event: Callable[[PipelineStateSnapshot], None] | None,
    ) -> PipelineProgressSnapshot:
        previous = self._run_states.get(snapshot.run_id)
        terminal = snapshot.model_copy(update={
            "revision": previous.revision + 1 if previous else 1,
            "updated_at": datetime.now(timezone.utc),
        }, deep=True)
        self._run_states[terminal.run_id] = terminal.model_copy(deep=True)
        self._notify_callback(terminal, on_event)
        return terminal

    @staticmethod
    def _notify_callback(
        snapshot: PipelineProgressSnapshot,
        on_event: Callable[[PipelineStateSnapshot], None] | None,
    ) -> None:
        if on_event is None:
            return
        try:
            on_event(AnalysisPipeline._result_state(snapshot))
        except Exception:
            logger.exception("pipeline progress callback failed for %s", snapshot.run_id)

    def _publish_state(
        self,
        run_id: str,
        tracker: RunTracker,
        on_event: Callable[[PipelineStateSnapshot], None] | None,
    ) -> None:
        # Terminal events are emitted only AFTER the result is fully assembled.
        if tracker.finished_at is not None:
            return
        stored = self._run_states.get(run_id)
        count = stored.findings_count if stored is not None else 0
        state = self._snapshot(
            run_id,
            tracker,
            count,
            revision=(stored.revision + 1) if stored is not None else 1,
            updated_at=datetime.now(timezone.utc),
            analysis_status=stored.analysis_status if stored else None,
            is_degraded=stored.is_degraded if stored else None,
            degradation_reasons=stored.degradation_reasons if stored else (),
            failed_blocks=stored.failed_blocks if stored else (),
            fallbacks_used=stored.fallbacks_used if stored else (),
        )
        self._run_states[run_id] = state.model_copy(deep=True)
        self._notify_callback(state, on_event)

    @staticmethod
    def _snapshot(
        run_id: str,
        tracker: RunTracker,
        findings_count: int,
        *,
        status: PipelineRunStatus | None = None,
        partial: bool = False,
        revision: int = 0,
        updated_at: datetime | None = None,
        analysis_status: str | None = None,
        is_degraded: bool | None = None,
        degradation_reasons: Sequence[str] = (),
        failed_blocks: Sequence[str] = (),
        fallbacks_used: Sequence[str] = (),
    ) -> PipelineProgressSnapshot:
        stages = [
            PipelineStageSnapshot(**stage.to_dict()) for stage in tracker.stages()
        ]
        if status is None and tracker.finished_at is None:
            status = PipelineRunStatus.RUNNING
        if status is None:
            degraded = partial or any(
                stage.state in {StageState.DEGRADED, StageState.FAILED}
                for stage in tracker.stages()
            )
            status = (
                PipelineRunStatus.PARTIAL
                if degraded
                else PipelineRunStatus.COMPLETE
            )
        active = next(
            (
                stage.name
                for stage in tracker.stages()
                if stage.state is StageState.RUNNING
            ),
            None,
        )
        observed_degradation = [
            f"stage:{stage.name}:{stage.state}:{stage.note}".rstrip(":")
            for stage in stages
            if stage.state in {StageState.DEGRADED.value, StageState.FAILED.value}
        ]
        if is_degraded is None:
            is_degraded = bool(observed_degradation)
        if not degradation_reasons and observed_degradation:
            degradation_reasons = observed_degradation
        return PipelineProgressSnapshot(
            run_id=run_id,
            status=status,
            revision=revision,
            updated_at=updated_at or datetime.now(timezone.utc),
            started_at=tracker.started_at,
            finished_at=tracker.finished_at,
            current_stage=active,
            stages=stages,
            findings_count=findings_count,
            analysis_status=analysis_status,
            is_degraded=is_degraded,
            degradation_reasons=list(degradation_reasons),
            failed_blocks=list(failed_blocks),
            fallbacks_used=list(fallbacks_used),
        )


def _findings_stage(
    name: StageName, findings: list
) -> tuple[StageName, StageState, str]:
    """Чем кончилась ветка проверок — по её собственным находкам.

    Ветка, которой в прогоне не было, отличается от ветки без находок тем, что
    её источников нет вовсе. Ветка, которая была, но отмечена сбоем, называется
    `DEGRADED` — иначе провал модели выглядел бы как «претензий нет».

    :param name: имя этапа
    :param findings: находки ветки в её собственном типе
    :return: имя, состояние и причина для журнала
    """
    if not findings:
        return name, StageState.SKIPPED, "ветки в прогоне не было"
    broken = [
        found
        for found in findings
        if str(getattr(found, "status", "")) != CXNodeStatus.SUCCESS.value
    ]
    deviations = sum(len(found.deviations) for found in findings)
    if broken:
        return (
            name,
            StageState.DEGRADED,
            f"источников {len(findings)}, без ответа {len(broken)}",
        )
    return (
        name,
        StageState.DONE if deviations else StageState.EMPTY,
        f"источников {len(findings)}, отклонений {deviations}",
    )

