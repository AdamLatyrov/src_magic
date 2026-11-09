"""Цикл обработки заданий из очереди.

Собирает инфраструктурные блоки в один цикл: снимает задание из БД, скачивает кадры
из S3, гонит граф, пишет результат.

Пометка о происхождении (`LEGACY_SUSPECTED`). Шапка раньше говорила, что цикл
добавлен «без правок существующего каркаса» одной строчкой в `lifecycle.py`. Оба
утверждения устарели: файл давно живёт своей жизнью, а подключение реестра
описано в `service/lifecycle.py`. Верить телу кода, а не этой истории.

Что здесь важно знать тому, кто будет править:

1. Цикл берёт ОДНО задание за раз. Параметр `max_concurrent` принимается и
   пишется в журнал, но в цикле не участвует — см. `QueueRunner.__init__`.
2. Остановка не ждёт текущего задания: `stop` снимает флаг и возвращается сразу.
   Срок `EVAL_SHUTDOWN_GRACE_SEC` до этого класса не доходит — `stop_all()` из
   `service_lifespan` не вызывается.
3. Отмена задания снимает только цикл heartbeat; сам прогон графа доигрывает.
4. Отказ модели на любом узле не прерывает прогон, а даёт пустой результат и
   закрытие `done` — см. `_run_graph`.

Разбор по всем четырём пунктам — `docs/project-spec/09-maintenance/
improvements.md` (I-003, I-006, I-008, I-010).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from src.integration.core.config import Settings
from src.integration.durable.jobs import JobLease, QueuePort, SafeJobError
from src.integration.persist.binding import BindingPort, BindingUnavailable
from src.integration.persist.result import (
    ResultWriteError,
    ResultWriter,
    RunContext,
    context_from_result,
)
from src.integration.schemas.durable import validate_result
from src.integration.service.execution_snapshot import (
    SnapshotRejected,
    execution_settings,
    fingerprint,
    validate_snapshot,
)
from src.integration.service.runtime import GraphRuntime
from src.integration.storage.frames import FrameSource
from src.pipeline.aef.exceptions import (
    GigaChatModelNotAvailable,
    MalformedAnswer,
)
from src.pipeline.dto.agent.result import AgentResult
from src.pipeline.schemas.nodes import CXFrame
from src.pipeline.schemas.result import AnalyzeResult
from src.pipeline.schemas.state.agent import AgentStateSchema

logger = logging.getLogger(__name__)


def _recursion_limit(frames: int) -> int:
    """Сколько супершагов LangGraph разрешить прогону на пути из `frames` кадров.

    Скан идёт отдельным супершагом на каждый кадр, а после него встают ветки с
    проверками и сведение — их число задаётся конфигом, а не кадрами. Потолок
    берётся с запасом на кадры: молчаливый обрыв на середине пути дороже, чем
    счётчик, который никто не видит.

    :param frames: сколько кадров у пути
    :return: предел рекурсии для `ainvoke`
    """
    return max(50, frames * 4 + 50)


def _with_snapshot(settings: Any, snapshot: Any) -> Any:
    """Настройки прогона со значениями из снимка исполнения задания.

    Снимок едет в задании и называет значения по именам переменных этапов
    (`LLM_CHECK_CX_CRITERIA`, `LLM_MAX_TOKENS` и т. п.). Раньше их читала
    глобальная переменная контекста, и прогон заглядывал в неё изнутри; теперь
    прогон получает готовый объект настроек, а снимок накладывается здесь, до
    вызова. Разница только в моменте: значения снимка по-прежнему главнее
    значений процесса, и задание, поставленное до перезапуска, считается тем,
    чем его ставили.

    Обход идёт по `alias` полей: второй список имён пришлось бы держать в
    согласии с настройками этапов.

    :param settings: настройки процесса (`GraphSettings`)
    :param snapshot: снимок исполнения из задания; `None` — значений не было
    :return: настройки со значениями снимка либо те же настройки
    """
    if not snapshot:
        return settings
    values = execution_settings(snapshot)
    if not values:
        return settings

    def stage_with_values(stage: Any) -> Any:
        patch = {}
        for name, field in type(stage).model_fields.items():
            alias = field.alias
            if alias and alias in values:
                patch[name] = values[alias]
        return stage.model_copy(update=patch) if patch else stage

    patch = {}
    for name, field in type(settings).model_fields.items():
        if field.alias and field.alias in values:
            patch[name] = values[field.alias]
    updated = settings.model_copy(update=patch) if patch else settings

    for name, field in type(updated).model_fields.items():
        stage = getattr(updated, name, None)
        if stage is None or not hasattr(type(stage), "model_fields"):
            continue
        patched = stage_with_values(stage)
        if patched is not stage:
            updated = updated.model_copy(update={name: patched})
    return updated


class QueueRunner:
    """Единственный Runner: цикл «взял задание -> прогон -> записал результат».

    Цикл последовательный: одно задание на процесс, пока оно не доиграет —
    следующее не берётся. Параллельность даётся репликами, а не этим классом.
    """

    name = "queue"

    def __init__(
        self,
        settings: Settings,
        runtime: GraphRuntime,
        repo: QueuePort,
        writer: ResultWriter,
        frame_source: FrameSource,
        binder: BindingPort | None = None,
        worker_id: str | None = None,
        max_concurrent: int = 4,
    ) -> None:
        """Собирает исполнителя.

        ВНИМАНИЕ по `max_concurrent`. Параметр сохраняется в `_max_concurrent`,
        попадает в журнал при старте и больше НИ НА ЧТО не влияет: `_process_next`
        берёт одно задание и не возвращается, пока то не доиграет. Так было не
        всегда задумано, но так работает сейчас.

        Практическое следствие: планировать пропускную способность по этому числу
        нельзя — промах будет ровно в него раз. Разбор — `docs/project-spec/
        09-maintenance/improvements.md`, пункт I-006.

        :param settings: настройки процесса
        :param runtime: собранный граф с провайдером и справочником
        :param repo: порт очереди — снятие, heartbeat, завершение, отказ
        :param writer: писатель результата
        :param frame_source: источник кадров (S3)
        :param binder: привязка прогона к пути каталога, если результат её не назвал
        :param worker_id: имя работника для журнала и лизы
        :param max_concurrent: НЕ ИСПОЛЬЗУЕТСЯ, см. предупреждение выше
        """
        self._settings = settings
        self._runtime = runtime
        self._repo = repo
        self._writer = writer
        self._frame_source = frame_source
        self._binder = binder
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._max_concurrent = max_concurrent
        self._running = False
        self._lease: JobLease | None = None
        self._refused = False

    # ── Runner protocol ────────────────────────────────────────────────

    def ready(self) -> bool:
        return True  # готов с момента создания

    async def start(self) -> None:
        """Запускает бесконечный цикл обслуживания очереди.

        Строка `max_concurrent=%d` в журнале намеренно оставлена как есть: она
        отражает переданный параметр, а не фактическую параллельность, которой
        нет. Пусть лучше видно расхождение, чем настройка, которая тихо не работает.
        """
        self._running = True
        logger.info(
            "queue runner started: worker_id=%s max_concurrent=%d",
            self._worker_id,
            self._max_concurrent,
        )
        while self._running:
            try:
                await self._process_next()
            except Exception:
                logger.exception("queue loop fatal")
                await asyncio.sleep(10)

    async def stop(self, grace_s: float) -> None:
        """Сигнализирует выход: новое задание не возьмёт, текущее НЕ дождётся.

        ВНИМАНИЕ — `grace_s` только пишется в журнал. Метод не ждёт ни
        завершения `_process_next`, ни окончания `ainvoke`: он снимает флаг и
        возвращается немедленно. Если процесс погасят сразу после вызова,
        прогон оборвётся посреди работы, результат не запишется, а задание
        останется `running` со старой лизой до её истечения.

        Так же ведёт себя весь путь остановки: `stop_all()` из `service/lifespan`
        не вызывается вовсе. Срок `EVAL_SHUTDOWN_GRACE_SEC` (в образе 900 с) из
        README до кода не доезжает.

        Вывод для эксплуатации: выкатывать надо на пустой очереди. Разбор —
        `docs/project-spec/09-maintenance/improvements.md`, пункт I-008;
        `docs/project-spec/01-architecture/containers.md`, раздел «Остановка».

        :param grace_s: срок, который здесь ни на что не влияет
        """
        logger.info("queue runner stopping with grace=%.0fs", grace_s)
        self._running = False

    # ── Main loop ──────────────────────────────────────────────────────

    async def _process_next(self) -> None:
        try:
            # 1. Снять следующее свободное задание
            lease = await self._repo.claim_next(worker_id=self._worker_id)
            if lease is None:
                await asyncio.sleep(2)
                return

            self._lease = lease
            self._refused = False
            logger.info(
                "claim job_id=%s run_id=%s phase=%s",
                lease.job_id, lease.run_id, lease.phase,
            )

            # 2. Пустой путь считать нечем. Раньше такой прогон завершался `done`
            # с нулём находок и выглядел как исправный: отказ обязан называть
            # причину, иначе «кадры не дошли» неотличимо от «интерфейс чист».
            # Проверка стоит до запуска heartbeat: отказ записывается сразу, и
            # аренду незачем держать ради задания, которое уже не выполнится.
            if not self._frame_keys(lease.input_json):
                await self._fail_job(
                    SafeJobError(
                        code="no_frames",
                        message="В задании нет кадров пути — анализировать нечего",
                        retryable=False,
                        phase="input",
                        detail="NoFrames",
                    )
                )
                return

            # 3. Поддерживать аренду на heartbeat
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                # 4. Чем считать задание — снимок из постановки, если он был
                snapshot = self._execution_snapshot(lease.input_json)

                # 5. Скачать кадры из S3 и собрать графовый state
                frames = await self._download_frames(lease.input_json)
                screenshot_keys = self._extract_screenshot_keys(lease.input_json)

                # 6. Прогон графа
                result_dict = await self._run_graph(
                    frames, screenshot_keys, lease.run_id, snapshot
                )
                # 7. Записать результат прогона в runs/оценку/находки/экраны.
                # Пишем до complete(): задание считается done только после того,
                # как результат лёг в базу. Обратный порядок оставлял бы `done`
                # над пустыми таблицами — тот же молчаливый успех, что и на
                # пустом пути.
                await self._persist_result(result_dict, lease)
                if self._refused:
                    return
                validated_json = validate_result(result_dict)
                new_status = await self._repo.complete(
                    self._lease, validated_json
                )
                logger.info(
                    "job completed: job_id=%s status=%s",
                    lease.job_id, new_status,
                )

            except asyncio.CancelledError:
                heartbeat_task.cancel()
                await asyncio.sleep(0)
                raise

            finally:
                heartbeat_task.cancel()
                await asyncio.sleep(0)

        except Exception as exc:
            logger.exception("queue loop error")
            await self._fail_job(SafeJobError.from_exception(exc))

    @staticmethod
    def _execution_snapshot(
        input_json: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Снимок исполнения из задания — проверенный до прогона.

        Задание могло быть поставлено другой сборкой процесса: снимок едет через
        JSON и до захвата никем не проверялся. Непригодный снимок — отказ задания
        (`failed_terminal`), а не прогон на значениях по умолчанию: молча
        посчитать не тем хуже, чем не посчитать вовсе.

        :param input_json: вход задания
        :return: снимок исполнения; `None`, если постановка его не приложила
        :raises SafeJobError: снимок есть, но считать по нему нельзя
        """
        snapshot = (input_json.get("params") or {}).get("analysis_execution")
        if snapshot is None:
            return None
        try:
            validate_snapshot(snapshot)
        except SnapshotRejected as exc:
            logger.error("job refused: снимок исполнения непригоден (%s)", exc)
            raise SafeJobError(
                code="analysis_execution_invalid",
                message=f"Снимок исполнения непригоден: {exc}",
                retryable=False,
                detail=type(exc).__name__,
            ) from exc
        logger.info("execution snapshot applied fingerprint=%s", fingerprint(snapshot))
        return dict(snapshot)

    # ── Frame download ─────────────────────────────────────────────────

    @staticmethod
    def _frame_keys(input_json: dict[str, Any]) -> list[str]:
        """Ключи кадров пути из постановки задания.

        Постановка кладёт манифест материала (`l1_manifest`): кадры перечислены
        артефактами с ординалом и ссылкой в хранилище. Плоского `frame_keys` у неё
        нет, и чтение только его означало пустой путь — граф завершался за секунду
        пустым результатом, а задание при этом считалось выполненным.

        Кадры берутся строго по ординалу: порядок экранов — часть пути, а не
        порядок, в котором пришёл ответ хранилища.

        :param input_json: вход задания
        :return: ключи кадров по возрастанию ординала
        """
        artifacts = ((input_json.get("l1_manifest") or {}).get("artifacts")) or []
        frames = sorted(
            (item for item in artifacts if item.get("role") == "frame"),
            key=lambda item: item.get("ordinal") or 0,
        )
        keys = [item["artifact_ref"] for item in frames if item.get("artifact_ref")]
        if keys:
            return keys
        return list(input_json.get("frame_keys", []))

    async def _download_frames(self, input_json: dict[str, Any]) -> list[CXFrame]:
        keys = self._frame_keys(input_json)
        if not keys:
            return []

        tasks = [self._frame_source.frame(key, step=idx) for idx, key in enumerate(keys)]

        try:
            return await asyncio.gather(*tasks)
        except Exception:
            logger.exception("failed to download frames")
            raise

    @staticmethod
    def _extract_screenshot_keys(input_json: dict[str, Any]) -> list[str]:
        return QueueRunner._frame_keys(input_json)

    # ── Graph execution ────────────────────────────────────────────────

    async def _run_graph(
        self,
        frames: list[CXFrame],
        screenshot_keys: list[str],
        run_id: Any,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Прогон графа на кадрах задания.

        Начальное состояние называет те же типы, что и схема состояния: поля-словари
        (`cx_findings`, `journey_frictions`) начинаются пустыми словарями, а не
        списками. Сведение этих полей в графе — `operator.or_`, и список в качестве
        начального значения роняет прогон на первом же супершаге, до обращения к
        контуру.

        Зависимости узлов уходят через `configurable`, а не замыканием сборки: обёртка
        `@node` берёт их оттуда по именам аргументов функции узла. Запуск графа без
        `configurable` роняет прогон на первом узле с недостающими аргументами.

        `recursion_limit` считается по числу кадров: скан идёт отдельным супершагом на
        каждый кадр, и умолчание LangGraph (25) обрывает путь уже на третьем десятке
        экранов.

        Прогон считает по снимку исполнения из задания, а не по настройкам процесса:
        за время ожидания в очереди процесс мог перезапуститься на другом файле
        настроек, и прогон обязан остаться тем, чем его ставили.

        :param frames: кадры пути в порядке следования
        :param screenshot_keys: ключи снимков материала
        :param run_id: прогон, к которому относится задание
        :param snapshot: снимок исполнения из задания; `None` — настроек не было
        :return: результат прогона для записи в задание
        """
        state = {
            "run_id": str(run_id),
            "frames": frames,
            "screenshot_keys": screenshot_keys,
            "criteria_model": self._runtime.criteria,
            "frame_readouts": [],
            "cx_findings": {},
            "journey_frictions": {},
            "autofill_findings": None,
            "screen_findings": None,
        }
        config = {
            "recursion_limit": _recursion_limit(len(frames)),
            "configurable": {
                "gigachat_provider": self._runtime.provider,
                "settings": _with_snapshot(self._runtime.settings.graph, snapshot),
            },
        }

        try:
            final_state = await self._runtime.graph.ainvoke(state, config)
        except (GigaChatModelNotAvailable, MalformedAnswer) as exc:
            # ВНИМАНИЕ — здесь рождается «тихий ноль». Прогон НЕ прерывается:
            # в `final_state` попадает начальное состояние, `AgentResult`
            # считает по нему пустой результат, а задание закрывается `done`.
            # Клиент получает балл без находок и не может отличить «модель
            # недоступна» от «интерфейс чист»; единственный след остаётся
            # в журнале строкой ниже.
            #
            # Помечать ли деградацию в самом результате — решение владельца
            # (Q-006). Разбор — `docs/project-spec/09-maintenance/`
            # `improvements.md`, пункт I-003; запрет — `I-13` в
            # `02-domain/invariants.md`.
            logger.warning(
                "graph degraded for run %s: %s", run_id, exc
            )
            final_state = state

        result = AgentResult(
            state=AgentStateSchema.model_validate(final_state),
            criteria=self._runtime.criteria,
            analyzed_by=self._runtime.settings.graph.summarize_journey.stage_model,
        ).get_result()
        return result.model_dump(mode="json")

    # ── Result persist ─────────────────────────────────────────────────

    async def _persist_result(
        self, result_dict: dict[str, Any], lease: JobLease
    ) -> None:
        """Пишет результат прогона в `runs`/оценку/находки/экраны.

        Пишем до `complete()`: задание считается `done` только после того, как
        результат лёг в базу. Обратный порядок оставлял бы `done` над пустыми
        таблицами — тот же молчаливый успех, что и на пустом пути.

        Прогон результата сверяется с прогоном задания: граф получает `run_id`
        из состояния, и результат по чужому прогону записал бы данные в контекст,
        которого задание не касается. Такой результат — отказ задания, а не
        исключение: `SafeJobError` — значение, его записывают, а не бросают.

        :param result_dict: результат прогона в JSON-форме
        :param lease: снятое задание — носитель прогона
        """
        result = AnalyzeResult.model_validate(result_dict)
        if lease.run_id is not None and uuid.UUID(str(result.run_id)) != uuid.UUID(
            str(lease.run_id)
        ):
            await self._fail_job(
                SafeJobError(
                    code="run_id_mismatch",
                    message="Результат пришёл по чужому прогону",
                    retryable=False,
                    phase="persist",
                    detail="RunIdMismatch",
                )
            )
            self._refused = True
            return
        try:
            context = await self._run_context(result, lease)
        except (BindingUnavailable, ResultWriteError) as exc:
            await self._fail_job(
                SafeJobError(
                    code="no_binding",
                    message=f"Прогон не к чему привязать: {exc}",
                    retryable=False,
                    phase="persist",
                    detail=type(exc).__name__,
                )
            )
            self._refused = True
            return
        ids = await self._writer.persist(result, context)
        logger.info(
            "result persisted: run_id=%s evaluation_id=%s journey_id=%s app_id=%s",
            ids.run_id, ids.evaluation_id, context.journey_id, context.app_id,
        )

    async def _run_context(self, result: AnalyzeResult, lease: JobLease) -> RunContext:
        """Привязка прогона: сначала сам результат, потом задание.

        Распознавание по кадрам знает имена пути и приложения, но не
        идентификаторы каталога, поэтому штатный носитель привязки — задание:
        у него есть продукт постановки, а у каталога — путь и приложение этого
        продукта. Отсутствие читателя каталога — отказ задания, а не прогон без
        привязки: `runs.journey_id`/`app_id` обязательны, и писать некуда.

        :param result: результат прогона
        :param lease: снятое задание — носитель продукта постановки
        :return: контекст прогона для записи
        :raises BindingUnavailable: каталог не дал однозначной привязки
        :raises ResultWriteError: привязки нет и доставать её не откуда
        """
        context = context_from_result(result)
        if context is not None:
            return context
        if self._binder is None:
            raise ResultWriteError(
                "результат не называет путь и приложение, а читателя каталога "
                "не подставили"
            )
        return await self._binder.resolve(
            job_id=lease.job_id, input_json=lease.input_json
        )

    # ── Heartbeat ──────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        assert self._lease is not None
        while self._running and self._lease is not None:
            try:
                status = await self._repo.heartbeat(
                    self._lease, phase="analyzing", progress=""
                )
                if status.cancel_requested:
                    logger.info("job cancelled by heartbeat")
                    break
            except Exception as exc:  # noqa: BLE001 — сбой heartbeat не должен ронять цикл
                logger.warning("heartbeat failed: %s", exc)
            await asyncio.sleep(30)

    # ── Failure handling ───────────────────────────────────────────────

    async def _fail_job(self, error: SafeJobError) -> None:
        if self._lease is None:
            return
        try:
            status = await self._repo.fail(self._lease, error)
            logger.error(
                "job failed: job_id=%s status=%s error=%s",
                self._lease.job_id, status, error,
            )
        except Exception:
            logger.exception(
                "queue: не удалось записать ошибку (возможно БД недоступна)"
            )
