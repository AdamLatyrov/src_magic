"""Статусы этапов прогона: чем кончился каждый шаг и чем — прогон целиком.

Прогон длинный и платный: скан кадров, сведение описаний, ветки проверок,
сведение находок, сверка с прошлым. Пока он идёт, снаружи видно только «задание
в работе», а после — список находок либо ошибку. Из этого не понять, где именно
потратилось время и что не состоялось: пустой список одинаково значит «претензий
нет» и «модель не ответила» (разбор — `docs/project-spec/09-maintenance/improvements.md`,
I-003).

Здесь — запись о том, что произошло, без знаний о том, куда её положить. Сбор
журнала ничего не пишет на диск и не говорит с базой: он держит записи в памяти,
а отдаёт их вызывающей стороне вместе с ответом. Кто их сохранит — работник
очереди, ноутбук или лаборатория — решает вызывающая сторона; пайплайн только
называет факты.

Записи не заменяют журнал приложения: журнал читает человек при разборе, а
статусы — код, который показывает ход задания. Одно и то же событие попадает и
туда, и туда.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Optional


CURRENT_PIPELINE_STAGE: ContextVar[str | None] = ContextVar(
    "current_pipeline_stage", default=None
)
CURRENT_PIPELINE_NODE: ContextVar[str | None] = ContextVar(
    "current_pipeline_node", default=None
)


class StageState(StrEnum):
    """Состояние одного этапа.

    Разделение важнее самих имён: `DONE` — работа сделана; `EMPTY` — сделана, но
    не нашла ничего; `DEGRADED` — модель не ответила, результат собран обходным
    путём; `FAILED` — этап сорвался и без него результат неполон; `SKIPPED` — этап
    не запускался по решению прогона (например, путь не признан путём).
    """

    RUNNING = "running"
    DONE = "done"
    EMPTY = "empty"
    DEGRADED = "degraded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageName(StrEnum):
    """Этапы прогона в том порядке, в каком их ждёт снаружи.

    Имена совпадают с метками журнала (`LogLabel`) там, где этап им соответствует:
    одно имя на две системы проще сверять при разборе.
    """

    SCAN_JOURNEY_FRAME = "scan_journey_frame"
    SUMMARIZE_JOURNEY = "summarize_journey"
    CRITERIA_MODEL_FRAMES = "criteria_model_frames"
    SCREEN = "screen"
    JOURNEY_FRICTIONS = "journey_frictions"
    AUTOFILL = "autofill"
    CONSOLIDATE_FINDINGS = "consolidate_findings"
    DEDUP_FINDINGS = "dedup_findings"
    RELEASE_FRAMES = "release_frames"


_ORDER = {name: index for index, name in enumerate(StageName)}


@dataclass
class StageStatus:
    """Один этап: в каком состоянии, когда начался и кончился, чем объясняется.

    `note` — короткая причина, а не текст ошибки: её читают в интерфейсе и в
    отчёте. Полный разбор сбоя остаётся в журнале, где лежит исключение.
    """

    name: str
    state: StageState = StageState.RUNNING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    note: str = ""
    counters: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> Optional[float]:
        """Сколько длился этап; `None`, пока он не кончился."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Запись в том виде, в каком её отдают наружу: только JSON-значения."""
        return {
            "name": self.name,
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
            "duration_s": self.duration_s,
            "note": self.note,
            "counters": self.counters,
        }


class RunTracker:
    """Журнал этапов одного прогона — в памяти, без записи наружу.

    Создаётся на прогон, живёт до конца ответа и отдаётся вызывающей стороне
    целиком. Потокобезопасности здесь нет намеренно: граф исполняет узлы в одном
    цикле событий, и запись этапа идёт из того же цикла.

    Этапы, которых прогон не запускал, в журнале всё равно появляются — со
    состоянием `SKIPPED` при `finish`. Иначе «этап не попал в журнал» пришлось бы
    читать как «этап не было», а это разные вещи: прогон мог пройти мимо ветки,
    а мог потерять запись.
    """

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._stages: dict[str, StageStatus] = {}
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._on_change = on_change

    def _notify(self) -> None:
        """Уведомить фасад после синхронного изменения журнала."""
        if self._on_change is not None:
            self._on_change()

    # ── Запись ─────────────────────────────────────────────────────────

    def begin_run(self) -> None:
        """Отметить начало прогона целиком."""
        self.started_at = _now()
        self._notify()

    def begin(self, name: str, *, note: str = "") -> StageStatus:
        """Отметить начало этапа; повторный вызов начинает этап заново.

        Повтор допустим: скан кадра идёт циклом, и этап «скан» за прогон
        начинается столько раз, сколько кадров. Начало перебивает время, а
        состояние возвращается в `RUNNING` — последнее слово за последним ходом.

        :param name: имя этапа из `StageName`
        :param note: что именно делается
        :return: запись этапа — её можно дополнить счётчиками
        """
        stage = self._stages.get(name)
        if stage is None:
            stage = StageStatus(name=name)
            self._stages[name] = stage
        stage.state = StageState.RUNNING
        stage.started_at = _now()
        stage.finished_at = None
        stage.note = note
        self._notify()
        return stage

    def finish(
        self,
        name: str,
        state: StageState,
        *,
        note: str = "",
        counters: Optional[dict[str, Any]] = None,
    ) -> StageStatus:
        """Закрыть этап: состояние, причина, счётчики.

        :param name: имя этапа из `StageName`
        :param state: чем этап кончился
        :param note: короткая причина — её читают снаружи
        :param counters: числа этапа: сколько кадров, блоков, обращений
        :return: запись этапа
        """
        stage = self.begin(name) if name not in self._stages else self._stages[name]
        stage.state = state
        stage.finished_at = _now()
        stage.note = note
        if counters:
            stage.counters.update(counters)
        self._notify()
        return stage

    def count(self, name: str, **counters: Any) -> StageStatus:
        """Дописать счётчики этапа, не меняя его состояния.

        :param name: имя этапа из `StageName`
        :param counters: что прибавилось
        :return: запись этапа
        """
        stage = self._stages.get(name)
        if stage is None:
            stage = self.begin(name)
        stage.counters.update(counters)
        self._notify()
        return stage

    def step(self, name: str, *, note: str = "", by: int = 1) -> StageStatus:
        """Один шаг этапа: прибавить к счётчику `steps` и обновить пояснение.

        Так скан отмечает каждый разобранный кадр, а ветка блоков — каждое
        обращение к контуру. Состояние остаётся `RUNNING`, а время начала
        берётся с первого шага: снаружи видно, сколько этап уже идёт и на чём
        он стоит сейчас.

        :param name: имя этапа из `StageName`
        :param note: что делается прямо сейчас
        :param by: насколько прибавить счётчик шагов
        :return: запись этапа
        """
        stage = self._stages.get(name)
        if stage is None or stage.state is not StageState.RUNNING:
            stage = self.begin(name, note=note)
        stage.counters["steps"] = stage.counters.get("steps", 0) + by
        if note:
            stage.note = note
        self._notify()
        return stage

    # ── Чтение ─────────────────────────────────────────────────────────

    def stage(self, name: str) -> Optional[StageStatus]:
        """Запись этапа; `None` — прогон этот этап ещё не трогал."""
        return self._stages.get(name)

    def state_of(self, name: str) -> Optional[StageState]:
        """Состояние этапа; `None` — этапа в журнале нет."""
        stage = self._stages.get(name)
        return stage.state if stage is not None else None

    def stages(self) -> list[StageStatus]:
        """Все этапы в порядке прогона — так их читает интерфейс."""
        return sorted(
            self._stages.values(),
            key=lambda stage: _ORDER.get(StageName(stage.name), len(_ORDER)),
        )

    def finish_run(self, *, failed: bool = False) -> None:
        """Закрыть прогон; при аварии активные этапы становятся `FAILED`."""
        self.finished_at = _now()
        for stage in self._stages.values():
            if stage.state is StageState.RUNNING:
                stage.state = StageState.FAILED if failed else StageState.SKIPPED
                stage.finished_at = self.finished_at
                stage.note = ("прогон прервался" if failed else "этап не завершился")
        self._notify()

    def to_dict(self) -> dict[str, Any]:
        """Журнал в том виде, в каком его отдают наружу."""
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
            "duration_s": (
                (self.finished_at - self.started_at).total_seconds()
                if self.started_at and self.finished_at
                else None
            ),
            "stages": [stage.to_dict() for stage in self.stages()],
        }


def _now() -> datetime:
    """Текущее время в UTC — снаружи его читают по-разному, перевод — их дело."""
    return datetime.now(timezone.utc)
