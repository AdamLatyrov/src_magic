"""Живые отметки этапов: узел графа отмечается в журнале до и после работы.

Журнал этапов (`pipeline.status`) пишет итоги после прогона: чем кончился этап и
сколько длился. Этого хватает отчёту, но не тому, кто смотрит на прогон снаружи
— работнику очереди, ноутбуку, человеку у экрана. Пока граф идёт, они видят
только «задание в работе».

Здесь узел оборачивается в отметку: перед вызовом этап начинается, после —
получает состояние и время. Обёртка ничего не решает про сам узел: она не
перехватывает ошибки и не меняет ответ, только отмечает факт начала и конца.
Сбой узла отмечается `FAILED` и уходит дальше как был — иначе политика повторов
и обработчик графа его не увидят.

Имя этапа берётся из имени узла: `scan_journey_frame` и `summarize_journey`
совпадают с `StageName`, а узлы проверок начинаются с имени ветки и
приводятся к общему этапу. Чужие узлы (обработчики ошибок, служебные)
пропускаются: в журнале им делать нечего.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from inspect import iscoroutinefunction

from langchain_core.runnables import RunnableConfig

from src.pipeline.status import (
    CURRENT_PIPELINE_NODE,
    CURRENT_PIPELINE_STAGE,
    RunTracker,
    StageName,
    StageState,
)

logger = logging.getLogger(__name__)


def stage_of(node_name: str) -> StageName | None:
    """Этап журнала по имени узла графа; `None` — узел в журнал не пишется.

    Узлы ветки блоков называются `<метка ветки>_<ключ блока>`, а поэкранной —
    `summarize_screen`. Обе ветки в журнале — один этап: снаружи важно «идёт
    проверка блоков», а не какой именно блок сейчас спрашивают.

    :param node_name: имя узла в графе
    :return: этап журнала либо `None`
    """
    for name in StageName:
        if node_name == name.value:
            return name
    if node_name in ("check_cx_criteria_block",) or node_name.startswith(StageName.CRITERIA_MODEL_FRAMES.value):
        return StageName.CRITERIA_MODEL_FRAMES
    if node_name in (
        StageName.SCREEN.value,
        "summarize_screen", "check_screen",
    ):
        return StageName.SCREEN
    if node_name.startswith(StageName.JOURNEY_FRICTIONS.value) or node_name in (
        "map_journey_frictions", "check_journey_friction",
    ):
        return StageName.JOURNEY_FRICTIONS
    if node_name == "check_autofill":
        return StageName.AUTOFILL
    return None


def tracked(fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Обернуть узел графа живой отметкой этапа.

    Узел получает зависимости из `config["configurable"]`, и журнал приходит
    туда же под именем `tracker`. Журнала нет — узел работает как обычно: прогон
    не обязан вести отметки, чтобы работать.

    :param fn: функция узла
    :return: та же функция с отметками в журнале
    """
    if not iscoroutinefunction(fn):
        raise NotImplementedError

    async def wrapper(state, config: RunnableConfig) -> dict:
        tracker: RunTracker | None = (config.get("configurable") or {}).get(
            "tracker"
        )
        stage = stage_of(fn.__name__)
        stage_token = CURRENT_PIPELINE_STAGE.set(stage.value if stage else None)
        node_token = CURRENT_PIPELINE_NODE.set(fn.__name__)
        try:
            if tracker is None or stage is None:
                return await fn(state, config)

            tracker.begin(stage, note=f"узел {fn.__name__}")
            tracker.step(stage, note=f"узел {fn.__name__}")
            answer = await fn(state, config)
            if stage is StageName.SCAN_JOURNEY_FRAME:
                tracker.count(stage, frames_processed=(
                    len(state.frame_readouts) + len(answer.get("frame_readouts", []))
                ))
            tracker.finish(stage, StageState.DONE, note=f"узел {fn.__name__}")
            return answer
        except Exception:
            if tracker is not None and stage is not None:
                tracker.finish(
                    stage,
                    StageState.FAILED,
                    note=f"узел {fn.__name__} сорвался",
                )
            raise
        finally:
            CURRENT_PIPELINE_NODE.reset(node_token)
            CURRENT_PIPELINE_STAGE.reset(stage_token)

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__doc__ = fn.__doc__
    return wrapper
