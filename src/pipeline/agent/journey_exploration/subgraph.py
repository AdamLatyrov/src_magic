"""Сборка цикла исследования пути: скан идёт кадр за кадром.

Узел кладётся в тот же `StateGraph`, что и блоки оценки (см.
`pipeline.agent.criteria_model_frames.subgraph`): состояние у них общее. Имя узла называет
сборщик строкой: по нему к узлу ведут переходы, оно видно в стриме прогона и в
журнале.

Чужие узлы здесь не регистрируются: цикл знает только имя узла, который
продолжит прогон после последнего кадра (`finish`), и получает его снаружи. Так
скану не известен маршрут прогона, и сборщик графа (`pipeline.agent.graph`) остаётся
единственным местом, где решают, что идёт после кадров.

Цикл держит условный переход: узел скана возвращает себя, пока в пути остаются
неописанные кадры и правило досрочной остановки молчит. Обработчик исчерпанных
попыток есть у каждого узла, и его выход ведёт туда же, куда выход успешного
прохода, — иначе деградированный кадр становится тупиком, за которым молча
обрывается остальной маршрут.

Здесь только сборка: чем деградирует сорванный кадр, знает узел ветки
(`pipeline.agent.journey_exploration.nodes: mark_frame_unread`), а не сборка, и в файле
остаётся одна функция — сборщик цикла.
"""

from langgraph.graph import StateGraph

from src.pipeline.aef.patterns.tracked import tracked
from src.pipeline.aef.patterns.retry import (
    error_handler_node,
    llm_retry_policy,
)
from src.pipeline.agent.journey_exploration.nodes import (
    mark_frame_unread,
    scan_journey_frame,
)
from src.pipeline.settings.api.graph import JourneyExplorationSettings
from src.pipeline.utils.journey import stop_reason

__all__ = ["build_journey_exploration_subgraph"]


def build_journey_exploration_subgraph(
    graph: StateGraph, settings: JourneyExplorationSettings, finish: str
) -> None:
    """Цикл описания кадров в общий граф; вход в него — `"scan_journey_frame"`.

    Имя узла названо здесь строкой и оно же ожидается в переходах сборщика
    графа (`pipeline.agent.graph`): узел в графе адресуют по имени, а не по функции.
    Узел вооружён политикой повторов (`pipeline.aef.patterns.retry`), а когда попытки
    кончаются, вместо падения прогона срабатывает обработчик — он оставляет в
    состоянии кадр без описания, и цикл идёт дальше.

    Пока в пути остаются неописанные кадры и молчит правило досрочной остановки,
    узел возвращает себя; когда кадры кончились или правило сработало, переход
    ведёт в `finish` — тот же переход рисуется и выходу обработчика исчерпанных
    попыток, иначе деградированный кадр становится тупиком. Правила считает
    политика приёмки (`pipeline.utils.journey: stop_reason`): скан обрывается, когда
    первых `stop_after` разобранных кадров не похожи на путь (или когда контур не
    ответил ни по одному из них), и кончается, когда описаны все кадры пути.

    :param graph: `StateGraph` прогона — узел добавляется в него
    :param settings: настройки исследования пути: правила остановки скана,
        длина контекста, параметры обращений к контуру
    :param finish: имя узла, который продолжит прогон после последнего кадра
    """
    scan = "scan_journey_frame"
    graph.add_node(
        scan,
        tracked(scan_journey_frame),
        retry_policy=llm_retry_policy(),
        error_handler=mark_frame_unread,
    )

    for source in (scan, error_handler_node(scan)):
        graph.add_conditional_edges(
            source,
            lambda state: (
                finish
                if len(state.frame_readouts) >= len(state.frames)
                or stop_reason(state.frame_readouts, settings.stop_after)
                else scan
            ),
            {scan: scan, finish: finish},
        )
