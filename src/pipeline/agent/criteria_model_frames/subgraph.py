"""Сборка сабграфа блоков критериальной модели: цепочка узлов по справочнику.

Ветка кадров: каждый блок спрашивается по изображениям пути — единственная ветка
проверки блоков, которая стоит в графе прогона.

Узлы кладутся в тот же `StateGraph`, что и остальные узлы прогона (см.
`pipeline.agent.journey_exploration.subgraph`): состояние у них общее. Имена узлам
даёт ветка — по ключам блоков справочника (`block_node_name` из
`pipeline.agent.criteria_model_frames.nodes`), — и возвращает их вызывающей стороне,
чтобы та повела прогон дальше: блоков сколько угодно, и граф о их составе не
знает.

Здесь только сборка: чем вооружён узел и чем он деградирует записано в том же
`add_node`, которое ставит узел, — помощник на один блок ветке не нужен, и в
файле остаются только функции сборки.

Цепочка обязана дожить до конца и после сбоя блока: у каждого узла есть
обработчик исчерпанных попыток, и его выход ведёт туда же, куда выход успешного
узла, — иначе отказ одного блока обрывал бы оценку остальных.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.pipeline.aef.patterns.tracked import tracked
from src.pipeline.aef.patterns.retry import (
    degradation_handler,
    error_handler_node,
    llm_retry_policy,
)
from src.pipeline.agent.criteria_model_frames.nodes import (
    block_node_name,
    check_cx_criteria_block,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import CheckCxCriteriaSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import CXNodeStatus
from src.pipeline.utils.block_findings import findings_update

__all__ = [
    "build_criteria_frames_graph",
    "build_criteria_model_frames_subgraph",
]


def build_criteria_model_frames_subgraph(
    graph: StateGraph, settings: CheckCxCriteriaSettings
) -> tuple[str, str]:
    """Цепочка узлов блоков в общий граф; возвращает имя первого и последнего.

    Имя узла даёт ключ блока, действие одно на все блоки: `check_cx_criteria_block`
    знает, какой блок спрашивать, по имени собственного узла. Узел вооружён
    политикой повторов (`pipeline.aef.patterns.retry`) — он поднимает только то, что
    повтором и лечится, а когда попытки кончаются, вместо падения прогона
    срабатывает обработчик: он пишет блоку статус `error`, и цепочка идёт дальше.

    Собранная здесь цепь обязана дожить до конца и после сбоя блока: у каждого
    узла есть обработчик исчерпанных попыток, и его выход ведёт туда же, куда
    и успешный блок. За последним блока обработку дописывает сборщик графа — им
    же и `error_handler_node(...)` нужен для перехода к сведению.

    :param graph: общий граф прогона — узлы добавляются в него
    :param settings: настройки ветки: состав блоков и их параметры
    :return: имя первого узла цепи и имя последнего
    """
    blocks = settings.blocks()
    if not blocks:
        raise ValueError(f"В {settings.criteria_model_path} нет блоков")

    previous = None
    for block in blocks:
        name = block_node_name(block.key)
        graph.add_node(
            name,
            tracked(check_cx_criteria_block),
            retry_policy=llm_retry_policy(),
            error_handler=degradation_handler(
                findings_update(block.key, CXNodeStatus.ERROR),
                f"{LogLabel.CHECK_CX_CRITERIA_FRAMES.value}: блок {block.key}",
            ),
        )
        if previous is not None:
            graph.add_edge(previous, name)
            graph.add_edge(error_handler_node(previous), name)
        previous = name

    return block_node_name(blocks[0].key), previous


def build_criteria_frames_graph(
    settings: CheckCxCriteriaSettings,
) -> CompiledStateGraph:
    """Граф только ветки кадров: независимая проверка цепочки блоков по кадрам.

    Собирается без скана и сведения — по состоянию, где описаний нет: цепочка
    блоков здесь проверяется сама по себе, ровно как её собирает прогон.

    :param settings: настройки ветки блоков
    :return: скомпилированный граф цепочки
    """
    graph = StateGraph(AgentStateSchema)
    first, last = build_criteria_model_frames_subgraph(graph, settings)
    graph.add_edge(START, first)
    graph.add_edge(last, END)
    graph.add_edge(error_handler_node(last), END)

    return graph.compile()
