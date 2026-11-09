"""Сборка сабграфа проблем пути: цепочка шести проверок и их разметка.

Каждая проверка спрашивает контур один раз и отвечает только за свой предмет,
поэтому узлы ставятся цепочкой — тем же способом, что и цепочка блоков
(`pipeline.agent.criteria_model_frames.subgraph`): имена даёт сборщик, а граф о
составе проверок не знает. Замыкает цепочку разметка находок справочником — по
тому же образцу, что и пара `check_screen` → `summarize_screen`: поиск отделён от
адреса, и адрес выдаётся один раз на все найденное.

Ветка запускается рядом с проверкой блоков: её первый узел продолжает тот же
узел сведения описаний кадров, а последний приходит туда же, куда и блоки, — в
сведение находок. Параллельны здесь именно ветки, а не проверки внутри них.

Узлы кладутся в тот же `StateGraph`, что и остальные узлы прогона: состояние у
них общее, а ответы каждая проверка пишет в собственный ключ, поэтому гонка
между ветками невозможна.

Здесь только сборка: чем вооружён узел и чем он деградирует записано в том же
`add_node`, которое ставит узел, — помощник на одну проверку ветке не нужен, и
в файле остаются только функции сборки.

Цепочка обязана дожить до конца и после сбоя проверки: у каждого узла есть
обработчик исчерпанных попыток, и его выход ведёт туда же, куда выход успешного
узла, — иначе отказ одной проверки обрывал бы остальные пять.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.pipeline.aef.patterns.tracked import tracked
from src.pipeline.aef.patterns.retry import (
    degradation_handler,
    error_handler_node,
    llm_retry_policy,
)
from src.pipeline.agent.journey_frictions.nodes import (
    check_journey_friction,
    check_node_name,
    map_journey_frictions,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import GraphSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import CXNodeStatus
from src.pipeline.utils.friction_findings import frictions_update

__all__ = [
    "build_journey_frictions_graph",
    "build_journey_frictions_subgraph",
]


def build_journey_frictions_subgraph(
    graph: StateGraph, settings: GraphSettings
) -> tuple[str, str]:
    """Цепочка проверок пути в общий граф; возвращает имя первого и последнего.

    Порядок узлов — порядок проверок в конфигурации: он читается в стриме прогона
    и в журнале, поэтому перестановка проверок остаётся решением конфига, а не
    правкой сборки. Имя узла даёт ключ проверки, действие одно на все проверки,
    а вооружена она той же политикой повторов, что и остальные диалоговые узлы
    (`pipeline.aef.patterns.retry`): когда попытки кончаются, прогон не падает —
    обработчик пишет проверке статус `error`, и цепочка идёт дальше.

    Последним узлом встаёт разметка: проверки искали без справочника, и адрес —
    код дефекта, его блок и его зона — их находки получают разом, когда весь поиск
    закончен. Разметка идёт именно в конце цепи, а не после каждой проверки,
    потому что справочник один на все шесть ответов и резать его молча нельзя.

    :param graph: общий граф прогона — узлы добавляются в него
    :param settings: настройки графа: состав проверок и настройки разметки
    :return: имя первого узла цепи и имя последнего
    """
    checks = settings.journey_frictions.checks
    if not checks:
        raise ValueError(
            f"в ветке {LogLabel.CHECK_JOURNEY_FRICTIONS.value} нет ни одной "
            "проверки"
        )

    previous = None
    for check in checks:
        name = check_node_name(check.check_key)
        graph.add_node(
            name,
            tracked(check_journey_friction),
            retry_policy=llm_retry_policy(),
            error_handler=degradation_handler(
                frictions_update(check.check_key, CXNodeStatus.ERROR),
                f"{LogLabel.CHECK_JOURNEY_FRICTIONS.value}"
                f": проверка {check.check_key}",
            ),
        )
        if previous is not None:
            graph.add_edge(previous, name)
            graph.add_edge(error_handler_node(previous), name)
        previous = name

    mapping = LogLabel.MAP_JOURNEY_FRICTIONS.value
    graph.add_node(
        mapping,
        tracked(map_journey_frictions),
        retry_policy=llm_retry_policy(),
        error_handler=degradation_handler(
            {},
            f"{mapping}: разметка находок проверок пути",
        ),
    )
    graph.add_edge(previous, mapping)
    graph.add_edge(error_handler_node(previous), mapping)

    return check_node_name(checks[0].check_key), mapping


def build_journey_frictions_graph(
    settings: GraphSettings,
) -> CompiledStateGraph:
    """Граф только ветки проблем пути: независимая проверка цепочки по кадрам.

    Собирается без скана и сведения — по состоянию, где атрибуции может не быть:
    цепочка проверяется сама по себе, ровно как её собирает прогон.

    :param settings: настройки графа — из них берётся ветка и её разметка
    :return: скомпилированный граф цепочки
    """
    graph = StateGraph(AgentStateSchema)
    first, last = build_journey_frictions_subgraph(graph, settings)
    graph.add_edge(START, first)
    graph.add_edge(last, END)
    graph.add_edge(error_handler_node(last), END)

    return graph.compile()
