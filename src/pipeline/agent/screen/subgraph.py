"""Сборка сабграфа поэкранной ветки: проход окном и разметка находок.

Ветка — цепочка из двух узлов (`pipeline.agent.criteria_model_frames.subgraph`,
`pipeline.agent.journey_frictions.subgraph`): проход разбирает кадры по одному окну и
отвечает описаниями и сырыми находками, разметка кладёт эти находки в справочник и
возвращает их с кодом, блоком и зоной. Разведены они именно поэтому: у прохода во
вопросе картинки и ни одного кода, у разметки — весь каталог текстом и ни одной
картинки.

Состав ветки граф не знает: имена даёт сборщик, а прогон собирает все проверяющие
ветки одним циклом по первой и последней вершине.

Узлы кладутся в тот же `StateGraph`, что и остальные узлы прогона: состояние у них
общее, а пишет каждая ветка в собственный ключ (`screen_findings`), поэтому гонка
между ветками невозможна. Вооружены узлы той же политикой повторов, что остальные
диалоговые узлы, а когда попытки кончаются, прогон продолжает соседняя ветка:
обработчик прохода пишет ему статус `error`, и он попадает в причины деградации
отчёта вместо того, чтобы оборвать маршрут (`pipeline.aef.patterns.retry`).

Обработчик разметки findings не трогает. Сырые находки уже лежат в состоянии, и
заменить их пустым контейнером значило бы потерять всё, что проход увидел на
кадрах, из-за этапа, который только подписывает находки кодами.

Здесь только сборка: чем вооружён узел и чем он деградирует записано в том же
`add_node`, которое ставит узел.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.pipeline.aef.patterns.tracked import tracked
from src.pipeline.aef.patterns.retry import (
    degradation_handler,
    error_handler_node,
    llm_retry_policy,
)
from src.pipeline.agent.screen.nodes import check_screen, summarize_screen
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import GraphSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import CXNodeStatus
from src.pipeline.utils.screen_findings import screen_update

__all__ = [
    "build_screen_graph",
    "build_screen_subgraph",
]


def build_screen_subgraph(
    graph: StateGraph, settings: GraphSettings
) -> tuple[str, str]:
    """Поэкранная ветка в общий граф; возвращает имя первого и последнего узла.

    Имена узлов дают метки журнала: по ним находка подписывается в перечне сырья
    сведения. Первый узел — проход, последний — разметка: ветка из двух узлов
    встаёт в тот же цикл сборки, что и цепочки блоков и проверок пути.

    :param graph: общий граф прогона — узлы добавляются в него
    :param settings: настройки графа — из них берётся модель каждого узла
    :return: имя первого узла цепи и имя последнего
    """
    screen = LogLabel.CHECK_SCREEN.value
    graph.add_node(
        screen,
        tracked(check_screen),
        retry_policy=llm_retry_policy(),
        error_handler=degradation_handler(
            screen_update(status=CXNodeStatus.ERROR),
            f"{LogLabel.CHECK_SCREEN.value}: поэкранный проход",
        ),
    )

    mapping = LogLabel.SUMMARIZE_SCREEN.value
    graph.add_node(
        mapping,
        tracked(summarize_screen),
        retry_policy=llm_retry_policy(),
        error_handler=degradation_handler(
            {},
            f"{LogLabel.SUMMARIZE_SCREEN.value}: разметка поэкранного прохода",
        ),
    )
    graph.add_edge(screen, mapping)
    graph.add_edge(error_handler_node(screen), mapping)
    return screen, mapping


def build_screen_graph(
    settings: GraphSettings,
) -> CompiledStateGraph:
    """Граф только поэкранной ветки: независимая проверка прохода и разметки.

    Собирается без скана, распознавания и сведения — по состоянию, где кадров и
    атрибуции может не быть: ветку проверяют саму по себе, ровно как её собирает
    прогон, и её можно тестировать, не поднимая весь маршрут.

    :param settings: настройки графа — из них берётся модель каждого узла
    :return: скомпилированный граф ветки
    """
    graph = StateGraph(AgentStateSchema)
    first, last = build_screen_subgraph(graph, settings)
    graph.add_edge(START, first)
    graph.add_edge(last, END)
    graph.add_edge(error_handler_node(last), END)

    return graph.compile()
