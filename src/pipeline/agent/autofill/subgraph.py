"""Сборка сабграфа фокусного прохода автозаполнения.

Ветка — один узел: вопрос про предзаполнение задаётся один раз на весь путь
(сколько групп в этом обращении, решает настройка `batch_frames`, а не состав
графа), поэтому у сборки нет ни цепи, ни цикла. Возвращает она то же, что и
соседние сборщики, — имя первого и имя последнего узла: граф собирает все
проверяющие ветки одним циклом и о их внутреннем устройстве не знает.

Узел кладётся в тот же `StateGraph`, что и остальные узлы прогона: состояние у
них общее, а ответ ветка пишет в собственный ключ, поэтому гонка между ветками
невозможна. Вооружён узел той же политикой повторов, что остальные диалоговые
узлы, а когда попытки кончаются, прогон продолжает соседняя ветка: обработчик
пишет проходу статус `error`, и он попадает в причины деградации отчёта вместо
того, чтобы оборвать маршрут (`pipeline.aef.patterns.retry`).

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
from src.pipeline.agent.autofill.nodes import check_autofill
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import AutofillSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import CXNodeStatus
from src.pipeline.utils.autofill_findings import autofill_update

__all__ = [
    "build_autofill_graph",
    "build_autofill_subgraph",
]


def build_autofill_subgraph(
    graph: StateGraph, settings: AutofillSettings
) -> tuple[str, str]:
    """Узел фокусного прохода в общий граф; возвращает его же дважды.

    Имя узла даёт метка журнала ветки: по нему находка подписывается в перечне
    сырья сведения. Первый и последний узлы совпадают — ветка из одного узла
    встаёт в тот же цикл сборки, что и цепочки блоков и проверок пути.

    :param graph: общий граф прогона — узел добавляется в него
    :param settings: настройки ветки: модель, промпты, блок-адресат находок
    :return: имя узла дважды — начало и конец ветки
    """
    name = LogLabel.CHECK_AUTOFILL.value
    graph.add_node(
        name,
        tracked(check_autofill),
        retry_policy=llm_retry_policy(),
        error_handler=degradation_handler(
            autofill_update(status=CXNodeStatus.ERROR),
            f"{LogLabel.CHECK_AUTOFILL.value}: проход автозаполнения",
        ),
    )
    return name, name


def build_autofill_graph(
    settings: AutofillSettings,
) -> CompiledStateGraph:
    """Граф только фокусного прохода: независимая проверка ветки по кадрам.

    Собирается без скана, распознавания и сведения — по состоянию, где кадров и
    атрибуции может не быть: ветку проверяют саму по себе, ровно как её собирает
    прогон, и её можно тестировать, не поднимая весь маршрут.

    :param settings: настройки ветки
    :return: скомпилированный граф ветки
    """
    graph = StateGraph(AgentStateSchema)
    first, last = build_autofill_subgraph(graph, settings)
    graph.add_edge(START, first)
    graph.add_edge(last, END)
    graph.add_edge(error_handler_node(last), END)

    return graph.compile()
