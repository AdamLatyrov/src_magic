"""Граф прогона: сначала путь, потом проверяющие ветки, одно сведение.

Здесь единственное место, где решают порядок: что за чем идёт и чем
заканчивается отказ каждой из сторон. Ветки собираются своими сборщиками и друг
о друге не знают ни слова — каждой хватило бы собственного графа.

Исследование пути (`pipeline.agent.journey_exploration`) стоит до блоков: скан
описывает кадры по одному, а сведение называет по описаниям продукт и сценарий
— их читают блоки оценки, переставая угадывать домен каждый сам. Тип пути в
состоянии остаётся (его печатают журнал и отчёт), но в вопрос веткам не уходит:
ссылка на то, чем путь является, снимала с проверки целые блоки. Отрицательный
вердикт ведёт сразу к снятию кадров (`release_frames`), и ни один блок не
спрашивается; вердикта нет — прогон идёт вслепую, потому что «контур не ответил» и
«пути нет» значат разное: атрибуции в состоянии просто нет, когда скан не утверждал
ничего. Снятие кадров — единственный выход графа: какой бы стороной он ни кончился,
загруженные в контур ссылки прогон обязан оставить за собой.

Проверяющие ветки идут рядом, продолжая одну и ту же развилку после сведения
описаний: блоки справочника по кадрам (`pipeline.agent.criteria_model_frames`), шесть
проверок проблем пути (`pipeline.agent.journey_frictions`), фокусный проход
автозаполнения (`pipeline.agent.autofill`) и поэкранная ветка
(`pipeline.agent.screen`) — она одна даёт описание каждого кадра и находки с
координатами пина. Ветка сама двухузловая: проход проходит путь сканирующим
окном и спрашивает по одному кадру за обращение, подавая справочник только как
примеры отклонений, а разметка (`summarize_screen`) берёт эти сырые находки и
ищет им код в том же справочнике целиком. Разведены они потому, что у прохода
вопрос с картинками и без единого кода, а у разметки — весь каталог текстом и
ни одной картинки. Так же устроена и ветка проблем пути: её цепочка замкнута
разметкой (`map_journey_frictions`), и проверки тоже ищут, не зная ни блоков, ни
кодов. Параллельны именно ветки,
а не проверки внутри них, и сходятся они в одно сведение находок. Ответы лежат в
разных ключах состояния, поэтому сведение добавлено с `defer`: оно откладывается
до момента, когда некому больше работать, и спрашивается ровно один раз — даже
когда вторая ветка отдалась ошибкой и не пришла вовсе. Обычный переход из двух
источников так не умеет: он ждёт все объявленные и, не дождавшись, не срабатывает
никогда.

Ветка блоков одна и она кадровая (`pipeline.agent.criteria_model_frames`): блок
справочника спрашивается изображениями пути, а батчи режет сам узел. Имена узлов
даёт сама ветка — своя функция в её `nodes.py`, которую читают и сборщик, и
узел, — поэтому цепь, собранная этой веткой, не может достаться узлу другой.
Вопроса по описаниям кадров вместо изображений в графе нет: блоки справочника
дают годный вердикт только тогда, когда видят экран.

Диалоговые узлы вооружены одинаково (`pipeline.aef.patterns.retry`): политика
повторов на временные сбои и обработчик исчерпанных попыток, пишущий деградацию
вместо обрыва прогона. LangGraph из обработчика переходов не рисует, поэтому у
каждого обработчика ребро тянут здесь же и туда же, куда ведёт успешный узел:
невытянутое ребро превращает деградированный узел в тупик, за которым молча
обрывается весь остальной маршрут. Выход обработчика сведения описаний
повторяет маршрут успешного ответа — отсутствие вердикта продолжает прогон
вслепую, а не отказывает в нём.

Узлы адресуются по имени, поэтому имя названо строкой и обязано совпадать с
именем функции-узла (`pipeline.agent.nodes`): под ним узел показывает себя в стриме
прогона и в журнале. Ошибку в имени видно сразу — `compile()` не находит узел,
к которому ведёт переход.

Ответ прогона граф не собирает: по состоянию его формирует DTO
`pipeline.dto.agent.result.AgentResult`.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.pipeline.aef.patterns.retry import (
    degradation_handler,
    error_handler_node,
    llm_retry_policy,
)
from src.pipeline.aef.patterns.tracked import tracked
from src.pipeline.agent.autofill import build_autofill_subgraph
from src.pipeline.agent.criteria_model_frames import (
    build_criteria_model_frames_subgraph,
)
from src.pipeline.agent.journey_exploration import (
    build_journey_exploration_subgraph,
)
from src.pipeline.agent.journey_frictions import (
    build_journey_frictions_subgraph,
)
from src.pipeline.agent.nodes import (
    consolidate_findings,
    release_frames,
    summarize_journey,
)
from src.pipeline.agent.screen import build_screen_subgraph
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import GraphSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.utils.journey import verdict_of


def build_graph(settings: GraphSettings) -> CompiledStateGraph:
    """Граф прогона: исследование пути, проверяющие ветки, сведение находок.

    Скан пути идёт циклом, и каждый кадр занимает отдельный супершаг. У LangGraph
    по умолчанию лимит равен 25, поэтому прямой вызов этого графа должен передать
    больший `recursion_limit`. Публичный фасад `AnalysisPipeline` уже задаёт
    рассчитанный лимит при запуске; при прямом вызове графа его нужно передать
    самостоятельно.

    :param settings: настройки графа — из них берётся состав блоков и проверок
    :return: скомпилированный граф
    """
    graph = StateGraph(AgentStateSchema)

    blocks_first, blocks_last = build_criteria_model_frames_subgraph(
        graph, settings.check_cx_criteria
    )
    frictions_first, frictions_last = build_journey_frictions_subgraph(
        graph, settings
    )
    autofill_first, autofill_last = build_autofill_subgraph(
        graph, settings.autofill
    )
    screen_first, screen_last = build_screen_subgraph(graph, settings)

    build_journey_exploration_subgraph(
        graph, settings.journey_exploration, "summarize_journey"
    )
    graph.add_node(
        "summarize_journey",
        tracked(summarize_journey),
        retry_policy=llm_retry_policy(),
        error_handler=degradation_handler(
            {"journey_details": None},
            f"{LogLabel.SUMMARIZE_JOURNEY.value}: сведение описаний кадров",
        ),
    )

    graph.add_node(
        "consolidate_findings",
        tracked(consolidate_findings),
        defer=True,
        retry_policy=llm_retry_policy(),
        error_handler=degradation_handler(
            {"cx_consolidation": None},
            f"{LogLabel.CONSOLIDATE_FINDINGS.value}: сведение находок",
        ),
    )

    graph.add_node("release_frames", tracked(release_frames))

    graph.add_edge(START, "scan_journey_frame")

    branches = [blocks_first, screen_first, frictions_first, autofill_first]
    for source in (
        "summarize_journey",
        error_handler_node("summarize_journey"),
    ):
        graph.add_conditional_edges(
            source,
            lambda state: (
                "release_frames"
                if verdict_of(state.journey_details) is False
                else branches
            ),
            {branch: branch for branch in branches}
            | {"release_frames": "release_frames"},
        )

    for last in (blocks_last, screen_last, frictions_last, autofill_last):
        graph.add_edge(last, "consolidate_findings")
        graph.add_edge(error_handler_node(last), "consolidate_findings")
    for source in (
        "consolidate_findings",
        error_handler_node("consolidate_findings"),
    ):
        graph.add_edge(source, "release_frames")
    graph.add_edge("release_frames", END)

    return graph.compile()
