"""Construction агента: провайдер, справочник и собранный граф — один раз на процесс.

Провайдер держит авторизацию контура и политику повторов, справочник читается с
диска, а граф собирается из настроек — всё это дороже одного прогона и дешевле
процесса. Собирать граф на задание значило бы перечитывать промпты на каждый
кадр; заводить два провайдера на две параллельные ветки — значило бы иметь две
авторизации к контуру в одном процессе.

Ветка блоков проверяется здесь, а не вспоминается по памяти запускающего: узлы
ветки отличаются от прочих только меткой, и граф без них не упал бы — блоки
просто отсутствовали бы в состоянии, а находки деградировали бы в `error` на
верно, по мнению сборки, графе. Проверяется поэтому состав узлов, а не
настройка: справочник читается из настроек той ветки, которая стоит в графе.

`recursion_limit` сюда не входит: он зависит от числа кадров задания, то есть от
конфигурации конкретного прогона, а не процесса (фаза 3).
"""

import logging
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from src.integration.core.config import Settings
from src.integration.settings.api.gigachat import GigaChatSettings
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.agent.criteria_model_frames.nodes import (
    block_node_name as frames_node_name,
)
from src.pipeline.agent.graph import build_graph
from src.pipeline.schemas.nodes import CXCriteriaModel
from src.pipeline.settings.api.graph import (
    CheckCxCriteriaSettings,
    GraphSettings,
)

logger = logging.getLogger(__name__)


class BranchInvariantError(RuntimeError):
    """В собранном графе нет узлов проверки блоков по кадрам."""


@dataclass(frozen=True)
class GraphRuntime:
    """Собранный агент: тем процессом он вооружён на все задания.

    Attributes:
        settings: настройки этого процесса.
        graph: скомпилированный граф прогона.
        provider: один провайдер контура на процесс.
        criteria: справочник блоков той ветки, что реально стоит в графе.
    """

    settings: Settings
    graph: CompiledStateGraph
    provider: GigaChatProvider
    criteria: CXCriteriaModel


def build_provider(settings: GigaChatSettings) -> GigaChatProvider:
    """Провайдер контура по настройкам ГигаЧата.

    `retry_limit` на единицу больше числа повторов: первая попытка повтором не
    считается, а контур считает их вместе — расхождение на одну попытку видно
    только по платным обращениям.

    :param settings: настройки контура этого процесса
    :return: провайдер, готовый к первому запросу
    """
    return GigaChatProvider(
        base_url=settings.base_url,
        authorization=settings.auth,
        retry_limit=settings.max_retries + 1,
        retry_increment=settings.retry_delay,
        request_timeout=settings.timeout,
    )


def criteria_of(
    graph: CompiledStateGraph, settings: GraphSettings
) -> CheckCxCriteriaSettings:
    """Настройки проверки блоков, которая действительно стоит в собранном графе.

    Ветка узнаётся по именам её узлов, а не по памяти сборки: имена даёт сама
    ветка (`block_node_name`), поэтому справочник прогона читается из тех же
    настроек, по которым граф спрашивает блоки, — а не из тех, что просто
    описаны в конфиге.

    :param graph: собранный граф прогона
    :param settings: настройки графа
    :raises BranchInvariantError: узлов проверки блоков в графе нет
    :return: настройки ветки блоков
    """
    nodes = {node.id for node in graph.get_graph().nodes.values()}
    asked = [
        frames_node_name(block.key)
        for block in settings.check_cx_criteria.blocks()
    ]
    if not any(name in nodes for name in asked):
        raise BranchInvariantError(
            "в графе нет ни одного узла проверки блоков по кадрам — ожидали "
            f"имена вида `{asked[0]}`"
        )
    logger.info("criteria nodes %d of %d", len(asked), len(nodes))
    return settings.check_cx_criteria


def build_graph_runtime(
    settings: Settings | None = None,
    criteria: CXCriteriaModel | None = None,
) -> GraphRuntime:
    """Собирает агентскую часть процесса: граф, справочник, провайдер.

    Ни одного обращения к модели здесь нет: собирают каркас и проверяют его на
    машине разработчика без доступа к контуру.

    Справочник приходит готовым: он живёт только в базе, а чтение её —
    асинхронное, поэтому приложение читает его раньше и передаёт сюда. Без
    справочника сборка отказывает: граф спрашивает блоки уже на этом шаге, и
    подставлять вместо базы файл значило бы вернуть ту редакцию каталога,
    которой на стенде нет.

    :param settings: настройки процесса; по умолчанию — как у процесса целиком
    :param criteria: справочник из базы
    :raises BranchInvariantError: собран не тот граф
    :raises ValueError: справочник не передан
    :return: собранный агент
    """
    settings = settings or Settings()
    if criteria is None:
        raise ValueError(
            "справочник не передан: он читается из базы, а не из настроек"
        )
    settings.graph.bind_criteria_model(criteria)
    graph = build_graph(settings.graph)
    return GraphRuntime(
        settings=settings,
        graph=graph,
        provider=build_provider(settings.gigachat),
        criteria=criteria,
    )
