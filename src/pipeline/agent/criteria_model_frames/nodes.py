"""Узел проверки блока: батч кадров пути на один вопрос контуру.

Ветка кадров: вопрос строится по изображениям пути — `state.kept_frames` уходят в
него содержимым с картинками, делясь на батчи по документированный лимит контура
(`CheckCxCriteriaSettings.batches`); отфильтрованные кадры ветка не видит, их файл
уже снят с контура. Ответ разбирается общим разбором (`pipeline.utils.block_findings`).

Обращения идут строго последовательно: блок за блоком цепочкой узлов, а внутри
блока — батч за батчем одним циклом. Параллельности в этой ветке нет намеренно:
одновременные вопросы складывались в очередь контура (429), а сорванная половина
вопросов неотличима от блока без находок. Последовательный ход дороже по времени
и дешевле по срывам, и каждый его шаг подписан номером батча — и в вопросе, и в
журнале, и в записях лапы.

Узел делает то же, что узлы `pipeline.agent.nodes`: собирает вопрос по настройкам,
спрашивает контур и кладёт его ответ в состояние как есть. Разбора ответа здесь
нет: форму находкам даёт `pipeline.utils.block_findings`, а баллы и разметку —
DTO по состоянию после графа.
"""

import logging

from langchain_core.runnables import RunnableConfig

from src.pipeline.aef.exceptions import MalformedAnswer
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.aef.patterns.retry import (
    failure_of,
    is_shape_failure,
    llm_retryable,
    repeat_recoverable,
)
from src.pipeline.aef.patterns.utils import node
from src.pipeline.schemas.nodes import CXBlock, CXBlockFindings
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import (
    CheckCxCriteriaSettings,
    GraphSettings,
)
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import CXItemStatus, CXNodeStatus
from src.pipeline.settings.params.retry import (
    BATCH_ATTEMPTS,
    BATCH_RETRY_PAUSE_S,
)
from src.pipeline.utils.block_findings import (
    findings_of,
    findings_update,
    merge_batches,
)
from src.pipeline.utils.journey import screen_tally, screen_texts

logger = logging.getLogger(__name__)


def block_node_name(block_key: str) -> str:
    """Имя узла этой ветки для блока справочника.

    Имя несёт метку ветки: `criteria_model_frames_cxv3_1` — блок, проверенный по
    кадрам, и прогон обязан различать узлы в стриме и в журнале. Разбирает имя
    тот же модуль, который его даёт, — узел и сборщик смотрят в одну функцию.

    :param block_key: ключ блока из справочника
    :return: имя узла для `add_node`
    """
    return (
        f"{LogLabel.CHECK_CX_CRITERIA_FRAMES.value}"
        f"_{block_key.replace(':', '_')}"
    )


def block_of_node(
    criteria: CheckCxCriteriaSettings, node_name: str
) -> CXBlock:
    """Блок, который проверяет узел с таким именем.

    Узел узнаёт своё имя из конфигурации запуска, и обратный поиск идёт
    перебором справочника: ключ блока разбирать из имени нельзя — он держит
    разделитель, который в имя узла не переносится.

    :param criteria: настройки ветки — по ним справочник блоков
    :param node_name: имя узла в графе
    :return: блок справочника
    :raises ValueError: имени узла нет среди блоков справочника
    """
    for block in criteria.blocks():
        if block_node_name(block.key) == node_name:
            return block
    raise ValueError(
        f"в {criteria.criteria_model_path} нет блока для узла {node_name}"
    )


@node
async def check_cx_criteria_block(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
    config: RunnableConfig,
) -> dict:
    """Проверка одного блока критериальной модели, батчами по несколько кадров.

    Узел один на все блоки: какой блок проверять, он узнаёт по собственному имени в
    графе — имя даёт ветка (`block_node_name`), и здесь же оно разбирается обратно в
    блок, — а настройки приходят из конфигурации вызова. Так состав узлов остаётся
    делом сборки графа, а здесь остаётся одна плоская функция, годная для любого
    состава справочника.

    Кадры уходят батчами (`CheckCxCriteriaSettings.batches`), как в проверках проблем
    пути: контур держит в поле зрения документированные десять изображений и молча
    снимает лишние, поэтому вопрос на одиннадцать кадров — это ответ по десяти, а
    одиннадцатый экран при этом считается просмотренным. Вердикты батчей складывает
    `merge_batches`: проблема, увиденная в одном батче, не перестаёт быть проблемой от
    того, что остальные батчи её не видели.

    Обращения идут одно за другим, а не параллельно, и порядок их подписан номером
    батча в самом вопросе (`[ЧАСТЬ ПУТИ]`): без номера батч из десяти экранов
    неотличим от вопроса про весь путь, и «этого в пути нет» оказывается выводом об
    экранах, до которых вопрос не дошёл.

    Сбой узел делит сам: повтором лечится только форма ответа да временные отказы
    контура — они поднимаются наверх, и политика из `pipeline.aef.patterns.retry`
    перевызывает узел; всё остальное гасится здесь в статус `error`, как и
    раньше. Исчерпанные попытки дописывает обработчик
    `pipeline.agent.criteria_model_frames.subgraph`. Батч, сорванный не по вине транспорта,
    не стоит всего блока: его счётчик ложится в `dropped`, а находки отвеченных
    батчей остаются в прогоне.

    :param state: состояние прогона: материалы пути и его кадры
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся проверка блоков
    :param config: конфигурация запуска графа
    :return: находки этого блока в состояние прогона
    """
    criteria = settings.check_cx_criteria
    node_name = (config.get("metadata") or {}).get("langgraph_node")
    block_settings = criteria.for_block(block_of_node(criteria, node_name))
    block = block_settings.block
    frames = state.kept_frames
    if not frames:
        logger.warning(
            "%s: блок %s не спрашивается — с анализа снят каждый кадр пути",
            LogLabel.CHECK_CX_CRITERIA_FRAMES,
            block.key,
        )
        return findings_update(block.key, CXNodeStatus.EMPTY_RESPONSE)
    try:
        groups = block_settings.batches(frames)
        # Тексты всех кадров пути уходят в каждое обращение: кадров в батче
        # умещается десять, а экранов анкетирования на пути сорок — без следа
        # блок не видит длины пути и отвечает "ok" по первому впечатлению.
        trace = screen_texts(state.frame_readouts)
        tally = screen_tally(state.frame_readouts)
        answers: list[tuple[CXBlockFindings, str]] = []
        broken: list[Exception] = []
        # Следующее обращение начинается после ответа на предыдущее: параллельных
        # вопросов в ветке блоков нет.
        for number, group in enumerate(groups, start=1):
            try:
                answers.append(
                    await _ask_batch(
                        block_settings,
                        gigachat_provider,
                        group,
                        state,
                        len(frames),
                        trace,
                        tally,
                        number,
                        len(groups),
                    )
                )
            except Exception as error:
                broken.append(error)
                if not is_shape_failure(error) and not llm_retryable(error):
                    # Неповторяемый отказ одинаков на каждом батче: спотыкаться о
                    # него оставшейся частью пути незачем.
                    logger.warning(
                        "%s: блок %s остановлен на батче %s из %s (%s)",
                        LogLabel.CHECK_CX_CRITERIA_FRAMES,
                        block.key,
                        number,
                        len(groups),
                        failure_of(error),
                    )
                    break

        if not answers:
            error = broken[0]
            if is_shape_failure(error) or llm_retryable(error):
                logger.warning(
                    "%s: блок %s — ни один из %s батчей не отвечен (%s); "
                    "обращение к контуру повторится",
                    LogLabel.CHECK_CX_CRITERIA_FRAMES,
                    block.key,
                    len(groups),
                    failure_of(error),
                )
                # Форму ответа узел оборачивает сам: повтором она достижима
                # только через `MalformedAnswer`.
                if is_shape_failure(error):
                    raise MalformedAnswer(
                        f"блок {block.key} не отвечен: {failure_of(error)}"
                    ) from error
                raise error
            raise error

        findings = merge_batches(
            block,
            [part for part, _ in answers],
            broken=len(groups) - len(answers),
        )
        failed = sum(
            1 for item in findings.items if item.status == CXItemStatus.FAIL
        )
        not_applicable = sum(
            1 for item in findings.items if item.status == CXItemStatus.NA
        )
        why = findings.emptiness_note()
        logger.info(
            "%s: блок %s, кадров %s, батчей %s, отвечено %s, сорвано %s, "
            "вердиктов %s (fail %s, na %s), дефектов %s%s, модель %s",
            LogLabel.CHECK_CX_CRITERIA_FRAMES,
            block.key,
            f"{len(frames)} из {len(state.frames)}",
            len(groups),
            len(answers),
            len(groups) - len(answers),
            len(findings.items),
            failed,
            not_applicable,
            len(findings.deviations),
            f" — {why}" if why else "",
            answers[0][1],
        )
        return findings_update(block.key, findings=findings)

    except Exception as ex:
        if llm_retryable(ex) or is_shape_failure(ex):
            logger.warning(
                "%s: блок %s — %s; обращение к контуру повторится",
                LogLabel.CHECK_CX_CRITERIA_FRAMES,
                block.key,
                failure_of(ex),
            )
            raise
        logger.error(
            "%s: Failed to generate response check_cx_criteria for block %s. "
            "Exception: %s",
            LogLabel.CHECK_CX_CRITERIA_FRAMES,
            block.key,
            failure_of(ex),
            exc_info=True,
        )
        return findings_update(block.key, CXNodeStatus.ERROR)


async def _ask_batch(
    block_settings: CheckCxCriteriaSettings,
    gigachat_provider: GigaChatProvider,
    group: list,
    state: AgentStateSchema,
    frames_count: int,
    trace: str = "",
    tally: str = "",
    number: int = 1,
    count: int = 1,
) -> tuple[CXBlockFindings, str]:
    """Один батч кадров блока: обращение к контуру и его разбор.

    Разбор стоит наравне с обращением: ответ без JSON — сбой батча, а не «в этой
    части пути проблем нет». Сделано это внутри обращения нарочно: сорванный батч не
    имеет права утянуть находки остальных, они уже заплачены.

    Повтор стоит того, что стоит одно обращение: неразобранный ответ и очередь
    контура лечатся именно повтором. В прогоне 047 терялось по батчу на блок —
    контур обрезал ответ на середине JSON, батч закрывался счётчиком `dropped`, и
    одиннадцать экранов пути оставались разбранными в один. Без локального повтора
    потери были бы хуже: узел перевызывает себя целиком только когда не отвечен ВЕСЬ
    блок, а пока отвечен хоть один батч, политика графа не просыпается — сорванный
    батч молча остался бы непроверенной частью справочника.

    Неповторяемое (нет модели, права, 4xx кроме очереди) поднимается сразу:
    повторить его — деньги на ветер, а причину узел обязан увидеть.

    :param block_settings: настройки блока — они собирают вопрос
    :param gigachat_provider: провайдер контура — им задан вопрос
    :param group: кадры этого батча
    :param state: состояние прогона — материалы, атрибуция, кадры
    :param frames_count: число кадров пути — батч подписан частью именно им
    :param trace: тексты всех кадров пути — по ним батч считает экраны и поля
    :param tally: счёт пути, посчитанный кодом: кадров и экранов
    :param number: который по счёту батч
    :param count: сколько батчей на блок
    :return: находки батча и имя отвечавшей модели
    :raises Exception: батч остался без ответа после всех попыток — форма
        ответа (`MalformedAnswer`) либо повторяемый сбой контура
    """
    return await repeat_recoverable(
        lambda: _ask_once(
            block_settings,
            gigachat_provider,
            group,
            state,
            frames_count,
            trace,
            tally,
            number,
            count,
        ),
        subject=(
            f"{LogLabel.CHECK_CX_CRITERIA_FRAMES}: блок "
            f"{block_settings.block.key}, батч {number} из {count} "
            f"(кадры {group[0].step}-{group[-1].step})"
        ),
        attempts=BATCH_ATTEMPTS,
        pause_s=BATCH_RETRY_PAUSE_S,
    )


async def _ask_once(
    block_settings: CheckCxCriteriaSettings,
    gigachat_provider: GigaChatProvider,
    group: list,
    state: AgentStateSchema,
    frames_count: int,
    trace: str = "",
    tally: str = "",
    number: int = 1,
    count: int = 1,
) -> tuple[CXBlockFindings, str]:
    """Одно обращение к контуру за одним батчем блока и разбор его ответа.

    :param block_settings: настройки блока — они собирают вопрос
    :param gigachat_provider: провайдер контура — им задан вопрос
    :param group: кадры этого батча
    :param state: состояние прогона — материалы, атрибуция, кадры
    :param frames_count: число кадров пути — батч подписан частью именно им
    :param trace: тексты всех кадров пути — по ним батч считает экраны и поля
    :param tally: счёт пути, посчитанный кодом: кадров и экранов
    :param number: который по счёту батч
    :param count: сколько батчей на блок
    :return: находки батча и имя отвечавшей модели
    :raises MalformedAnswer: ответа нет или он не разбирается
    """
    response = await gigachat_provider.chat(
        block_settings.chat_request(
            state.material,
            group,
            state.journey_details,
            total=frames_count,
            trace=trace,
            tally=tally,
            number=number,
            count=count,
        )
    )
    answer = response.message.content or ""
    edges = f"кадры {group[0].step}–{group[-1].step}"
    if not answer.strip():
        raise MalformedAnswer(
            f"контур ответил пустотой на {edges} "
            f"блока {block_settings.block.key} (батч {number} из {count})"
        )
    try:
        findings = findings_of(
            block_settings.block, LogLabel.CHECK_CX_CRITERIA_FRAMES, answer
        )
    except ValueError as ex:
        raise MalformedAnswer(
            f"ответ блока {block_settings.block.key} "
            f"({edges}, батч {number} из {count}) не разобран: {ex}"
        ) from ex
    logger.info(
        "%s: блок %s, батч %s из %s (%s) отвечен, модель %s",
        LogLabel.CHECK_CX_CRITERIA_FRAMES,
        block_settings.block.key,
        number,
        count,
        edges,
        response.used_model.name,
    )
    return findings, response.used_model.name
