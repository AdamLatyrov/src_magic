"""Узлы графа: разговор с моделью и ничего больше.

В этом файле стоят исключительно функции под декоратором `@node` — своих функций,
констант и разборов здесь нет намеренно. Узел делает одно: собирает вопрос по
настройкам, спрашивает контур и кладёт его ответ в состояние; ответ распознавания
проходит через политику приёмки `pipeline.utils.journey`, находки блоков и сведение
ложатся как пришли.

Разбором ответа ведает тот, кому он нужен: распознавание пути разбирает состояние
(`AgentStateSchema.journey_details`, политика приёмки — `pipeline.utils.journey`),
находки блоков — `pipeline.utils.block_findings`, а баллы, разметку блоков и
причины деградации — DTO `pipeline.dto.agent.result.AgentResult`. Политика приёмки вне
графа поэтому, что состояние собирают и без прогона: по записанному ответу модели,
при сверке распознавания; вердикт и причины обязаны получаться одними и теми же при
любом способе наполнения состояния.
"""

import logging

from src.pipeline.aef.exceptions import MalformedAnswer
from src.pipeline.aef.models import ChatRequest
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.aef.patterns.retry import (
    failure_of,
    is_shape_failure,
    llm_retryable,
)
from src.pipeline.aef.patterns.utils import node
from src.pipeline.schemas.nodes import (
    CXConsolidatedFindings,
    CXJourneyDetails,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import GraphSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import NO_ANSWER
from src.pipeline.utils.findings import raw_findings_listing
from src.pipeline.utils.journey import (
    describe,
    from_scan,
    refusal_note,
    stop_reason,
)

logger = logging.getLogger(__name__)


@node
async def summarize_journey(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Сведение описаний кадров в атрибуцию пути: продукт, сценарий, тип пути.

    Вердикт «путь ли это» берётся не от модели сведения, а от скана
    (`AgentStateSchema.scan_verdict`): кадры видел он, и только они могут
    сказать, что перед нами интерфейс, а не фотография. Модель здесь описывает
    путь по текстам, поэтому вопрос ей задаётся лишь когда путь подтверждён:
    «не путь» объясняется словами модели с кадров (`refusal_note`), а молчание
    скана остаётся молчанием — прогон идёт вслепую, а не отклоняется.

    Продукт и сценарий — то, ради чего здесь поднимается весь скан: блоки оценки
    читают атрибуцию из состояния и перестают угадывать домен каждый сам. Второе —
    сквозной путь (`CXJourneyDetails.path`): маршрут, переходы, где показаны
    условия и чего материал не проверяет. Атрибуцию и путь читают проверяющие
    ветки — блоки справочника спрашивают экраны, но переходы между ними видны
    только по всему пути, поэтому путь и ложится в вопрос рядом с атрибуцией.

    Сбой делится как в остальных диалоговых узлах: неразборчивый ответ и
    временный отказ перевызываются, постоянное гасится здесь атрибуцией без
    названий — подтверждённый путём путь остаётся путём, а отсутствие названий
    видно по `degraded`.

    :param state: состояние прогона: материалы, кадры и описания кадров
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся сведение описаний
    :return: атрибуция пути в состояние прогона
    """
    scan = settings.journey_exploration
    verdict = state.scan_verdict

    if verdict is not True:
        details = from_scan(
            verdict,
            None,
            refusal_note(
                state.frame_readouts,
                stop_reason(state.frame_readouts, scan.stop_after),
            ),
        )
        logger.info(
            "%s: %s — контур не спрашивается, %s",
            LogLabel.SUMMARIZE_JOURNEY,
            (
                "путь не подтверждён"
                if verdict is False
                else "кадров с ответом нет"
            ),
            describe(details),
        )
        return {"journey_details": details}

    try:
        named = await gigachat_provider.chat_structured_output(
            request_options=settings.summarize_journey.chat_request(
                material=state.material,
                frames_text=state.scan_evidence,
                coverage=state.scan_coverage,
            ),
            structured_output=CXJourneyDetails,
            method="format_instructions",
        )
    except Exception as ex:
        if is_shape_failure(ex):
            logger.warning(
                "%s: ответ контура не разобран по схеме (%s); обращение "
                "к контуру повторится",
                LogLabel.SUMMARIZE_JOURNEY,
                failure_of(ex),
            )
            raise MalformedAnswer(
                f"сведение описаний не разобрано: {failure_of(ex)}"
            ) from ex
        if llm_retryable(ex):
            logger.warning(
                "%s: %s; обращение к контуру повторится",
                LogLabel.SUMMARIZE_JOURNEY,
                failure_of(ex),
            )
            raise
        logger.error(
            "%s: Failed to generate response summarize_journey. Exception: %s",
            LogLabel.SUMMARIZE_JOURNEY,
            failure_of(ex),
            exc_info=True,
        )
        details = from_scan(verdict, None).model_copy(
            update={"degraded": NO_ANSWER}
        )
        logger.info("%s: %s", LogLabel.SUMMARIZE_JOURNEY, describe(details))
        return {"journey_details": details}

    details = from_scan(verdict, named)
    logger.info("%s: %s", LogLabel.SUMMARIZE_JOURNEY, describe(details))
    return {"journey_details": details}


@node
async def consolidate_findings(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """
    Узел сведения находок: сводит отклонения обеих веток в единый ответ модели.

    Ветки проверялись вслепую друг к другу и дублируют одни и те же проблемы: блок
    справочника и проверка проблем пути видят один экран по-своему. Здесь сырьё
    обеих веток уходит модели один раз — одним перечнем однотипных отклонений
    (`AgentStateSchema.findings`), без кадров: описания и улики уже добыты на
    кадрах. Модель отвечает по схеме `CXConsolidatedFindings` — форму ответа задаёт
    она же, вместе с инструкцией формата, которую контур дописывает в сообщение.

    Дедупликация — ради чего ветки сводят вместе: одна проблема, найденная блоком и
    проверкой пути, должна описываться одной находкой и штрафоваться один раз. Пока
    сведение видело только блоки, трения уходили в отчёт отдельной секцией без
    штрафа, и развести их было некому.

    Узел только говорит с моделью. Баллы, разметка блоков и причины деградации — дело
    DTO `pipeline.dto.agent.result.AgentResult`, который собирают по состоянию после графа;
    не состоялось сведение — в состоянии пусто, и DTO возьмёт сырые отклонения обеих
    веток сам.

    Сбой делится как в узле распознавания (`pipeline.aef.patterns.retry`): неразборчивый
    ответ и временный отказ контура перевызываются, постоянное гасится здесь в
    пустоте состояния — отсутствующее сведение для прогона нормально.

    :param state: состояние прогона: материалы, кадры, находки блоков и справочник
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся параметры сведения
    :return: сводный ответ модели в состояние
    """
    try:
        sources = state.findings
        logger.info(
            "%s: сведение %s отклонений по %s источникам",
            LogLabel.CONSOLIDATE_FINDINGS,
            state.raw_findings_count,
            len(sources),
        )
        chat_request = ChatRequest(
            options=settings.consolidate_findings.chat_options,
            model=settings.consolidate_findings.stage_model,
            messages=settings.consolidate_findings.messages(
                material=state.material,
                raw_findings=raw_findings_listing(
                    state.criteria_model,
                    sources,
                ),
                checks=state.checks_listing,
                block_keys=state.block_keys_listing,
                finding_count=state.raw_findings_count,
                details=state.journey_details,
            ),
        )
        result = await gigachat_provider.chat_structured_output(
            request_options=chat_request,
            structured_output=CXConsolidatedFindings,
            method="format_instructions",
        )
        logger.info(
            "%s: модель свела %s сырых находок в %s",
            LogLabel.CONSOLIDATE_FINDINGS,
            state.raw_findings_count,
            len(result.issues),
        )
        return {"cx_consolidation": result}

    except Exception as ex:
        if is_shape_failure(ex):
            logger.warning(
                "%s: ответ контура не разобран по схеме (%s); обращение "
                "повторится",
                LogLabel.CONSOLIDATE_FINDINGS,
                failure_of(ex),
            )
            raise MalformedAnswer(
                f"ответ сведения не разобран: {failure_of(ex)}"
            ) from ex
        if llm_retryable(ex):
            logger.warning(
                "%s: %s; обращение к контуру повторится",
                LogLabel.CONSOLIDATE_FINDINGS,
                failure_of(ex),
            )
            raise
        logger.error(
            "%s: Failed to generate response consolidate_findings. Exception: %s",
            LogLabel.CONSOLIDATE_FINDINGS,
            failure_of(ex),
            exc_info=True,
        )
        return {"cx_consolidation": None}


@node
async def release_frames(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Снимает ссылки кадров, загруженные в контур: прогон их больше не смотрит.

    Скан загружает кадр один раз и оставляет ссылку на него (`CXFrame.ref`), чтобы
    то же изображение не платило загрузку на каждый вопрос блоков и проверок.
    Файл после прогона остаётся висеть в контуре: никто кроме прогона его не
    снимет, а ссылок на него к этому моменту не осталось ни в одном живом вопросе.

    Узел стоит последним переходом графа, а не в `finally` у вызывающей стороны:
    кадры обязаны сняться и когда путь не признан путём, и когда ветка деградировала
    ошибкой, — иначе каждая сорванная проверка оставляла бы на стороне контура по
    пучку файлов. Неснятую ссылку видно из состояния: узел чистит их только по
    удачном снятии, а частичный сбой называет журналом.

    :param state: состояние прогона — кадры пути с оставшимися ссылками контура
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — тем же клиентом кадры и грузили
    :return: кадры пути без ссылок; пусто, если снимать нечего или ссылок не снять
    """
    refs = [frame.ref for frame in state.frames if frame.ref]
    if not refs:
        return {}

    scan = settings.journey_exploration.scan_frame
    try:
        await gigachat_provider.delete_frames(
            scan.chat_options, scan.stage_model, refs
        )
    except Exception as ex:  # noqa: BLE001
        logger.error(
            "%s: %s ссылок кадров не снято (%s) — файлы остались на стороне "
            "контура",
            LogLabel.RELEASE_FRAMES,
            len(refs),
            ex,
            exc_info=True,
        )
        return {}

    logger.info(
        "%s: ссылок кадров снято: %s из %s",
        LogLabel.RELEASE_FRAMES,
        len(refs),
        len(state.frames),
    )
    return {
        "frames": [
            frame.model_copy(update={"ref": ""}) for frame in state.frames
        ]
    }
