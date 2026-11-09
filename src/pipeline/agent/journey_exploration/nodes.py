"""Узел исследования пути: что изображено на одном кадре.

Узел делает то же, что узлы `pipeline.agent.nodes`: собирает вопрос по настройкам,
спрашивает контур и кладёт его ответ в состояние как есть. Правил здесь нет: чем
остановиться, чем объяснить отказ и чем деградировать сорванному кадру — в
политике приёмки `pipeline.utils.journey`, там же и строка журнала о ходе скана. Здесь
два узла цикла: сам скан и обработчик исчерпанных попыток, которому номер кадра
открывается только из состояния.

Цикл по кадрам держит граф, а не узел: за проход описывается один кадр, а его
номер — это число описаний в состоянии
(`AgentStateSchema.frame_readouts`). Так проход кадра виден в стриме прогона, а
сбой одного кадра стоит потерянного описания, а не всего пути.

Кадр загружается в контур один раз и живёт в состоянии ссылкой на загруженный
файл (`CXFrame.ref`): разбор следующего кадра прикладывает уже загруженных
предшественников, а после скана те же ссылки берут блоки справочника и проверки
пути. Снятый фильтром кадр ссылку теряет сразу — его изображение больше не нужно
прогону, а осиротевший файл на стороне контура никто кроме него не удалит.
Остальные ссылки снимает узел `pipeline.agent.nodes.release_frames` в конце прогона.

Номера кадров при фильтрации не сдвигаются: описание получает каждый кадр пути,
в том числе отфильтрованный. Иначе номер следующего кадра поехал бы вместе с
пропуском, а находки отчёта адресуют экран именно этим номером.
"""

import logging
from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig

from src.pipeline.aef.exceptions import MalformedAnswer
from src.pipeline.aef.models import RetryOptions
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.aef.patterns.retry import (
    failure_of,
    is_shape_failure,
    llm_retryable,
)
from src.pipeline.aef.patterns.utils import node
from src.pipeline.schemas.nodes import CXFrame, CXFrameReadout
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import (
    GraphSettings,
    JourneyExplorationSettings,
    ScanJourneyFrameSettings,
)
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.utils.journey import describe_scan, stop_reason, unread

logger = logging.getLogger(__name__)


def _window(
    scan: ScanJourneyFrameSettings,
    state: AgentStateSchema,
    frames: Sequence[CXFrame],
    index: int,
) -> list[CXFrame]:
    """Окно разбираемого кадра: он сам и два предыдущих из оставшихся в анализе.

    Отфильтрованные кадры из окна выпадает: их описания не нужны, а изображение
    снятого кадра прогон больше не держит в контуре. Из-за этого соседом иногда
    оказывается кадр через пропуск — подписи с номерами пути это и разрешают.

    :param scan: настройки скана — по ним считается окно
    :param state: состояние прогона — по описаниям видно, какие кадры отфильтрованы
    :param frames: кадры пути с уже добытыми ссылками контура
    :param index: положение разбираемого кадра в пути, с нуля
    :return: кадры окна в порядке пути, разбираемый — последним
    """
    kept = [
        frame
        for frame in frames[: index + 1]
        if frame.step not in state.filtered_steps
    ]
    return scan.window(kept, len(kept) - 1)


async def _keep_ref(
    gigachat_provider: GigaChatProvider,
    scan: ScanJourneyFrameSettings,
    frames: list[CXFrame],
    index: int,
) -> CXFrame:
    """Загружает кадр в контур один раз и оставляет ссылку на него в состоянии.

    Каждый кадр проходит через этот вызов ровно один раз: у кадра, разобранного
    раньше, ссылка уже есть, и повторная загрузка только стоила бы round-trip.
    Сбой загрузки кадру не стоит — байты уходят в вопрос по-прежнему, просто
    следующий вопрос догрузит их снова.

    :param gigachat_provider: провайдер контура модели
    :param scan: настройки скана — модель и параметры обращения
    :param frames: кадры пути — кадр в них обновляется на месте
    :param index: положение разбираемого кадра, с нуля
    :return: разбираемый кадр — со ссылкой, если она добыта
    """
    frame = frames[index]
    if frame.ref:
        return frame
    try:
        ref = await gigachat_provider.upload_frame(
            scan.chat_options, scan.stage_model, frame.data_url
        )
    except Exception as ex:  # noqa: BLE001
        logger.warning(
            "%s: кадр %s — ссылка контура не получена (%s), кадр уйдёт в вопрос "
            "байтами",
            LogLabel.SCAN_JOURNEY_FRAME,
            frame.step,
            failure_of(ex),
        )
        return frame
    loaded = frame.model_copy(update={"ref": ref})
    frames[index] = loaded
    return loaded


async def _drop_ref(
    gigachat_provider: GigaChatProvider,
    scan: ScanJourneyFrameSettings,
    frames: list[CXFrame],
    index: int,
) -> None:
    """Снимает ссылку кадра, которая прогону больше не нужна.

    Снятие здесь всегда вторично: зовут либо из разбора ответа, либо из тракта
    сбоя, и сбой удаления не имеет права выглядеть как сбой контура — он заглушает
    настоящую причину, по которой кадр не разобран. Осиротевший файл при этом
    остаётся заботой контура: кадр к нему больше не сошлётся.

    :param gigachat_provider: провайдер контура модели
    :param scan: настройки скана — тем же клиентом грузили файл
    :param frames: кадры пути — у кадра в них убирается ссылка
    :param index: положение кадра, с нуля
    """
    frame = frames[index]
    if not frame.ref:
        return
    try:
        await gigachat_provider.delete_frames(
            scan.chat_options, scan.stage_model, [frame.ref]
        )
    except Exception as ex:  # noqa: BLE001
        logger.warning(
            "%s: кадр %s — ссылка %s не снята (%s)",
            LogLabel.SCAN_JOURNEY_FRAME,
            frame.step,
            frame.ref,
            ex,
        )
    frames[index] = frame.model_copy(update={"ref": ""})


async def _ask_frame(
    gigachat_provider: GigaChatProvider,
    explore: JourneyExplorationSettings,
    state: AgentStateSchema,
    window: Sequence[CXFrame],
    missing: Sequence[str] = (),
    doubt: str = "",
) -> CXFrameReadout:
    """Одно обращение к контуру про кадр окна; `missing`/`doubt` — повод переспроса.

    :param gigachat_provider: провайдер контура модели
    :param explore: настройки исследования пути — из них берутся окно и вопрос
    :param state: состояние прогона: материалы, кадры и описания
    :param window: кадры окна в порядке пути, разбираемый — последний
    :param missing: пустые поля предыдущего ответа — приходят в вопрос
        требованием к повторному ответу
    :param doubt: чем неудовлетворён предыдущий ответ при заполненных полях
    :return: описание кадра с номером кадра пути вместо номера из ответа
    """
    frame = window[-1]
    scan = explore.scan_frame
    readout = await gigachat_provider.chat_with_structure_output_with_frames(
        request_options=scan.chat_request(
            material=state.material,
            window=window,
            center_step=frame.step,
            context=state.scan_context(explore.context_last),
            missing=missing,
            doubt=doubt,
        ),
        structured_output=CXFrameReadout,
        method="format_instructions",
    )
    # Номера кадров в состоянии обязаны совпадать с файлами пути: модель могла
    # подписать кадр соседа из окна, а номера читают и блоки оценки, и отчёт.
    return readout.model_copy(update={"step": frame.step})


async def _reask_frame(
    gigachat_provider: GigaChatProvider,
    explore: JourneyExplorationSettings,
    state: AgentStateSchema,
    window: Sequence[CXFrame],
    flawed: CXFrameReadout,
) -> CXFrameReadout:
    """Один переспрос про неудовлетворительный ответ — и принятие ответа как есть.

    Поводов два, и оба — нарушение контракта, а не сбой контура: пустое
    обязательное поле (`blank_text`) и переписанный соседний кадр
    (`off_frame_text` — названного заголовка нет в тексте). Второй тот же вопрос
    ничего бы не дал: при `temperature = 0` контур повторяет себя дословно, и в
    прогоне 009 кадр сжёг пять одинаковых обращений, чтобы кончить записью
    `no_answer`. Поэтому в переспрос входит само замечание — чего не хватило или
    с какого изображения переписывать.

    Принимается только второй ответ без всякого замечания: ответ с другим
    дефектом улучшением не считается, а первый содержит то, что модель уже
    увидела на кадре. Кадр не остаётся пустым ни в одном случае.

    :param gigachat_provider: провайдер контура модели
    :param explore: настройки исследования пути
    :param state: состояние прогона
    :param window: кадры окна того же кадра — переспрос идёт по ним же
    :param flawed: первый ответ кадра, принятой формой, но с замечанием
    :return: ответ кадра — второй, если он исправился, иначе первый
    """
    frame = window[-1]
    logger.warning(
        "%s: кадр %s — %s; кадр переспрошен по существу",
        LogLabel.SCAN_JOURNEY_FRAME,
        frame.step,
        flawed.fault,
    )
    try:
        again = await _ask_frame(
            gigachat_provider,
            explore,
            state,
            window,
            flawed.missing_fields,
            flawed.fault if flawed.suspect_text else "",
        )
    except Exception as ex:
        # Сбой переспроса не имеет права стоить первого ответа: пустой кадр
        # хотя бы описан, а запись сбоя не сказала бы об этом ничего.
        logger.warning(
            "%s: кадр %s — переспрос не состоялся (%s), принят первый ответ",
            LogLabel.SCAN_JOURNEY_FRAME,
            frame.step,
            ex,
        )
        return flawed

    if not again.needs_reask:
        logger.info(
            "%s: кадр %s — переспрос исправил ответ (%s)",
            LogLabel.SCAN_JOURNEY_FRAME,
            frame.step,
            flawed.fault,
        )
        return again

    logger.warning(
        "%s: кадр %s — %s и после переспроса; принят первый ответ, отметку "
        "дописала схема, не модель",
        LogLabel.SCAN_JOURNEY_FRAME,
        frame.step,
        again.fault or flawed.fault,
    )
    return flawed


@node
async def scan_journey_frame(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Описание одного кадра: кадры окна `[i-2, i-1, i]`, ответ про центр.

    Какой кадр разбирать, узел узнаёт из состояния — по числу уже описанных, —
    поэтому вызов повторяется графом до последнего кадра; куда вести прогон
    дальше, решает маршрут `pipeline.agent.journey_exploration.subgraph`. Описаний
    больше, чем кадров, быть не может: на последнем кадре маршрут выходит из
    цикла, а пустой путь до узла не доходит вообще. Описание получает каждый кадр
    пути, и отфильтрованный тоже: номер следующего кадра не имеет права
    ездить вместе с пропусками, его читают находки отчёта.

    Кадр перед вопросом загружается в контур и остаётся в состоянии ссылкой на
    этот файл (`CXFrame.ref`) — по ней же его прикладывают следующие окна и все
    проверяющие ветки после скана.

    Сбой делится как в остальных диалоговых узлах
    (`pipeline.aef.patterns.retry`): неразборчивый ответ и временный отказ контура
    перевызываются, постоянное гасится здесь записью `unread` — кадр обязан
    остаться в состоянии, иначе номер следующего кадра не сдвинется и цикл
    встанет на этом кадре навсегда. Перевызов означает новый вопрос с новым
    обращением к контуру, поэтому ссылку, добытую в сорванной попытке, узел снимает
    сам: иначе каждая попытка оставила бы по файлу, на который больше никто не
    сошлётся.

    Пустое обязательное поле и переписанный не с того кадра текст — не сбой, а
    нарушение ответа: такой кадр переспрашивается один раз по существу
    (`_reask_frame`) и остаётся в состоянии любым из двух ответов.

    :param state: состояние прогона: материалы и кадры пути, описания
        разобранных кадров
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся сканирование кадров
    :return: описание следующего кадра и кадры пути с добытой ссылкой
    """
    explore = settings.journey_exploration
    scan = explore.scan_frame
    index = len(state.frame_readouts)
    if index >= len(state.frames):
        return {}

    frames = list(state.frames)
    frame = await _keep_ref(gigachat_provider, scan, frames, index)
    window = _window(scan, state, frames, index)
    try:
        readout = await _ask_frame(gigachat_provider, explore, state, window)
    except Exception as ex:
        await _drop_ref(gigachat_provider, scan, frames, index)
        if is_shape_failure(ex):
            logger.warning(
                "%s: кадр %s — ответ контура не разобран по схеме (%s); "
                "обращение к контуру повторится",
                LogLabel.SCAN_JOURNEY_FRAME,
                frame.step,
                failure_of(ex),
            )
            raise MalformedAnswer(
                f"описание кадра {frame.step} не разобран: {failure_of(ex)}"
            ) from ex
        if llm_retryable(ex):
            logger.warning(
                "%s: кадр %s — %s; обращение к контуру повторится",
                LogLabel.SCAN_JOURNEY_FRAME,
                frame.step,
                failure_of(ex),
            )
            raise
        logger.error(
            "%s: Failed to generate response for frame %s. Exception: %s",
            LogLabel.SCAN_JOURNEY_FRAME,
            frame.step,
            failure_of(ex),
            exc_info=True,
        )
        return {
            "frame_readouts": [unread(frame.step)],
            "frames": frames,
        }

    # Кадр отвечен по форме, но ответ неудовлетворительный: поле пустое либо
    # переписан соседний кадр. Прежде чем принять такой ответ, узел один раз
    # спрашивает именно про это замечание.
    if readout.needs_reask:
        readout = await _reask_frame(
            gigachat_provider, explore, state, window, readout
        )

    if readout.need_filter:
        # Кадр снят с анализа: его изображение прогону больше не нужно — ни в
        # следующем окне, ни в проверяющих ветках. Номер остаётся: описание этого
        # кадра в состоянии есть, и счёт следующего кадра оно не сдвигает.
        # Сам кадр здесь ещё числится неразбранным, поэтому его сам себя он и
        # не снял: отнятая единица — это он.
        left = len(state.kept_frames) - 1
        await _drop_ref(gigachat_provider, scan, frames, index)
        logger.warning(
            "%s: кадр %s отфильтрован (%s) — изображение снято с анализа, "
            "в анализе осталось %d кадров из %d",
            LogLabel.SCAN_JOURNEY_FRAME,
            frame.step,
            readout.filter_note or "причины кадр не назвал",
            left,
            len(state.frames),
        )

    passed = [*state.frame_readouts, readout]
    logger.info(
        "%s: %s",
        LogLabel.SCAN_JOURNEY_FRAME,
        describe_scan(
            passed,
            len(state.frames),
            stop_reason(passed, explore.stop_after),
        ),
    )
    return {"frame_readouts": [readout], "frames": frames}


@node
async def mark_frame_unread(
    state: AgentStateSchema, config: RunnableConfig
) -> dict:
    """Сорванный кадр: запись об ошибке вместо пропажи кадра.

    Узел стоит на месте обработчика исчерпанных попыток, поэтому какой кадр
    остался без описания, он узнаёт из состояния — по числу уже разобранных, тем
    же счётом, что и сам скан. Готовым словарём здесь не деградируют: номер кадра
    знает только состояние. Без записи цикл стоял бы на этом кадре вечно, а в
    отчёте пропущенный кадр был бы неотличим от несуществующего.

    :param state: состояние прогона: кадры пути и разобранные описания
    :param config: конфигурация запуска — в ней ошибка упавшего узла
    :return: описание кадра без ответа либо пусто, если кадров не осталось
    """
    index = len(state.frame_readouts)
    if index >= len(state.frames):
        return {}
    node_error = (config.get("configurable") or {}).get(
        RetryOptions.NODE_ERROR_KEY
    )
    logger.warning(
        "%s: кадр %s — без результата (%s)",
        LogLabel.SCAN_JOURNEY_FRAME,
        state.frames[index].step,
        getattr(node_error, "error", node_error),
    )
    return {"frame_readouts": [unread(state.frames[index].step)]}
