"""Узлы поэкранной ветки: проход окна кадров и разметка его находок справочником.

Ветка кадров: кадры уходят в вопрос содержимым с картинками, как в соседних ветках
(`pipeline.agent.autofill.nodes`, `pipeline.agent.journey_frictions.nodes`,
`pipeline.agent.criteria_model_frames`). Отличие в вопросе: он разбирает ОДИН кадр —
даёт ему описание и перечисляет то, что мешает на нём самом, а по два кадра с
каждого края приложены только контекстом. Контекст нужен затем, что элемент,
обрезанный краем экрана, перестаёт быть нарушением, когда он показан на соседнем
кадре; без соседей проход штрафовал бы путь за фрагменты чужих экранов.

Разбираемый кадр — середина окна (`ScreenSettings.window`), и каждый кадр пути
становится им ровно один раз: сколько неразбранных кадров, столько и обращений.
Обращения идут последовательно, а не одним `gather`: окно с картинками дороже
одного кадра, а одновременные вопросы складываются в очередь контура (429) — там,
где сорванными остаются ровно те окна, которые не успели ответить.

Ответ прохода — сырые находки: что видно на экране, где это лежит и чем
доказывается. Ни блока, ни кода в них нет, и это не недосмотр, а порядок работы со
справочником: в вопросе прохода критерии стоят примерами того, что замечать, а
полный справочник с кодами и зонами приходит вторым узлом — `summarize_screen`.
Там находки уже текстом (изображений в этом обращении нет), а код находят по
описанию, а не по кадру: кадры своё дело сделали, и платить за них ещё раз
значило бы платить за матчинг, в котором картинки только мешают.

Оба узла делают то же, что узлы `pipeline.agent.nodes`: собирают вопрос по настройкам,
спрашивают контур и кладут его ответ в состояние. Разбором ответа прохода ведает
`pipeline.utils.screen_findings`, разметкой — `pipeline.utils.screen_mapping`, формой отчёта
— DTO после графа.
"""

import logging
from collections import Counter

from src.pipeline.aef.exceptions import MalformedAnswer
from src.pipeline.aef.models import ChatResponse
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.aef.patterns.retry import (
    failure_of,
    is_shape_failure,
    llm_retryable,
    repeat_recoverable,
)
from src.pipeline.aef.patterns.utils import node
from src.pipeline.schemas.nodes import (
    CXFrame,
    CXJourneyDetails,
    CXPinnedDeviation,
    CXScreenFindings,
    CXScreenNote,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import (
    GraphSettings,
    ScreenSettings,
    SummarizeScreenSettings,
)
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    NOT_ASKED_PREFIX,
    UNMATCHED_PREFIX,
    CXNodeStatus,
    CXZone,
)
from src.pipeline.utils.screen_findings import screen_findings, screen_update
from src.pipeline.utils.screen_mapping import screen_mapping

logger = logging.getLogger(__name__)


@node
async def check_screen(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Поэкранный разбор окном: что за экран и что на нём мешает клиенту.

    Каждое окно отвечает про один кадр — свой, — поэтому покрыт весь путь: вывод
    «чего-то не хватает» про экран снят разбором, а вывод «этого нет на всём
    маршруте» одному окну недоступен в принципе, и такие выводы остались одной
    `journey_frictions`, где весь путь перед глазами.

    Прохода без кадров нет: судить здесь не по чему, и вместо обращения к контуру
    узел пишет пропуск отметкой `not_asked:` — «не спрашивали» обязано отличаться
    от «спросили и чисто».

    Сбой делится как в соседних ветках: неразборчивый ответ и временный отказ
    контура поднимаются наверх политике повторов, остальное гасится здесь. Когда
    часть окон сорвана, узел оставляет разобранные кадры и пишет счётчик в
    `degraded` — молчание прохода не должно читаться как «кадров нет вовсе». Тем
    же счётчиком помечается недобор описаний: модель обязана описать каждый
    разбираемый кадр, и сколько из них осталось без описания, прогон обязан видеть
    (`summaries:11/15`).

    :param state: состояние прогона: материалы, кадры и атрибуция пути
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся эта ветка
    :return: находки и описания прохода в состояние прогона
    """
    screen = settings.screen
    label = LogLabel.CHECK_SCREEN
    frames_count = len(state.frames)
    asked = state.kept_frames

    if not asked:
        skip = (
            "для анализа не осталось кадров"
            if frames_count
            else "кадров пути нет"
        )
        logger.info("%s: проход не запускается — %s", label, skip)
        return screen_update(degraded=f"{NOT_ASKED_PREFIX}{skip}")

    windows = len(asked)
    answers: list[tuple[CXScreenFindings, str]] = []
    broken: list[Exception] = []
    for number, index in enumerate(range(windows), start=1):
        target = asked[index]
        try:
            answers.append(
                await _ask_window(
                    screen,
                    gigachat_provider,
                    screen.window(asked, index),
                    target,
                    state,
                    frames_count,
                    number,
                    windows,
                )
            )
        except Exception as error:
            broken.append(error)
            if not is_shape_failure(error) and not llm_retryable(error):
                logger.warning(
                    "%s: проход остановлен на кадре %s из %s (%s)",
                    label,
                    target.step,
                    windows,
                    failure_of(error),
                )
                break

    failed = windows - len(answers)
    if not answers:
        error = broken[0]
        if is_shape_failure(error) or llm_retryable(error):
            logger.warning(
                "%s: ни одно из %s окон не отвечено (%s); обращение к контуру "
                "повторится",
                label,
                windows,
                failure_of(error),
            )
            if is_shape_failure(error):
                raise MalformedAnswer(
                    f"поэкранный проход не отвечен: {failure_of(error)}"
                ) from error
            raise error
        logger.error(
            "%s: Failed to generate response check_screen. Exception: %s",
            label,
            failure_of(error),
            exc_info=error,
        )
        return screen_update(status=CXNodeStatus.ERROR)

    findings = _merge(label.value, [part for part, _ in answers])
    model = answers[0][1]
    degraded = _degraded(failed, windows, findings, frames_count)
    counts = Counter(item.zone for item in findings.deviations)
    zones = ", ".join(f"{zone}={counts.get(zone, 0)}" for zone in CXZone)
    logger.info(
        "%s: кадров %s, окон %s (сорвано %s), описаний %s, находок %s (%s)%s,"
        " модель %s",
        label,
        f"{len(asked)} из {frames_count}",
        windows,
        failed,
        len(findings.screens),
        len(findings.deviations),
        zones,
        f"; {degraded}" if degraded else "",
        model,
    )
    return screen_update(findings=findings, degraded=degraded)


@node
async def summarize_screen(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Находки прохода — размеченные кодами справочника.

    Проход описывает видимое и не знает ни кодов, ни зон; здесь те же находки, уже
    текстом, ставятся в справочник: совпало описание с дефектом — находка получает
    его код, блок и зону, не совпало — остаётся внекаталожной, со своей зоной и
    без кода. Мимо справочника находка не выбрасывается: найденное на экране
    дороже, чем удобство сверенного со словарём перечня
    (`pipeline.utils.screen_mapping`).

    Разметка молчит, когда молчал проход: находок нет — спрашивать не о чем, и
    узел уходит своей отметкой. Сбой разметки находок не трогает тем более: узел
    не поднимает его наверх нарочно, потому что обработчик ветки деградирует весь
    её результат, а терять здесь нечего — прогон остаётся с сырыми находками, у
    которых нет кода, но есть текст, улика и пин.

    :param state: состояние прогона: находки поэкранного прохода и атрибуция пути
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся эта ветка
    :return: размеченные находки в состоянии прогона
    """
    found = state.screen_findings
    label = LogLabel.SUMMARIZE_SCREEN
    if found is None or not found.deviations:
        why = (
            "поэкранного прохода не было"
            if found is None
            else "проход не назвал ни одной находки"
        )
        logger.info("%s: разметка не запускается — %s", label, why)
        return screen_update(
            findings=found, degraded=f"{NOT_ASKED_PREFIX}{why}"
        )

    mapping = settings.summarize_screen
    try:
        marked, model = await _ask_mapping(
            mapping, gigachat_provider, found, state
        )
    except Exception as error:
        logger.error(
            "%s: Failed to generate response summarize_screen. Exception: %s",
            label,
            failure_of(error),
            exc_info=error,
        )
        return screen_update(
            findings=found.model_copy(
                update={
                    "degraded": _marks(
                        found.degraded,
                        f"mapping_failed:{type(error).__name__}",
                    )
                }
            )
        )

    matched = sum(1 for item in marked.deviations if item.code)
    unmatched = len(marked.deviations) - matched
    degraded = _marks(
        marked.degraded,
        (
            f"{UNMATCHED_PREFIX}{unmatched}/{len(marked.deviations)}"
            if unmatched
            else ""
        ),
    )
    logger.info(
        "%s: находок %s, с кодом %s, без кода %s%s, модель %s",
        label,
        len(marked.deviations),
        matched,
        unmatched,
        f"; {degraded}" if degraded else "",
        model,
    )
    return screen_update(
        findings=marked.model_copy(update={"degraded": degraded})
    )


def _marks(*parts: str) -> str:
    """Строка `degraded` из отметок ветки: новая отметка не стирает старую.

    :param parts: отметки пустые пропускаются
    :return: отметки, пустая строка — если отметок не было
    """
    return "; ".join(part for part in parts if part)


def _degraded(
    broken: int,
    windows: int,
    findings: CXScreenFindings,
    frames_count: int,
) -> str:
    """Чем плох отвеченный проход: сорванные окна и недобор описаний кадров.

    Пустой перечень находок при живом контуре значит «видимых проблем не было» —
    это штатный ответ прохода. А вот кадр без описания штатным не бывает: описание
    промпт требует на каждом разбираемом кадре, и молчать об их числе значило бы
    выдавать неполный разбор за полный.

    :param broken: сколько окон осталось без ответа
    :param windows: сколько окон было
    :param findings: сводный итог разобранных окон
    :param frames_count: число кадров пути
    :return: строка `degraded`, пустая, если придраться не к чему
    """
    parts: list[str] = []
    if broken:
        parts.append(f"batches_failed:{broken}/{windows}")
    if len(findings.screens) < frames_count:
        parts.append(f"summaries:{len(findings.screens)}/{frames_count}")
    return "; ".join(parts)


async def _ask_window(
    screen: ScreenSettings,
    gigachat_provider: GigaChatProvider,
    window: list[CXFrame],
    target: CXFrame,
    state: AgentStateSchema,
    frames_count: int,
    number: int = 1,
    count: int = 1,
) -> tuple[CXScreenFindings, str]:
    """Одно окно кадров, спрашиваемое заново, пока сбой лечится повтором.

    Окна идут последовательно, и политикой графа частичная потеря не
    прикрывается: узел перевызывает себя целиком только когда не отвечено ВСЁ.
    Пока одно окно отвечено, сорванное молча оставалось бы непросмотренной частью
    пути — его кадр не попал бы ни в одно описание экрана и ни в одну находку.

    :param screen: настройки ветки
    :param gigachat_provider: провайдер контура модели
    :param window: кадры этого окна, разбираемый — в середине
    :param target: разбираемый кадр окна
    :param state: состояние прогона — заметка эксперта и атрибуция пути
    :param frames_count: число кадров всего пути — за его пределами экрана нет
    :param number: которое по счёту окно
    :param count: сколько окон легло на проход
    :return: находки и описания окна и имя отвечавшей модели
    :raises Exception: окно осталось без ответа после всех попыток
    """
    return await repeat_recoverable(
        lambda: _ask_contour(
            screen,
            gigachat_provider,
            window,
            target,
            state,
            frames_count,
            number,
            count,
        ),
        subject=(
            f"{LogLabel.CHECK_SCREEN}: кадр {target.step} "
            f"({number} из {count})"
        ),
    )


async def _ask_contour(
    screen: ScreenSettings,
    gigachat_provider: GigaChatProvider,
    window: list[CXFrame],
    target: CXFrame,
    state: AgentStateSchema,
    frames_count: int,
    number: int = 1,
    count: int = 1,
) -> tuple[CXScreenFindings, str]:
    """Одно окно кадров: обращение к контуру и его разбор.

    Разбор стоит наравне с обращением: ответ без JSON и ответ, где нет перечня
    `screens`, — сбой окна, а не «на пути проблем нет». Ответ про соседний кадр
    разбором снимается, а не переводится в находку соседнего экрана: у того экрана
    будет своё обращение.

    :param screen: настройки ветки
    :param gigachat_provider: провайдер контура модели
    :param window: кадры этого окна, разбираемый — в середине
    :param target: разбираемый кадр окна
    :param state: состояние прогона — заметка эксперта и атрибуция пути
    :param frames_count: число кадров всего пути — за его пределами экрана нет
    :param number: которое по счёту окно — им подписан вопрос
    :param count: сколько окон легло на проход
    :return: находки и описания окна и имя отвечавшей модели
    :raises MalformedAnswer: ответ есть, но он не разобран
    """
    response: ChatResponse = await gigachat_provider.chat(
        screen.chat_request(
            state.material,
            window,
            target,
            state.journey_details,
            total=frames_count,
            number=number,
            count=count,
        )
    )
    try:
        findings = screen_findings(
            response.message.content or "", frames_count, target.step
        )
    except ValueError as ex:
        raise MalformedAnswer(
            f"ответ поэкранного окна не разобран: {ex}"
        ) from ex
    return findings, response.used_model.name


async def _ask_mapping(
    mapping: SummarizeScreenSettings,
    gigachat_provider: GigaChatProvider,
    found: CXScreenFindings,
    state: AgentStateSchema,
) -> tuple[CXScreenFindings, str]:
    """Разметка находок, спрашиваемая заново, пока сбой лечится повтором.

    Разбор входит в повторяемое обращение: ответ без перечня `matches` — не «кода
    не нашлись», а неразобранный ответ, и спрашивается он заново на месте, а не
    уходит вниз отметкой о деградации.

    :param mapping: настройки разметки
    :param gigachat_provider: провайдер контура модели
    :param found: сырые находки прохода
    :param state: состояние прогона — атрибуция пути
    :return: размеченные находки и имя отвечавшей модели
    :raises Exception: разметка осталась без ответа после всех попыток
    """
    return await repeat_recoverable(
        lambda: _ask_reference(
            mapping, gigachat_provider, found, state.journey_details
        ),
        subject=(
            f"{LogLabel.SUMMARIZE_SCREEN}: {len(found.deviations)} находок"
            " прохода"
        ),
    )


async def _ask_reference(
    mapping: SummarizeScreenSettings,
    gigachat_provider: GigaChatProvider,
    found: CXScreenFindings,
    details: CXJourneyDetails,
) -> tuple[CXScreenFindings, str]:
    """Одно обращение разметки: полный справочник, перечень находок, разбор.

    Кадров здесь нет нарочно: справочник читают по описанию находки, а не по
    картинке, и пять изображений на каждое обращение добавили бы в вопрос только
    цену. Разбором ведает `pipeline.utils.screen_mapping`: он ставит код, блок и зону
    справочника и не трогает ни текст находки, ни её пин.

    :param mapping: настройки разметки
    :param gigachat_provider: провайдер контура модели
    :param found: сырые находки прохода
    :param details: атрибуция пути — она подписывает вопрос
    :return: размеченные находки и имя отвечавшей модели
    :raises MalformedAnswer: ответ есть, но он не разобран
    """
    response: ChatResponse = await gigachat_provider.chat(
        mapping.chat_request(found.deviations, found.screens, details)
    )
    try:
        marked = screen_mapping(
            response.message.content or "",
            found,
            mapping.load_criteria_model(),
        )
    except ValueError as ex:
        raise MalformedAnswer(
            f"ответ разметки поэкранного прохода не разобран: {ex}"
        ) from ex
    return marked, response.used_model.name


def _merge(source_key: str, parts: list[CXScreenFindings]) -> CXScreenFindings:
    """Ответы окон в один итог ветки.

    Номера экранов модель берёт из подписей `[кадр N]`, а подписывает их узел
    глобальными номерами кадров: складывать смещение окна, как делает ядро на своих
    батчах, здесь не нужно, и находки окон просто складываются. Описания одного
    кадра из разных окон остаются первым — второе самоповтор, а не второе мнение.

    :param source_key: ключ источника — метка ветки
    :param parts: итоги разобранных окон
    :return: общий итог прохода
    """
    deviations: list[CXPinnedDeviation] = []
    notes: dict[int, CXScreenNote] = {}
    dropped: Counter[str] = Counter()
    for part in parts:
        deviations += list(part.deviations)
        dropped.update(part.dropped)
        for note in part.screens:
            notes.setdefault(note.step, note)
    return CXScreenFindings(
        source_key=source_key,
        deviations=deviations,
        screens=[notes[step] for step in sorted(notes)],
        dropped=dict(dropped),
    )
