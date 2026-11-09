"""Узел проверки проблемы пути: окно кадров на один вопрос контуру.

Ветка кадров по проблемам пути: вопрос строится по изображениям пути —
`state.kept_frames` уходят в него содержимым с картинками, как в соседней ветке
блоков (`pipeline.agent.criteria_model_frames.nodes`); отфильтрованные кадры ветка не
видит. Узел один на все шесть проверок: какую проверку делать, он узнаёт по
собственному имени в графе.

Путь делится на окна (`CheckJourneyFrictionSettings.batches`) и окна идут одно за
другим, как в ветке блоков и в поэкранном проходе: контур держит в поле зрения
документированные десять изображений и молча снимает лишние, поэтому вопрос на
сорок экранов — это ответ по первым десяти, за который заплачено как за сорок.
Структура всего маршрута остаётся в вопросе текстом (`route_trace`), чтобы окно в
десять кадров не выдавало себя за весь путь.

Второй узел ветки — `map_journey_frictions`: проверки ищут то, что мешает, и при
этом не знают справочника, а адрес — код дефекта, его блок и его зона — приходит к
их находкам отдельно, когда находки стали текстом. Порядок тот же, что у пары
`check_screen` → `summarize_screen`, и политика разметки у них общая
(`pipeline.utils.deviation_mapping`).

Узел делает то же, что узлы `pipeline.agent.nodes`: собирает вопрос по настройкам,
спрашивает контур и кладёт его ответ в состояние. Разбором ответа ведает
`pipeline.utils.friction_findings`, а формой отчёта — DTO после графа.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Optional

from langchain_core.runnables import RunnableConfig

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
    CXDeviation,
    CXFrictionFindings,
    CXJourneyDetails,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import (
    CheckJourneyFrictionSettings,
    GraphSettings,
    JourneyFrictionsSettings,
    MapJourneyFrictionsSettings,
)
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    DROP_PHANTOM_QUOTE,
    DROP_SERIES_MERGED,
    UNMATCHED_PREFIX,
    CXNodeStatus,
    CXZone,
)
from src.pipeline.utils.friction_findings import (
    friction_findings,
    frictions_update,
    merge_frictions,
)
from src.pipeline.utils.journey import route_trace, value_trace
from src.pipeline.utils.journey_mapping import friction_mapping

logger = logging.getLogger(__name__)


def check_node_name(check_key: str) -> str:
    """Имя узла этой ветки для одной проверки.

    Имя несёт метку ветки: `journey_frictions_length` и
    `criteria_model_frames_cxv3_1` отвечают за разное, и прогон каждой ветки
    обязан читаться в стриме и в журнале отдельно. Ключ проверки входит в имя
    целиком — разбор имени поэтому ищет перебором, а не режет строку.

    :param check_key: ключ проверки из конфигурации ветки
    :return: имя узла для `add_node`
    """
    return f"{LogLabel.CHECK_JOURNEY_FRICTIONS.value}_{check_key}"


def check_of_node(
    frictions: JourneyFrictionsSettings, node_name: str
) -> CheckJourneyFrictionSettings:
    """Проверку, которую делает узел с таким именем.

    Узел узнаёт своё имя из конфигурации запуска; обратный поиск перебором,
    а не разбором имени: ключ проверки может содержать что угодно.

    :param frictions: настройки ветки — по ним состав проверок
    :param node_name: имя узла в графе
    :return: настройки проверки этого узла
    :raises ValueError: имени узла нет среди описанных проверок
    """
    for check in frictions.checks:
        if check_node_name(check.check_key) == node_name:
            return check
    raise ValueError(
        f"среди проверок {LogLabel.CHECK_JOURNEY_FRICTIONS} нет узла "
        f"{node_name}"
    )


@node
async def check_journey_friction(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
    config: RunnableConfig,
) -> dict:
    """Проверка пути на одну из шести проблем, окнами по несколько кадров.

    Узел общий: какую проверку делать, он узнаёт по своему имени в графе — имя
    даёт ветка (`check_node_name`), и здесь же оно разбирается обратно в проверку,
    — а настройки приходят из конфигурации вызова. Так состав проверок остаётся
    делом сборки графа, а здесь остаётся одна плоская функция.

    Кадры уходят окнами: проблемы пути живут между экранами, и окно подпирается
    перечнем всего пути текстом — по нему видно, что десять показанных экранов есть
    середина сорокаэкранного маршрута, а не весь путь. Номера кадров при этом
    глобальные, поэтому находки окон складываются без пересборки.

    Сбой узел делит сам (`pipeline.aef.patterns.retry`): неразборчивый ответ и
    временный отказ контура поднимаются наверх и перевызываются, остальное
    гасится здесь в статус `error`. Неповторяемый отказ останавливает проверку на
    своём окне: следующие окна отвечали бы тем же отказом, а их несостоявшиеся
    обращения — та же потеря покрытия, что сорванный ответ. Молчание сорванного
    окна не читается чистотой: его счётчик ложится в `degraded`, иначе прогон, где
    половина окон не отвечена, печатался бы как чистый. Исчерпанные попытки
    дописывает обработчик `pipeline.agent.journey_frictions.subgraph`, и цепочка идёт
    дальше — отказ одной проверки не имеет права лишать прогона остальных пяти.

    :param state: состояние прогона: материалы, кадры и атрибуция пути
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся состав проверок
    :param config: конфигурация запуска графа — в ней имя этого узла
    :return: находки этой проверки в состояние прогона
    """
    frictions = settings.journey_frictions
    node_name = (config.get("metadata") or {}).get("langgraph_node")
    check = check_of_node(frictions, node_name)
    frames = state.kept_frames
    if not frames:
        logger.warning(
            "%s: проверка %s не спрашивается — с анализа снят каждый кадр пути",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
        )
        return frictions_update(check.check_key, CXNodeStatus.EMPTY_RESPONSE)

    groups = check.batches(frames)
    label = LogLabel.CHECK_JOURNEY_FRICTIONS
    answers: list[tuple[CXFrictionFindings, str]] = []
    broken: list[Exception] = []
    for number, group in enumerate(groups, start=1):
        try:
            answers.append(
                await _ask_group(
                    check,
                    frictions,
                    gigachat_provider,
                    group,
                    state,
                    len(frames),
                )
            )
        except Exception as error:
            broken.append(error)
            if not is_shape_failure(error) and not llm_retryable(error):
                # Неповторяемый отказ одинаков на каждом окне: спотыкаться о
                # него оставшейся частью пути незачем.
                logger.warning(
                    "%s: проверка %s остановлена на окне %s из %s (%s)",
                    label,
                    check.check_key,
                    number,
                    len(groups),
                    failure_of(error),
                )
                break

    if not answers:
        error = broken[0]
        if is_shape_failure(error) or llm_retryable(error):
            logger.warning(
                "%s: проверка %s — ни одно из %s окон не отвечено (%s); "
                "обращение к контуру повторится",
                label,
                check.check_key,
                len(groups),
                failure_of(error),
            )
            if is_shape_failure(error):
                raise MalformedAnswer(
                    f"проверка {check.check_key} не отвечена: "
                    f"{failure_of(error)}"
                ) from error
            raise error
        logger.error(
            "%s: Failed to generate response check_journey_friction for "
            "check %s. Exception: %s",
            label,
            check.check_key,
            failure_of(error),
            exc_info=error,
        )
        return frictions_update(check.check_key, CXNodeStatus.ERROR)

    lost = len(groups) - len(answers)
    failed = f"batches_failed:{lost}/{len(groups)}" if lost else ""
    findings = merge_frictions(
        check, [part for part, _ in answers], degraded=failed
    )
    _report(
        check,
        findings,
        len(frames),
        len(state.frames),
        len(groups),
        lost,
        answers[0][1],
    )
    return frictions_update(check.check_key, findings=findings)


async def _ask_group(
    check: CheckJourneyFrictionSettings,
    frictions: JourneyFrictionsSettings,
    gigachat_provider: GigaChatProvider,
    group: list,
    state: AgentStateSchema,
    frames_count: int,
) -> tuple[CXFrictionFindings, str]:
    """Одно окно проверки, спрашиваемое заново, пока сбой лечится повтором.

    Окна идут одно за другим (цикл узла), и политикой графа частичная потеря не
    прикрывается: узел перевызывает себя целиком только когда не отвечено НИ ОДНО
    окно. Пока одно окно отвечено, сорванное молча оставалось бы непросмотренной
    половиной пути — в прогоне 047 так уходило окно у `route_length` и у
    `disclosure`, то есть ровно у тех проверок, что находят самые тяжёлые зоны.

    :param check: настройки проверки
    :param frictions: настройки ветки — собирают вопрос по общим правилам
    :param gigachat_provider: провайдер контура модели
    :param group: кадры этого окна
    :param state: состояние прогона — заметка эксперта, атрибуция и след путей
    :param frames_count: число кадров пути — по нему окно подписано частью
    :return: находки окна и имя отвечавшей модели
    :raises Exception: окно осталось без ответа после всех попыток
    """
    return await repeat_recoverable(
        lambda: _ask_window(
            check,
            frictions,
            gigachat_provider,
            group,
            state,
            frames_count,
        ),
        subject=(
            f"{LogLabel.CHECK_JOURNEY_FRICTIONS}: проверка {check.check_key}, "
            f"окно {group[0].step}-{group[-1].step}"
        ),
    )


async def _ask_window(
    check: CheckJourneyFrictionSettings,
    frictions: JourneyFrictionsSettings,
    gigachat_provider: GigaChatProvider,
    group: list,
    state: AgentStateSchema,
    frames_count: int,
) -> tuple[CXFrictionFindings, str]:
    """Одно окно кадров: обращение к контуру и его разбор.

    Разбор стоит наравне с обращением: ответ без JSON — сбой окна, а не «на этом
    участке пути проблем нет». Сделано это внутри задачи нарочно: сорванное окно не
    имеет права утянуть находки остальных, они уже заплачены.

    :param check: настройки проверки
    :param frictions: настройки ветки — собирают вопрос по общим правилам
    :param gigachat_provider: провайдер контура модели
    :param group: кадры этого окна
    :param state: состояние прогона — заметка эксперта, атрибуция и след путей
    :param frames_count: число кадров пути — по нему окно подписано частью
    :return: находки окна и имя отвечавшей модели
    :raises MalformedAnswer: ответ есть, но он не разобран
    """
    steps = [frame.step for frame in group]
    response: ChatResponse = await gigachat_provider.chat(
        frictions.chat_request(
            check.check_key,
            state.material,
            group,
            state.journey_details,
            route=route_trace(state.frame_readouts),
            total=frames_count,
            values=value_trace(state.frame_readouts, hidden=steps),
            here=value_trace(state.frame_readouts, only=steps),
        )
    )
    answer = response.message.content or ""
    try:
        findings = friction_findings(check, answer, state.frame_readouts)
    except ValueError as ex:
        raise MalformedAnswer(
            f"ответ окна {check.check_key} (кадры {group[0].step}–"
            f"{group[-1].step}) не разобран: {ex}"
        ) from ex
    return findings, response.used_model.name


def _report(
    check: CheckJourneyFrictionSettings,
    findings: CXFrictionFindings,
    asked: int,
    frames_count: int,
    groups: int,
    lost: int,
    model: str,
) -> None:
    """Одной строкой — чем закончилась проверка: разбор, кандидаты, находки.

    Разбор печатается рядом с числом находок нарочно: ноль находок при разборе
    «ноль кадров» и ноль находок при разборе на все экраны — два разных итога, и в
    журнале они обязаны читаться разными строками. Кандидат без находки и без
    причины — то место, где проверка признала проблему и сняла её молча.

    :param check: настройки проверки — для подписи в строке
    :param findings: сведённый итог окон
    :param asked: кадров, ушедших в вопросы
    :param frames_count: число кадров пути
    :param groups: сколько окон спрашивали контур
    :param lost: сколько окон не отвечено — сорвано или не состоялось
    :param model: имя отвечавшей модели
    """
    counts = Counter(item.zone for item in findings.deviations)
    zones = ", ".join(f"{zone}={counts.get(zone, 0)}" for zone in CXZone)
    parts = [
        f"разобрано кадров {len(findings.covered_steps)} из {asked}",
        f"кандидатов {len(findings.candidate_steps)}",
    ]
    if findings.copied_steps:
        parts.append(
            f"кадров переписано с чужого экрана: {len(findings.copied_steps)}"
        )
    if check.units:
        parts.append(
            f"разобрано мер "
            f"{len(check.units) - len(findings.unchecked_units)}"
            f" из {len(check.units)}"
        )
    if findings.units_with_finding:
        parts.append(
            f"мер с находкой: {', '.join(findings.units_with_finding)}"
        )
    if findings.units_without_deviation:
        parts.append(
            "мер названа находкой без отклонения: "
            f"{', '.join(findings.units_without_deviation)}"
        )
    if findings.units_norm_with_shortfall:
        parts.append(
            "мера свёрнута нормой там, где её же строка назвала нехватку: "
            f"{', '.join(findings.units_norm_with_shortfall)}"
        )
    if findings.units_without_frame:
        parts.append(
            f"мер сведено без кадра: {', '.join(findings.units_without_frame)}"
        )
    if findings.units_reused_evidence:
        parts.append(
            "мер сведено одним основанием: "
            f"{', '.join(findings.units_reused_evidence)}"
        )
    if findings.candidates_without_finding:
        parts.append(
            f"кандидатов без находки "
            f"{len(findings.candidates_without_finding)}"
        )
    series = findings.dropped.get(DROP_SERIES_MERGED, 0)
    if series:
        parts.append(f"повторов рядом свёрнуто: {series}")
    phantoms = findings.dropped.get(DROP_PHANTOM_QUOTE, 0)
    if phantoms:
        parts.append(f"цитат, которых нет на подписанных кадрах: {phantoms}")
    if findings.unchecked_units:
        parts.append(f"мер без разбора: {', '.join(findings.unchecked_units)}")
    if findings.degraded:
        parts.append(findings.degraded)
    why = findings.emptiness_note()
    logger.info(
        "%s: проверка %s, кадров %s, окон %s (не отвечено %s), проблем %s (%s); "
        "%s%s, модель %s",
        LogLabel.CHECK_JOURNEY_FRICTIONS,
        check.check_key,
        f"{asked} из {frames_count}",
        groups,
        lost,
        len(findings.deviations),
        zones,
        "; ".join(parts),
        f" — {why}" if why else "",
        model,
    )


@node
async def map_journey_frictions(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Находки шести проверок — размеченные кодами справочника.

    Проверки описывают видимое и не знают адресов: здесь те же находки, уже
    текстом, ставятся в справочник — совпало описание с дефектом, находка получает
    его код, блок и зону; не совпало — остаётся внекаталожной, со своей зоной и без
    кода. Мимо справочника находка не выбрасывается: найденное на пути дороже, чем
    удобство сверенного со словарём перечня (`pipeline.utils.journey_mapping`).

    Разметка молчит, когда молчали проверки: находок нет — спрашивать не о чем.
    Сбой разметки находок не трогает: узел не поднимает его наверх нарочно, потому
    что обработчик ветки деградирует весь её результат, а терять здесь нечего —
    прогон остаётся с сырыми находками, у которых нет кода, но есть текст, улика и
    экран. Без кода их берёт на себя сведение (`defer`-проход), поэтому отказ
    разметки стоит прогону адреса, а не находок.

    :param state: состояние прогона: находки шести проверок и атрибуция пути
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся эта ветка
    :return: размеченные находки проверок в состоянии прогона
    """
    frictions = settings.journey_frictions
    label = LogLabel.MAP_JOURNEY_FRICTIONS
    groups = [
        (check.check_key, found)
        for check in frictions.checks
        if (found := state.journey_frictions.get(check.check_key)) is not None
        and found.deviations
    ]
    if not groups:
        why = (
            "проверок проблем пути в прогоне не было"
            if not state.journey_frictions
            else "ни одна проверка не назвала ни одной находки"
        )
        logger.info("%s: разметка не запускается — %s", label, why)
        return {}

    titles = [
        (_check_title(frictions, key), found.deviations)
        for key, found in groups
    ]
    try:
        marked, model = await _ask_mapping(
            settings.map_journey_frictions,
            gigachat_provider,
            titles,
            groups,
            state,
        )
    except Exception as error:
        logger.error(
            "%s: Failed to generate response map_journey_frictions. "
            "Exception: %s",
            label,
            failure_of(error),
            exc_info=error,
        )
        return _frictions_update(
            {
                key: found.model_copy(
                    update={
                        "degraded": _marks(
                            found.degraded,
                            f"mapping_failed:{type(error).__name__}",
                        )
                    }
                )
                for key, found in groups
            }
        )

    total = sum(len(found.deviations) for _, found in groups)
    matched = sum(
        1 for part in marked.values() for item in part.deviations if item.code
    )
    left = _unmapped(marked)
    logger.info(
        "%s: находок %s, с кодом %s, без кода %s%s, модель %s",
        label,
        total,
        matched,
        total - matched,
        "".join(f"; {key} {mark}" for key, mark in left.items()),
        model,
    )
    return _frictions_update(
        {
            key: part.model_copy(
                update={"degraded": _marks(part.degraded, left.get(key, ""))}
            )
            for key, part in marked.items()
        }
    )


def _check_title(frictions: JourneyFrictionsSettings, check_key: str) -> str:
    """Подпись проверки в перечне находок: ключ и её предмет.

    Ключа одного мало: разметка сверяет смысл, а «disclosure» ничего не говорит о
    том, что проверялось, — предмет берётся из того же конфига, что задаёт вопрос
    проверки. Пустое имя в конфиге — обычное дело для новой проверки, тогда остаётся
    один ключ.

    :param frictions: настройки ветки — в них имена проверок
    :param check_key: ключ проверки, чьи находки подписываются
    :return: строка `ключ — предмет` либо один ключ
    """
    name = frictions.check(check_key).name.strip()
    return f"{check_key} — {name}" if name else check_key


def _marks(*parts: str) -> str:
    """Строка `degraded` из отметок: новая отметка не стирает старую.

    :param parts: отметки, пустые пропускаются
    :return: отметки, пустая строка — если отметок не было
    """
    return "; ".join(part for part in parts if part)


def _unmapped(marked: dict[str, CXFrictionFindings]) -> dict[str, str]:
    """Сколько находок у каждой проверки осталось без кода.

    Дробь считается по проверке: общая «6 из 40» скрыла бы проверку, которая не
    размечилась вовсе, за спинами тех, где кода нашли всем. Внекаталожная находка —
    штатный итог сверки с каталогом, поэтому отметка видна в журнале и в `degraded`,
    но деградацией прогона не считается (тот же смысл, что `unmatched:` у поэкранной
    разметки).

    :param marked: размеченные находки проверок
    :return: отметка `unmatched:i/n` по каждой проверке, где остались без кода
    """
    left: dict[str, str] = {}
    for key, part in marked.items():
        count = sum(1 for item in part.deviations if not item.code)
        if count:
            left[key] = f"{UNMATCHED_PREFIX}{count}/{len(part.deviations)}"
    return left


def _frictions_update(findings: dict[str, CXFrictionFindings]) -> dict:
    """Размеченные проверки — в состояние: чужие ключи остаются нетронутыми.

    :param findings: находки размеченных проверок
    :return: запись в состояние прогона
    """
    return {"journey_frictions": findings}


async def _ask_mapping(
    mapping: MapJourneyFrictionsSettings,
    gigachat_provider: GigaChatProvider,
    titles: Sequence[tuple[str, Sequence[CXDeviation]]],
    groups: list[tuple[str, CXFrictionFindings]],
    state: AgentStateSchema,
) -> tuple[dict[str, CXFrictionFindings], str]:
    """Разметка находок, спрашиваемая заново, пока сбой лечится повтором.

    Разбор входит в повторяемое обращение: ответ без перечня `matches` — не «кода
    не нашлись», а неразобранный ответ, и спрашивается он заново на месте.

    :param mapping: настройки разметки
    :param gigachat_provider: провайдер контура модели
    :param titles: перечень находок — тем же порядком, что и `groups`
    :param groups: `(ключ проверки, её находки)` — на них раскладывается ответ
    :param state: состояние прогона — атрибуция пути
    :return: размеченные находки проверок и имя отвечавшей модели
    :raises Exception: разметка осталась без ответа после всех попыток
    """
    return await repeat_recoverable(
        lambda: _ask_reference(
            mapping, gigachat_provider, titles, groups, state.journey_details
        ),
        subject=(
            f"{LogLabel.MAP_JOURNEY_FRICTIONS}: "
            f"{sum(len(items) for _, items in titles)} находок проверок"
        ),
    )


async def _ask_reference(
    mapping: MapJourneyFrictionsSettings,
    gigachat_provider: GigaChatProvider,
    titles: Sequence[tuple[str, Sequence[CXDeviation]]],
    groups: list[tuple[str, CXFrictionFindings]],
    details: Optional[CXJourneyDetails],
) -> tuple[dict[str, CXFrictionFindings], str]:
    """Одно обращение разметки: полный справочник, перечень находок, разбор.

    Кадров здесь нет нарочно: справочник читают по описанию находки, а не по
    картинке, и десять изображений на каждое обращение добавили бы в вопрос только
    цену. Разбором ведает `pipeline.utils.journey_mapping`: он ставит код, блок и зону
    справочника и не трогает ни текст находки, ни её номер экрана.

    :param mapping: настройки разметки
    :param gigachat_provider: провайдер контура модели
    :param titles: перечень находок по проверкам
    :param groups: находки проверок — на них раскладывается ответ
    :param details: атрибуция пути — она подписывает вопрос
    :return: размеченные находки и имя отвечавшей модели
    :raises MalformedAnswer: ответ есть, но он не разобран
    """
    response: ChatResponse = await gigachat_provider.chat(
        mapping.chat_request(titles, details)
    )
    try:
        marked = friction_mapping(
            response.message.content or "",
            groups,
            mapping.load_criteria_model(),
        )
    except ValueError as ex:
        raise MalformedAnswer(
            f"ответ разметки проверок пути не разобран: {ex}"
        ) from ex
    return marked, response.used_model.name
