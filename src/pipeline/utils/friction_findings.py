"""Разбор ответа проверки проблем пути: отклонения и чем заполнится состояние.

Здесь и то, как ответ проверки ложится в состояние прогона: узел говорит с
контуром (`pipeline.agent.journey_frictions.nodes`), а форму записи выбирают эти
функции — ту же, что и находки блоков (`pipeline.utils.block_findings`): отклонение
одной формы (`pipeline.schemas.nodes.CXDeviation`), контейнер одной формы
(`pipeline.schemas.nodes.CXCheckFindings`). Сведение поэтому получает находки обеих
веток одним перечнем и дедуплицирует их между собой, а не два несходных списка.

Ответу прогона они же дают список трений: состояние graph-слой не покидает, а
`Friction` из ответа собирает `pipeline.dto.agent.result.AgentResult` по этим правилам.

Форма строга, значения мягкие. Отсутствующий перечень `deviations` — сбой формы, а
не «ноль проблем»: молчаливый пустой список означал бы, что проверка прошла
чисто, хотя модель просто ответила не тем словом. Невозможная зона стоит только
самой находке: она мягчает, остальные остаются.

Адреса у находки на этом этапе нет вовсе — ни блока, ни кода дефекта. Разбор не
сверяет находку со справочником и не спрашивает у неё раздела: адрес ставит
разметчик после всех проверок (`pipeline.utils.journey_mapping`), и ни выдуманный
раздел, ни отсутствие раздела не снимает находки — найти отклонение важнее, чем
заранее решить, заслуживает ли оно места в отчёте.
"""

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from src.pipeline.schemas.nodes import (
    CXDeviation,
    CXFrameObservation,
    CXFrameReadout,
    CXFrictionFindings,
    CXUnitReview,
)
from src.pipeline.schemas.result import Friction
from src.pipeline.scoring.zones import zone_to_severity
from src.pipeline.settings.api.graph import CheckJourneyFrictionSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    DROP_BAD_STEP,
    DROP_NOT_OBJECT,
    DROP_PHANTOM_QUOTE,
    DROP_SERIES_MERGED,
    DROP_UNKNOWN_CODE,
    PATH_EXCUSE,
    SELF_DISMISSED_MARK,
    UNIT_FINDING,
    UNIT_NA,
    UNIT_NORM,
    UNIT_WORDS,
    CXNodeStatus,
    CXSeverity,
    CXZone,
)
from src.pipeline.utils.deviations import quotes_of, step_of, text_of
from src.pipeline.utils.evidence import frame_texts, verified_quotes
from src.pipeline.utils.json_reply import json_payload

logger = logging.getLogger(__name__)


def friction_findings(
    check: CheckJourneyFrictionSettings,
    answer: str,
    readouts: Sequence[CXFrameReadout] = (),
) -> CXFrictionFindings:
    """Ответ проверки в форму отклонений пути.

    Отброшенное не исчезает молча: причина считается в `dropped` найденного
    контейнера. Находка здесь не отбрасывается вовсе: блок, который проверка могла
    бы назвать, ей неизвестен, а выдуманная подпись раздела не имеет права стоить
    находки. В прогоне 014 проверка навигации принесла 17 отклонений, 10 из них
    были подписаны чужим блоком и исчезали молча — теперь адрес вообще не у дел до
    той поры, пока его не поставит разметчик.

    Разбор кадров (`observations`) читается рядом с находками и не мешая им: по
    нему видно, до скольких экранов дошла проверка и какие меры она не назвала ни
    на одном кадре. Пустой перечень при пустом разборе значит «не смотрели», а не
    «чисто», — и остаётся видимым в состоянии и в журнале.

    С `readouts` сюда приходит ещё одна проверка — подлинность улики: цитата
    обязана читаться на кадре, которому она себя приписала (`pipeline.utils.evidence`).
    Проверка отвечает картинками и подписывает кадр номером из вопроса, а скан уже
    переписал с этого кадра текст дословно. В прогоне 047 доверие поставило `black`
    на пару «кадр 3: „Комиссия до 1%" — кадр 6: списание 410 ₽ с 25 000 ₽», которой
    в материале нет: кадр 3 был экраном выбора группы инвалидности, а величин
    распознавание не сняло с пути ни одной. Без этой сверки выдумка держит чёрную
    зону — проверке довольно непустого перечня улик.

    :param check: настройки проверки — её меры и имя для журнала
    :param answer: ответ модели
    :param readouts: описания кадров пути — по ним сверяются цитаты; пусто — сверять
        нечем, улики остаются как пришли
    :return: находки этой проверки
    """
    payload = json_payload(answer, "с перечнем проблем пути")
    raw = payload.get("deviations")
    if not isinstance(raw, list):
        if payload.get("observations") or payload.get("units"):
            logger.warning(
                "%s: проверка %s вернула разбор без перечня deviations — "
                "отклонений по этому окну нечитается",
                LogLabel.CHECK_JOURNEY_FRICTIONS,
                check.check_key,
            )
            raw = []
        else:
            raise ValueError(
                f"в ответе проверки {check.check_key} нет ни перечня deviations,"
                " ни разбора кадров"
            )

    dropped: Counter[str] = Counter()
    texts = frame_texts(readouts)
    found: list[CXDeviation] = []
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        found.append(_deviation(check, item, texts, dropped))

    observations = _observations(check, payload.get("observations"), dropped)
    reviews = _unit_reviews(
        check,
        payload.get("units") or payload.get("unit_reviews"),
        dropped,
    )
    missing = _missing_units(check, observations, reviews)
    if missing:
        logger.warning(
            "%s: проверка %s не разобрала меры %s — по ответу они не смотрены",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            ", ".join(missing),
        )

    note = text_of(payload.get("note"))
    if not found and PATH_EXCUSE.search(note):
        logger.warning(
            "%s: проверка %s объяснила пустоту тем, чем путь является (%s), а "
            "не материалом — проверка по кадрам этого пути не состоялась",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            note.strip()[:120],
        )
        note = f"{note.strip()} — {SELF_DISMISSED_MARK}"

    findings = CXFrictionFindings(
        source_key=check.check_key,
        deviations=found,
        note=note,
        dropped=dict(dropped),
        observations=observations,
        unit_reviews=reviews,
        unchecked_units=missing,
    )
    if findings.unexplained:
        logger.warning(
            "%s: проверка %s вернула пустой перечень без объяснения — чем она "
            "проверяла путь, из ответа не читается%s",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            _losses(findings.dropped),
        )
    return findings


def _observations(
    check: CheckJourneyFrictionSettings, raw: object, dropped: Counter[str]
) -> list[CXFrameObservation]:
    """Разбор кадров: что модель видела на каждом экране и к чему это отнесла.

    Форма мягкая: перечня может не быть вовсе (контур ответил по-старому), и это не
    сбой ответа — это непроверенный путь, который видно по нулевому покрытию. Запись
    без номера экрана бесполезна — её не к чему приклеить, — и уходит в счётчик.

    :param check: настройки проверки — для лога
    :param raw: значение `observations` из ответа
    :param dropped: счётчик потерь разборщика, пополняется здесь
    :return: разборы кадров в порядке ответа
    """
    if not isinstance(raw, list):
        return []

    seen: list[CXFrameObservation] = []
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        step = step_of(item.get("step"))
        if step <= 0:
            dropped[DROP_BAD_STEP] += 1
            continue
        seen.append(
            CXFrameObservation(
                step=step,
                seen=text_of(item.get("seen")),
                values=[
                    value
                    for value in (
                        text_of(part) for part in quotes_of(item.get("values"))
                    )
                    if value
                ],
                unit=text_of(item.get("unit")).upper().replace(" ", ""),
                candidate=bool(item.get("candidate")),
                why=text_of(item.get("why")),
            )
        )
    return seen


def _unit_reviews(
    check: CheckJourneyFrictionSettings, raw: object, dropped: Counter[str]
) -> list[CXUnitReview]:
    """Разбор по меркам проверки: одна строка на каждую меру всего пути.

    Подпись кадра шифром показывает итог экрана и ничего не говорит о том, видел ли
    проверку другой экран: на пути из двадцати четырёх кадров модель ставит первую
    меру на все кадры и считает проверку сданной. Здесь итог держит сама мера, и
    прогон видит, что меру сняли, — а не то, что её не было.

    :param check: настройки проверки — для лога
    :param raw: значение `unit_reviews` из ответа
    :param dropped: счётчик потерь разборщика, пополняется здесь
    :return: разборы мер в порядке ответа
    """
    if not isinstance(raw, list):
        return []

    seen: list[CXUnitReview] = []
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        unit = text_of(item.get("unit")).upper().replace(" ", "")
        if not unit:
            dropped[DROP_UNKNOWN_CODE] += 1
            continue
        seen.append(
            CXUnitReview(
                unit=unit,
                frames=sorted(
                    {
                        step
                        for step in (
                            step_of(part)
                            for part in quotes_of(item.get("frames"))
                        )
                        if step > 0
                    }
                ),
                result=_unit_result(item),
                why=text_of(item.get("why")),
            )
        )
    return seen


def _unit_result(item: dict) -> str:
    """Итог меры одним из трёх значений: контур пишет его своими словами.

    «Нет нарушений» содержит слово «нарушений», и прочитанное по одному этому слову
    выглядело бы находкой — поэтому отрицание читается раньше: снятая мера остаётся
    снятой.
    """
    unit_negation = ("нет", "не ", "отсутств", "не обнаруж", "не видно")
    named = text_of(item.get("result")).casefold().replace("_", " ")

    if item.get("candidate") and not named:
        return UNIT_FINDING
    for result, words in UNIT_WORDS.items():
        if not any(word in named for word in words):
            continue
        if result == UNIT_FINDING and any(
            word in named for word in unit_negation
        ):
            return UNIT_NORM
        return result
    return ""


def _missing_units(
    check: CheckJourneyFrictionSettings,
    observations: list[CXFrameObservation],
    reviews: list[CXUnitReview],
) -> list[str]:
    """Меры проверки, о которых ответ не сказал ничего.

    Мера считается смотренной по любому из двух следов: разборе самой меры или
    подписи кадра. Пустой перечень мер у проверки снимает сверку — ей нечего искать.

    :param check: настройки проверки — её меры
    :param observations: разборы кадров этого окна
    :param reviews: разборы мер этого окна
    :return: шифры мер, которые ответ не назвал
    """
    named = {item.unit for item in reviews} | {
        item.unit for item in observations
    }
    return [unit for unit in check.units if unit not in named]


def merge_frictions(
    check: CheckJourneyFrictionSettings,
    parts: list[CXFrictionFindings],
    degraded: str = "",
) -> CXFrictionFindings:
    """Ответы окон пути в один итог проверки.

    Номера кадров модель берёт из подписей `[кадр N]`, а подписывает их узел
    глобальными номерами: окна отвечают об одних и тех же экранах без смещения, и
    находки групп просто складываются. Повтором считается только повтор одного и
    того же экрана — тот же кадр, названный вторым окном (см. `_series_of`):
    соседний экран — уже другое место пути и другая находка, иначе прогон теряет
    находки пачками. Разбор окон объединяется по кадрам: кадр остаётся с той
    записью, где он стал кандидатом.

    :param check: настройки проверки — её ключ и меры
    :param parts: итоги разобранных окон в порядке пути
    :param degraded: чем плох прогон окон (`batches_failed:i/j`)
    :return: общий итог проверки по всем окнам
    """
    deviations: list[CXDeviation] = []
    notes: dict[int, CXFrameObservation] = {}
    units: dict[str, CXUnitReview] = {}
    dropped: Counter[str] = Counter()
    explained: list[str] = []
    for part in parts:
        dropped.update(part.dropped)
        for deviation in part.deviations:
            kept = _series_of(deviations, deviation)
            if kept is None:
                deviations.append(deviation)
                continue
            dropped[DROP_SERIES_MERGED] += 1
            deviations[deviations.index(kept)] = _merge_series(kept, deviation)
        for item in part.observations:
            kept = notes.get(item.step)
            if kept is None or (item.candidate and not kept.candidate):
                notes[item.step] = item
        for item in part.unit_reviews:
            units[item.unit] = _merge_unit(units.get(item.unit), item)
        note = part.note.strip()
        if note and note not in explained:
            explained.append(note)

    seen = [notes[step] for step in sorted(notes)]
    reviews = [units[unit] for unit in sorted(units)]
    return CXFrictionFindings(
        source_key=check.check_key,
        deviations=deviations,
        note="; ".join(explained),
        dropped=dict(dropped),
        observations=seen,
        unit_reviews=reviews,
        unchecked_units=_missing_units(check, seen, reviews),
        degraded=degraded,
    )


def _series_key(deviation: CXDeviation, series_key_part: int = 44) -> str:
    """Узнаватель серии: конец формы текста без названных величин и без адреса.

    Жалоба стоит в конце строки («… → неясно, что произойдёт»), а в начале —
    названные кнопка и экран, которые как раз и меняются от кадра к кадру: «Далее»,
    «Продолжить», «Редактировать» при одной и той же беде. Форма берётся с конца, а
    не с начала — именно поэтому. Пустая форма не сливается ни с чем: сравнить
    нечего.

    Блок в узнаватель не входит: на этом шаге у находки его ещё нет — адрес
    появляется после разметки, а окна одной проверки сводятся до неё.

    :param deviation: находка
    :return: хвост формы текста, по которому узнаётся серия
    """
    series_quoted = re.compile(r"[«\"'][^«\"'»]{1,40}[»\"']")
    series_words = re.compile(r"[^а-яёa-z]+")
    shape = series_quoted.sub(" ", deviation.text.casefold())
    return series_words.sub(" ", shape).strip()[-series_key_part:]


def _series_of(
    deviations: list[CXDeviation],
    deviation: CXDeviation,
) -> CXDeviation | None:
    """Найденная раньше находка того же экрана, если она есть.

    Повтором считается один и тот же кадр, названный двумя окнами на стыке: окна
    перекрываются подписями, и экран 12 из конца одного окна — тот же экран 12 из
    начала следующего. Соседний экран повтором не считается и остаётся отдельной
    находкой: «нет кнопки отмены» на входе в приложение и перед подписанием
    договора решаются по-разному, и одна находка на двоих стоит прогону семи
    отклонений (прогон 014: 14 отклонений стало двумя, и журнал назвал это
    «серия свёрнута»).

    :param deviations: отклонения, уже собранные сводкой
    :param deviation: разбираемое отклонение
    :return: находка того же экрана или None
    """
    shape = _series_key(deviation)
    if not shape:
        return None
    for found in deviations:
        same_screen = deviation.step > 0 and found.step == deviation.step
        if same_screen and _series_key(found) == shape:
            return found
    return None


def _merge_series(
    kept: CXDeviation,
    item: CXDeviation,
    max_series_evidence: int = 8
) -> CXDeviation:
    """Вторая запись ряда легла в первую: остаётся худшая зона и оба кадра.

    Экран выбирается худшим из ряда — на него и смотреть: если одна и та же мелочь
    названа на соседних экранах одного окна, находка обязана стоять там, где цена
    выше. Кадры ряда перечислены в цитатах.

    :param kept: находка ряда, уже стоящая в итоге
    :param item: ещё одна запись того же ряда
    :return: общая находка ряда
    """
    kept_zone = CXZone.parse(kept.zone) or CXZone.YELLOW
    item_zone = CXZone.parse(item.zone) or CXZone.YELLOW
    winner = kept if kept_zone.at_least(item_zone) else item
    related = winner.related_step or kept.related_step or item.related_step
    quotes = list(dict.fromkeys(kept.evidence + item.evidence))
    return winner.model_copy(
        update={
            "related_step": related,
            "evidence": quotes[:max_series_evidence],
        }
    )


def _merge_unit(kept: CXUnitReview | None, item: CXUnitReview) -> CXUnitReview:
    """Одна мера в двух окнах пути: остаётся сильный итог и оба набора кадров.

    Окно отвечает только за свои кадры, поэтому «норма» здесь значит «на этих кадрах
    ничего», а не «мера снята»: если та же мера в другом окне дала находку, находка
    и остаётся итогом меры по пути.

    :param kept: что о мере уже известно из прежних окон, если известно
    :param item: разбор этой меры в новом окне
    :return: общий разбор меры
    """
    if kept is None:
        return item
    strong = {UNIT_FINDING: 2, UNIT_NORM: 1, UNIT_NA: 0, "": -1}
    winner = item if strong[item.result] > strong[kept.result] else kept
    reasons = [reason for reason in (kept.why, item.why) if reason]
    return CXUnitReview(
        unit=kept.unit,
        frames=sorted(set(kept.frames) | set(item.frames)),
        result=winner.result,
        why="; ".join(dict.fromkeys(reasons)),
    )


def _losses(dropped: dict[str, int]) -> str:
    """Хвост строки журнала: сколько отклонений разбор выбросил и по какой причине.

    :param dropped: счётчик потерь разборщика
    :return: пустая строка, если терять было нечего
    """
    if not dropped:
        return ""
    counts = ", ".join(f"{why}={n}" for why, n in sorted(dropped.items()))
    return f" — и то не дошли все: отброшено ({counts})"


def frictions_update(
    check_key: str,
    status: CXNodeStatus = CXNodeStatus.SUCCESS,
    findings: CXFrictionFindings | None = None,
) -> dict:
    """Одна проверка — остальное в состоянии остаётся нетронутым.

    :param check_key: ключ проверки, чьи находки пишутся
    :param status: чем закончился узел, когда находок нет
    :param findings: находки проверки — иначе на её ключ ложится только статус
    :return: запись в состояние прогона
    """
    if findings is None:
        findings = CXFrictionFindings(source_key=check_key, status=status)
    return {"journey_frictions": {check_key: findings}}


def frictions_out(
    checks: dict[str, CXFrictionFindings], frames_count: int
) -> list[Friction]:
    """Находки шести проверок — в список трений ответа прогона.

    Порядок ответа повторяет порядок проверок: цепочка собирается конфигом и идёт
    последовательно, поэтому находки приходят к отчёту в том же порядке, в каком
    их читает стрим прогона.

    Зона уходит зоной находки, а `severity` выводится из неё теми же весами, что и
    у находок блоков (`pipeline.utils.findings`): слово модели о тяжести калибровке
    проверки не перебивает, иначе одна и та же зона покраснила бы отчёт по-разному
    в двух ветках. Код и блок — те, что поставил разметчик; у незаматченной находки
    их нет, и трение остаётся внекаталожным — но подписывается своей проверкой, чем
    и отличается от «адрес не удалось найти» и от находки блока. Номер экрана
    становится индексом кадра, как в `_steps`: улика за пределами пути кадра не
    имеет и уходит без привязки.

    :param checks: находки проверок, ключ — ключ проверки
    :param frames_count: число кадров пути — по нему отсекаются номера экранов
    :return: трения пути для ответа прогона
    """
    frictions: list[Friction] = []
    for check in checks.values():
        if CXNodeStatus(check.status) is not CXNodeStatus.SUCCESS:
            continue
        for found in check.deviations:
            text = found.text.strip()
            if not text:
                continue
            zone = found.zone
            frictions.append(
                Friction(
                    step=_screen(found.step, frames_count),
                    zone=zone,
                    severity=zone_to_severity(zone) or CXSeverity.MID,
                    criterion=found.criterion or check.source_key,
                    code=found.code,
                    text=text,
                    rec=found.rec.strip(),
                    evidence="; ".join(found.evidence),
                )
            )
    return frictions


def _screen(step: int, frames_count: int) -> int:
    """Экран находки в индекс кадра ответа; -1 — кадр не назван или его нет.

    :param step: номер экрана из подписи `[кадр N]`, 1-based; 0 — не назван
    :param frames_count: число кадров пути
    :return: 0-based индекс кадра или -1 без привязки
    """
    if not 1 <= step <= frames_count:
        return -1
    return step - 1


def _deviation(
    check: CheckJourneyFrictionSettings,
    item: dict,
    texts: Mapping[int, str],
    dropped: Counter[str],
) -> CXDeviation:
    """Одно отклонение: зона сверяется с перечнем, улика — с кадром, адрес не смотрят.

    Кода и блока у находки нет, и поле `criterion` из ответа не читается: проверка
    не знает справочника, поэтому её подпись раздела была бы или выдумкой, или
    подсказкой из прошлого ревизии каталога. Адрес находке поставит разметчик
    (`pipeline.utils.journey_mapping`) — по всему справочнику, а не по одному разделу.
    Молча выбрасывать из-за этого находку значило бы терять найденное: проверка
    держит в себе предмет, а не право на штраф.

    Цитата, которой нет на подписанном кадре, уходит из `evidence` счётчиком, а не
    остаётся доказательством, и на этом основании зона находки снижается: находка с
    выдуманной уликой не может стоить прогону чёрный пояс того блока, которому её
    положит разметчик. Саму находку гейт не снимает — кандидат остался разобран,
    эксперту его показывают жёлтым.

    :param check: настройки проверки — её калибровка зон и ключ для журнала
    :param item: элемент перечня `deviations` из ответа модели
    :param texts: тексты кадров пути по номерам (`frame_texts`)
    :param dropped: счётчик потерь разборщика, пополняется здесь
    :return: отклонение без адреса — с текстом, проверенной уликой и зоной
    """
    step = step_of(item.get("step"))
    evidence, phantom = verified_quotes(
        quotes_of(item.get("evidence")), texts, default=step
    )
    if phantom:
        dropped[DROP_PHANTOM_QUOTE] += len(phantom)
        logger.warning(
            "%s: проверка %s назвала кадр %s цитатой, которой на нём нет (%s) — "
            "улика снята, тяжесть снижена",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            step or "не назван",
            "; ".join(part[:60] for part in phantom),
        )
    return CXDeviation(
        zone=_zone_of(
            check, item.get("zone"), bool(evidence), unproven=bool(phantom)
        ),
        step=step,
        related_step=step_of(item.get("related_step")),
        severity=text_of(item.get("severity")),
        text=text_of(item.get("text")),
        rec=text_of(item.get("rec")),
        evidence=evidence,
    )


def _zone_of(
    check: CheckJourneyFrictionSettings,
    raw: object,
    has_evidence: bool,
    unproven: bool = False,
) -> CXZone:
    """Зона находки: неизвестная — `yellow`, `black` без цитаты — `red`.

    Калибровка тяжести в промпте обещает, что худшая зона ставится только с
    дословной цитатой. Слово модели исполняет здесь код: зона решает штраф, а
    цена выдуманной улики — весь балл блока, которому легла находка.

    `unproven` значит, что цитата у находки была и оказалась выдумкой: улику сняла
    сверка с кадром (`pipeline.utils.evidence`). Такая находка не выше `yellow` — ровно
    той зоной промпт отвечает на «кандидат виден, а надпись кадра не переписать
    дословно». Разница с `black` без цитаты в том, что там модель честно не нашла
    что цитировать, а здесь назвала то, чего на экране нет.

    :param check: настройки проверки — для лога
    :param raw: зона из ответа модели
    :param has_evidence: есть ли хоть одна цитата, прочитанная на кадре
    :param unproven: все названные цитаты на подписанном кадре не читались
    :return: зона находки
    """
    try:
        zone = CXZone(str(raw or "").strip().lower())
    except ValueError:
        logger.warning(
            "%s: проверка %s прислала зону %r вне перечня — считаем yellow",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            raw,
        )
        return CXZone.YELLOW

    if unproven and not has_evidence and zone in (CXZone.BLACK, CXZone.RED):
        logger.warning(
            "%s: проверка %s держала зону %s цитатой, которой на кадре нет — "
            "снижаем до yellow",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            zone.value,
        )
        return CXZone.YELLOW

    if zone is CXZone.BLACK and not has_evidence:
        logger.warning(
            "%s: проверка %s поставила black без дословной цитаты — снижаем до red",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
        )
        return CXZone.RED

    if unproven and not has_evidence and zone in (CXZone.BLACK, CXZone.RED):
        logger.warning(
            "%s: проверка %s держала зону %s цитатой, которой на кадре нет — "
            "снижаем до yellow",
            LogLabel.CHECK_JOURNEY_FRICTIONS,
            check.check_key,
            zone.value,
        )
        return CXZone.YELLOW

    return zone
