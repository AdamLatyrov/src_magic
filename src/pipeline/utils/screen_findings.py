"""Разбор ответа поэкранного прохода: описания кадров, находки с пинами и секция `screens`.

Здесь и то, как ответ прохода ложится в состояние прогона: узел говорит с контуром
(`pipeline.agent.screen.nodes`), а форму записи выбирают эти функции — той же формой,
что и находки блоков (`pipeline.utils.block_findings`), проверок проблем пути
(`pipeline.utils.friction_findings`) и прохода автозаполнения
(`pipeline.utils.autofill_findings`): отклонение одной формы
(`pipeline.schemas.nodes.CXPinnedDeviation`), контейнер одной формы
(`pipeline.schemas.nodes.CXScreenFindings`). Сведение поэтому видит находки четырёх
веток одним перечнем и дедуплицирует их между собой: одна и та же комиссия,
названная блоком `cxv3:5`, проверкой `disclosure` и этим проходом, штрафуется
один раз.

Отличие прохода — в том, что у него два выхода. Отклонения идут в сведение и в
балл, а описания кадров (`summary`) в сведение не идут: они питают секцию `screens`
ответа прогона (`pipeline.schemas.result.ScreenOut`), где рисуются вместе с пинами.
Держать их в одном перечне отклонений значило бы штрафовать прогон за описание
экрана.

Форма строга, значения мягкие. Отсутствующий перечень `screens` — сбой формы, а не
«путь чист»: молчаливый пустой список означал бы, что кадры разобрали, хотя модель
ответила не тем словом. Окно спрашивается ради одного кадра, поэтому элемент ответа
про другой экран снимается счётчиком: тот кадр будет разобран своим обращением, и
второй ответ о нём — самоповтор с другими координатами. Находка без текста стоит
только самой находке: она отбрасывается со счётчиком в `dropped`, остальные
остаются, а описания кадров не теряются вместе с ними.

Адреса у находки здесь нет: ни блока, ни кода дефекта проход не выбирает — он
смотрит на экран и описывает то, что мешает клиенту. Код, блок и зону по справочнику
находке даёт следующий этап (`pipeline.utils.screen_mapping`), когда находки уже есть.
"""

import logging
from collections import Counter

from src.pipeline.schemas.nodes import (
    CXPinnedDeviation,
    CXScreenFindings,
    CXScreenNote,
)
from src.pipeline.schemas.result import ScreenFinding, ScreenOut
from src.pipeline.scoring.zones import zone_to_severity
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    DROP_BAD_STEP,
    DROP_NOT_OBJECT,
    PIN_CENTER,
    CXNodeStatus,
    CXSeverity,
    CXZone,
)
from src.pipeline.utils.deviations import pct_of, quotes_of, step_of, text_of
from src.pipeline.utils.json_reply import json_payload

logger = logging.getLogger(__name__)


def screen_findings(
    answer: str, frames_count: int, target_step: int
) -> CXScreenFindings:
    """Ответ про окно кадра — в находки пути и описание разбираемого кадра.

    Окно спрашивается ради одного кадра, и элемент ответа про соседа снимается целым
    элементом: этот сосед будет разобран своим обращением, а два ответа об одном
    экране дали бы две находки с разными координатами и разной тяжестью. Кадр,
    которого в пути нет, не бывает уликой: такое отклонение снимается, а описание
    того же кадра — нет, иначе один выдуманный номер экрана стоил бы прогону
    описания всех разобранных экранов.

    :param answer: ответ модели
    :param frames_count: сколько кадров в пути — за его пределами экрана нет
    :param target_step: разбираемый кадр этого обращения
    :return: находки и описания прохода
    :raises ValueError: если в ответе нет перечня `screens`
    """
    payload = json_payload(answer, "с перечнем экранов")
    raw = payload.get("screens")
    if not isinstance(raw, list):
        raise ValueError("в ответе поэкранного прохода нет перечня screens")

    dropped: Counter[str] = Counter()
    found: list[CXPinnedDeviation] = []
    notes: list[CXScreenNote] = []
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        step = step_of(item.get("step"))
        if not 1 <= step <= frames_count:
            dropped[DROP_BAD_STEP] += 1
            continue
        if step != target_step:
            dropped["answer_about_context_frame"] += 1
            logger.warning(
                "%s: кадр %s отвечен вместо разбираемого %s — элемент снят, "
                "у этого экрана будет своё обращение",
                LogLabel.CHECK_SCREEN,
                step,
                target_step,
            )
            continue
        summary = text_of(item.get("summary")).strip()
        notes.append(CXScreenNote(step=step, summary=summary))
        for finding in item.get("findings") or []:
            if not isinstance(finding, dict):
                dropped[DROP_NOT_OBJECT] += 1
                continue
            deviation = _deviation(finding, step, dropped)
            if deviation is not None:
                found.append(deviation)

    findings = CXScreenFindings(
        source_key=LogLabel.CHECK_SCREEN.value,
        deviations=found,
        screens=notes,
        dropped=dict(dropped),
    )
    if findings.unexplained:
        logger.warning(
            "%s: проход вернул ни находок, ни описаний кадров — кадр он смотрел "
            "или молчал, из ответа не читается%s",
            LogLabel.CHECK_SCREEN,
            _losses(findings.dropped),
        )
    elif findings.dropped:
        logger.warning(
            "%s: разбор прохода отбросил часть ответа%s",
            LogLabel.CHECK_SCREEN,
            _losses(findings.dropped),
        )
    return findings


def screen_update(
    findings: CXScreenFindings | None = None,
    status: CXNodeStatus = CXNodeStatus.SUCCESS,
    degraded: str = "",
) -> dict:
    """Итог прохода — остальное в состоянии остаётся нетронутым.

    :param findings: находки прохода — иначе на ветку ложится только статус
    :param status: чем закончился узел, когда находок нет
    :param degraded: чем плох ответ при статусе `success` (сорванные группы кадров,
        недобранные описания)
    :return: запись в состояние прогона
    """
    if findings is None:
        findings = CXScreenFindings(
            source_key=LogLabel.CHECK_SCREEN.value,
            status=status,
            degraded=degraded,
        )
    elif degraded and not findings.degraded:
        findings = findings.model_copy(update={"degraded": degraded})
    return {"screen_findings": findings}


def screens_out(
    found: CXScreenFindings | None, frames_count: int
) -> list[ScreenOut]:
    """Находки и описания прохода — в секцию `screens` ответа прогона.

    Порядок — порядок кадров пути, а не порядок ответа модели: раскадровка читает
    секцию по экранам, и экран обязан встать на своё место. Шаг становится
    0-based индексом кадра (как у `Friction.step`). Зона приходит та, что осталась
    у находки после разметки: код справочника перебивает мнение прохода, и
    раскадровка красит находку той же тяжестью, какой её оштрафовал балл
    (`pipeline.utils.screen_mapping`); у незаматченной находки остаётся её собственная.

    Кадр попадает в выдачу, даже если описания и находок нет: экран обязан
    остаться видимым разобранному прогону, иначе пропуск выглядел бы как «кадра
    не было».

    :param found: итог прохода — пусто, если прохода не было
    :param frames_count: число кадров пути — за его пределами экрана нет
    :return: экраны пути в их порядке
    """
    if found is None:
        return []

    by_step: dict[int, list[ScreenFinding]] = {}
    for deviation in found.deviations:
        by_step.setdefault(deviation.step, []).append(_finding_out(deviation))
    summaries = {note.step: note.summary for note in found.screens}
    steps = sorted(set(by_step) | set(summaries))
    return [
        ScreenOut(
            step=step - 1,
            summary=summaries.get(step, ""),
            findings=by_step.get(step, []),
        )
        for step in steps
        if 1 <= step <= frames_count
    ]


def _deviation(
    item: dict,
    step: int,
    dropped: Counter[str],
) -> CXPinnedDeviation | None:
    """Одна находка кадра: текст обязателен, зона сверяется с правилом.

    Ни блока, ни кода у находки нет: проход не выбирает дефект из справочника, он
    описывает видимое. Адресом находку обеспечит разметка (`summarize_screen`), а
    здесь её слово стоит только о тяжести — ту проход калибрует по кадру.

    Координаты остаются обязательными: пин — единственное, чем эта находка
    отличается от находки любой другой ветки, и без места на экране она перестала
    бы быть подсказкой человеку, который смотрит раскадровку. Центр кадра берётся
    там, где место указать нечем (`pct_of`), — показать находку в углу хуже.

    Номер экрана берётся из подписи экрана, а не из находки: промпт просит
    координаты, а не второй номер кадра, и выдуманный номер внутри находки не
    имеет права переставить её на другой экран.

    :param item: элемент перечня `findings` из ответа модели
    :param step: номер экрана элемента ответа, уже сверенный с путём
    :param dropped: счётчик потерь разборщика
    :return: отклонение либо `None`, если оно не прошло проверку
    """
    text = text_of(item.get("text")).strip()
    if not text:
        dropped["finding_without_text"] += 1
        return None
    evidence = quotes_of(item.get("evidence"))
    return CXPinnedDeviation(
        zone=_zone_of(item.get("zone"), bool(evidence)),
        step=step,
        text=text,
        rec=text_of(item.get("rec")).strip(),
        evidence=evidence,
        x=pct_of(item.get("x"), PIN_CENTER),
        y=pct_of(item.get("y"), PIN_CENTER),
    )


def _zone_of(raw: object, has_evidence: bool) -> CXZone:
    """Зона находки: неизвестная — `yellow`, `black` без цитаты — `red`.

    Промпт обещает, что чёрная зона ставится только когда состав нарушения виден
    на кадре и подтверждён дословной цитатой. Слово модели исполняет здесь код:
    зона решает штраф, а цена выдуманной улики — весь блок, которому легла находка
    (то же правило, что у проверок проблем пути).

    :param raw: зона из ответа модели
    :param has_evidence: есть ли хоть одна непустая цитата
    :return: зона находки
    """
    try:
        zone = CXZone(str(raw or "").strip().lower())
    except ValueError:
        logger.warning(
            "%s: проход прислал зону %r вне перечня — считаем yellow",
            LogLabel.CHECK_SCREEN,
            raw,
        )
        return CXZone.YELLOW

    if zone is CXZone.BLACK and not has_evidence:
        logger.warning(
            "%s: проход поставил black без дословной цитаты — снижаем до red",
            LogLabel.CHECK_SCREEN,
        )
        return CXZone.RED

    return zone


def _finding_out(deviation: CXPinnedDeviation) -> ScreenFinding:
    """Отклонение прохода — в находку экрана: цитата строкой, тяжесть из зоны.

    :param deviation: отклонение с пином
    :return: находка экрана для секции `screens`
    """
    zone = deviation.zone
    return ScreenFinding(
        zone=zone,
        severity=zone_to_severity(zone) or CXSeverity.INFO,
        criterion=deviation.criterion,
        code=deviation.code,
        text=deviation.text,
        rec=deviation.rec,
        evidence="; ".join(deviation.evidence),
        x=deviation.x,
        y=deviation.y,
    )


def _losses(dropped: dict[str, int]) -> str:
    """Хвост строки журнала: сколько ответов разбор выбросил и по какой причине.

    :param dropped: счётчик потерь разборщика
    :return: пустая строка, если терять было нечего
    """
    if not dropped:
        return ""
    counts = ", ".join(f"{why}={n}" for why, n in sorted(dropped.items()))
    return f" — и то не дошли все: отброшено ({counts})"
