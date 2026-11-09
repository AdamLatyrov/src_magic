"""Разбор ответа фокусного прохода автозаполнения.

Здесь и то, как ответ прохода ложится в состояние прогона: узел говорит с
контуром (`pipeline.agent.autofill.nodes`), а формой записи выбирают эти функции —
той же формой, что и находки блоков (`pipeline.utils.block_findings`) и проверок
проблем пути (`pipeline.utils.friction_findings`): отклонение одной формы
(`pipeline.schemas.nodes.CXDeviation`), контейнер одной формы
(`pipeline.schemas.nodes.CXCheckFindings`). Сведение поэтому видит находки трёх
веток одним перечнем и дедуплицирует их между собой: одна и та же анкета,
названная блоком справочника и этим проходом, штрафуется один раз.

Справочника ветка не знает — ни своего перечня подкритериев, ни адреса блока.
Находка доезжает до сведения внекаталожной, как доезжает до своей разметки
находимка проверки пути: код, блок и зону ей даёт сведение, которое видит весь
справочник сразу (`pipeline.utils.findings._issue`). Адрес из настроек здесь был — и
записывал весь проход в один раздел справочника, каким бы ни оказался повтор
ввода, а опечатка в ключе молча снимала находке блок; без адреса терять
нечего, кроме заранее выбранного раздела.

Форма строга, значения мягкие. Отсутствующий перечень `findings` — сбой формы, а
не «предзаполнять было нечего»: молчаливый пустой список означал бы чистый путь
там, где модель ответила не тем словом. Непригодное отдельное отклонение стоит
только его самого — оно отбрасывается со счётчиком в `dropped`, остальные
остаются.

Три правила прохода код исполняет сам, потому что промпт здесь просьба, а не
запрет: зона всегда `yellow` (ручной ввод известного банку значения мешает, но
не блокирует путь, и красная зона означала бы штраф блока по всей строгости),
номер экрана обязан принадлежать пути (у кадра, которого нет, не бывает улики),
и находка, чей вывод — «ввод этого поля нормален», в отчёт не идёт: иначе
«оправдание» банка штрафовало бы прогон.
"""

import logging
import re
from collections import Counter

from src.pipeline.schemas.nodes import CxAutofillFindings, CXPinnedDeviation
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    DROP_BAD_STEP,
    DROP_INPUT_JUSTIFIED,
    DROP_NOT_OBJECT,
    PIN_CENTER,
    CXNodeStatus,
    CXZone,
)
from src.pipeline.utils.deviations import pct_of, step_of, text_of
from src.pipeline.utils.json_reply import json_payload

logger = logging.getLogger(__name__)


def autofill_findings(answer: str, frames_count: int) -> CxAutofillFindings:
    """Ответ прохода в форму находок пути.

    :param answer: ответ модели
    :param frames_count: сколько кадров в пути — за его пределами экрана нет
    :return: находки прохода
    :raises ValueError: если в ответе нет перечня `findings`
    """
    payload = json_payload(answer, "с перечнем незаполненных полей")
    raw = payload.get("findings")
    if not isinstance(raw, list):
        raise ValueError("в ответе прохода нет перечня findings")

    dropped: Counter[str] = Counter()
    found: list[CXPinnedDeviation] = []
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        deviation = _deviation(item, frames_count, dropped)
        if deviation is not None:
            found.append(deviation)

    findings = CxAutofillFindings(
        source_key=LogLabel.CHECK_AUTOFILL.value,
        deviations=found,
        note=text_of(payload.get("note")),
        dropped=dict(dropped),
    )
    if findings.unexplained:
        logger.warning(
            "%s: проход вернул пустой перечень без объяснения — чем он проверял "
            "путь, из ответа не читается%s",
            LogLabel.CHECK_AUTOFILL,
            _losses(findings.dropped),
        )
    elif findings.dropped:
        logger.warning(
            "%s: разбор прохода отбросил часть находок%s",
            LogLabel.CHECK_AUTOFILL,
            _losses(findings.dropped),
        )
    return findings


def autofill_update(
    findings: CxAutofillFindings | None = None,
    status: CXNodeStatus = CXNodeStatus.SUCCESS,
    degraded: str = "",
) -> dict:
    """Итог прохода — остальное в состоянии остаётся нетронутым.

    :param findings: находки прохода — иначе на ветку ложится только статус
    :param status: чем закончился узел, когда находок нет
    :param degraded: чем плох ответ при статусе `success` (сорванные группы кадров)
    :return: запись в состояние прогона
    """
    if findings is None:
        findings = CxAutofillFindings(
            source_key=LogLabel.CHECK_AUTOFILL.value,
            status=status,
            degraded=degraded,
        )
    elif degraded and not findings.degraded:
        findings = findings.model_copy(update={"degraded": degraded})
    return {"autofill_findings": findings}


def _deviation(
    item: dict,
    frames_count: int,
    dropped: Counter[str],
) -> CXPinnedDeviation | None:
    """Одно отклонение прохода: экран сверяется с путём, зона — с правилом.

    Кода и адреса у находки нет: проход не выбирает дефект из справочника и не
    знает блоков — их находке даёт сведение, куда она приходит внекаталожной.

    :param item: элемент перечня `findings` из ответа модели
    :param frames_count: число кадров пути
    :param dropped: счётчик потерь разборщика
    :return: отклонение либо `None`, если оно не прошло проверку
    """
    text = text_of(item.get("text")).strip()
    step = step_of(item.get("step"))
    if not text or not 1 <= step <= frames_count:
        dropped[DROP_BAD_STEP] += 1
        return None
    blob = f"{text} {text_of(item.get('rec'))}".lower()
    input_justified = re.compile(
        r"оправдан|это нормально|ввод нормален|не является проблемой"
        r"|не проблема|так положено|требовани\w+ безопасност|по законодательств",
        re.IGNORECASE,
    )
    if input_justified.search(blob):
        dropped[DROP_INPUT_JUSTIFIED] += 1
        return None
    return CXPinnedDeviation(
        zone=_zone_of(item.get("zone")),
        step=step,
        text=text,
        rec=text_of(item.get("rec")).strip(),
        x=pct_of(item.get("x"), PIN_CENTER),
        y=pct_of(item.get("y"), PIN_CENTER),
    )


def _zone_of(raw: object) -> CXZone:
    """Зона находки — всегда `yellow`; слово модели здесь только для журнала.

    Промпт запрещает этой ветке красную и чёрную зоны: недозаполненное поле
    стоит клиенту труда, а не всего пути. Зона решает штраф блока, поэтому её
    исполняет код, а не просьба в вопросе.

    :param raw: зона из ответа модели
    :return: жёлтая зона
    """
    value = str(raw or "").strip().lower()
    if value and value != CXZone.YELLOW.value:
        logger.warning(
            "%s: проход прислал зону %r — жёлтая ветка штраф не повышает",
            LogLabel.CHECK_AUTOFILL,
            raw,
        )
    return CXZone.YELLOW


def _losses(dropped: dict[str, int]) -> str:
    """Хвост строки журнала: сколько находок разбор выбросил и по какой причине.

    :param dropped: счётчик потерь разборщика
    :return: пустая строка, если терять было нечего
    """
    if not dropped:
        return ""
    counts = ", ".join(f"{why}={n}" for why, n in sorted(dropped.items()))
    return f" — и то не дошли все: отброшено ({counts})"
