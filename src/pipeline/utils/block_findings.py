"""Разбор ответа блока критериальной модели: вердикты и отклонения.

Здесь и то, как находки блока ложатся в состояние прогона: узел говорит с
контуром и пишет ответ как есть, а формой записи выбирают эти функции. Разбор
один на все обращения ветки блоков (`pipeline.agent.criteria_model_frames`): ответ
блока не зависит от того, сколькими обращениями его спрашивали.

Канон, по которому сверяется ответ, берётся из блока справочника
(`pipeline.schemas.nodes.CXBlock`), а не из настроек ветки: состоянию известен
справочник прогона, и `findings_of()` разбирает ответ по нему одному — так
находки собираются свойством состояния (`AgentStateSchema.criteria_model_findings`)
там, где никаких настроек узла уже нет. Метка журнала приходит отдельным
аргументом: по ней в логе видно, какой узел отвечал.

Отклонения блок отдаёт общей формой (`pipeline.schemas.nodes.CXDeviation`) — той же,
что и проверки проблем пути: сведение сравнивает находки блока с находками
проверок, а не два несходных перечня. Отличается у блока источник кода и зоны —
их называет справочник, а не слово модели.
"""

import logging
from collections import Counter
from collections.abc import Sequence

from src.pipeline.schemas.nodes import (
    CXBlock,
    CXBlockAnswer,
    CXBlockFindings,
    CXDeviation,
    CXSubVerdict,
)
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    ANSWER_MARKS,
    DROP_AMBIGUOUS_CODE,
    DROP_BATCH_BROKEN,
    DROP_CLEAN_WITHOUT_CANDIDATES,
    DROP_NO_VERDICT,
    DROP_NOT_OBJECT,
    DROP_UNKNOWN_CODE,
    DROP_UNKNOWN_STATUS,
    DROP_UNKNOWN_SUBCRITERION,
    EVIDENCE_NUMBER,
    SELF_DISMISSED_MARK,
    UNCATALOGUED,
    UNPROVEN_OK_MARK,
    CXItemStatus,
    CXNodeStatus,
    CXZone,
)
from src.pipeline.utils.catalog import locate_in
from src.pipeline.utils.deviations import quotes_of, step_of, text_of
from src.pipeline.utils.json_reply import json_payload

logger = logging.getLogger(__name__)


def findings_of(
    block: CXBlock,
    log_label: LogLabel,
    answer: str | CXBlockAnswer,
) -> CXBlockFindings:
    """Ответ блока в форму оценки: вердикты подкритериев и отклонения.

    Ответ приходит и текстом (его отдаёт контур), и уже разобранным по форме
    (`CXBlockAnswer`): форма строга в обоих случаях, а значения мягкие здесь.
    Отсутствие `items` — сбой формы, а не «ноль проблем»: молчаливый пустой
    список дал бы блоку полный балл. Битый статус или чужой ключ стоят только
    своего подкритерия — он становится `na` и выпадает из знаменателя, весь
    блок не роняется.

    Всё, что разбор отбросил, остаётся счётчиком в `dropped`: вердикт без
    которого модель осталась — это молча снятый счёта подкритерий, и без
    счётчика блок с половиной непроверенного справочника печатает полный балл.

    :param block: блок справочника — по нему канон подкритериев и зон
    :param log_label: метка ветки для журнала разбора
    :param answer: ответ модели — текстом или разобранным по схеме формы
    :return: оценка блока
    :raises ValueError: если в ответе нет перечня `items`
    """
    payload = (
        answer.model_dump(mode="json")
        if isinstance(answer, CXBlockAnswer)
        else json_payload(answer, "с оценкой блока")
    )
    dropped: Counter[str] = Counter()
    verdicts = _verdicts(block, log_label, payload.get("items"), dropped)
    _dismissed(verdicts, block, log_label)
    _unproven_ok(verdicts, block, log_label, dropped)
    missing = [
        key
        for key in block.required_keys
        if key not in {verdict.key for verdict in verdicts}
    ]
    if missing:
        dropped[DROP_NO_VERDICT] += len(missing)
        logger.warning(
            "%s: блок %s ответил не по всем подкритериям, без вердикта: %s",
            log_label,
            block.key,
            ", ".join(missing),
        )

    deviations = _deviations(
        block, log_label, payload.get("deviations"), dropped
    )
    note = text_of(payload.get("note"))
    _clean_without_candidates(
        verdicts, deviations, note, block, log_label, dropped
    )
    _verdict_failures(verdicts, deviations, block, log_label)

    return CXBlockFindings(
        source_key=block.key,
        items=_marked(verdicts),
        deviations=deviations,
        note=note,
        dropped=dict(dropped),
    )


def block_status_update(block_key: str, status: CXNodeStatus) -> dict:
    """Чем закончился узел блока, когда ответа нет.

    ВНИМАНИЕ — функция не вызывается ниоткуда. Узлы блоков пишут статус внутри
    находок (`findings_update(block.key, CXNodeStatus.ERROR)` в
    `pipeline.agent.criteria_model_frames`), и поле состояния `criteria_model_status`
    остаётся пустым всегда. Читатели поля (`pipeline.dto.agent.result`) честно получают
    пустой словарь и не падают, но обещанного «блок с сорванным обращением
    отличим от блока без находок» через это поле не происходит.

    Оставлена на случай, если статус понадобится отдельно от находок. Разбор —
    `docs/project-spec/09-maintenance/improvements.md`, пункт I-004.

    :param block_key: ключ блока, о котором сообщаем
    :param status: чем закончился узел
    :return: запись в состояние прогона
    """
    return {"criteria_model_status": {block_key: status}}


def findings_update(
    block_key: str,
    status: CXNodeStatus = CXNodeStatus.SUCCESS,
    findings: CXBlockFindings | None = None,
) -> dict:
    """Один блок находок — остальное в состоянии остаётся нетронутым.

    Ими пишут узлы блоков: блок проверяется несколькими обращениями по батчам
    кадров, поэтому итог блока собирает узел, а не состояние.

    :param block_key: ключ блока, чьи находки пишутся
    :param status: чем закончился узел, когда находок нет
    :param findings: находки блока — иначе на его ключ ложится только статус
    :return: запись в состояние прогона
    """
    if findings is None:
        findings = CXBlockFindings(source_key=block_key, status=status)
    return {"cx_findings": {block_key: findings}}


def merge_batches(
    block: CXBlock,
    parts: Sequence[CXBlockFindings],
    broken: int = 0,
) -> CXBlockFindings:
    """Вердикты и отклонения одного блока по всем батчам кадров — одним итогом.

    Батч отвечает только за свои кадры, поэтому статус подкритерия решается по
    всем батчам вместе: "fail" любого батча перебивает "ok" остальных (проблему
    увидели — значит она есть), а "na" не перебивает просмотренное "ok". Номера
    кадров в отклонениях глобальные, поэтому находки батчей просто складываются:
    пересобирать номера не нужно.

    Молчание сорванного батча не имеет права выглядеть чистотой: оно остаётся
    счётчиком в `dropped`, и прогон, где половина кадров не отвечена, печатается
    деградировавшим, а не пустым.

    :param block: блок справочника — по нему канон-порядок подкритериев
    :param parts: находки батчей в порядке пути
    :param broken: сколько батчей осталось без ответа
    :return: сведённая оценка блока
    """
    dropped: Counter[str] = Counter()
    for part in parts:
        dropped.update(part.dropped)
    if broken:
        dropped[DROP_BATCH_BROKEN] += broken

    verdicts: dict[str, list[CXSubVerdict]] = {}
    for part in parts:
        for item in part.items:
            verdicts.setdefault(item.key, []).append(item)
    items: list[CXSubVerdict] = []
    for key in block.required_keys:
        answers = verdicts.get(key) or []
        if not answers:
            continue
        strongest = max(
            answers,
            key=lambda answer: (
                CXItemStatus.parse(answer.status) or CXItemStatus.NA
            ).rank,
        )
        notes = list(
            dict.fromkeys(
                answer.note.strip()
                for answer in answers
                if answer.note.strip()
            )
        )
        items.append(
            strongest.model_copy(update={"note": _joined_notes(notes)})
        )

    deviations: list[CXDeviation] = []
    seen: set[tuple[str, int | None, str]] = set()
    for part in parts:
        for deviation in part.deviations:
            mark = (
                deviation.code,
                deviation.step,
                " ".join(deviation.text.lower().split())[:80],
            )
            if mark in seen:
                continue
            seen.add(mark)
            deviations.append(deviation)

    notes = list(
        dict.fromkeys(part.note.strip() for part in parts if part.note.strip())
    )
    answered = [part for part in parts if part.items]
    return CXBlockFindings(
        source_key=block.key,
        items=items,
        deviations=deviations,
        note="; ".join(notes[:2]),
        status=answered[0].status if answered else CXNodeStatus.ERROR,
        dropped=dict(dropped),
    )


def _dismissed(
    verdicts: list[CXSubVerdict], block: CXBlock, log_label: LogLabel
) -> None:
    """Назвать вслух подкритерии, снятые ссылкой на тип или сценарий пути.

    Основание запрещено промптом обеих веток, но промпт — просьба, а не запрет:
    пока такой вердикт не назван в журнале, отказ от проверки неотличим от её
    результата.
    """
    keys = [
        verdict.key for verdict in verdicts if verdict.dismissed_by_attribution
    ]
    if not keys:
        return
    logger.warning(
        "%s: блок %s вместо проверки по материалу снял подкритерии со ссылкой "
        "на тип или сценарий пути (%d): %s",
        log_label,
        block.key,
        len(keys),
        ", ".join(keys),
    )


def _unproven_ok(
    verdicts: Sequence[CXSubVerdict],
    block: CXBlock,
    log_label: LogLabel,
    dropped: Counter[str],
) -> None:
    """Назвать вердикты "ok", в которых нет ни номера кадра, ни счёта.

    Промпт требует улики от каждого статуса, а код сверял её только у "fail":
    "ok" проходил результатом проверки, не заплатив ни штрафом, ни снятым
    подкритерием. По пяти слепкам так выглядели 88 вердиктов из 201 — на них
    держался полный балл блока, который в отчёте нечем проверить.

    Статус здесь не меняется: соблюдён подкритерий или нет решает кадр, а не
    формулировка, и выдумывать "na" ради статистики значило бы снимать счёта
    проверенное. Отметка нужна ровно затем, чтобы доля таких вердиктов читалась в
    журнале и в печати прогона.
    """
    keys = [verdict.key for verdict in verdicts if verdict.unproven_ok]
    if not keys:
        return
    dropped["ok_without_evidence"] += len(keys)
    logger.warning(
        "%s: блок %s объявил соблюдёнными без ссылки на кадр или счёт (%d): %s",
        log_label,
        block.key,
        len(keys),
        ", ".join(keys),
    )


def _clean_without_candidates(
    verdicts: Sequence[CXSubVerdict],
    deviations: Sequence[CXDeviation],
    note: str,
    block: CXBlock,
    log_label: LogLabel,
    dropped: Counter[str],
) -> None:
    """Назвать чистый ответ блока, где не перебран ни один кандидат.

    Пустой перечень отклонений промпт разрешает только с поимёнными кандидатами в
    `note` блока, иначе «проверено и чисто» неотличимо от «блок не смотрели».
    `pipeline.schemas.nodes.CXCheckFindings.unexplained` ловит здесь одно полное
    молчание: пояснение «все экраны соответствуют паттернам» без единого номера
    кадра проходит как объяснение. Считается только там, где хоть один подкритерий
    объявлен соблюдённым, — блок, отвеченный целиком "na", чистоту не заявляет.
    """
    if deviations or EVIDENCE_NUMBER.search(note):
        return
    if not any(verdict.status == CXItemStatus.OK for verdict in verdicts):
        return
    dropped[DROP_CLEAN_WITHOUT_CANDIDATES] += 1
    logger.warning(
        "%s: блок %s ответил чисто и без кандидатов — пояснение %r не называет "
        "ни одного просмотренного кадра",
        log_label,
        block.key,
        note or "пусто",
    )


def _marked(verdicts: list[CXSubVerdict]) -> list[CXSubVerdict]:
    """Пометка в `note` там, где вердикт не подтверждён просмотром материала.

    Пометка дописана к заметке модели, а не поставлена вместо статуса: `na` из
    «материал не покрыл» и `na` из «этому пути не положено» различаются только
    текстом заметки — и именно её читают отчёт и разбор прогона. Так же различаются
    теперь два вида "ok": «на кадре 7 показано X» и «всё в порядке».
    """
    marked: list[CXSubVerdict] = []
    for verdict in verdicts:
        note = verdict.note.strip()
        mark = (
            SELF_DISMISSED_MARK
            if verdict.dismissed_by_attribution
            else UNPROVEN_OK_MARK
            if verdict.unproven_ok
            else ""
        )
        if not mark or mark in note:
            marked.append(verdict)
            continue
        marked.append(
            verdict.model_copy(
                update={"note": f"{note} — {mark}" if note else mark}
            )
        )
    return marked


def _joined_notes(notes: Sequence[str]) -> str:
    """Заметки батчей одного подкритерия — одной строкой без повтора отметок.

    Отметку несёт вердикт, а не каждое его обращение: два батча дали одну формулу
    без ссылки на кадр, и «не подтверждено» должно читаться один раз.
    """
    marks: list[str] = []
    texts: list[str] = []
    for note in notes:
        text = note.strip()
        for mark in ANSWER_MARKS:
            if mark in text:
                marks.append(mark)
                text = text.replace(f" — {mark}", "").replace(mark, "")
        text = text.strip().strip(" ;—")
        if text:
            texts.append(text)
    body = "; ".join(list(dict.fromkeys(texts))[:2])
    tail = "; ".join(mark for mark in ANSWER_MARKS if mark in marks)
    return " — ".join(part for part in (body, tail) if part)


def _verdicts(
    block: CXBlock,
    log_label: LogLabel,
    raw: object,
    dropped: Counter[str],
) -> list[CXSubVerdict]:
    """Вердикты в порядке справочника: канонные ключи, по одному на ключ."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"в ответе блока {block.key} нет перечня items")

    collected: dict[str, CXSubVerdict] = {}
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        key = text_of(item.get("key"))
        if key not in block.subcriteria:
            dropped[DROP_UNKNOWN_SUBCRITERION] += 1
            logger.warning(
                "%s: подкритерия %s нет в блоке %s, вердикт отброшен",
                log_label,
                key,
                block.key,
            )
            continue
        collected[key] = CXSubVerdict(
            key=key,
            status=_status_of(
                key, item.get("status"), block, log_label, dropped
            ),
            note=text_of(item.get("note")),
        )
    return [collected[key] for key in block.required_keys if key in collected]


def _status_of(
    key: str,
    raw: object,
    block: CXBlock,
    log_label: LogLabel,
    dropped: Counter[str],
) -> CXItemStatus:
    """Статус подкритерия; неизвестное значение — `na`, не ошибка блока.

    `na` стоит блоку дешевле `fail`, но в знаменатель не входит, и под ним
    прячется и отказ модели формулироваться: потому такой статус и считается
    отдельно (`status_out_of_enum`).
    """
    status = CXItemStatus.parse(raw)
    if status is not None:
        return status
    dropped[DROP_UNKNOWN_STATUS] += 1
    logger.warning(
        "%s: блока %s подкритерий %s вернул статус %r — считаем na",
        log_label,
        block.key,
        key,
        raw,
    )
    return CXItemStatus.NA


def _deviations(
    block: CXBlock,
    log_label: LogLabel,
    raw: object,
    dropped: Counter[str],
) -> list[CXDeviation]:
    """Отклонения блока: код и зона сверяются со справочником блока.

    Зона берётся из справочника по коду, а не из слова модели: код решает штраф,
    и собственное мнение контура о критичности не должно его перебивать.
    Отклонение принадлежит блоку, который его нашёл, — `criterion` получает блок,
    а не то, что могло придти в ответ.

    Внекаталожная находка остаётся: служба `uncatalogued` — не выдуманный код, а
    признанное «вижу проблему, кода под неё в справочнике нет». Она идёт без кода
    и с жёлтой зоной — тяжелее её назначает справочник, а не догадка разбора.
    Код, которого в справочнике нет и который не помечен службой, остаётся
    сброшенным: иначе выдуманный шифр начал бы решать штраф. Сброшенной оказывается
    и метка, которую блок назвал дважды: коротким кодом различается только дефект
    внутри своей проверки (`pipeline.utils.catalog`), и какая из двух проверок найдена —
    нечем понять, а молча взять штраф первой значит оштрафовать за то, чего блок не
    называл. Разобранный адрес отдаёт и свою проверку: ниже по потоку
    (`pipeline.utils.findings`) код того же блока ищется среди его проверок заново, и без
    названной проверки он слипся бы с меткой соседней проверки. Проваленный
    подкритерий при этом не исчезает: его словами отвечает `_verdict_failures`.
    """
    deviations: list[CXDeviation] = []
    for item in raw or []:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        named = text_of(item.get("code"))
        found = locate_in([block], named)
        if not found:
            if named.strip().lower() != UNCATALOGUED:
                dropped[DROP_UNKNOWN_CODE] += 1
                logger.warning(
                    "%s: кода %s нет в перечне блока %s, находка отброшена",
                    log_label,
                    named,
                    block.key,
                )
                continue
            deviations.append(
                CXDeviation(
                    criterion=block.key,
                    code="",
                    zone=CXZone.YELLOW,
                    step=step_of(item.get("step")),
                    text=text_of(item.get("text")),
                    evidence=quotes_of(item.get("evidence")),
                )
            )
            continue
        if len(found) > 1:
            dropped[DROP_AMBIGUOUS_CODE] += 1
            logger.warning(
                "%s: код %s есть в %s проверках блока %s (%s) — какую из них нашёл "
                "блок, нечем различить, находка отброшена",
                log_label,
                named,
                len(found),
                block.key,
                ", ".join(here.ref for here in found),
            )
            continue
        defect = found[0].defect
        deviations.append(
            CXDeviation(
                criterion=block.key,
                code=defect.code,
                check_key=found[0].check.key,
                zone=defect.zone,
                step=step_of(item.get("step")),
                text=text_of(item.get("text")),
                evidence=quotes_of(item.get("evidence")),
            )
        )
    return deviations


def _verdict_failures(
    verdicts: Sequence[CXSubVerdict],
    deviations: list[CXDeviation],
    block: CXBlock,
    log_label: LogLabel,
) -> None:
    """Проваленный подкритерий без отклонения не имеет права исчезнуть.

    Модель обязана увязать "fail" с кодом из перечня, но перечень описывает не
    всё, что видно на экране: если код не нашёлся, ответ уже называет проблему
    словами, и эти слова — сырьё сведения. Без этой дописки прогон теряет ровно
    те находки, которым не хватает шифра: статус учтён в балле, а в отчёте о нём
    ничего не читают — так проваленный подкритерий неотличим от проверенного.

    Названный код закрывает все проверки, где он живёт: метка различает дефект
    только внутри своей проверки, и решить, какую из двух имел в виду блок, нечем.
    Считать покрытой каждую из них — значит рискнуть лишней внекаталожной находкой;
    обратное решение молча снимало бы с отчёта проваленный подкритерий.
    """
    covered = {
        here.check.key
        for item in deviations
        for here in locate_in([block], item.code)
    }
    failed = [
        verdict
        for verdict in verdicts
        if verdict.status == CXItemStatus.FAIL
        and verdict.key not in covered
        and verdict.note.strip()
    ]
    if not failed:
        return
    logger.info(
        "%s: блок %s провалил подкритерии без кода дефекта — они идут сведению "
        "внекаталожной находкой: %s",
        log_label,
        block.key,
        ", ".join(verdict.key for verdict in failed),
    )
    for verdict in failed:
        deviations.append(
            CXDeviation(
                criterion=block.key,
                code="",
                zone=CXZone.YELLOW,
                text=verdict.note.strip(),
            )
        )
