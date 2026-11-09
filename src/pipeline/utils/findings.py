"""Разбор ответа сведения: единый список находок и фолбэк из сырья веток.

Сведение кормят находками обеих проверяющих веток — блоков справочника и проверок
проблем пути: у отклонения одна форма (`pipeline.schemas.nodes.CXDeviation`), поэтому
здесь по нему идёт один цикл, а не два. Разница между ветками видна только по
справочнику: у отклонения с кодом дефекта подпись и симптом берёт модель блока,
внекаталожная находка проверки пути печатается своими словами.

Дедупликация — часть этого же файла: одна и та же проблема, найденная блоком и
проверкой пути, описывается одним Issue и штрафуется один раз.
"""

import logging
import re
from collections.abc import Sequence
from typing import Optional

from src.pipeline.schemas.nodes import (
    CXBlock,
    CXCheckFindings,
    CXConsolidatedEvidence,
    CXConsolidatedFinding,
    CXConsolidatedFindings,
    CXCriteriaModel,
    CXDeviation,
)
from src.pipeline.schemas.result import Issue, IssueEvidence
from src.pipeline.scoring.zones import zone_to_severity
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    UNCATALOGUED,
    CXConfidence,
    CXFindingKind,
    CXNodeStatus,
    CXSeverity,
    CXZone,
)
from src.pipeline.utils.catalog import (
    DefectAt,
    block_of,
    locate_in,
    owner_block_of,
    unique,
)

logger = logging.getLogger(__name__)


def raw_findings_listing(
    model: CXCriteriaModel, sources: Sequence[CXCheckFindings]
) -> str:
    """Сырые находки обеих веток для промпта сведения.

    Источники идут одним перечнем — блоки справочника, затем проверки проблем
    пути: модели нужно сравнить находки между собой, а не читать два отчёта,
    поэтому одна и та же проблема двух веток стоит здесь соседними строками, а не
    в разных разделах.

    Зону и код печатаем по справочнику: они уже сверены с блоком, и пересобирать их
    по своему вкусу модель не должна. Улика идёт дословной цитатой — по ней
    консолидация решает доказательность находки.

    :param model: критериальная модель — по ней подпись дефекта у улики
    :param sources: находки веток, каждый источник подписан своим ключом
    :return: перечень сырья по источникам
    """
    chunks: list[str] = []
    for source in sources:
        if not source.deviations:
            continue
        lines = [_source_heading(model, source)]
        if source.note:
            lines.append(f"Чем проверено: {source.note}")
        for deviation in source.deviations:
            lines.append(_deviation_line(model, source, deviation))
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) if chunks else "  (нет находок)"


def _source_heading(model: CXCriteriaModel, source: CXCheckFindings) -> str:
    """Подпись источника находок: блок справочника или проверка проблем пути.

    Ключ источника ищется в модели: блок его находит и отдаёт своё название, ключ
    проверки пути в модели отсутствует — такая находка подписывается своей
    проверкой.
    """
    block = _block_of_source(model, source)
    if block is None:
        return f"### Проверка пути {source.source_key}"
    return f"### Блок {block.key} — {block.name}"


def _block_of_source(
    model: CXCriteriaModel, source: CXCheckFindings
) -> Optional[CXBlock]:
    """Блок, которому принадлежит источник; у проверки проблем пути его нет."""
    try:
        return model.block(source.source_key)
    except KeyError:
        return None


def _block_of_deviation(
    model: CXCriteriaModel, source: CXCheckFindings, deviation: CXDeviation
) -> Optional[CXBlock]:
    """Блок находки: блок источника либо блок, который назвала сама находка.

    У ветки блоков адрес стоит в ключе источника, и он же задаёт подпись раздела.
    У фокусных проходов источника в справочнике нет — они отвечают за весь путь, —
    и адрес приходит в самой находке: и поэкранная ветка, и проверки проблем пути
    получают его на своей разметке (`pipeline.utils.deviation_mapping.mark`), и без этого
    перехода код находки остался бы текстом: ни подписи дефекта в сырье, ни
    `defect_code` и справочной зоны в фолбэке отчёта.

    Размеченная находка приносит блок вместе с кодом, и он бывает блоком другого
    блока — проверке пути путь виден целиком. Внекаталожной находке блока дать
    нечем: у проверки пути это остаётся её собственный ключ, и в отчёте такая находка
    живёт вне блоков.
    """
    block = _block_of_source(model, source)
    if block is not None:
        return block
    return (
        block_of(model, deviation.criterion) if deviation.criterion else None
    )


def _deviation_line(
    model: CXCriteriaModel, source: CXCheckFindings, deviation: CXDeviation
) -> str:
    """Одно отклонение в перечне сырья: где видно, чем доказано и к чему зовёт."""
    block = _block_of_deviation(model, source, deviation)
    # Подпись дефекта берётся там, где её различают: по одному коду ответ даёт
    # несколько строк справочника, и чужой подписью находку снабжать нечем.
    address = (
        _defect_address(block, deviation.code, deviation.check_key)
        if block is not None
        else None
    )
    known = address.defect if address is not None else None
    # Внекаталожная находка подписывается служебным кодом: тот же ответ ждём и в
    # `defect_code`, а пустая подпись читалась бы как испорченное сырьё.
    signature = (
        f"{deviation.code} — {known.defect}"
        if known
        else (deviation.code or UNCATALOGUED)
    )
    symptom = f" Проявляется так: {known.symptom}." if known else ""
    where = f"экран {deviation.step}" if deviation.step else "уровень пути"
    pair = (
        f", второй экран {deviation.related_step}"
        if deviation.related_step
        else ""
    )
    # Адрес у ненайденного кода отсутствует — пустая часть списка читалась бы как
    # испорченная строка («зона red, »), поэтому блок печатается только когда он есть.
    marks = f"зона {deviation.zone}"
    if deviation.criterion:
        marks += f", {deviation.criterion}"
    lines = [f"  - [{where}{pair}, {marks}] {signature}.{symptom}"]
    if deviation.text:
        lines.append(f"      находка: {deviation.text}")
    for quote in deviation.evidence:
        # Цитата уже подписана кадром и взята в кавычки самой моделью — оборачивать
        # её ещё раз значило бы путать вложенные кавычки в перечне.
        lines.append(f"      улика: {quote}")
    if deviation.rec:
        lines.append(f"      рекомендация: {deviation.rec}")
    return "\n".join(lines)


def consolidated_issues(
    model: CXCriteriaModel, answer: CXConsolidatedFindings, frames_count: int
) -> list[Issue]:
    """Ответ сведения в валидированные находки прогона.

    Форма строга уже на стороне контура — ответ приходит по схеме структурированного
    вывода. Значения остаются мягкими и здесь: значение вне каноны приводится к
    допустимому, а не роняет находку — находка без зоны штрафуется максимальным
    весом, и терять её дороже, чем чинить.

    :param model: критериальная модель — справочник зон и подкритериев
    :param answer: сводный ответ модели
    :param frames_count: число кадров пути — по нему отсекаются номера экранов
    :return: уникальные находки прогона
    """
    issues = [
        issue
        for item in answer.issues
        if (issue := _issue(model, item, frames_count)) is not None
    ]
    return dedup_issues(issues)


def issues_from_findings(
    model: CXCriteriaModel,
    sources: Sequence[CXCheckFindings],
    frames_count: int,
) -> list[Issue]:
    """Находки из сырья обеих веток — фолбэк, когда сведение не состоялось.

    Форма отклонения одна, поэтому фолбэк ветки не различает: находка проверки
    пути становится проблемой того блока, который проверка сама и назвала, а
    отсутствие кода делает её внекаталожной — ровно как в сведённом ответе.
    `confidence` брать неоткуда, и она остаётся пустой: догадка о доказательности
    была бы опаснее максимального веса, который видно в отчёте.

    :param model: критериальная модель
    :param sources: находки веток, каждый источник подписан своим ключом
    :param frames_count: число кадров пути
    :return: адресуемые находки прогона
    """
    issues: list[Issue] = []
    for source in sources:
        for deviation in source.deviations:
            block = _block_of_deviation(model, source, deviation)
            address = (
                _defect_address(block, deviation.code, deviation.check_key)
                if block is not None
                else None
            )
            known = address.defect if address is not None else None
            zone = _gated_zone(
                known.zone if known else _model_zone(deviation.zone),
                bool(deviation.evidence),
            )
            steps = [
                step
                for step in (deviation.step, deviation.related_step)
                if step
            ]
            issues.append(
                Issue(
                    severity=zone_to_severity(zone) or CXSeverity.MID,
                    criterion=deviation.criterion or source.source_key,
                    title=(
                        known.defect
                        if known
                        else deviation.code or deviation.text.strip()
                    ),
                    text=deviation.text or (known.defect if known else ""),
                    rec=deviation.rec,
                    steps=_steps([str(step) for step in steps], frames_count),
                    evidence=_raw_evidence(deviation),
                    defect_code=known.code if known else None,
                    defect_ref=address.ref if address else None,
                    defect_name=known.defect if known else None,
                    check_key=address.check.key if address else None,
                    zone=zone,
                )
            )
    return dedup_issues(issues)


def _raw_evidence(deviation: CXDeviation) -> list[IssueEvidence]:
    """Улики сырого отклонения: цитата и экран, с которого она взята.

    Номер кадра модель пишет внутри цитаты («кадр 12: „…"») — по нему улика и
    привязывается; без номера берётся экран самого отклонения.

    :param deviation: сырое отклонение
    :return: улики находки с номерами экранов 1-based
    """
    quote_screen = re.compile(r"^\D*(\d+)")
    evidence: list[IssueEvidence] = []
    for quote in deviation.evidence:
        found = quote_screen.match(quote)
        screen = int(found.group(1)) if found else deviation.step
        evidence.append(IssueEvidence(screen=screen or None, quote=quote))
    return evidence


def dedup_issues(issues: list[Issue]) -> list[Issue]:
    """Дедуп находок: один и тот же дефект на тех же экранах — одна проблема.

    Кодом сверяется размеченная находка: у неё есть `defect_code`, и две ветки,
    назвавшие один код на одном наборе экранов, говорят об одном нарушении, какими
    бы разными словами его ни описали. Этим же кодом сворачиваются и соседние
    проверки пути: они больше не разложены по блокам справочника и смотрят на один
    путь шестью способами, поэтому одна беда теперь находится дважды чаще, чем
    раньше, — и именно код, а не текст, решает, что это была одна беда.

    Внекаталожной находке сверять нечего, и она остаётся при прежнем ключе
    (блок, начало текста): без адреса два разных описания одной проблемы
    различить нечем, а молча слить их было бы хуже, чем показать обе.

    :param issues: находки прогона — сведённые либо из сырья обеих веток
    :return: находки без повторов, в порядке входа
    """
    seen: set[tuple] = set()
    unique: list[Issue] = []
    for issue in issues:
        if issue.defect_code:
            key: tuple = (
                issue.defect_code,
                tuple(sorted(set(issue.steps))),
            )
        else:
            key = (
                issue.criterion or "",
                " ".join((issue.text or "").lower().split())[:80],
            )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def covered_by(issue: Issue, reported: Sequence[Issue]) -> bool:
    """Попадает ли сырая находка в сводный ответ — тем же экраном того же блока.

    Сведение имеет право слить дубли, но не имеет права молча выбросить находку:
    в прогоне 047 из тринадцати сырых отклонений до отчёта дошли четыре, и по
    остальным девяти блок остался «соблюдён», хотя проверка пути назвала экран,
    номер и цитату. Текст сведённой находки переписан своими словами, поэтому
    по тексту покрытие ищут в последнюю очередь: опознаются код дефекта и экран,
    которые сведение обязано сохранить.

    Блок требуют друг от друга только находки со справочным адресом. У
    внекаталожной находки фокусного прохода адреса нет, и подписывает её ключ
    источника («screen», «autofill») — которого в сведённом ответе не бывает ни у
    чего, кроме как у самой модели. Требовать тогда равных блоков значило бы
    считать такую находку потерванной всегда: прогон платил бы за неё дважды
    штрафом и всегда ставил бы `consolidation_dropped`.

    :param issue: сырая находка (`issues_from_findings`)
    :param reported: проблемы сводного ответа
    :return: есть ли сырой находке место в сведённом списке
    """
    for item in reported:
        addressed = issue.defect_code is not None or bool(issue.check_key)
        if addressed and (item.criterion or "") != (issue.criterion or ""):
            continue
        if issue.defect_code and issue.defect_code == item.defect_code:
            return True
        if set(issue.steps) & set(item.steps):
            return True
        if issue.check_key and issue.check_key == item.check_key:
            return True
    return False


def _issue(
    model: CXCriteriaModel, item: CXConsolidatedFinding, frames_count: int
) -> Optional[Issue]:
    """Одна находка сведённого ответа: форма из схемы, значения мягкие.

    Зона каталогизированной находки берётся ИЗ СПРАВОЧНИКА — мнение модели её не
    перебивает; у внекаталожной остаётся зона модели. Чёрная без дословной цитаты
    опускается до красной.
    """
    text = item.text.strip()
    if not text:
        return None
    kind = _finding_kind(item.kind)
    if kind == CXFindingKind.NOISE.value:
        return None  # шум и придирки в отчёт не идут

    evidence = _evidence(item.evidence)
    block, check_key = _binding(
        model, block_of(model, item.criterion), item.check_key
    )
    address = (
        _defect_address(block, item.defect_code, check_key)
        if block is not None
        else None
    )
    known = address.defect if address is not None else None
    zone = known.zone if known is not None else _model_zone(item.zone)
    zone = _gated_zone(zone, bool(evidence))
    return Issue(
        severity=zone_to_severity(zone) or CXSeverity.MID,
        criterion=block.key if block is not None else "",
        title=item.title,
        text=text,
        rec=item.rec,
        steps=_steps(item.steps, frames_count),
        evidence=evidence,
        confidence=_confidence(item.confidence),
        kind=kind,
        defect_code=known.code if known is not None else None,
        defect_ref=address.ref if address is not None else None,
        defect_name=known.defect if known is not None else None,
        # справочник знает, какой проверке принадлежит названный код; ответ
        # сводит их сам и может разминуться — адрес решает справочник
        check_key=address.check.key if address is not None else check_key,
        zone=zone,
    )


def _defect_address(
    block: CXBlock, raw: str, check_key: Optional[str]
) -> Optional[DefectAt]:
    """Адрес дефекта блока: названная проверка точнее голой метки.

    И сведение, и разметка отвечают кодом, а проверку называют свою: сведение —
    своим полем `check_key`, разметка — проверкой найденной строки справочника
    (`pipeline.utils.deviation_mapping.mark`). Эта пара различает дефект там, где
    короткая метка уже не различает. Без пары голая метка даёт адрес только тогда,
    когда во всём блоке встречается один раз: выбрать один из двух молча значило бы
    выписать находке штраф одной из проверок наугад.
    """
    named = str(raw or "").strip()
    if not named or named.lower() == UNCATALOGUED:
        return None
    if check_key:
        found = locate_in([block], f"{check_key}:{named}")
        if found:
            return found[0]
    return unique(locate_in([block], named))


def _finding_kind(kind: str) -> Optional[str]:
    """Вид находки из канона; значение вне канона остаётся пустым."""
    try:
        return CXFindingKind(kind.strip().lower()).value
    except ValueError:
        return None


def _binding(
    model: CXCriteriaModel, block: Optional[CXBlock], raw: object
) -> tuple[Optional[CXBlock], Optional[str]]:
    """Привязка находки: блок и подкритерий.

    Осознанный отказ (`none`) и ключ вне модели дают `None` — находка остаётся в блоке
    как «Другое» и разметку блока не трогает. Ключ точнее блока: он один, блок из него
    выводится однозначно, поэтому чужой ключ привязку не роняет — по нему чинится блок.
    """
    key = str(raw or "").strip()
    if key.lower() in ("", "none", "null", UNCATALOGUED):
        return block, None
    owner = owner_block_of(model, key)
    if owner is None:
        logger.warning(
            "%s: подкритерия %s нет в модели, находка без подкритерия",
            LogLabel.CONSOLIDATE_FINDINGS,
            key,
        )
        return block, None
    if block is not None and owner.key != block.key:
        logger.warning(
            "%s: подкритерий %s из блока %s, а criterion=%s — берём блок подкритерия",
            LogLabel.CONSOLIDATE_FINDINGS,
            key,
            owner.key,
            block.key,
        )
    return owner, key


def _evidence(
    items: list[CXConsolidatedEvidence],
) -> list[IssueEvidence]:
    """Улики находки: экран + дословная цитата; без цитаты улики нет."""
    evidence: list[IssueEvidence] = []
    for item in items:
        quote = item.quote.strip()
        if not quote:
            continue
        try:
            screen: Optional[int] = int(item.screen)
        except (TypeError, ValueError):
            screen = None
        evidence.append(IssueEvidence(screen=screen, quote=quote))
    return evidence


def _steps(values: list[str], frames_count: int) -> list[int]:
    """Номера экранов 1-based → индексы кадров; вылезшие за пути отсекаются."""
    steps: list[int] = []
    for value in values:
        try:
            index = int(str(value)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < frames_count and index not in steps:
            steps.append(index)
    return steps


def _confidence(confidence: str) -> Optional[CXConfidence]:
    """Уверенность модели; неизвестный уровень остаётся пустым."""
    return CXConfidence.parse(confidence)


def _model_zone(zone: CXZone | str | None) -> CXZone:
    """Зона внекаталожной находки; значение не из канона — жёлтая."""
    return CXZone.parse(zone) or CXZone.YELLOW


def _gated_zone(
    zone: CXZone | str | None, has_evidence: bool
) -> Optional[CXZone]:
    """Стоп-кран только при доказательной цитате: чёрная без улики → красная.

    Зона приходит то членом перечисления, то своим значением (`use_enum_values`
    оставляет в модели строку), поэтому сверка идёт после `parse`.
    """
    parsed = CXZone.parse(zone)
    if parsed is CXZone.BLACK and not has_evidence:
        return CXZone.RED
    return parsed


def findings_status(found: Optional[CXCheckFindings]) -> CXNodeStatus:
    """Чем закончилась проверка; источника без находок нет среди успешных."""
    if found is None:
        return CXNodeStatus.ERROR
    return CXNodeStatus(found.status)
