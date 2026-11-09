"""Сверка находок текущего прогона с находками прошлого прогона.

Прошлый прогон даёт формулировки, текущий — факты: что видно на экранах сегодня.
Поэтому в итоговом списке от прошлой находки берутся `title`, `text` и `rec`
(формулировки прошли экспертизу), а шаги, зона, код, подкритерий и улики — из
новой. Находка из совпадения встаёт на место прошлой, чтобы список читался как
продолжение прошлого прогона, а не как новый отчёт.

Правила совпадения сформулированы в промпте этапа, здесь — их исполнение.
Сведение решается в коде, а не моделью: модель называет только, какие новые
находки дублируют какую прошлую, а кому что достанется, считает этот модуль.
Так у правила «на одну прошлую — одна находка» есть единственный исполнитель,
и ошибка модели не плодит дублей в списке штрафов.

Прошлых находок приходит один прогон — последний удачный по продукту. Групп из
нескольких прошлых не бывает, и правило «несколько прошлых на одну новую» здесь
не нужно.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.dedup.listing import current_listing, previous_listing
from src.pipeline.llm.gateway import ask
from src.pipeline.schemas.nodes import CXDedupGroups
from src.pipeline.schemas.result import Issue
from src.pipeline.settings.api.graph import DedupFindingsSettings
from src.pipeline.settings.params.logging import LogLabel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DedupResult:
    """Итог сверки: список находок и счётчики того, что с ними случилось.

    Счётчики нужны журналу и отчёту: по ним видно, сколько прошлых находок
    вернулось из прошлого прогона, сколько новых погашено как дубли и не
    пришлось ли собирать результат без модели.
    """

    issues: list[Issue] = field(default_factory=list)
    restored: int = 0
    merged: int = 0
    added: int = 0
    duplicated: int = 0
    model_calls: int = 0
    degraded: bool = False


async def dedup(
    previous: Sequence[Issue],
    current: Sequence[Issue],
    *,
    settings: Optional[DedupFindingsSettings] = None,
    provider: Optional[GigaChatProvider] = None,
) -> DedupResult:
    """Сводит находки текущего прогона с находками прошлого.

    Обе находки приходят аргументами: где их взяли — дело вызывающей стороны.
    В базу и в файлы метод не ходит.

    Прошлые с вердиктом `not_ok` в сверке не участвуют: эксперт их снял, и
    поднимать их обратно в список проблем нельзя.

    :param previous: находки прошлого прогона в их порядке
    :param current: находки текущего прогона в их порядке
    :param settings: настройки этапа; собирает их вызывающая сторона
    :param provider: провайдер контура; собирает его вызывающая сторона
    :return: итоговый список находок и счётчики сверки
    """
    stage = settings or _stage()
    # Снятое экспертизой не возвращается даже когда провал сегодняшнего прогона
    # не нашёл ничего: прошлая находка — не факт о пути, а вердикт человека.
    kept = [issue for issue in previous if not _rejected(issue)]

    if not kept and not current:
        return DedupResult()

    # Первый прогон по продукту либо чистый прошлый: модель тут не нужна,
    # платить за сверку нечего.
    if not kept:
        logger.info(
            "%s: прошлых находок нет — новый прогон идёт как есть (%s)",
            LogLabel.DEDUP_FINDINGS,
            len(current),
        )
        return DedupResult(issues=list(current), added=len(current))

    if not current:
        logger.info(
            "%s: находок сегодня нет — поднято всё прошлое (%s)",
            LogLabel.DEDUP_FINDINGS,
            len(kept),
        )
        return DedupResult(issues=list(kept), restored=len(kept))

    # Сначала бесплатное: совпадение по адресу дефекта видно без модели.
    pairs = _by_reference(kept, current)
    # В промпт уходят только те новые, чей адрес не совпал: остальные уже
    # получили пару, и второй раз спрашивать про них — платить за ответ,
    # который у нас есть.
    uncovered = [index for index in range(len(current)) if index not in pairs]

    groups: dict[int, list[int]] = {}
    restored: set[int] = set()
    model_calls = 0
    degraded = False

    if uncovered:
        try:
            answer = await ask(
                stage,
                CXDedupGroups,
                variables=stage.variables(
                    previous_findings=previous_listing(kept),
                    current_findings=current_listing(
                        [current[index] for index in uncovered]
                    ),
                    previous_count=len(kept),
                    current_count=len(uncovered),
                ),
                provider=provider,
            )
            model_calls = 1
            groups, restored = _groups_of(answer, len(kept), uncovered)
        except Exception as ex:  # noqa: BLE001
            # Пустой ответ модели не отменяет сверку: прошлое поднимается по
            # коду. Лучше список с дублем, чем прогон, потерявший историю из-за
            # чужого сбоя. Молча это не проходит — флаг `degraded` в результате
            # и предупреждение в журнале.
            logger.warning(
                "%s: сверка по смыслу не состоялась (%s) — прошлые находки "
                "подняты по коду, новые добавлены все",
                LogLabel.DEDUP_FINDINGS,
                ex,
            )
            degraded = True
            restored = set(range(len(kept))) - set(pairs)
    else:
        # Всё совпало по адресу: модель не нужна, все прошлые либо в паре,
        # либо поднимаются.
        restored = set(range(len(kept))) - set(pairs)

    # Пары по адресу досыпаются в те же группы, что модель назвала по смыслу:
    # для сборки результата это одно и то же — «эта новая дублирует эту прошлую».
    for index, previous_index in pairs.items():
        groups.setdefault(previous_index, []).append(index)

    return _result(kept, current, groups, restored, model_calls, degraded)


def _stage() -> DedupFindingsSettings:
    """Настройки этапа сверки приходят аргументом, а не из настроек процесса.

    Сверку вызывают и работник, и ноутбук; настроек процесса у ноутбука нет, а
    прогон не должен знать, откуда они пришли.

    :raises ValueError: настройки этапа не переданы
    """
    raise ValueError(
        "настройки этапа сверки не переданы: их собирает вызывающая сторона"
    )


def _rejected(issue: Issue) -> bool:
    """Находка снята экспертизой: поднимать её обратно нельзя."""
    return (issue.verified_verdict or "") == "not_ok"


def _by_reference(
    previous: Sequence[Issue], current: Sequence[Issue]
) -> dict[int, int]:
    """Совпадения, которые видно без модели: одна ссылка на дефект — одна беда.

    Берётся первая свободная прошлая находка с той же ссылкой, иначе с той же
    парой «код дефекта и подкритерий». Ссылки у находок нет — пара не находится,
    такую решает модель.

    :param previous: прошлые находки
    :param current: находки текущего прогона
    :return: номер новой находки → номер прошлой, на которую она похожа
    """
    pairs: dict[int, int] = {}
    taken: set[int] = set()
    for current_index, issue in enumerate(current):
        for previous_index, candidate in enumerate(previous):
            if previous_index in taken:
                continue
            if same_reference(issue, candidate):
                pairs[current_index] = previous_index
                taken.add(previous_index)
                break
    return pairs


def same_reference(left: Issue, right: Issue) -> bool:
    """Находки указывают на один дефект справочника (строгий адрес)."""
    if left.defect_ref and right.defect_ref:
        return left.defect_ref == right.defect_ref
    if left.defect_code and right.defect_code and left.check_key and right.check_key:
        return (left.defect_code, left.check_key) == (
            right.defect_code,
            right.check_key,
        )
    return False


def _groups_of(
    answer: CXDedupGroups, previous_count: int, uncovered: Sequence[int]
) -> tuple[dict[int, list[int]], set[int]]:
    """Группы ответа модели в номерах списков — с отбраковкой её ошибок.

    Отбраковываются: номер прошлой вне перечня, номер новой вне перечня, новая,
    попавшая в две группы. Группа с повтором отбрасывается целиком: разобрать,
    какая из двух ссылок верна, нечем.

    :param answer: ответ модели
    :param previous_count: сколько прошлых находок ушло в промпт
    :param uncovered: номера новых находок, которые модель видела (1-based)
    :return: группы в номерах списков и прошлые без дублей
    """
    # Ответ модели называет номера из перечня, а перечень — это только те новые,
    # которые ушли в промпт. Здесь номер возвращается в индекс общего списка.
    allowed_current = {number: index for number, index in enumerate(uncovered, 1)}
    groups: dict[int, list[int]] = {}
    restored: set[int] = set()
    seen_current: dict[int, int] = {}
    dropped_current: set[int] = set()

    for group in answer.groups:
        previous_number = group.previous
        # Номер прошлой вне перечня означает, что модель выдумала находку.
        # Доверять остальным номерам такой группы тоже нельзя: она собрана
        # вокруг несуществующей ссылки.
        if previous_number is None or not 1 <= previous_number <= previous_count:
            logger.warning(
                "%s: модель назвала прошлую находку %s вне перечня — группа "
                "отброшена",
                LogLabel.DEDUP_FINDINGS,
                previous_number,
            )
            dropped_current.update(group.current)
            continue
        previous_index = previous_number - 1
        # Одна прошлая — одна группа. Вторая группа на ту же прошлую значит, что
        # модель сама себе противоречит; какая из двух верна, отсюда не видно.
        if previous_index in groups:
            logger.warning(
                "%s: прошлая находка П%s пришла в двух группах — обе отброшены",
                LogLabel.DEDUP_FINDINGS,
                previous_number,
            )
            dropped_current.update(groups.pop(previous_index))
            dropped_current.update(group.current)
            continue
        members: list[int] = []
        for current_number in group.current:
            index = allowed_current.get(current_number)
            if index is None:
                logger.warning(
                    "%s: модель назвала новую находку %s вне перечня — номер "
                    "отброшен",
                    LogLabel.DEDUP_FINDINGS,
                    current_number,
                )
                continue
            # Новая уже стоит в другой группе — значит одна проблема записана
            # дважды. Обе группы распускаются, находка остаётся сама: свести её
            # в чужую формулировку дороже, чем показать лишний раз.
            if index in seen_current:
                logger.warning(
                    "%s: новая находка Т%s попала в две группы — обе группы "
                    "отброшены",
                    LogLabel.DEDUP_FINDINGS,
                    current_number,
                )
                dropped_current.add(index)
                _drop(groups, index)
                members = []
                continue
            seen_current[index] = previous_index
            members.append(index)
        if members:
            groups[previous_index] = members

    # Находку могли сначала записать в группу, а потом распустить её по ошибке
    # в другой: такие не остаются нигде, и в итог они идут как новые.
    dropped = set(seen_current) - {i for m in groups.values() for i in m}
    dropped_current.update(dropped)
    if dropped_current:
        logger.info(
            "%s: после отбраковки ответа модели без группы остались %s новых — "
            "пойдут в список сами",
            LogLabel.DEDUP_FINDINGS,
            len(dropped_current),
        )

    for number in answer.restored:
        if 1 <= number <= previous_count:
            restored.add(number - 1)
    return groups, restored


def _drop(groups: dict[int, list[int]], index: int) -> None:
    """Снимает находку из группы, где она есть."""
    for previous_index, members in list(groups.items()):
        if index in members:
            members.remove(index)
            if not members:
                groups.pop(previous_index)


def _result(
    previous: Sequence[Issue],
    current: Sequence[Issue],
    groups: dict[int, list[int]],
    restored: set[int],
    model_calls: int,
    degraded: bool,
) -> DedupResult:
    """Собирает итоговый список: прошлые в их порядке, затем новые без пар.

    Слитая находка берёт формулировку у прошлой, а шаги, зону, код, подкритерий
    и улики — у новой. Если в группе несколько новых, факты берутся у новой с
    наименьшим номером: это первая по порядку прогона, а остальные — её дубли.

    :param previous: прошлые находки
    :param current: новые находки
    :param groups: прошлая находка → новые находки того же смысла
    :param restored: прошлые без дублей
    :param model_calls: сколько раз спрашивали модель
    :param degraded: собирали ли результат без модели
    :return: итог сверки
    """
    issues: list[Issue] = []
    merged = 0
    duplicated = 0
    consumed: set[int] = set()

    # Прошлые идут в своём порядке: читатель отчёта знает прошлый прогон и ищет
    # знакомые находки на знакомых местах. Слитая находка стоит на месте своей
    # прошлой, а не в конце.
    for previous_index, issue in enumerate(previous):
        members = groups.get(previous_index)
        if not members:
            issues.append(issue)
            continue
        # Несколько новых на одну прошлую — одна находка. Факты берём у первой
        # по порядку прогона: остальные её дубли, и их номера экранов ничего
        # нового к дефекту не добавляют.
        source = current[min(members)]
        issues.append(_wording_of(issue, source))
        merged += 1
        duplicated += len(members) - 1
        consumed.update(members)

    # Всё, что не ушло в группу, остаётся своим: сегодняшний прогон нашёл это
    # сам, и выбрасывать его находки сверка не вправе.
    added = 0
    for current_index, issue in enumerate(current):
        if current_index in consumed:
            continue
        issues.append(issue)
        added += 1

    result = DedupResult(
        issues=issues,
        restored=len(restored),
        merged=merged,
        added=added,
        duplicated=duplicated,
        model_calls=model_calls,
        degraded=degraded,
    )
    logger.info(
        "%s: прошлых поднято %s, слито %s, новых добавлено %s, дублей погашено "
        "%s%s",
        LogLabel.DEDUP_FINDINGS,
        result.restored,
        result.merged,
        result.added,
        result.duplicated,
        " (без модели)" if degraded else "",
    )
    return result


def _wording_of(previous: Issue, current: Issue) -> Issue:
    """Находка прошлого с фактами сегодняшнего: формулировка прошлая, улики новые.

    Формулировку (`title`, `text`, `rec`) не трогаем: её правил человек, и она
    уже прошла экспертизу. Всё остальное — факт о нынешнем материале: экраны,
    зона, код, улики. Старая цитата могла ссылаться на кадр, которого сегодня
    нет, поэтому `evidence` тоже берётся у новой.
    """
    # Без экспертного подтверждения не закрепляем старую формулировку:
    # pending означает только ожидание решения, а не одобрение текста.
    if previous.review_status not in ("approved", "accepted"):
        return current.model_copy(update={
            "id": previous.id or current.id,
            "review_status": "pending",
        })
    return previous.model_copy(
        update={
            "steps": list(current.steps),
            "zone": current.zone,
            "severity": current.severity,
            "criterion": current.criterion,
            "check_key": current.check_key,
            "defect_code": current.defect_code,
            "defect_ref": current.defect_ref,
            "defect_name": current.defect_name,
            "evidence": list(current.evidence),
            "confidence": current.confidence,
            "kind": current.kind,
        }
    )
