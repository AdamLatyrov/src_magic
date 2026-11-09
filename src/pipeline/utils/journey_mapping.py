"""Разметка находок проверок пути: описание находки — в код справочника.

Шесть проверок ищут проблему, не зная, где она описана в справочнике: у их находок
нет ни блока, ни кода, ни взятой из каталога зоны
(`pipeline.utils.friction_findings`). Здесь у них появляется адрес — код дефекта, его
блок и его зона, — и только. Тексты, улики, номера экранов и зоны незаматченных
находок остаются те же: их поставил тот, кто смотрел кадры, а разметка кадров не
видит и не имеет права ни переформулировать находку, ни передвинуть её.

Разбор ответа общий с поэкранной разметкой и живёт в `pipeline.utils.deviation_mapping`;
здесь — то, чем отличается эта ветка: находки приходят шестью проверками, а
размечаются одним обращением. Номера в перечне сквозные (шесть раз платить за
справочник незачем), поэтому ответ размечается сначала плоско, а потом раскладывается
обратно по проверкам — и потери разметки считаются по той проверке, чью находку они
задели, а не одним общим счётчиком на шесть.

Незаматченная находка остаётся при своей зоне и без кода: «вижу проблему, кода под
неё нет» — обычный итог сверки с каталогом. Такая находка не теряет ничего, кроме
адреса: сведение кладёт её на проверку, которая её нашла
(`pipeline.utils.findings.issues_from_findings`).
"""

from collections import Counter
from collections.abc import Sequence

from src.pipeline.schemas.nodes import CXCriteriaModel, CXFrictionFindings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.utils.deviation_mapping import (
    codes_of,
    mark,
    report_coverage,
)


def friction_mapping(
    answer: str,
    groups: Sequence[tuple[str, CXFrictionFindings]],
    model: CXCriteriaModel,
) -> dict[str, CXFrictionFindings]:
    """Ответ разметки — на находки всех шести проверок сразу.

    Проверки идут в том же порядке, в каком их находки уходили в вопрос: узел
    собирает перечень тем же перечнем групп, что разбирает здесь, иначе код лег бы
    на чужую находку. Пустая проверка в перечень не входила — её ключ возвращается
    без изменений.

    :param answer: ответ модели — перечень соответствий номерам находок
    :param groups: `(ключ проверки, её находки)` в порядке вопроса
    :param model: критериальная модель — по ней код сверяется с блоком и зоной
    :return: находки каждой проверки, где у заматченных есть адрес справочника
    :raises ValueError: если в ответе нет перечня `matches`
    """
    total = sum(len(findings.deviations) for _, findings in groups)
    # Снятые при разборе соответствия потери глобальны: номер вне перечня не
    # принадлежит ни одной проверке, поэтому он остаётся только в журнале
    # `codes_of`, а счётчик каждой проверки считает потери своего окна.
    codes, _global = codes_of(answer, total, LogLabel.MAP_JOURNEY_FRICTIONS)
    marked: dict[str, CXFrictionFindings] = {}
    offset = 0
    for key, findings in groups:
        count = len(findings.deviations)
        mine = {
            number - offset: code
            for number, code in codes.items()
            if offset < number <= offset + count
        }
        losses: Counter[str] = Counter(findings.dropped)
        deviations = mark(
            findings.deviations,
            mine,
            model,
            LogLabel.MAP_JOURNEY_FRICTIONS,
            losses,
        )
        # Покрытие считается по проверке: «разметка не досчиталась кодов» у одной
        # проверки не должно читаться как провал всех шести.
        report_coverage(deviations, losses, LogLabel.MAP_JOURNEY_FRICTIONS)
        marked[key] = findings.model_copy(
            update={"deviations": deviations, "dropped": dict(losses)}
        )
        offset += count
    return marked
