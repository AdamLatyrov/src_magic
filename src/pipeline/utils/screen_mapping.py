"""Разметка находок поэкранного прохода: описание находки — в код справочника.

Ответ прохода описывает видимое и не знает адресов: ни блока, ни кода у его находки
нет (`pipeline.utils.screen_findings`). Здесь находке даётся адрес — код дефекта, его
блок и его зона, — и только. Тексты, улики, номера экранов и координаты остаются те
же: их поставил тот, кто смотрел кадр, а разметка кадров не видит — ей бы иначе
пришлось вторично оплачивать весь путь картинками, и она вернула бы координаты,
которые никто не видел.

Сам разбор ответа общий для обеих размечаемых веток и живёт в
`pipeline.utils.deviation_mapping`: та же сверка номера с перечнем, тот же отказ брать
код вне справочника и то же правило «зону называет справочник». Здесь остаётся
только то, чем поэкранная ветка отличается: её находка держит пин и описание
экрана, и размеченный перечень возвращается в контейнер прохода целиком.

Незаматченная находка остаётся со своей зоной и без кода — тот же смысл, что
служебное `uncatalogued`: «вижу проблему, кода под неё нет» не выбрасывается, а
находит адрес позже, в сведении (`pipeline.utils.findings`).
"""

from collections import Counter

from src.pipeline.schemas.nodes import CXCriteriaModel, CXScreenFindings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.utils.deviation_mapping import (
    codes_of,
    mark,
    report_coverage,
)


def screen_mapping(
    answer: str, found: CXScreenFindings, model: CXCriteriaModel
) -> CXScreenFindings:
    """Ответ разметки — на те же находки, что пришли из прохода.

    :param answer: ответ модели — перечень соответствий находкам
    :param found: итог прохода: находки с пинами и описания кадров
    :param model: критериальная модель — по ней код сверяется с блоком и зоной
    :return: итог прохода, где у заматченных находок есть код, блок и зона
        справочника
    :raises ValueError: если в ответе нет перечня `matches`
    """
    codes, dropped = codes_of(
        answer, len(found.deviations), LogLabel.SUMMARIZE_SCREEN
    )
    losses: Counter[str] = Counter(found.dropped)
    losses.update(dropped)
    deviations = mark(
        found.deviations,
        codes,
        model,
        LogLabel.SUMMARIZE_SCREEN,
        losses,
    )
    report_coverage(deviations, losses, LogLabel.SUMMARIZE_SCREEN)
    return found.model_copy(
        update={"deviations": deviations, "dropped": dict(losses)}
    )
