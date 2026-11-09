"""Разбор ответа о том, что это за клиентский путь: вердикт и его полнота.

Здесь вся политика приёмки ответа распознавания. Сам узел её не знает: он только
спрашивает контур и кладёт его ответ в состояние как есть, а чем этот ответ
считать — вердиктом, пустотой или сбоем узла — решают `from_scan` и `verdict_of`
ниже. Сюда же легли правила по описаниям кадров (`unread`, `stop_reason`,
`refusal_note`, `describe_scan`):
их читают и маршрут цикла скана, и узел сведения, а узлам графа заглядывать внутрь
друг друга незачем. По той же причине разбор живёт вне графа: состояние собирают и
без прогона (переигрывание записанного ответа, сверка распознавания в ноутбуках), и
политика приёмки обязана выходить одной и той же при любом способе наполнения
состояния.

Форму ответа задаёт структурированный вывод (`src/schemas/nodes.py`), поэтому
текст модели здесь больше не разбирается: состоянию нечего вытаскивать из JSON,
когда контур обязан ответить схемой.
"""

import re
from collections.abc import Sequence
from typing import Optional

from src.pipeline.schemas.nodes import CXFrameReadout, CXJourneyDetails
from src.pipeline.settings.params.nodes import (
    INCOMPLETE,
    NAME_IN_ROUTE,
    NO_ANSWER,
    VALUE_FOUND,
    VALUE_FRAMES,
    VALUE_IN_FRAME,
    CXNodeStatus,
)


def named_values(text: str) -> list[str]:
    """Величины одного кадра, как их переписал скан: порядком и без повторов.

    :param text: дословный текст кадра из описания скана
    :return: величины кадра, не больше `VALUE_IN_FRAME`
    """
    named: list[str] = []
    for raw in VALUE_FOUND.findall(text):
        value = " ".join(raw.split())
        if value and value not in named:
            named.append(value)
    return named[:VALUE_IN_FRAME]


def from_scan(
    verdict: Optional[bool],
    details: Optional[CXJourneyDetails],
    reason: str = "",
) -> Optional[CXJourneyDetails]:
    """Атрибуция пути по описаниям кадров: вердикт скана, названия — от модели.

    Кадры видел скан, он и решает, путь это или нет; модель по текстовым
    описаниям называет компанию, продукт, сценарий и тип пути. Здесь они
    склеиваются и не спорят: `is_journey` всегда берётся из `verdict`, остальное
    — из ответа модели. Форма строга только в вердикте, названия мягкие: пустая
    строка лучше выдуманного продукта.

    * вердикта нет — ни один кадр не был отвечен: атрибуции нет вовсе, прогон
      идёт вслепую вместо отклонения;
    * «не путь» — ответ самодостаточен: причин продуктом не объясняют, а вот
      слова модели о материале стоит сохранить (`note`, иначе — `reason`);
    * подтверждённый путь без продукта и сценария остаётся путём — названия могут
      и не найтись в описаниях, нехватка видна по `degraded`.

    :param verdict: вердикт сканирования: `True`, `False` или `None` — молчание
    :param details: что модель назвала по описаниям кадров; пусто — ответа не было
    :param reason: чем скан объяснил «не путь» (слова модели с кадров)
    :return: атрибуция пути либо `None`, если вердикта нет
    """
    if verdict is None:
        return None

    accepted = (details or CXJourneyDetails(is_journey=verdict)).model_copy(
        update={"is_journey": verdict}
    )
    if not verdict:
        no_journey_note = (
            "Ни на одном кадре не видно интерфейса приложения или сайта."
        )
        return accepted.model_copy(
            update={
                "organization": "",
                "basis": "",
                "product": "",
                "scenario": "",
                "journey_type": None,
                "degraded": "",
                "note": accepted.note.strip()
                or reason.strip()
                or no_journey_note,
            }
        )
    if is_incomplete(accepted):
        return accepted.model_copy(update={"degraded": INCOMPLETE})
    return accepted


def unread(step: int) -> CXFrameReadout:
    """Описание кадра, которого контур не дал.

    Кадр обязан остаться в состоянии: без записи цикл сканирования стоял бы на
    нём вечно (номер следующего кадра считается по числу описаний), а в отчёте
    пропущенный кадр был бы неотличим от несуществующего.

    :param step: номер кадра пути, по которому ответа нет
    :return: описание кадра со статусом ошибки узла
    """
    return CXFrameReadout(
        step=step, status=CXNodeStatus.ERROR, degraded=NO_ANSWER
    )


def stop_reason(readouts: Sequence[CXFrameReadout], stop_after: int) -> str:
    """Пора ли оборвать скан; пусто — продолжать до последнего кадра.

    Оба условия считают по ПЕРВЫМ `stop_after` разобранных кадров и оба молчат
    при `stop_after<=0`: выключенное правило не имеет права урезать путь.

    :param readouts: описания кадров, разобранные до текущего момента
    :param stop_after: сколько первых разобранных кадров без пути терпим; 0 —
        правило выключено
    :return: причина остановки текстом либо пусто, если повод не найден
    """
    if stop_after <= 0:
        return ""

    answered = [readout for readout in readouts if readout.answered]
    silent = [readout for readout in readouts if not readout.answered]

    if len(answered) >= stop_after and not any(
        readout.is_journey for readout in answered
    ):
        return (
            f"клиентский путь не найден среди первых {len(answered)}"
            " разобранных кадров"
        )
    if len(silent) >= stop_after:
        return f"контур не ответил ни по одному из первых {len(silent)} кадров"
    return ""


def refusal_note(readouts: Sequence[CXFrameReadout], reason: str) -> str:
    """Почему прогон отклонён: слова модели с кадров, а не наша догадка.

    :param readouts: описания кадров прогона
    :param reason: причина остановки скана текстом
    :return: объяснение для вердикта «не путь»
    """
    seen = [
        readout.why_not_journey.strip()
        for readout in readouts
        if readout.answered and readout.is_journey is False
    ]
    explained = next((note for note in seen if note), "")
    return "; ".join(part for part in (reason, explained) if part)


def route_trace(readouts: Sequence[CXFrameReadout]) -> str:
    """Экраны пути одной строкой: номер кадра и то, чем он назван.

    Строка даёт структуру маршрута там, где изображений не показывают: проверка
    видит окно в десять кадров и по нему не различит, длинный путь или короткий,
    был ли экран первым и возвращались ли на него. Перечень экранов отвечает на это
    текстом, а улику по-прежнему берут только с приложенного кадра — поэтому здесь
    только названия, без описаний и без цитат.

    :param readouts: описания кадров прогона в порядке пути
    :return: строка вида `1 · Вход → 2 · Кошелёк`, пустая — если кадров нет
    """
    parts: list[str] = []
    for readout in readouts:
        if readout.need_filter:
            continue
        name = " ".join((readout.screen or "").split())[:NAME_IN_ROUTE]
        parts.append(f"{readout.step} · {name or 'экран не назван'}")
    return " → ".join(parts)


def screen_texts(readouts: Sequence[CXFrameReadout]) -> str:
    """Тексты кадров всего пути, каждый со своим номером и целиком.

    Изображения в вопросе блока есть и сами — но только кадров этого батча, а их
    умещается столько, сколько держит одно обращение
    (`max_images_per_request`), тогда как экранов анкетирования на пути сорок.
    След не заменяет кадры, а доносит надписи тех, которых в батче не было: по
    нему ветка блоков считает длину пути и число поля́ там, где её изображений
    на весь путь не хватило.

    Судит ветка по тому, что увидела: в прогоне 047 она вернула десять "ok",
    переписав заголовок подкритерия, — считать экраны и поля ей было нечем.

    Текст кадра идёт целиком, без среза: условие и последствия сидят в хвосте
    длинного экрана. В прогоне 002 перечень обрывался на «Ваш годовой доход по
    этому счету —», и вместе с хвостом из вопроса выпадали шкала вычета и строка
    «если вы закроете ИИС раньше срока владения, вычеты на взнос придется вернуть
    ФНС» — проверка объявила подкритерий соблюдённым по экрану, которого ей не
    показали.

    :param readouts: описания кадров прогона в порядке пути
    :return: строки вида `[кадр 3 · Анкета] <текст>`, пустая — если текстов нет
    """
    lines: list[str] = []
    for readout in readouts:
        if readout.need_filter:
            continue
        text = " ".join((readout.text_on_screen or "").split())
        if not text:
            continue
        name = " ".join((readout.screen or "").split())[:NAME_IN_ROUTE]
        title = f"{readout.step} · {name}" if name else f"{readout.step}"
        lines.append(f"[кадр {title}] {text}")
    return "\n".join(lines)


def screen_tally(readouts: Sequence[CXFrameReadout]) -> str:
    """Счёт пути, посчитанный кодом: сколько кадров и сколько экранов.

    Мера половины подкритериев — число («4 — лучший UX, 8+ — плохо»), а счёт по
    десяти приложенным кадрам модель не ведёт: в прогоне 047 она поставила "ok"
    на одиннадцатиэкранном пути, пересказав меру вместо счёта. Здесь число
    считает код по тем же надписям экрана, которые скан переписал с кадра:
    соседние кадры с одной надписью — один экран (скролл, подгрузка, ввод в ту
    же форму). Модель этот счёт сверяет с мерой, а не добывает.

    :param readouts: описания кадров прогона в порядке пути
    :return: строка вида `кадров: 11 · экранов: 9`, пустая — если кадров нет
    """
    kept = [readout for readout in readouts if not readout.need_filter]
    screens = 0
    previous: str | None = None
    for readout in kept:
        name = " ".join((readout.screen or "").split()).casefold()
        if not name or name != previous:
            screens += 1
        previous = name
    if not kept:
        return ""
    return f"кадров: {len(kept)} · экранов: {screens}"


def value_lines(
    readouts: Sequence[CXFrameReadout], steps: Optional[set[int]] = None
) -> str:
    """Кадры со своими величинами одной строкой; `steps` — только эти кадры.

    :param readouts: описания кадров прогона в порядке пути
    :param steps: номера кадров, которые нужны; пусто — все, что не снято с анализа
    :return: строка вида `12 · 108,59 ₽ · 10 000 ₽; 15 · …`, пустая — если величин нет
    """
    lines: list[str] = []
    for readout in readouts:
        if readout.need_filter:
            continue
        if steps is not None and readout.step not in steps:
            continue
        named = named_values(readout.text_on_screen)
        if named:
            lines.append(f"{readout.step} · " + " · ".join(named))
    return "; ".join(lines[:VALUE_FRAMES])


def value_trace(
    readouts: Sequence[CXFrameReadout],
    hidden: Sequence[int] = (),
    only: Sequence[int] = (),
) -> str:
    """Величины пути по кадрам — с номером каждого кадра.

    Проверка идёт окнами по десять кадров, а меры пути сравнивают величины, которые
    живут на разных концах маршрута: остаток на счёте из начала пути и сумму, которую
    оформляют в конце. Окно такой сверки сделать не может — второго кадра оно не
    видит, и снятая по этой причине мера выглядела бы как «путь чист». Здесь лежат
    величины, переписанные сканом дословно, и каждая — со своим номером кадра: сверять
    по ним можно, а уликой остаётся приложенный кадр — это сказано в самом блоке
    вопроса.

    Величины приложенных кадров (`only`) нужны по другой причине: на окне в шесть
    экранов модель переписывает набор величин первого кадра остальным и сводит меру
    вслепую — в прогоне 014 шесть подряд кадров получили «10 000 ₽ · 10,8%», хотя на
    двух из них лежал остаток 108,59 ₽, та самая пара, ради которой мера и придумана.
    Перечень величин своего кадра не даёт подменить один кадр другим.

    :param readouts: описания кадров прогона в порядке пути
    :param hidden: номера кадров, которые в перечень не входят — их модель видит
    :param only: номера кадров, которые нужны; пусто — все, кроме `hidden`
    :return: строка вида `12 · 108,59 ₽ · 10 000 ₽; 15 · …`, пустая — если величин нет
    """
    wanted = set(only)
    if wanted:
        return value_lines(readouts, steps=wanted)
    shown = set(hidden)
    return value_lines(
        [readout for readout in readouts if readout.step not in shown]
    )


def describe_scan(
    readouts: Sequence[CXFrameReadout], frames_total: int, reason: str = ""
) -> str:
    """Одна строка журнала о состоянии скана: сколько пройдено и чем пахнет.

    :param readouts: описания кадров, разобранные до текущего момента
    :param frames_total: сколько кадров в пути всего
    :param reason: причина остановки, если скан оборван
    :return: строка для лога
    """
    answered = [readout for readout in readouts if readout.answered]
    journey = sum(1 for readout in answered if readout.is_journey)
    refused = sum(1 for readout in answered if readout.is_journey is False)
    verdict = "путь" if journey else ("не путь" if refused else "")
    tail = f", причина: {reason}" if reason else ""
    return (
        f"кадров {len(readouts)} из {frames_total}, с описанием"
        f" {len(answered)}, кадров пути {journey}"
        f"{f', вердикт: {verdict}' if verdict else ''}{tail}"
    )


def is_incomplete(details: CXJourneyDetails) -> bool:
    """Не хватает ли деталям подтверждённого вердикта опоры.

    Отрицательный вердикт самодостаточен: «это не путь» — готовый ответ, объяснять
    его продуктом и сценарием нечем. Утвердительный без продукта и сценария остаётся
    вердиктом — на нём прогон и идёт, — но названия пути взять неоткуда, и это видно
    по `degraded` в ответе прогона.

    Названное имя без цитаты с кадра неполно по той же причине: `basis` — единственное,
    чем название проверяется. Прогон 032 называл продуктом вывеску раздела («Долевой
    капитал»), хотя в том же материале оставались надписи «Страховые покрытия» и
    «Выгодоприобретатели»; домены от этого расходились, и ниже по конвейеру на расхождение
    ссылались уже как на причину снять проверку.
    """
    if not details.is_journey:
        return False

    detail_keys = ("product", "scenario")
    if not all(getattr(details, key).strip() for key in detail_keys):
        return True

    frame_citation = re.compile(r"кадр[а-яё]*\s*[№#]?\s*\d+", re.IGNORECASE)
    return bool(details.product.strip()) and not frame_citation.search(
        details.basis
    )


def verdict_of(
    details: Optional[CXJourneyDetails],
) -> Optional[bool]:
    """Подтверждённый вердикт «путь / не путь»; `None` — вердикта нет.

    Останавливать прогон имеет право только успешный ответ узла. Узел без ответа
    тоже хранит `is_journey=False` (выдумывать подтверждённый путь нельзя), и читать
    это поле напрямую значило бы обрывать прогон при сбое контура.

    :param details: атрибуция пути из состояния прогона
    :return: `True` — путь, `False` — не путь, `None` — вердикта нет
    """
    if (
        details is None
        or CXNodeStatus(details.status) is not CXNodeStatus.SUCCESS
    ):
        return None
    return details.is_journey


def describe(details: Optional[CXJourneyDetails]) -> str:
    """Одна строка журнала о результате распознавания: вердикт и что распознано.

    Строка собирает то, что состояние уже знает, и ничего к нему не добавляет:
    печатать результат распознавания принято там, где его впервые читают. Тип пути
    печатается здесь, хотя в вопрос веток он не уходит: сверять распознавание по
    журналу всё равно нужно по всем четырём меткам, а ветки читают три.

    :param details: атрибуция пути из состояния прогона
    :return: одна строка о вердикте, статусе и распознанных названиях
    """
    if details is None:
        return "распознавание не запускалось"

    reason = f", degraded={details.degraded}" if details.degraded else ""
    recognized = details.context().replace("\n", "; ") or "деталей нет"
    journey_type = details.journey_type_value
    if journey_type:
        recognized += f"; Тип пути: {journey_type}"
    return (
        f"is_journey={details.is_journey}, статус {details.status}{reason}: "
        f"{recognized}"
    )
