"""Подлинность улики: цитата обязана читаться в тексте разобранного кадра.

Цитата — единственное, чем держится чёрная и красная зона, и до сих пор код
смотрел только на то, что перечень улик непустой (`_zone_of`, `_gated_zone`).
Выдуманная цитата проходила этой же сверкой: в прогоне 047 проверка доверия
поставила `black` на пару «кадр 3: „Комиссия до 1%" — кадр 6: списание 410 ₽ с
25 000 ₽», хотя распознавание не переписало с кадров этого пути ни одной величины,
кадр 3 был экраном выбора группы инвалидности, а названных чисел не было ни в
материале, ни в описаниях кадров. Балл блока заплатил за это четверть шкалы.

Здесь одна проверка, общая для веток: номер кадра из строки улики и сама надпись
сверяются с текстом, который скан снял с этого кадра (`frame_readouts`). Сверка
терпелива к оформлению — регистр, кавычки, переносы строк и знаки препинания тексты
скана и ответ модели разводят — и строга к содержанию: чего в тексте кадра нет, то
не улика, а число, спрятанное внутри другого числа («410» в «4100»), — тем более.

Кадр без текста (распознавание молчало, кадр снят с анализа) неверифицируем: такая
сверка ничего не знает об улике и не имеет права её снимать. Снимает улику и
выбирает тяжесть вызывающая ветка — здесь только ответ «читается или нет».
"""

from collections.abc import Iterable, Mapping, Sequence

from src.pipeline.schemas.nodes import CXFrameReadout
from src.pipeline.settings.params.evidence import (
    NOT_WORD,
    NUMBER_RUNS,
    OMISSION,
    QUOTE_SCREEN,
    QUOTED,
)


def frame_texts(readouts: Sequence[CXFrameReadout]) -> Mapping[int, str]:
    """Тексты кадров по номерам — то, с чем сверяется цитата.

    Рядом с переписанным текстом экрана стоит и его название (`screen`): вопрос
    показывает перечень экранов (`[ЧАСТЬ ПУТИ]`), и им же модель подписывает свой
    экран — «Загрузка документов» вместо надписи «Загрузите документы» есть тот же
    экран, а не выдумка. Название берётся только своего кадра: подпись чужого
    экрана этой поблажкой не оправдывается.

    Снимается с анализа только то, что снято с разбора пути: такой кадр не может
    быть ни уликой, ни опровержением, и сверять по нему нельзя.

    :param readouts: описания кадров прогона
    :return: нормализованный текст кадра по его номеру
    """
    return {
        readout.step: flat(
            " ".join((readout.text_on_screen or "", readout.screen or ""))
        )
        for readout in readouts
        if not readout.need_filter
    }


def flat(text: str) -> str:
    """Текст одними буквами и цифрами: оформление двух текстов сводит к одному.

    :param text: любой из двух текстов одного экрана
    :return: строка без пробелов, знаков и регистра
    """
    return NOT_WORD.sub("", str(text or "").casefold().replace("ё", "е"))


def digit_runs(text: str) -> set[str]:
    """Ряды цифр сглаженного текста: «25 000» и «25000» дают один ряд.

    Нужен рядом с проверкой вхождения: «410 ₽» читается внутри «4100 ₽» как
    подстрока, а уликой не является — выдуманное число прячется именно так.

    :param text: сглаженный текст кадра или осколок цитаты (`flat`)
    :return: ряды цифр этого текста
    """
    return set(NUMBER_RUNS.findall(text))


def quote_screen(quote: str, default: int) -> int:
    """Номер кадра, которому улика себя приписала; 0 — кадра нет.

    :param quote: строка улики («кадр 3: „Продолжить"»)
    :param default: экран самой находки, когда номер в строке не назван
    :return: номер экрана 1-based либо 0
    """
    found = QUOTE_SCREEN.match(str(quote or ""))
    return int(found.group(1)) if found else default


def quoted_text(quote: str) -> str:
    """Сама надпись из строки улики: кавычки, а не подпись кадра.

    Цитата без кавычек берётся целиком за вычетом подписи «кадр N:» — модель
    подписывает кадр то с двоеточием, то без него.

    :param quote: строка улики
    :return: надписанная часть строки
    """
    raw = str(quote or "")
    found = QUOTED.search(raw)
    if found:
        return found.group(1)
    _, _, tail = raw.partition(":")
    return tail or raw


def reads_on_frame(quote: str, texts: Mapping[int, str], default: int) -> bool:
    """Читается ли цитата на кадре, которому она приписана.

    True значит и «читается», и «сверить было нечем»: неверифицируемая улика
    остаётся в силе, потому что молчание распознавания не опровергает кадр.
    False ставится только тогда, когда текст кадра есть и надписи в нём нет.

    :param quote: строка улики
    :param texts: тексты кадров по номерам (`frame_texts`)
    :param default: экран находки — если номер в строке не назван
    :return: есть ли надпись на подписанном кадре
    """
    screen = quote_screen(quote, default)
    text = texts.get(screen, "")
    if not text:
        return True
    # Осколки пропускаются ДО сглаживания: многоточие сглаживатель и съедает.
    parts = [flat(part) for part in OMISSION.split(quoted_text(quote))]
    fragments = [part for part in parts if part]
    if not fragments:
        # Подпись кадра без надписи («кадр 6») доказательством не была и раньше.
        return True
    on_frame = digit_runs(text)
    return all(
        part in text and digit_runs(part) <= on_frame for part in fragments
    )


def verified_quotes(
    quotes: Iterable[str], texts: Mapping[int, str], default: int
) -> tuple[list[str], list[str]]:
    """Цитаты, разделённые на прочитанные на кадре и на выдуманные.

    :param quotes: строки `evidence` находки
    :param texts: тексты кадров по номерам (`frame_texts`)
    :param default: экран находки для цитат без своего номера
    :return: (улики в силе, улики, которых на кадре нет)
    """
    kept: list[str] = []
    phantom: list[str] = []
    for quote in quotes:
        where = phantom if not reads_on_frame(quote, texts, default) else kept
        where.append(quote)
    return kept, phantom
