"""JSON в ответе модели: одна выжимка объекта на весь контур разбора."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def json_payload(answer: str, expected: str) -> dict:
    """Объект из ответа модели.

    Ответ ищется жадно от первой `{` до последней `}`: модель оборачивает JSON в
    markdown и добавляет пояснения вне объекта.

    Не читающийся объект разбирается по целому началу (`_whole_answer`) — см. его
    докстринг про оборванные ответы контура. Если не читается и целое начало,
    остаётся исходная ошибка строгого разбора: неразобранный ответ должен быть
    ошибкой, а не подменой.

    :param answer: ответ модели
    :param expected: что именно ожидается в ответе — попадает в текст ошибки
    :return: разобранный JSON-объект
    :raises ValueError: в ответе нет JSON либо он не читается даже целым началом
    """
    JSON_ANSWER = re.compile(r"\{.*\}", re.DOTALL)
    matched = JSON_ANSWER.search(answer or "")
    strict_error: ValueError | None = None
    if matched is not None:
        try:
            return json.loads(matched.group(0))
        except ValueError as error:
            strict_error = error

    whole = _whole_answer(answer)
    if whole is None:
        if strict_error is not None:
            raise strict_error
        if (answer or "").find("{") < 0:
            raise ValueError(f"в ответе модели нет JSON {expected}")
        raise ValueError(
            f"ответ модели оборван на самом начале — целого начала нет {expected}"
        )
    logger.warning(
        "ответ модели оборван — разобрано целое начало: %s зн. из %s, "
        "отброшено %s (%s)",
        whole[1],
        len(answer),
        len(answer) - whole[1],
        expected,
    )
    return whole[0]


def _whole_answer(answer: str) -> tuple[dict, int] | None:
    """Целое начало оборванного ответа: сгенерированное до обрыва.

    Контур обрывает длинные ответы на середине — прямо посреди элемента
    перечня или с лишней закрывающей скобкой на месте потерянной. Строгий разбор
    такой ответ отбрасывает целиком, и узел перевызывает контур: а повтор ничего
    не меняет, ответ той же длины обрывается на том же месте (проверено на
    журнале вызовов — две попытки одной проверки дали ту же колонку). При этом
    всё содержательное — вердикты и находки — сгенерировано ДО обрыва, и цена
    ошибки в том, что ветка теряет весь экран ради хвоста, который модель не
    успела дописать.

    Здесь ничего не додумывается: снимается недписанный хвост (в том числе
    незакрытая строка) и дописываются ровно те скобки, которые открыты в целой
    части. Если целого начала нет — возвращается None, и разбор остаётся ошибкой.

    :param answer: ответ модели целиком
    :return: разобранный объект и число знаков целого начала либо None
    """
    PAIRS = {"[": "]", "{": "}"}
    start = (answer or "").find("{")
    if start < 0:
        return None
    fragment = answer[start:]
    stack: list[str] = []
    cuts: list[tuple[int, tuple[str, ...]]] = []
    in_string = escaped = False
    for position, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in PAIRS:
            stack.append(PAIRS[char])
        elif char in "}]":
            if not stack or char != stack[-1]:
                break
            stack.pop()
            if stack:
                cuts.append((position + 1, tuple(stack)))
        elif char == "," and stack:
            cuts.append((position, tuple(stack)))
    for cut, opened in reversed(cuts):
        candidate = fragment[:cut].rstrip().rstrip(",") + "".join(
            reversed(opened)
        )
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload, start + cut
    return None
