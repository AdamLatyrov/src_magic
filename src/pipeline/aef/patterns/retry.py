"""Ретраи диалоговых узлов: за что переспросить контур и чем деградировать.

Модуль лежит рядом с провайдером не случайно: предикат повторимости знает
ровно про сбои контура (`pipeline.aef.exceptions`), транспорт и форму ответа —
про то, что описывает этот слой, и ничего не знает про конкретный граф.
Числа местных повторов — `pipeline.settings.params.retry`, числа повторов узла и
константы контракта графа — `RetryOptions` (`pipeline.aef.models`).

Обращения к контуру падают одинаково, кто бы о нём ни спрашивал, поэтому
правило «что повторять» собрано здесь одним предикатом: узел сам решает, что
из его разбора заслуживает повтора, и поднимает наверх `MalformedAnswer`
(`pipeline.aef.exceptions`).

Обработчик — узел сам по себе, и langgraph не рисует из него переходов: кому
что делать после деградации, решает сборщик графа (`error_handler_node`).
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.types import RetryPolicy

from src.pipeline.aef.exceptions import (
    GigaChatException,
    GigaChatModelNotAvailable,
    GigaChatTooManyRequests,
    MalformedAnswer,
)
from src.pipeline.aef.models import RetryOptions
from src.pipeline.settings.params.retry import (
    ASK_ATTEMPTS,
    ASK_RETRY_PAUSE_S,
    SHAPE_RETRY_PAUSE_S,
)

logger = logging.getLogger(__name__)


def error_handler_node(node_name: str) -> str:
    """Имя узла-обработчика, который langgraph заводит для узла `node_name`.

    Обработчик генерируется при добавлении узла, но никаких переходов из него
    graph не рисует: не вытянуть ребро — и деградированный узел становится
    тупиком, за которым молча обрывается весь остальной маршрут (проверено на
    живом прогоне).
    """
    return f"{RetryOptions.HANDLER_PREFIX}{node_name}"


def is_shape_failure(exc: Exception) -> bool:
    """Похожа ли ошибка на неразборчивый ответ, а не на сбой обращения.

    :param exc: ошибка, прервавшая узел
    :return: True — ответ есть, но он не проходит схеме
    """
    return isinstance(exc, RetryOptions.SHAPE_FAILURES)


def failure_of(exc: Exception) -> str:
    """Сбой обращения в одну строку, всегда с названием ошибки.

    У сетевых сбоев сообщение пустое — `httpx.ConnectError` из `httpcore`
    приходит без текста, и отказ, собранный через `{exc}`, читается как «проверка
    не отвечена: », то есть как пустой ответ модели вместо сбоя обращения.

    :param exc: ошибка, прервавшая узел
    :return: текст сбоя; пустому сообщению на место встаёт имя класса
    """
    text = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


def llm_retryable(exc: Exception) -> bool:
    """Достойно ли повторить обращение к контуру после этой ошибки.

    Ретраится то, что повтором и лечится: форма ответа, очередь контура (429),
    транспорт и 5xx. Постоянные ошибки (нет модели, права, 4xx) — нет: повтор
    них деньги на ветер.

    :param exc: ошибка, прервавшая узел
    :return: True — граф сделает ещё попытку
    """
    if isinstance(exc, MalformedAnswer):
        return True
    if isinstance(exc, GigaChatTooManyRequests):
        return True
    if isinstance(exc, GigaChatModelNotAvailable):
        return False
    if isinstance(exc, GigaChatException):
        # код ответа тонет в тексте сообщения — читаем его оттуда;
        # без кода считаем ошибку финальной, выдумывать ретрай нечем
        code = re.search(r"Code:\s*(\d{3})", str(exc))
        return code is not None and int(code.group(1)) >= 500
    return isinstance(exc, (httpx.RequestError, ConnectionError, TimeoutError))


def llm_retry_policy(
    options: RetryOptions | None = None,
    retry_on=llm_retryable,
) -> RetryPolicy:
    """Политика диалоговых узлов: по умолчанию до пяти попыток, паузы 1—2—4—8 с.

    Первая пауза маленькая нарочно: битый ответ — щедрость генерации, а не
    очередь контура, ждать минуту смысла нет; 429 тем временем прикрывается
    растущей паузой.

    :param options: числа повторов — по умолчанию `RetryOptions`
    :param retry_on: предикат повторимости — своя градация поверх общей
    :return: политика для `add_node(retry_policy=...)`
    """
    options = options or RetryOptions()
    return RetryPolicy(
        max_attempts=options.max_attempts,
        initial_interval=options.initial_interval,
        backoff_factor=options.backoff_factor,
        max_interval=options.max_interval,
        jitter=options.jitter,
        retry_on=retry_on,
    )


async def repeat_recoverable[T](
    ask: Callable[[], Awaitable[T]],
    subject: str,
    attempts: int = ASK_ATTEMPTS,
    pause_s: float = ASK_RETRY_PAUSE_S,
) -> T:
    """Одно обращение к контуру, повторённое ровно до предела его сбоя.

    Повторяет то, что повтором и лечится: форму ответа да очередь контура — по тому
    же предикату, по которому перевызывает узел политика графа. Неповторяемое (нет
    модели, права, 4xx кроме очереди) поднимается сразу: повторить его — деньги на
    ветер, а причину узел обязан увидеть.

    Живёт здесь, а не в узле: обращения к контуру падают одинаково, кто бы о нём ни
    спрашивал. Узлы задают свои вопросы батчами одним циклом — окно за окном, как
    ветка блоков, — и перевызывают себя целиком только когда не отвечен ВЕСЬ вопрос:
    пока отвечен хоть один, политика графа не просыпается, и сорванный вопрос молча
    оставался бы непроверенной частью справочника.

    :param ask: одно обращение — обращение к контуру и его разбор
    :param subject: о чём вопрос, в одну строку для лога
    :param attempts: сколько раз задать вопрос
    :param pause_s: пауза перед повтором очереди, растущая с попыткой
    :return: то, что вернуло обращение
    :raises Exception: сбой, который повтор не лечит, либо последний из повторимых
    """
    failed: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await ask()
        except Exception as ex:
            if not (is_shape_failure(ex) or llm_retryable(ex)):
                raise
            failed = ex
            if attempt < attempts:
                logger.warning(
                    "%s: %s — спрашивается заново (%s попытка из %s)",
                    subject,
                    failure_of(ex),
                    attempt + 1,
                    attempts,
                )
                await asyncio.sleep(pause_before_repeat(ex, attempt, pause_s))
    assert failed is not None
    raise failed


def pause_before_repeat(ex: Exception, attempt: int, pause_s: float) -> float:
    """Сколько ждать перед повтором после этого сбоя.

    Неразобранный ответ повторяют сразу, очередь контура — с паузой, растущей с
    попыткой: у щедрости генерации нет очереди, которой надо дождаться, а очередь за
    секунды не рассасывается. Узел оборачивает неразобранный ответ в
    `MalformedAnswer`, поэтому короткая пауза положена и ей, а не только сбоям схемы
    из `RetryOptions.SHAPE_FAILURES`.

    :param ex: сбой, из-за которого обращение повторяют
    :param attempt: какая попытка только что кончилась
    :param pause_s: шаг паузы для сбоя контура
    :return: пауза в секундах
    """
    if is_shape_failure(ex) or isinstance(ex, MalformedAnswer):
        return SHAPE_RETRY_PAUSE_S
    return pause_s * attempt


def degradation_handler(update: dict, subject: str):
    """Терминальный обработчик узла: пишет деградацию, не обрывая прогон.

    Вызывается графом, когда узел так и не закончился успешно — после
    исчерпанных попыток. Аргументом приходит вход упавшей задачи, сама ошибка
    лежит в конфигурации запуска под `RetryOptions.NODE_ERROR_KEY`; без неё
    лог был бы немым.

    :param update: поля состояния, которыми узел деградирует
    :param subject: имя узла для лога
    :return: обработчик для `add_node(error_handler=...)`
    """

    async def mark_degraded(_state, config: RunnableConfig) -> dict:
        node_error = (config.get("configurable") or {}).get(
            RetryOptions.NODE_ERROR_KEY
        )
        cause = getattr(node_error, "error", node_error)
        logger.warning(
            "%s: без результата (%s)",
            subject,
            failure_of(cause) if isinstance(cause, Exception) else cause,
        )
        return update

    return mark_degraded
