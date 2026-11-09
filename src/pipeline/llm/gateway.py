"""Шлюз к контуру модели: один вопрос по настройкам этапа → разобранный ответ.

Узлы графа спрашивают контур каждый по-своему: распознавание ловит ошибку формы
и перевызывает, сведение пишет журнал и гасит пустоту, ветки блоков оборачивают
это политикой повторов графа. Общий у них ровно один шаг — собрать запрос по
настройкам этапа и получить разобранный по схеме ответ. Шаг вынесен сюда, чтобы
его мог повторить всякий, кто работает с контуром вне графа: лаборатория,
ноутбук, разовый замер. Без этого слоя такой caller либо импортирует узел целиком,
либо копирует сборку запроса у себя — и правка промпта в одном месте перестаёт
доходить до другого.

Промпт и модель здесь не задаются: их берёт настройки этапа
(`BaseSummarizeSettings`), которые читают пару `prompts/<стадия>/` и снимок
исполнения задания. Провайдер, если не передан, собирается по настройкам контура
процесса (`integration.core.config.Settings.gigachat`) — тем же способом, каким его
собирает работник (`integration.service.runtime.build_provider`).

Отказ делится на два вида, как в узлах графа: ответ не разобран по схеме —
`MalformedAnswer` (деньги за него уплачены, перевызывать имеет смысл), прочее —
как есть, со своим текстом. Повторы шлюз не делает: их политика у вызывающей
стороны, потому что число попыток и паузы у графа, лаборатории и ноутбука разные.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.pipeline.aef.exceptions import MalformedAnswer
from src.pipeline.aef.models import ChatRequest
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.aef.patterns.retry import failure_of, is_shape_failure
from src.pipeline.settings.api.graph import BaseSummarizeSettings

logger = logging.getLogger(__name__)

Struct = TypeVar("Struct", bound=BaseModel)


async def ask(
    settings: BaseSummarizeSettings,
    schema: type[Struct],
    *,
    variables: Optional[dict[str, Any]] = None,
    messages: Optional[list[BaseMessage]] = None,
    provider: Optional[GigaChatProvider] = None,
) -> Struct:
    """Спрашивает контур по настройкам этапа и разбирает ответ по схеме.

    Вопрос собирается одним из двух способов: `variables` заполняют пару
    `prompts/<стадия>/system_prompt.yaml` + `user_prompt.yaml` этапа, `messages`
    задают сообщения напрямую — так спрашивает ноутбук, когда промпт ещё не
    отлит в файл. Оба сразу не передаются: два вопроса в одном запросе
    означали бы, что один из них молча не дошёл.

    Модель и параметры генерации берутся у самого этапа (`stage_model`,
    `chat_options`), поэтому снимок исполнения задания соблюдается: прогон,
    поставленный до перезапуска процесса, считается той моделью, что в снимке.

    :param settings: настройки этапа: промпт, модель, параметры генерации
    :param schema: схема ответа — форма, по которой контур возвращает JSON
    :param variables: значения переменных шаблона этапа
    :param messages: готовые сообщения вместо шаблона этапа
    :param provider: провайдер контура; собирает его вызывающая сторона
    :raises ValueError: не передан ни шаблон с переменными, ни сообщения,
        ни провайдер контура
    :raises MalformedAnswer: ответ получен, но по схеме не разложился
    :return: ответ контура, разобранный по схеме
    """
    if messages is None:
        if variables is None:
            raise ValueError(
                "вопрос не собран: передайте либо variables для промпта этапа, "
                "либо готовые messages"
            )
        messages = list(settings.chat_prompt_template.format_messages(**variables))
    elif variables is not None:
        # Оба сразу — признак того, что вызывающая сторона хотела передать одно,
        # а передала оба: молча взять первое значило бы спросить не о том.
        raise ValueError(
            "вопрос задан дважды: messages и variables вместе не передаются"
        )

    request = ChatRequest(
        options=settings.chat_options,
        model=settings.stage_model,
        messages=messages,
    )
    client = provider
    if client is None:
        raise ValueError(
            "провайдер контура не передан: его собирает вызывающая сторона"
        )
    try:
        return await client.chat_structured_output(
            request_options=request,
            structured_output=schema,
            method="format_instructions",
        )
    except Exception as ex:
        # Ответ не разложился по схеме — это не сбой обращения: текст получен,
        # деньги уплачены. Отдельный тип нужен, чтобы вызывающая сторона
        # отличала «модель ответила не по форме» от «контур недоступен».
        if is_shape_failure(ex):
            raise MalformedAnswer(
                f"ответ этапа не разобран по схеме {schema.__name__}: "
                f"{failure_of(ex)}"
            ) from ex
        raise


