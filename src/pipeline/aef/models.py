import json
import operator
from typing import Annotated, ClassVar, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ChatModel(BaseModel):
    """
    LLM, доступная для выбора пользователем.
    """

    name: str = Field(description="Имя модели")
    prompt_tokens_limit: Optional[int] = Field(
        description="Лимит токенов для запроса", default=None
    )
    completion_tokens_limit: Optional[int] = Field(
        description="Лимит токенов для ответа", default=None
    )


class ChatModels(BaseModel):
    """
    Список доступных моделей.
    """

    models: list[ChatModel] = Field(description="Список моделей")


class ChatOptions(BaseModel):
    """
    Параметры для модели.
    """

    model_config = ConfigDict(frozen=True)

    temperature: Optional[float] = Field(
        description="Температура", default=None
    )
    top_p: Optional[float] = Field(description="Топ-p", default=None)
    max_tokens: Optional[int] = Field(
        description="Максимальное количество токенов", default=None
    )
    repetition_penalty: Optional[float] = Field(
        description="Штраф за повторение текста", default=None
    )
    profanity_check: Optional[bool] = Field(
        description="Наличие проверки фильтром", default=None
    )
    reasoning_effort: Optional[str] = Field(
        description=(
            "Насколько модель рассуждает до ответа: off — не рассуждать. "
            "Рассуждение оплачивается из того же лимита токенов, что и ответ, "
            "а узлам нужен ответ заданной формы, а не ход мысли."
        ),
        default="off",
    )

    @staticmethod
    def get_default_configuration() -> dict:
        """
        Получить значения параметров по умолчанию.

        :return: словарь с параметрами по умолчанию
        """
        return {
            "temperature": 1.0,
            "top_p": 0.0,
            "max_tokens": 512,
            "repetition_penalty": 1.0,
            "profanity_check": True,
            "reasoning_effort": "off",
        }


class ChatRequest(BaseModel):
    """
    Запрос к модели.
    """

    options: Optional[ChatOptions] = Field(
        description="Параметры для модели", default=None
    )
    model: Optional[str] = Field(description="Имя модели", default="GigaChat")
    messages: Annotated[list[BaseMessage], operator.add]


class RetryOptions(BaseModel):
    """
    Параметры повторов диалогового узла: сколько раз и с какими паузами
    переспрашивать контур.

    Потолок попыток и паузы — настройки прогона; ключи, которыми граф и его
    обработчик передают друг другу ошибку и имя узла-обработчика, — факты
    контракта langgraph, они не настраиваются и лежат константами класса.
    """

    model_config = ConfigDict(frozen=True)

    NODE_ERROR_KEY: ClassVar[str] = "__pregel_node_error"
    """Ключ конфигурации запуска, под которым граф кладёт в неё ошибку узла
    (`langgraph.errors.NodeError`) для его обработчика; публичной константы у
    langgraph нет — держим здесь и сверяем тестом."""

    HANDLER_PREFIX: ClassVar[str] = "__error_handler__"
    """Префикс имени узла-обработчика, который langgraph заводит сам."""

    SHAPE_FAILURES: ClassVar[tuple[type[Exception], ...]] = (
        json.JSONDecodeError,
        ValidationError,
        OutputParserException,
    )
    """Чем выглядит сбой формы ответа: контур ответил, а ответ не сходится со
    схемой. Все три — ветви `ValueError`, поэтому перечень явный: `ValueError`,
    затесавшийся из бага кода, не должен сжигать попытки контура."""

    max_attempts: int = Field(
        description="Потолок обращений к контуру на узел", default=5
    )
    initial_interval: float = Field(
        description="Пауза перед второй попыткой, секунд", default=1.0
    )
    backoff_factor: float = Field(
        description="Во сколько раз растёт пауза между попытками", default=2.0
    )
    max_interval: float = Field(
        description="Потолок паузы между попытками, секунд", default=20.0
    )
    jitter: bool = Field(
        description="Разбрасывать ли паузу, чтобы повторы не совпадали",
        default=True,
    )


class Usage(BaseModel):
    """
    Использование модели.
    """

    prompt_tokens: int = Field(
        description="Количество токенов во входном запросе", default=0
    )
    completion_tokens: int = Field(
        description="Количество токенов в ответе", default=0
    )
    total_tokens: int = Field(
        description="Общее количество токенов", default=0
    )


class ModelInfo(BaseModel):
    """
    Информация о модели.
    """

    name: str = Field(description="Имя модели")
    version: str = Field(description="Версия модели")


class ChatResponse(BaseModel):
    """
    Ответ модели.
    """

    used_model: ModelInfo = Field(description="Информация о модели")
    usage: Usage = Field(description="Информация о использовании")
    timestamp: int = Field(description="Время выполнения запроса")
    message: BaseMessage = Field(description="Ответ модели")
    finish_reason: str = Field(description="Причина завершения запроса")
