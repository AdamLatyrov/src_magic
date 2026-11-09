"""
Набор утилит для работы с GigaChat API.
"""

import base64
import json
import logging
import os
import ssl
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from http import HTTPStatus
from typing import Any, Literal, NoReturn, Optional, TypeVar, cast

from gigachat import GigaChatAsyncClient
from gigachat.exceptions import ResponseError
from gigachat.models.chat_completions import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatModelOptions,
    ChatReasoning,
    ChatResponseFormat,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers.format_instructions import (
    JSON_FORMAT_INSTRUCTIONS,
)
from langchain_core.tools import BaseTool
from langchain_gigachat import GigaChat
from pydantic import BaseModel

from src.pipeline.aef.exceptions import (
    GigaChatException,
    GigaChatModelNotAvailable,
    GigaChatTooManyRequests,
    HealthCheckError,
)
from src.pipeline.aef.models import (
    ChatModel,
    ChatModels,
    ChatOptions,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    Usage,
)
from src.pipeline.aef.patterns.base import BaseConnector
from src.pipeline.settings.params.gigachat import (
    FILE_REF,
    IMAGE_EXTENSIONS,
    MESSAGE_ROLES,
    ReasoningEfforts,
)
from src.pipeline.utils.json_reply import json_payload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GigaChatAuthorization:
    """
    Параметры авторизации для работы с GigaChat.
    """

    auth_url: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    credentials: Optional[str] = None
    access_token: Optional[str] = None
    scope: Optional[str] = "GIGACHAT_API_CORP"
    verify: bool = False


Struct = TypeVar("Struct", bound=BaseModel)


def mtls_context(
    ca_file: str,
    cert_file: str,
    key_file: Optional[str] = None,
    key_password: Optional[str] = None,
) -> ssl.SSLContext:
    """Контекст TLS для входа по клиентскому сертификату.

    Контур ГигаЧата отбирает клиентский сертификат на рукопожатии TLS 1.2: при TLS 1.3 он уходит
    уже после handshake, и сервер обрывает соединение с `TLSV13_ALERT_CERTIFICATE_REQUIRED`.
    Сертификат загружается внутрь контекста, потому что httpx игнорирует параметр `cert`, когда в
    `verify` передан SSLContext.
    """
    context = ssl.create_default_context(cafile=ca_file)
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(
        certfile=cert_file, keyfile=key_file, password=key_password
    )
    return context


def client_ssl_context(
    auth: GigaChatAuthorization,
) -> Optional[ssl.SSLContext]:
    """Контекст TLS клиента: mTLS при клиентском сертификате, доверие машины иначе.

    SDK применяет `ca_bundle_file` поверх `ssl_context`, и контекст с потолком TLS теряется, а поле он
    добирает из GIGACHAT_CA_BUNDLE_FILE, если передан None. При входе по клиентскому сертификату
    доверяющий CA набор отдаётся только контекстом — поэтому рядом с ним передаётся пустая строка.

    Клиентского сертификата нет — проверять сервер всё равно чем-то нужно, и остаётся хранилище
    доверия машины: `httpx` без набора сверяет цепочку с публичным certifi, где корпоративного
    корня нет, и handshake рвётся на `CERTIFICATE_VERIFY_FAILED`. Названный набор — наш или из
    образа — сильнее: он доходит до SDK как есть, и контекст здесь лишний.
    """
    if auth.cert_file and auth.ca_file:
        return mtls_context(
            ca_file=auth.ca_file,
            cert_file=auth.cert_file,
            key_file=auth.key_file,
        )
    named = auth.ca_file or os.environ.get("GIGACHAT_CA_BUNDLE_FILE")
    if auth.verify and not named:
        return ssl.create_default_context()
    return None


def frame_from_data_url(url: str) -> tuple[str, bytes]:
    """Разбирает data-URL кадра: контур отдаёт кадры лишь загруженными файлами.

    :param url: data-URL вида `data:image/jpeg;base64,...`
    :return: тип содержимого и байты кадра
    """
    header, separator, encoded = str(url).partition(",")
    if (
        not separator
        or not header.startswith("data:")
        or ";base64" not in header
    ):
        raise ValueError(
            "GigaChat принимает кадры только как data-URL с base64, "
            f"получено `{header[:64]}`"
        )
    return header[len("data:") : header.index(";")], base64.b64decode(encoded)


def has_frames(messages: list[BaseMessage]) -> bool:
    """Есть ли в запросе кадр: такой запрос уходит по контракту V2.

    :param messages: сообщения запроса
    :return: True, если хотя бы одно сообщение содержит часть с изображением или
        со ссылкой на уже загруженный кадр
    """
    return any(
        isinstance(part, dict) and part.get("type") in {"image_url", FILE_REF}
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
    )


def v2_messages(
    messages: list[BaseMessage],
) -> tuple[list[dict], list[tuple[dict, str, bytes]]]:
    """Переводит сообщения LangChain в контракт V2 и собирает кадры под загрузку.

    V2 описывает содержимое частями: текст через `text`, кадр через `files` со ссылкой на загруженный
    файл. Ссылка появляется только после загрузки, поэтому на месте кадра остаётся пустая часть —
    собранные кадры идут строго в порядке их в запросе.

    Кадр, отданный частью `FILE_REF`, в заготовки не попадает: его файл загружен
    раньше и живёт до конца прогона, поэтому его заново ни грузят, ни удаляют после
    ответа — иначе чужой запрос стёр бы кадр, на который ещё сошлются следующие
    обращения (`pipeline.agent.nodes.release_frames` снимает ссылки один раз, в конце
    прогона).

    :param messages: сообщения запроса
    :return: сообщения контракта V2 и заготовки кадров (часть, тип содержимого, байты)
    """
    converted: list[dict] = []
    frames: list[tuple[dict, str, bytes]] = []
    for message in messages:
        role = MESSAGE_ROLES.get(message.type, "user")
        if isinstance(message.content, str):
            converted.append(
                {"role": role, "content": [{"text": message.content}]}
            )
            continue

        parts: list[dict] = []
        for part in message.content:
            if isinstance(part, str):
                parts.append({"text": part})
            elif part.get("type") == "text":
                parts.append({"text": part.get("text", "")})
            elif part.get("type") == FILE_REF:
                file_id = str(part.get("file_id") or "").strip()
                if not file_id:
                    raise ValueError(
                        "часть содержимого `file_ref` пришла без id файла"
                    )
                parts.append({"files": [{"id": file_id}]})
            elif part.get("type") == "image_url":
                media_type, raw = frame_from_data_url(part["image_url"]["url"])
                frame: dict[str, Any] = {"files": []}
                parts.append(frame)
                frames.append((frame, media_type, raw))
            else:
                raise ValueError(
                    f"Контур V2 не знает часть содержимого `{part.get('type')}`"
                )
        converted.append({"role": role, "content": parts})

    return converted, frames


def model_info(name: Optional[str]) -> ModelInfo:
    """Разбирает имя модели вида `имя:версия`.

    :param name: имя модели из ответа контура
    :return: имя и версия модели
    """
    model, _, version = (name or ":").partition(":")
    return ModelInfo(name=model, version=version)


def _v2_answer(completion: ChatCompletionResponse) -> str:
    answer = ""
    for message in completion.messages or []:
        if message.role != "assistant":
            continue
        answer += "".join(part.text or "" for part in message.content or [])
    return answer


def _reasoning(options: ChatOptions) -> Optional[ChatReasoning]:
    """Рассуждение модели в опциях контракта V2.

    Ответ модели идёт в том же лимите `max_tokens`, что и рассуждение: модель,
    которая думает вслух, снимает с ответа весь бюджет и отдаёт пустое
    содержимое с `finish_reason: length`. Узлам нужен ответ заданной формы,
    поэтому усилие по умолчанию — `off`.

    :param options: параметры модели из запроса
    :return: настройка рассуждения или None, если усилий не задано
    """
    if not options.reasoning_effort:
        return None
    return ChatReasoning(effort=options.reasoning_effort)


def _v1_reasoning_effort(options: ChatOptions) -> Optional[str]:
    """Усилие рассуждения для контракта V1 — каким его знает обёртка LangChain.

    `off` здесь теряет смысл: V1 перечисляет только `low`/`medium`/`high`, а
    обёртка отдаёт поле на контур при любом непустом значении — своё поле она
    держит строкой, но `gigachat.models.Chat` разбирает тело запроса по
    перечню и на `off` спотыкается ещё до обращения к контуру.

    :param options: параметры модели из запроса
    :return: значение из перечня V1 или None
    """
    effort = options.reasoning_effort
    return effort if ReasoningEfforts.accepts(effort) else None


async def _delete_frames(
    client: GigaChatAsyncClient, file_ids: list[str]
) -> None:
    for file_id in file_ids:
        try:
            await client.adelete_file(file_id)
        except ResponseError as e:
            logger.warning(
                "Не удалось удалить загруженный кадр %s: %s", file_id, e
            )


def _raise_gigachat_error(
    exc: ResponseError, model: Optional[str]
) -> NoReturn:
    status, content = exc.status_code, exc.content
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        raise GigaChatTooManyRequests() from exc
    elif status == HTTPStatus.NOT_FOUND:
        raise GigaChatModelNotAvailable(model or "Undefined") from exc
    else:
        raise GigaChatException(status, content) from exc


class GigaChatProvider(BaseConnector):
    """
    Класс для работы с GigaChat API.
    """

    def __init__(
        self,
        base_url: str,
        authorization: GigaChatAuthorization = GigaChatAuthorization(),
        retry_limit: int = 3,
        retry_increment: int = 0,
        request_timeout: int = 30,
    ):
        """
        :param base_url: базовый URL для запросов к API
        :param authorization: параметры авторизации
        :param retry_limit: максимальное количество попыток повторной отправки запроса
        :param retry_increment: задержка между попытками повторной отправки запроса
        :param request_timeout: таймаут ожидания ответа от сервера
        """
        super().__init__(
            base_url=base_url,
            retry_limit=retry_limit,
            retry_increment=retry_increment,
            request_timeout=request_timeout,
        )
        self.authorization = authorization

    async def health_check(self):
        """
        Проверка доступности GigaChat API
        """
        logger.info("GigaChat gRPC client health_check call")

        try:
            models: ChatModels = await self.list_models()
        except Exception as exc:
            logger.error(f"Gigachat gRPC health check failed: {repr(exc)}")
            raise HealthCheckError(
                http_url=self._base_url, exc=repr(exc)
            ) from exc

        logger.info("Available GigaChat models:")

        for model in models.models:
            logger.info(model)

        logger.info("GigaChat gRPC health_check was successful")

    async def list_models(self) -> ChatModels:
        """
        Возвращает массив объектов с данными доступных пользователю моделей

        :return: массив объектов с данными доступных пользователю моделей
        """
        gc_model = self.get_client(
            None, None, base_url=self._base_url, auth=self.authorization
        )

        try:
            models = await gc_model.aget_models()
            return ChatModels(
                models=[ChatModel(name=model.id_) for model in models.data]
            )
        except ResponseError as e:
            status, content = e.status_code, e.content
            if status == HTTPStatus.TOO_MANY_REQUESTS:
                raise GigaChatTooManyRequests() from e
            else:
                raise GigaChatException(status, content) from e

    @lru_cache(maxsize=10)
    def get_client(
        self,
        options: Optional[ChatOptions],
        model: Optional[str],
        base_url: str,
        auth: GigaChatAuthorization,
    ) -> GigaChat:
        """
        Возвращает клиент для работы с GigaChat.

        :param options: настройки для модели
        :param model: название модели
        :param base_url: базовый URL для запросов к API
        :param auth: параметры авторизации
        :return: клиент для работы с GigaChat
        """
        if options is None:
            options = ChatOptions(**ChatOptions.get_default_configuration())
        ssl_context = client_ssl_context(auth)
        return GigaChat(
            temperature=options.temperature,
            top_p=options.top_p,
            max_tokens=options.max_tokens,
            repetition_penalty=options.repetition_penalty,
            profanity_check=options.profanity_check,
            reasoning_effort=_v1_reasoning_effort(options),
            model=model,
            base_url=base_url,
            auth_url=auth.auth_url,
            cert_file=auth.cert_file,
            key_file=auth.key_file,
            ca_bundle_file="" if ssl_context else auth.ca_file,
            ssl_context=ssl_context,
            user=auth.username,
            password=auth.password,
            access_token=auth.access_token,
            credentials=auth.credentials,
            scope=auth.scope,
            timeout=self.request_timeout,
            verify_ssl_certs=auth.verify,
        )

    @lru_cache(maxsize=10)
    def get_sdk_client(
        self,
        options: Optional[ChatOptions],
        model: Optional[str],
        base_url: str,
        auth: GigaChatAuthorization,
    ) -> GigaChatAsyncClient:
        """
        Возвращает клиент SDK: кадров обёртка LangChain не видит.

        Обёртка собирает запрос контракта V1, где содержимое сообщения — одна строка, а кадры — полем
        `attachments`; из таких вложений контур читает только первое. Клиент SDK отдаёт контракт V2, где
        каждый кадр — отдельная часть содержимого со своим файлом.

        :param options: настройки для модели
        :param model: название модели
        :param base_url: базовый URL для запросов к API
        :param auth: параметры авторизации
        :return: асинхронный клиент GigaChat
        """
        if options is None:
            options = ChatOptions(**ChatOptions.get_default_configuration())
        ssl_context = client_ssl_context(auth)
        return GigaChatAsyncClient(
            model=model,
            profanity_check=options.profanity_check,
            base_url=base_url,
            auth_url=auth.auth_url,
            cert_file=auth.cert_file,
            key_file=auth.key_file,
            ca_bundle_file="" if ssl_context else auth.ca_file,
            ssl_context=ssl_context,
            user=auth.username,
            password=auth.password,
            access_token=auth.access_token,
            credentials=auth.credentials,
            scope=auth.scope,
            timeout=self.request_timeout,
            verify_ssl_certs=auth.verify,
        )

    async def upload_frame(
        self,
        options: Optional[ChatOptions],
        model: str,
        data_url: str,
    ) -> str:
        """Загружает кадр в Files API один раз и отдаёт id файла — ссылку на кадр.

        Обычный путь кадра (`_v2_completion`) грузит его внутри запроса и тут же
        удаляет: кадр нужен ровно одному обращению. Здесь кадр нужен всем последующим
        обращениям прогона — скан описывает его сам, потом отдаёт соседям окна, а
        дальше его видят блоки и проверки пути. Повторная загрузка того же кадра на
        каждое обращение — это лишний round-trip ради байтов, которые контур уже
        держит.

        Ссылкой пользуются частью содержимого `FILE_REF`; снятием занимается
        `pipeline.agent.nodes.release_frames` — файл, брошенный на стороне контура, никто
        кроме прогона не удалит.

        :param options: параметры модели — тем же клиентом, что отвечают на вопрос
        :param model: имя модели
        :param data_url: кадр как data-URL с base64
        :return: id загруженного файла
        """
        client = self.get_sdk_client(
            options=options,
            model=model,
            base_url=self._base_url,
            auth=self.authorization,
        )
        media_type, raw = frame_from_data_url(data_url)
        extension = IMAGE_EXTENSIONS.get(media_type, "bin")
        try:
            upload = await client.aupload_file(
                (f"frame.{extension}", raw, media_type), purpose="general"
            )
        except ResponseError as e:
            _raise_gigachat_error(e, model)
        return str(upload.id_)

    async def delete_frames(
        self,
        options: Optional[ChatOptions],
        model: str,
        file_ids: Sequence[str],
    ) -> None:
        """Снимает ссылки кадров, которые больше не нужны прогону.

        Кадр снимают не тогда, когда о нём спросили один раз, а когда прогон о нём
        забыл: отфильтрованный кадр вычёркивают сразу (`pipeline.agent.journey_exploration`),
        остальные снимает узел `pipeline.agent.nodes.release_frames` в конце прогона.
        Сбой снятия молча терпят: осиротевший файл на стороне контура не имеет права
        стоить прогону ответа, который уже собран.

        :param options: параметры модели — тем же клиентом, что грузили кадры
        :param model: имя модели
        :param file_ids: id загруженных файлов, повторы не уточвают
        """
        ids = list(dict.fromkeys(file_id for file_id in file_ids if file_id))
        if not ids:
            return
        client = self.get_sdk_client(
            options=options,
            model=model,
            base_url=self._base_url,
            auth=self.authorization,
        )
        await _delete_frames(client, ids)

    async def chat(
        self,
        request_options: ChatRequest,
        tools: Optional[list[BaseTool]] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        """
        Возвращает ответ модели с учетом переданных сообщений и настроек.

        Запрос с кадрами уходит по контракту V2 через клиент SDK, остальные — через обёртку LangChain.

        :param request_options: настройки для модели
        :param tools: инструменты, которые может использовать модель
        :param tool_choice: какую из инструментов использовать, если их несколько
        :return: ответ модели с учетом переданных сообщений и настроек
        """
        if has_frames(request_options.messages):
            return await self.chat_with_frames(request_options)

        gc_model = self.get_client(
            options=request_options.options,
            model=request_options.model,
            base_url=self._base_url,
            auth=self.authorization,
        )

        if tools:
            gc_model = gc_model.bind_tools(tools, tool_choice=tool_choice)
            runnable: Callable[[list[BaseMessage]], Awaitable[BaseMessage]] = (
                gc_model.ainvoke
            )
        else:
            runnable = partial(gc_model.ainvoke, functions=None)

        try:
            messages = [m.model_dump() for m in request_options.messages]
            chat = await runnable(messages)
            meta = chat.response_metadata or {}
            model, version = meta.get("model", ":").split(":", 1)
            return ChatResponse(
                message=chat,
                finish_reason=meta.get("finish_reason", "stop"),
                timestamp=int(time.time()),
                used_model=ModelInfo(name=model, version=version),
                usage=Usage(**meta.get("token_usage", {})),
            )
        except ResponseError as e:
            _raise_gigachat_error(e, request_options.model)

    async def chat_with_frames(
        self, request_options: ChatRequest
    ) -> ChatResponse:
        """
        Возвращает ответ модели по запросу с кадрами.

        Кадры сначала уходят в Files API, а после становятся частями содержимого контракта V2: в V1
        кадры передаются полем `attachments`, и контур читает из таких вложений только первое. Загруженные
        файлы удаляются сразу после ответа — контур хранит их между запросами.

        :param request_options: настройки для модели и сообщения с кадрами
        :return: ответ модели с учетом переданных сообщений и настроек
        """
        completion = await self._v2_completion(request_options)

        usage = completion.usage
        return ChatResponse(
            message=AIMessage(content=_v2_answer(completion)),
            finish_reason=completion.finish_reason or "stop",
            timestamp=int(time.time()),
            used_model=model_info(completion.model or request_options.model),
            usage=Usage(
                prompt_tokens=(usage.input_tokens or 0) if usage else 0,
                completion_tokens=(usage.output_tokens or 0) if usage else 0,
                total_tokens=(usage.total_tokens or 0) if usage else 0,
            ),
        )

    async def chat_with_structure_output_with_frames(
        self,
        request_options: ChatRequest,
        structured_output: type[Struct],
        method: Literal[
            "json_schema", "format_instructions"
        ] = "format_instructions",
    ) -> Struct:
        """Возвращает ответ по запросу с кадрами, разобранный в схему pydantic.

        Кадры живут только в контракте V2: в V1 из вложений читается первое, и
        обёртка LangChain сводит запрос именно к V1. Поэтому форма ответа
        собирается здесь, а не отдаётся обёртке, и способов два:

        - `format_instructions` — по умолчанию: схема уходит в сообщение
          текстом, как в одноимённом методе обёртки LangChain; ответ держится
          просьбой, а не контуром;
        - `json_schema` — нативный `response_format` контура; способ для
          контуров, где структурный вывод с файлами работает: на текущем
          ответ с кадрами приходит пустым или мимо схемы.

        Ответ валидируется схемой после ответа в любом способе: форма —
        рекомендация модели, а не гарантия, и несошедшая форма обязана быть
        ошибкой, а не молчаливой подменой.

        :param request_options: настройки для модели и сообщения с кадрами
        :param structured_output: схема pydantic для ответа модели
        :param method: способ задать форму ответа: `format_instructions` —
            схема текстом в запросе, `json_schema` — нативный
            `response_format`, пустой или мимо схемы на контурах, где
            структурный вывод с файлами не работает
        :return: ответ модели, разобранный в схему
        """
        if method == "json_schema":
            completion = await self._v2_completion(
                request_options,
                response_format=ChatResponseFormat(
                    type="json_schema",
                    schema_=structured_output.model_json_schema(),
                ),
            )
            answer = _v2_answer(completion)
        else:
            answer = await self._ask_with_format_instructions(
                request_options, structured_output
            )

        return structured_output.model_validate(
            json_payload(answer, "по схеме структурированного вывода")
        )

    async def _ask_with_format_instructions(
        self, request_options: ChatRequest, structured_output: type[Struct]
    ) -> str:
        """Спрашивает тем же транспортом V2, а форму кладёт в текст сообщения.

        Приём повторяет метод `format_instructions` обёртки LangChain — схема
        дописывается отдельным сообщением в конец запроса, — только поверх V2,
        где кадры остаются кадрами. Шаблон инструкции берётся тот же, из
        `langchain_core`, и сокращается так же, как в обёртке: без верхних
        `title` и `type`.

        :param request_options: настройки для модели и сообщения с кадрами
        :param structured_output: схема pydantic для ответа модели
        :return: ответ модели текстом
        """
        schema = structured_output.model_json_schema()
        reduced = {
            k: v for k, v in schema.items() if k not in ("title", "type")
        }
        instructions = JSON_FORMAT_INSTRUCTIONS.format(
            schema=json.dumps(reduced, ensure_ascii=False)
        )
        messages = list(request_options.messages)
        messages.append(HumanMessage(content=instructions))

        completion = await self._v2_completion(
            request_options.model_copy(update={"messages": messages})
        )
        return _v2_answer(completion)

    async def _v2_completion(
        self,
        request_options: ChatRequest,
        response_format: Optional[ChatResponseFormat] = None,
    ) -> ChatCompletionResponse:
        """Один транспорт контракта V2: кадры в Files API, ответ — completion.

        :param request_options: настройки для модели и сообщения с кадрами
        :param response_format: схема ответа — рядом с кадрами она живёт только
            в опциях модели этого контракта
        :return: ответ контура как пришёл
        """
        options = request_options.options or ChatOptions(
            **ChatOptions.get_default_configuration()
        )
        client = self.get_sdk_client(
            options=options,
            model=request_options.model,
            base_url=self._base_url,
            auth=self.authorization,
        )
        messages, frames = v2_messages(request_options.messages)
        uploaded: list[str] = []

        try:
            for number, (part, media_type, raw) in enumerate(frames, start=1):
                extension = IMAGE_EXTENSIONS.get(media_type, "bin")
                upload = await client.aupload_file(
                    (f"frame-{number}.{extension}", raw, media_type),
                    purpose="general",
                )
                uploaded.append(upload.id_)
                part["files"] = [{"id": upload.id_}]

            return await client.achat.create(
                ChatCompletionRequest(
                    model=request_options.model,
                    messages=messages,
                    model_options=ChatModelOptions(
                        temperature=options.temperature,
                        top_p=options.top_p,
                        max_tokens=options.max_tokens,
                        repetition_penalty=options.repetition_penalty,
                        reasoning=_reasoning(options),
                        response_format=response_format,
                    ),
                )
            )
        except ResponseError as e:
            _raise_gigachat_error(e, request_options.model)
        finally:
            await _delete_frames(client, uploaded)

    async def chat_structured_output(
        self,
        request_options: ChatRequest,
        structured_output: type[Struct],
        method: Literal[
            "function_calling", "json_mode", "format_instructions"
        ] = "function_calling",
    ) -> Struct:
        """Возвращает ответ модели, разобранный в схему pydantic.

        `format_instructions` идёт транспортом V2 — тем же, которым уходят кадры, —
        а не обёрткой LangChain: обёртка держит контракт V1, где содержимое
        сообщения — одна строка, а кадры читаются только первым вложением. Держать
        второй транспорт ради двух сводящих узлов незачем. Форма вопроса при этом
        не меняется: схема дописывается в конец запроса текстом, тем же шаблоном
        `langchain_core` и с теми же сокращениями, — прогон остаётся сравнимым с
        прежними прогонами.

        О чём надо помнить по этому контуру: с 13.10.2026 текстовый запрос к
        `GigaChat-3-Ultra` отбивается ответом 500 — вендор подтвердил это своим
        сбоем, и ждём починки с его стороны
        (`docs/gigachat-3-ultra-text-500.md`). Прогон до починки идёт на
        `Qwen3.5-397b`; у неё ход мысли приходит отдельным сообщением и снимает
        бюджет ответа, поэтому усилие рассуждения по умолчанию — `off`
        (`_reasoning`), а `_v2_answer` берёт из ответа только реплику ассистента.

        Ответ, который не разбирается, остаётся сбоем формы, а не сбоя обращения:
        деньги уплачены и текст получен, и перевызывать его имеет смысл только
        ради другой генерации.

        :param request_options: настройки для модели
        :param structured_output: структура для ответа модели
        :param method: метод взаимодействия с моделью
        :return: ответ модели с учетом переданных сообщений и настроек
        """
        if method == "format_instructions":
            answer = await self._ask_with_format_instructions(
                request_options, structured_output
            )
            try:
                payload = json_payload(
                    answer, "по схеме структурированного вывода"
                )
            except ValueError as ex:
                raise OutputParserException(str(ex)) from ex
            return structured_output.model_validate(payload)

        gc_model = self.get_client(
            options=request_options.options,
            model=request_options.model,
            base_url=self._base_url,
            auth=self.authorization,
        )
        gc_model = gc_model.with_structured_output(
            schema=structured_output, method=method
        )

        try:
            messages = [m.model_dump() for m in request_options.messages]
            result = cast(Struct, await gc_model.ainvoke(messages))
            return result
        except ResponseError as e:
            _raise_gigachat_error(e, request_options.model)
