import json


class NoResultsFound(Exception):
    """No results were found."""

    pass


class HealthCheckError(Exception):
    """Health check failed."""

    def __init__(self, http_url, exc):
        super().__init__(
            f"Health check for {http_url} wasn't successful: {exc}"
        )


class ClientError(Exception):
    """Client error (400-499) has occurred."""

    def __init__(self, error_code: int, http_url: str, content: bytes = b""):
        super().__init__(
            f"Client error: {str(error_code)} during {http_url} call", content
        )


class ServerError(Exception):
    """Server error (500-599) has occurred."""

    def __init__(self, error_code: int, http_url: str, content: bytes = b""):
        super().__init__(
            f"Server error: {error_code} during {http_url} call", content
        )


class RetryError(Exception):
    """Retry error has occurred."""

    def __init__(self, error_code, http_url):
        super().__init__(f"Retry error {error_code} during {http_url} call")


class BaseGigachatException(Exception):
    """Базовое исключение, связанное с работой GigaChat."""

    def __init__(self, details: str):
        self.details = details
        super().__init__(details)

    def to_json(self) -> str:
        return json.dumps({"error_description": self.details})


class GigaChatTooManyRequests(BaseGigachatException):
    """
    Исключение, связанное с ограничениями на количество
    запросов к GigaChat.
    """

    def __init__(self):
        details = (
            "The number of requests to GigaChat within the client CN has been exceeded. "
            "Increase the number of threads"
        )
        super().__init__(details)


class GigaChatModelNotAvailable(BaseGigachatException):
    """Исключение, связанное с отсутствием модели в GigaChat."""

    def __init__(self, model_name: str):
        details = f"The requested model {model_name} is not available or does not exist"
        super().__init__(details)


class GigaChatException(BaseGigachatException):
    """Исключение, связанное с ошибками в работе GigaChat."""

    def __init__(self, code: str, details: str):
        details = (
            f"Unexpected error in GigaChat. Code: {code}, Details: {details}"
        )
        super().__init__(details)


class MalformedAnswer(Exception):
    """Контур ответил, но ответ не разбирается по схеме.

    Это не сбой обращения: деньги уплачены, текст получен — но бесполезен.
    Для политики ретраев (`pipeline.aef.patterns.retry`) это транзиентная неудача:
    генерация неклейкая, следующий ответ может быть разборчивым.
    """
