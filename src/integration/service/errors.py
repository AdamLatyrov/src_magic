"""Конверт отказа сервиса и обработчик, который его отдаёт.

Наружу уходит `{code, message}` — никогда текст исключения. Текст исключения
работника читается в журнале процесса, а не в ответе: в него попадают пути,
ключи и содержимое ответов контура, и ровно поэтому ядро ограничивает свою
пометку регуляркой из имён классов (`durable_eval_jobs/domain.py:94-95`), а не
доверяет автору сообщения. Здесь те же ограничения: код — `snake_case`, пометка
— только технические имена, сообщение — короткие строки.
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Failure:
    """Отказ наружу.

    Attributes:
        code: устойчивый код отказа — по нему различают сбои, не читая журнала.
        message: что случилось, словами; без деталей обращения и секретов.
        detail: техническая пометка для расследования — имя класса исключения
            или код контура. Ограничена `_detail_pattern`: текста исключения в
            ней быть не может.
    """

    code: str
    message: str
    detail: str | None = None
    code_pattern = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
    detail_pattern = re.compile(r"[A-Za-z0-9_.:-]{1,120}")

    def __post_init__(self) -> None:
        if not self.code_pattern.fullmatch(self.code):
            raise ValueError("Failure.code должен быть snake_case")
        if not 1 <= len(self.message) <= 500:
            raise ValueError("Failure.message должен быть длиной 1..500")
        if self.detail is not None and not self.detail_pattern.fullmatch(
            self.detail
        ):
            raise ValueError("Failure.detail должен быть техническим кодом")

    def as_dict(self) -> dict[str, Any]:
        """Конверт для ответа: пустой пометки в нём нет."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ServiceError(Exception):
    """Отказ, который наружу идёт конвертом, а не текстом исключения.

    Attributes:
        status_code: чем отвечает сервис; 503 — «не готов», 400 — «отказали
            входные данные задания».
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 503,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = Failure(code=code, message=message, detail=detail)
        self.status_code = status_code


def detail_of(exc: BaseException) -> str | None:
    """Техническая пометка об отказе — имя класса, без текста исключения.

    Символы вне `_detail_pattern` вырезаются, а не поднимают исключение: пометка
    описывает уже случившийся отказ, и падать из-за её формата незачем — хуже
    отказа только отказ, который нечем объяснить.

    :param exc: исключение, о котором пишут в журнал
    :return: пометка для `Failure.detail`; пусто, если от имени ничего не осталось
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "", type(exc).__name__)
    return cleaned[:120] or None


async def failure_handler(request: "Request", exc: ServiceError):
    """Отдаёт `ServiceError` конвертом с тем кодом состояния, что назван в нём."""
    from fastapi.responses import JSONResponse

    logger.warning(
        "request refused path=%s code=%s detail=%s",
        request.url.path,
        exc.failure.code,
        exc.failure.detail or "",
    )
    return JSONResponse(
        status_code=exc.status_code, content=exc.failure.as_dict()
    )
