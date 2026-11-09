"""Типы между портами замены ядра: общая «валюта», не знающая про SQL/S3.

Порты (`QueuePort`/`ResultWriter`/...) обмениваются этими типами, чтобы верхний
слой не тянул ни БД, ни S3, ни ORM. Формы взяты из вендорных образцов
(`durable_eval_jobs/domain.py`): статусы жизненного цикла, безопасная ошибка,
валидация результата — то, что у ядра уже обкатано на проде.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Чувствительные/runtime-ключи, запрещённые в result_json и input_json: их наличие
# означает, что в данные просочились креды, заголовки или локальные пути — такие
# значения в БД и диагностику не пишем (см. threat-model Т1/Т3).
_FORBIDDEN_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "cookies",
        "credentials",
        "headers",
        "local_path",
        "password",
        "provider_headers",
        "raw_token",
        "refresh_token",
        "secret",
        "temp_path",
        "upload_file",
    }
)
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_DETAIL_RE = re.compile(r"[A-Za-z0-9_.:-]{1,120}")


class JobStatus(str, Enum):
    """Жизненный цикл задания — отражение констант ядра `EvalJobLifecycleStatus`."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


# Терминальные статусы, в которые переводит задание владелец лиза (см. QueuePort).
JobTerminalStatus = JobStatus  # done | cancelled | failed_retryable | failed_terminal

ResultJson = dict[str, Any]  # допускающий JSON-сериализацию результат задания


class InvalidJobInput(ValueError):
    """Вход/результат задания не прошёл проверку безопасности."""


def _is_json_object(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _validate_json_value(value: Any, *, path: str) -> None:
    """Проверяет значение на безопасные JSON-типы и отсутствие секретов.

    Отклоняет бинарные данные, локальные пути и чувствительные ключи — то, что
    не должно покидать процесс (см. `domain._validate_json_value` вендора).
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidJobInput(f"{path}: NaN/Infinity запрещены")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise InvalidJobInput(f"{path}: бинарные данные запрещены")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise InvalidJobInput(f"{path}: JSON key должен быть строкой")
            if key.strip().lower() in _FORBIDDEN_KEYS:
                raise InvalidJobInput(
                    f"{path}.{key}: чувствительное/runtime поле запрещено"
                )
            _validate_json_value(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json_value(nested, path=f"{path}[{index}]")
        return
    raise InvalidJobInput(
        f"{path}: значение типа {type(value).__name__} не является JSON"
    )


def validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Отделяет копию результата и отклоняет ложный пустой успех.

    Возвращает отвязанный JSON-object: caller не сможет изменить значение после
    записи, а в БД не уйдёт пустой `{}` под видом «успех».
    """
    if not _is_json_object(result):
        raise InvalidJobInput("result должен быть непустым JSON object")
    _validate_json_value(result, path="result")
    try:
        return json.loads(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise InvalidJobInput("result не сериализуется в JSON") from exc


@dataclass(frozen=True)
class SafeJobError:
    """Стабильная публичная ошибка без текста исключения, URL и секретов.

    Именно этот тип уходит в `error_json` задания: `code` — stable snake_case,
    `message` — 1..500 символов, `detail` — короткий технический код без пробелов
    и текста (имя класса исключения/код провайдера). Ни текста, ни URL, ни
    заголовков, ни тела ответа — формат ограничен `_SAFE_DETAIL_RE`.
    """

    code: str
    message: str
    retryable: bool
    phase: str | None = None
    detail: str | None = None

    @classmethod
    def from_exception(cls, exc: Exception, *, phase: str | None = None) -> SafeJobError:
        """Создаёт SafeJobError из исключения.

        Текст исключения в `message` НЕ попадает: в нём бывают URL контура, тело
        ответа и пути на диске, а `error_json` — публичное поле, которое читает
        шлюз. Наружу уходит только имя типа в `detail`; сам текст исключения
        остаётся в журнале процесса, где ему и место.
        """
        etype = type(exc).__name__
        return cls(
            code="unexpected_error",
            message="Внутренняя ошибка обработки задания",
            retryable=True,
            phase=phase,
            detail=etype,
        )

    def __post_init__(self) -> None:
        if not _SAFE_CODE_RE.fullmatch(self.code):
            raise ValueError("SafeJobError.code должен быть stable snake_case")
        if not self.message or len(self.message) > 500:
            raise ValueError("SafeJobError.message должен быть длиной 1..500")
        if self.phase is not None and len(self.phase) > 100:
            raise ValueError("SafeJobError.phase длиннее 100 символов")
        if self.detail is not None and not _SAFE_DETAIL_RE.fullmatch(self.detail):
            raise ValueError(
                "SafeJobError.detail должен быть коротким техническим кодом"
            )

    def as_dict(self) -> dict[str, Any]:
        """Публичное представление ошибки для `error_json`."""
        out: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.phase:
            out["phase"] = self.phase
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class HeartbeatResult:
    """Ответ на продление лиза: новый срок и запрошена ли отмена."""

    lease_expires_at: datetime
    cancel_requested: bool


@dataclass(frozen=True)
class JobLease:
    """Снятое задание: владение процессом на срок лиза.

    `worker_id` + `lease_token` — то, чем владелец доказывает, что он и никто
    другой держит задание (heartbeat/complete/fail). Сырой токен существует только
    здесь, в памяти процесса; в БД — его SHA-256.
    """

    job_id: Any  # uuid.UUID
    run_id: Any | None  # uuid.UUID | None
    kind: str
    input_json: dict[str, Any]
    worker_id: str
    lease_token: str
    claim_generation: int
    attempt: int
    phase: str = "claimed"