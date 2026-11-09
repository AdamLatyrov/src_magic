"""Объектное хранилище (S3/MinIO): настройки и boto3-адаптер порта `ObjectStorage`.

Реализация повторяет подход вендора (`vendor/.../cjxplorer-core/src/storage.py`):
обёртка над `boto3`, конфигурация через переменные окружения `S3_*`, бакет
приватный, наружная ссылка — короткоживущий presigned URL. Отличие — эталон здесь
наш (`integration.storage.frames.ObjectStorage`): код не зависит от вендора, а низ
(S3/MinIO-клиент через boto3) подставляется так же, как вендор его подставляет.

Настройки читаются из env с теми же именами и значениями по умолчанию, что и у
вендора, — стендовый контур S3 (переменные `S3_*`) подходит без изменений.
"""

from __future__ import annotations

import asyncio
import logging

from botocore import exceptions as botocore_exceptions
from botocore.client import Config
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class S3Settings(BaseSettings):
    """Настройки объектного хранилища: имена и дефолты повторяют вендора (`S3_*`)."""

    class Config:
        env_file = ".env"
        use_enum_values = True
        populate_by_name = True
        extra = "ignore"

    endpoint: str = Field(default="http://localhost:9000", alias="S3_ENDPOINT")
    public_endpoint: str = Field(
        default="http://localhost:9000", alias="S3_PUBLIC_ENDPOINT"
    )
    access_key: str = Field(default="cjxplorer", alias="S3_ACCESS_KEY")
    secret_key: str = Field(default="cjxplorer_dev_secret", alias="S3_SECRET_KEY")
    bucket: str = Field(default="cjxplorer", alias="S3_BUCKET")
    presign_ttl: int = Field(default=3600, alias="S3_PRESIGN_TTL")
    connect_timeout_s: float = Field(default=5, alias="S3_CONNECT_TIMEOUT_S")
    read_timeout_s: float = Field(default=30, alias="S3_READ_TIMEOUT_S")
    max_retries: int = Field(default=3, alias="S3_MAX_RETRIES")
    ca_bundle: str = Field(default="", alias="S3_CA_BUNDLE")


def s3_error_kind(exc: Exception) -> str:
    """Классификация сбоя S3: not_found | access_denied | timeout | unavailable | error.

    Нужна вызывающим, чтобы НЕ превращать любой сбой хранилища в 404: отсутствие
    объекта и недоступность сервиса — разные состояния. Копия логики вендора.

    ВНИМАНИЕ — этой градацией пока НИКТО не пользуется. Разбор нужен для
    вызывающего, но вызывающий её не спрашивает: `S3FrameStorage.load` пробрасывает
    исключение как есть, а `QueueRunner` превращает его в `SafeJobError` с
    `retryable=True` через `SafeJobError.from_exception`. В итоге `not_found`
    (битый ключ, объекта нет и не будет) жжёт попытки так же, как временная
    недоступность, вместо того чтобы стать `failed_terminal`.

    Что делать, если править: спросить здесь `s3_error_kind(exc)` и отдать
    `retryable=False` для `not_found` и `access_denied`. Разбор —
    `docs/project-spec/09-maintenance/improvements.md`, пункт I-009.
    """
    if isinstance(
        exc,
        (
            botocore_exceptions.ConnectTimeoutError,
            botocore_exceptions.ReadTimeoutError,
        ),
    ):
        return "timeout"
    if isinstance(
        exc,
        (
            botocore_exceptions.EndpointConnectionError,
            botocore_exceptions.ConnectionClosedError,
        ),
    ):
        return "unavailable"
    if isinstance(exc, botocore_exceptions.ClientError):
        err = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
        code = str(err.get("Code", ""))
        status = (exc.response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode")
        if code in ("NoSuchKey", "NoSuchBucket", "NoSuchUpload", "404", "NotFound"):
            return "not_found"
        if code in (
            "AccessDenied",
            "403",
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
        ):
            return "access_denied"
        if code in ("SlowDown", "ServiceUnavailable", "InternalError") or (
            isinstance(status, int) and status >= 500
        ):
            return "unavailable"
    return "error"


def _log_s3_failure(operation: str, key: str, exc: Exception) -> None:
    """Структурированный лог сбоя S3: операция/вид/ключ/тип — без секретов и URL."""
    logger.warning(
        "[s3] %s failed: kind=%s, key=%s, error=%s",
        operation,
        s3_error_kind(exc),
        key,
        type(exc).__name__,
    )


class S3ObjectStorage:
    """Адаптер `ObjectStorage` поверх boto3, конфигурируемый из env (`S3_*`).

    Создаёт клиент из настроек процесса; boto3-операции синхронные, поэтому
    читаются через `asyncio.to_thread`, чтобы не блокировать event loop — вендор
    держит S3-вызовы вне async-цикла точно так же (`asyncio.to_thread(...)`).
    """

    def __init__(self, settings: S3Settings | None = None) -> None:
        import boto3  # ленивый импорт: boto3 нужен только этому адаптеру

        self._settings = settings or S3Settings()
        self._client = boto3.client(
            "s3",
            endpoint_url=self._settings.endpoint,
            aws_access_key_id=self._settings.access_key,
            aws_secret_access_key=self._settings.secret_key,
            region_name="us-east-1",
            # Внутренний S3 стенда подписан корпоративным CA: без бандла проверка
            # сертификата падает на «unable to get local issuer certificate».
            verify=self._settings.ca_bundle or True,
            config=Config(
                signature_version="s3v4",
                connect_timeout=self._settings.connect_timeout_s,
                read_timeout=self._settings.read_timeout_s,
                retries={
                    "max_attempts": max(1, self._settings.max_retries),
                    "mode": "standard",
                },
            ),
        )

    async def read(self, key: str) -> bytes:
        """Читает объект целиком по ключу; пустой ключ — отказ."""
        if not key:
            raise ValueError("S3 object key пуст")
        try:
            resp = await asyncio.to_thread(
                self._client.get_object,
                Bucket=self._settings.bucket,
                Key=key,
            )
            body = resp["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except Exception as exc:  # noqa: BLE001 — лог + проброс типизированной причины
            _log_s3_failure("read", key, exc)
            raise

    def signed_url(self, key: str, *, expires_s: int | None = None) -> str:
        """Короткоживущая наружная ссылка на объект (presigned GET)."""
        ttl = expires_s if expires_s is not None else self._settings.presign_ttl
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._settings.bucket, "Key": key},
                ExpiresIn=ttl,
            )
        except Exception as exc:  # noqa: BLE001
            _log_s3_failure("signed_url", key, exc)
            raise


__all__ = ["S3Settings", "S3ObjectStorage", "s3_error_kind"]