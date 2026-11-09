"""Хранилище кадров: порт и реализации-адаптеры для S3.

Граф ждёт кадры в виде `CXFrame.data_url` (base64). Две ответственности:

* `FrameStorage` — достать байты по ключу и дать наружную ссылку;
* `FrameSource`  — превратить ключ кадра в объект графа `CXFrame` (упаковка байт
  в data_url), чтобы верхний слой не знал про base64 и S3.

`ObjectStorage` — абстрактный контракт (контракт `read`/`signed_url`); реальный
низ (boto3-адаптер `S3ObjectStorage` из `s3.py`, S3/MinIO-клиент с кредами из
настроек процесса `S3_*`) подставляется при применении — модуль сам клиент не
создаёт, креды в модуль не попадают (threat-model Т1). `S3FrameStorage` —
адаптер поверх `ObjectStorage`. Ключи кадров валидируются fail-closed
(threat-model Т6): `../`, `\\`, абсолютные пути и чужой namespace — отказ, без
обращения к хранилищу.
"""

from __future__ import annotations

import base64
import logging
import posixpath
import re
from pathlib import PurePosixPath
from typing import Protocol

from src.pipeline.schemas.nodes import CXFrame

logger = logging.getLogger(__name__)

# Расширение → MIME для data_url (кадры в прогоне — webp/png/jpg).
_MIME_BY_EXT = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
_DEFAULT_MIME = "image/webp"
# Имена файлов кадров, которые допускаем (web/экраны), без путей с переходом.
_FILE_RE = re.compile(r"[^.\\/][^\\/]*\.(webp|png|jpg|jpeg)")


class FrameStorage(Protocol):
    """Хранилище кадров по ключу: байты и наружная ссылка."""

    async def load(self, key: str) -> bytes:
        """Байты кадра по ключу; недопустимый ключ — отказ без обращения."""
        ...

    def media_url(self, key: str) -> str:
        """Наружная ссылка с коротким сроком жизни (подписанный URL)."""
        ...


class FrameSource(Protocol):
    """Ключ кадра → объект графа `CXFrame` с `data_url`."""

    async def frame(self, key: str, *, step: int) -> CXFrame:
        """Кадр для подачи в модель: `step` + base64-данные."""
        ...


class InvalidObjectKey(ValueError):
    """Ключ кадра не принадлежит допустимому namespace."""


class ObjectStorage(Protocol):
    """Минимальный контракт объектного хранилища, подставляемый при применении.

    Реальную реализацию (S3/MinIO-клиент с кредами из настроек процесса) даёт
    тот, кто применяет модуль; креды сюда не передаются — только готовый клиент.
    """

    async def read(self, key: str) -> bytes:
        """Читает объект целиком по ключу."""
        ...

    def signed_url(self, key: str, *, expires_s: int) -> str:
        """Короткоживущая наружная ссылка на объект."""
        ...


def is_allowed_key(key: str) -> bool:
    """Допустим ли ключ кадра: каноничный путь без перехода в чужой namespace.

    Принимаем ровно те раскладки, что эмитирует продукт:
    `runs/<run_id>/screenshots/<name>` (4 части) и
    `eval-input/<owner>/<...>/<name>` (4-5 частей). Всё прочее — отказ.
    """
    if (
        not key
        or key.startswith("/")
        or "\\" in key
        or ".." in key
        or posixpath.normpath(key) != key
    ):
        return False
    parts = key.split("/")
    if not parts or not _FILE_RE.fullmatch(parts[-1]):
        return False
    if parts[0] == "runs":
        return len(parts) == 4 and parts[2] == "screenshots" and bool(parts[3])
    if parts[0] == "eval-input":
        return len(parts) in {4, 5} and bool(parts[-1])
    return False


def _mime_for(key: str) -> str:
    suffix = PurePosixPath(key).suffix.lower()
    return _MIME_BY_EXT.get(suffix, _DEFAULT_MIME)


class S3FrameStorage:
    """Адаптер `FrameStorage` поверх `ObjectStorage`.

    Низ (`storage`) — готовый S3/MinIO-клиент с кредами из настроек процесса;
    модуль его не создаёт, а принимает готовым (креды в модуль не попадают, Т1).
    """

    def __init__(self, storage: ObjectStorage, *, expires_s: int = 3600) -> None:
        self._storage = storage
        self._expires_s = expires_s

    async def load(self, key: str) -> bytes:
        if not is_allowed_key(key):
            raise InvalidObjectKey(f"Недопустимый ключ кадра: {key!r}")
        return await self._storage.read(key)

    def media_url(self, key: str) -> str:
        if not is_allowed_key(key):
            raise InvalidObjectKey(f"Недопустимый ключ кадра: {key!r}")
        return self._storage.signed_url(key, expires_s=self._expires_s)


class S3FrameSource:
    """Адаптер `FrameSource`: ключ → `CXFrame.data_url` (упаковка байт).

    Байты берутся у `FrameStorage`; в персист/очередь не пишутся — только ключи
    (`screenshot_keys`), см. threat-model Т3.
    """

    def __init__(self, storage: FrameStorage) -> None:
        self._storage = storage

    async def frame(self, key: str, *, step: int) -> CXFrame:
        data = await self._storage.load(key)
        encoded = base64.b64encode(data).decode("ascii")
        return CXFrame(
            step=step,
            data_url=f"data:{_mime_for(key)};base64,{encoded}",
        )