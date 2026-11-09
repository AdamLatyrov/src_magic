"""Модуль хранилища кадров.

Порты `FrameStorage`/`FrameSource` (`frames.py`) и boto3-адаптер порта
`ObjectStorage` (`s3.py`) поверх S3/MinIO с настройками `S3_*` из env,
по мотиву вендора, но на наших интерфейсах.
"""

from src.integration.storage.frames import (
    FrameSource,
    FrameStorage,
    InvalidObjectKey,
    ObjectStorage,
    S3FrameSource,
    S3FrameStorage,
    is_allowed_key,
)
from src.integration.storage.s3 import (
    S3ObjectStorage,
    S3Settings,
    s3_error_kind,
)

__all__ = [
    "FrameSource",
    "FrameStorage",
    "InvalidObjectKey",
    "ObjectStorage",
    "S3FrameSource",
    "S3FrameStorage",
    "is_allowed_key",
    "S3ObjectStorage",
    "S3Settings",
    "s3_error_kind",
]