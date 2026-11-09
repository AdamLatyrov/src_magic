import base64
from pathlib import Path
from typing import Optional

from httpx import Limits, Timeout
from pydantic import Field

from src.pipeline.settings.core import CoreSettings


class BaseApiSettings(CoreSettings):
    base_path: str = Field(default="configs/private/")
    ssl: bool
    timeout: Optional[int] = None
    cert: Optional[str] = None
    key: Optional[str] = None
    ca: Optional[str] = None
    cp: Optional[str] = None
    max_keepalive_connections: Optional[int] = None
    max_connections: Optional[int] = None
    keepalive_expiry: Optional[float] = None

    @property
    def cp_(self):
        return base64.b64decode(self.cp).decode(encoding="UTF-8")

    @property
    def cert_path(self):
        return (
            Path.cwd().joinpath(self.base_path, self.cert).as_posix()
            if self.cert
            else None
        )

    @property
    def key_path(self):
        return (
            Path.cwd().joinpath(self.base_path, self.key).as_posix()
            if self.key
            else None
        )

    @property
    def ca_path(self):
        return (
            Path.cwd().joinpath(self.base_path, self.ca).as_posix()
            if self.ca
            else None
        )

    @property
    def verify_(self):
        return self.ca_path if (self.ssl and self.ca) else False

    @property
    def cert_(self):
        return (
            (self.cert_path, self.key_path)
            if (self.ssl and self.cert and self.key)
            else None
        )

    @property
    def httpx_params(self):
        return {
            "timeout": Timeout(self.timeout),
            "verify": self.verify_,
            "cert": self.cert_,
            "limits": Limits(
                max_keepalive_connections=self.max_keepalive_connections,
                max_connections=self.max_connections,
                keepalive_expiry=self.keepalive_expiry,
            ),
        }
