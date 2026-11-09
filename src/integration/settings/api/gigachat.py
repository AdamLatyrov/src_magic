from typing import Optional

from pydantic import Field

from src.integration.settings.api.base import BaseApiSettings
from src.pipeline.aef.patterns import GigaChatAuthorization
from src.pipeline.settings.params.gigachat import Scope


class GigaChatSettings(BaseApiSettings):
    base_url: str = Field(alias="GIGACHAT_BASE_URL")
    auth_url: Optional[str] = Field(default=None, alias="GIGACHAT_AUTH_URL")
    credentials: Optional[str] = Field(
        default=None, alias="GIGACHAT_CREDENTIALS"
    )
    scope: Optional[Scope] = Field(default=None, alias="GIGACHAT_SCOPE")
    timeout: int = Field(default=30.0, alias="GIGACHAT_TIMEOUT")
    max_retries: int = Field(default=2, alias="GIGACHAT_MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="GIGACHAT_RETRY_DELAY")

    @property
    def login_by_token(self) -> bool:
        """Вход идёт по ключу: `GIGACHAT_CREDENTIALS` задан и не пуст."""
        return bool((self.credentials or "").strip())

    @property
    def auth(self) -> GigaChatAuthorization:
        """Вход в контур: по ключу, если он задан, иначе по файлам сертификата.

        При ключе файлы секции конфига не участвуют вовсе: `cert_file` и `key_file`
        уходят пустыми строками — `None` заставил бы SDK добрать `GIGACHAT_CERT_FILE`
        из окружения, и handshake снова пошёл бы по клиентскому сертификату против
        контура, который его не ждёт. `ca_file` остаётся `None`: набор доверия серверу
        берёт не секция стенда, а окружение процесса — SDK читает
        `GIGACHAT_CA_BUNDLE_FILE`, а когда и его нет, провайдер строит контекст по
        умолчанию, то есть доверяет хранилищу машины
        (`pipeline.aef.patterns.gigachat_provider.client_ssl_context`).

        Скоуп контур сверяет с тем, что выдано ключу, поэтому он нужен токенному входу;
        на входе по сертификату токен берётся на клиентском эндпоинте, где скоуп не
        спрашивают, — там остаётся ровно то, что было до этой ветки.
        """
        token = self.credentials.strip() if self.credentials else ""
        if self.login_by_token:
            files = {"cert_file": "", "key_file": "", "ca_file": None}
            scope = self.scope or Scope.GIGACHAT_API_CORP.value
        else:
            files = {
                "cert_file": self.cert_path,
                "key_file": self.key_path,
                "ca_file": self.ca_path,
            }
            scope = self.scope
        return GigaChatAuthorization(
            credentials=token or None,
            auth_url=self.auth_url,
            verify=self.ssl,
            scope=scope,
            **files,
        )
