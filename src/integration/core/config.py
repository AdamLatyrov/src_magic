import os
from typing import Type

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from src.integration.settings.api.gigachat import GigaChatSettings
from src.integration.settings.api.persist import RunBindingSettings
from src.integration.settings.api.service import (
    DatabaseSettings,
    ServiceSettings,
)
from src.pipeline.settings.api.graph import GraphSettings
from src.pipeline.settings.core import CoreSettings
from src.pipeline.utils.external_sources import YAMLConfigSettingsSource


class Settings(CoreSettings):
    """Настройки процесса: сервисный слой, контур ГигаЧата и граф прогона.

    Секции `gigachat` и `graph` описаны в YAML (`CONFIG_PATH`) и обязательны: им
    нужен состав полей, которого в переменных окружения нет. Секции сервисного
    слоя собраны через `default_factory`: каждая из них читает свои переменные
    сама, поэтому процесс стартует и без строк в YAML — файл настроек стенда
    задаёт их переменными, а тестам живости хватает значений по умолчанию.
    """

    service: ServiceSettings = Field(default_factory=ServiceSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    binding: RunBindingSettings = Field(default_factory=RunBindingSettings)
    gigachat: GigaChatSettings
    graph: GraphSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        filepath = os.environ.get("CONFIG_PATH")
        yaml_settings = YAMLConfigSettingsSource(
            filepath=filepath, settings_cls=settings_cls
        )
        return init_settings, env_settings, yaml_settings, file_secret_settings


class LoggerSettings(CoreSettings):
    debug: bool = Field(default=False, alias="LOGGER_DEBUG")
    level: str = Field(default="INFO", alias="LOGGER_LEVEL")
    context_vars: list = ["request_id", "user_id", "call_id"]
    format: str = Field(
        default="%(asctime)s | %(name)-50s | %(levelname)-8s | request_id: %(request_id)-40s "
        "| call_id: %(call_id)-40s | %(message)s"
    )
    default_handlers: list = [
        "console",
        "json_file",
    ]
    log_max_bytes: int = Field(default=1048576, alias="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=1, alias="LOG_BACKUP_COUNT")
    log_file_path: str = Field(default="app.log", alias="LOG_PATH")
    logger_verbose: bool = Field(default=False, alias="LOGGER_VERBOSE")


settings = Settings()
logger_settings = LoggerSettings()
