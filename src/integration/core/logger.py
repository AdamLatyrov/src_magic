import json
import logging
from datetime import datetime
from logging import Filter, LogRecord
from typing import Any

from src.integration.core.config import LoggerSettings, logger_settings
from src.integration.core.context import LogParamsContext, context_var


class ContextFilter(Filter):
    def __init__(
        self, context_var: LogParamsContext, logger_settings: LoggerSettings
    ):
        super().__init__()
        self.context_var = context_var
        self.logger_settings = logger_settings

    def filter(self, record: LogRecord) -> bool:
        context_data = self.context_var.context

        for key, value in context_data.items():
            setattr(record, key, value)

        for field in self.logger_settings.context_vars:
            if not hasattr(record, field):
                setattr(record, field, "")

        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "asctime": datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "levelname": record.levelname,
            "request_id": getattr(record, "request_id", ""),
            "call_id": getattr(record, "call_id", ""),
            "message": record.getMessage(),
            "user_id": getattr(record, "user_id", ""),
            "logger_name": record.name,
            "thread_name": record.threadName,
            "event_type": "SberUserTranscriptionEvent",
        }

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)


def create_context_filter():
    return ContextFilter(
        context_var=context_var, logger_settings=logger_settings
    )


def create_json_formatter():
    return JsonFormatter()


LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "context_filter": {
            "()": create_context_filter,
        }
    },
    "formatters": {
        "standard": {"format": logger_settings.format},
        "json_formatter": {"()": create_json_formatter},
    },
    "handlers": {
        "console": {
            "level": logger_settings.level,
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "filters": ["context_filter"],
        },
        "json_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": logger_settings.log_file_path,
            "formatter": "json_formatter",
            "level": "INFO",
            "maxBytes": logger_settings.log_max_bytes,
            "backupCount": logger_settings.log_backup_count,
        },
    },
    "loggers": {
        "": {
            "handlers": logger_settings.default_handlers,
            "level": logger_settings.level,
            "propagate": True,
        },
        "httpx": {
            "handlers": logger_settings.default_handlers,
            "level": (
                logger_settings.level if logger_settings.debug else "WARNING"
            ),
            "propagate": False,
        },
        "httpcore": {
            "handlers": logger_settings.default_handlers,
            "level": (
                logger_settings.level if logger_settings.debug else "WARNING"
            ),
            "propagate": False,
        },
        "gigachat.client": {
            "handlers": logger_settings.default_handlers,
            "level": (
                logger_settings.level if logger_settings.debug else "WARNING"
            ),
            "propagate": False,
        },
        "aiohttp.access": {
            "handlers": logger_settings.default_handlers,
            "level": (
                logger_settings.level if logger_settings.debug else "WARNING"
            ),
            "propagate": False,
        },
        "aiokafka": {"level": "INFO", "handlers": [], "propagate": False},
    },
}
