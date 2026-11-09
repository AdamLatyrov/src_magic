import logging
import os
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

logger = logging.getLogger(__name__)


class YAMLConfigSettingsSource(PydanticBaseSettingsSource):
    """Раздел `configs/service_configs/*.yaml` как источник настроек.

    Секция конфига дополняется переменными окружения прогона: поле настроек узла
    берёт значение из переменной, имя которой записано его `alias`, поэтому список
    переменных живёт в описании полей, а не здесь. Порядок для одного поля: своя
    переменная, общая `LLM_MODEL` (только для поля `model`), строка конфига, значение
    поля по умолчанию. Переменная дополняет секцию, а не заменяет её: секция обязана
    быть описана в конфиге.
    """

    def __init__(self, filepath: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = self.get_app_config_path(filepath=filepath)
        self.file_content_json = yaml.safe_load(
            self.config_path.read_text(encoding="UTF8")
        )

    @staticmethod
    def get_app_config_path(filepath: str) -> Path:
        config_path = Path(__file__).resolve().parents[3].joinpath(filepath)
        if not config_path.exists():
            raise FileNotFoundError("Config file not found")
        return config_path

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        field_value = self.file_content_json.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        d: dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = self._with_environment(field, field_value)

        return d

    @classmethod
    def _with_environment(cls, field: FieldInfo, section: Any) -> Any:
        model = _settings_model(field.annotation)
        if model is not None and isinstance(section, dict):
            return cls._fill(model, section)

        item = _settings_model(_list_item(field.annotation))
        if item is not None and isinstance(section, list):
            return [cls._fill(item, row) for row in section]

        return section

    @classmethod
    def _fill(cls, model: type[BaseSettings], section: dict) -> dict:
        """Словарь секции плюс значения из окружения; саму секцию не собирает.

        Возвращается словарь, а не готовые настройки: секция проходит валидацию своей
        модели обычным путём, поэтому дефолты, `use_enum_values` и проверки самой
        модели остаются в силе.

        Общая переменная (`BaseSummarizeSettings.common_variables`) не перебивает
        строку конфига: она одна на все этапы, а калибровка у этапа своя. Пока это
        правило не действовало, `.env` с `LLM_MAX_TOKENS=4096` снимал лимит в 8192
        знака с ветки блоков — ответы блоков обрезались на середине JSON, окна
        считались сорванными, и прогон терял отклонения молча. Молчащий раздел
        конфига общая переменная по-прежнему дополняет, а своя переменная этапа и
        общая `LLM_MODEL` перебивают конфиг, как и обещано порядком источников.
        """
        filled = dict(section)

        for name, field in model.model_fields.items():
            nested = section.get(name)
            if isinstance(nested, (dict, list)):
                filled[name] = cls._with_environment(field, nested)

            variables = _variables(field)
            if not variables:
                continue

            own = _read(variables[0])
            common_model = name == "model" and not own
            value = own or (_read("LLM_MODEL") if name == "model" else None)
            if not value:
                continue
            if (
                not common_model
                and variables[0] in _common_variables(model)
                and _declared(section, field, name)
            ):
                logger.debug(
                    "%s: переменная %s пропущена — раздел конфига задал %s сам",
                    model.__name__,
                    variables[0],
                    name,
                )
                continue

            filled[field.alias or name] = value

        return filled


def _common_variables(model: type[BaseSettings]) -> frozenset[str]:
    """Переменные, общие для всех этапов: их объявляет базовый класс настроек узла.

    Список читается с класса, а не повторяется здесь: имена живут там, где описаны
    поля этапа, и второй список пришлось бы синхронизировать вручную.
    """
    return frozenset(getattr(model, "common_variables", frozenset()))


def _declared(section: dict, field: FieldInfo, name: str) -> bool:
    """Задал ли раздел конфига это поле сам — своим именем или именем переменной.

    Знак тот, что раздел отказался молчать: общая переменная приходит на помощь
    только туда, где конфига нет, и не подменяет то, что развёртывание назвало явно.
    """
    keys = {name, str(field.alias or ""), str(field.validation_alias or "")}
    return any(key in section for key in keys if key)


def _settings_model(annotation: Any) -> type[BaseSettings] | None:
    """Модель настроек в аннотации поля, если аннотация — она или `Optional` от неё."""
    if get_origin(annotation) in (UnionType, Union):
        for arg in get_args(annotation):
            model = _settings_model(arg)
            if model is not None:
                return model
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
        return annotation
    return None


def _list_item(annotation: Any) -> Any:
    """Тип элемента списка в аннотации поля."""
    if get_origin(annotation) in (list, tuple, set, frozenset):
        args = get_args(annotation)
        return args[0] if len(args) == 1 else None
    return None


def _variables(field: FieldInfo) -> tuple[str, ...]:
    """Имя переменной поля: его `alias`, либо каждый вариант `AliasChoices`.

    Поле без alias переменной не имеет — иначе переменными оказались бы служебные
    поля вроде `prompts_dir`.
    """
    alias = field.validation_alias or field.alias
    if isinstance(alias, str):
        return (alias,)
    choices = getattr(alias, "choices", None)
    if choices:
        return tuple(str(choice) for choice in choices)
    return ()


def _read(variable: str) -> str | None:
    """Значение переменной; пустая строка означает «переменной нет»."""
    value = os.environ.get(variable)
    if value is None or not value.strip():
        return None
    return value.strip()
