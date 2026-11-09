"""
Файл настроек запуска: строка файла главнее переменных процесса.

Настройки процесса приходят двумя путями: переменными самого запуска (образ,
compose, оболочка) и одним файлом, путь к которому назван в `ENV_FILE` — на
стенде это файл из секмана. Файл главнее: он источник правды стенда, и
заведённая в развёртывании переменная, даже пустая, не должна молча его перебить.

Разборка своя, а не python-dotenv: ради десятка строк `KEY=VALUE` в образе не
нужна ещё одна зависимость.

Поведение повторяет `launcher/envfile.py` — повтор сознательный и временный:
`launcher/` принадлежит образу на ядре, а `scripts/assemble.sh` копирует в него
только `vendor/`, `overlay/` и `patches/`. Общая библиотека заставила бы чинить
лаунчер ядра под дерево, которого в его образе нет; при переключении приёмки
`launcher/` уходит вместе с ядром, и здесь останется единственный разбор файла.
"""

import os
import re
from collections.abc import MutableMapping
from pathlib import Path


class EnvFileError(ValueError):
    """Файл настроек не читается как `KEY=VALUE`.

    Текст сообщения называет путь и строку, но не значение: в файле лежат
    секреты, а отказ с обрывком секрета в журнале — уже утечка.
    """


def read_env_file(path: Path) -> dict[str, str]:
    """Разбирает файл настроек в пары имя → значение; нет файла — пусто.

    Понимает пустые строки, комментарии `#`, префикс `export `, значение в
    одинарных или двойных кавычках (как есть, без экранирования) и комментарий
    после значения без кавычек.

    :param path: путь к файлу настроек
    :raises EnvFileError: строка без `=` или с недопустимым именем переменной
    :return: значения файла в порядке их в файле
    """
    _key_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _key_pattern.fullmatch(name):
            raise EnvFileError(
                f"{path}:{number}: ожидалась строка вида ИМЯ=значение"
            )
        values[name] = _value(value.strip())
    return values


def load_env_file(
    path: Path, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Записывает переменные файла в окружение, подменяя уже заданные.

    :param path: путь к файлу настроек
    :param environ: окружение, куда писать; по умолчанию — окружение процесса
    :raises EnvFileError: файл не читается как `KEY=VALUE`
    :return: имена переменных, взятых из файла, без значений
    """
    target = os.environ if environ is None else environ
    values = read_env_file(path)
    target.update(values)
    return list(values)


def load_startup_env(
    environ: MutableMapping[str, str] | None = None,
) -> tuple[Path, list[str]]:
    """Записывает в окружение файл настроек запуска.

    Файл, названный в `ENV_FILE`, обязан существовать: стенд, где секман не
    подложил настройки, иначе поднялся бы на умолчаниях и делал бы вид, что
    работает. Без `ENV_FILE` отсутствия `.env` в каталоге запуска — обычное дело.

    :param environ: окружение, куда писать; по умолчанию — окружение процесса
    :raises EnvFileError: файла из `ENV_FILE` нет или его строки не читаются
    :return: прочитанный путь и имена переменных файла, без значений
    """
    target = os.environ if environ is None else environ
    named = target.get("ENV_FILE", "").strip()
    path = Path(named) if named else Path(".env")
    if named and not path.is_file():
        raise EnvFileError(f"ENV_FILE={path}: файла нет")
    return path, load_env_file(path, target)


def _value(value: str) -> str:
    """Значение строки: снятые кавычки и отброшенный хвостовой комментарий."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.startswith("#"):
        return ""
    comment = value.find(" #")
    return value[:comment].rstrip() if comment >= 0 else value
