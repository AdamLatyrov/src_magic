"""Снимок исполнения прогона: чем именно считается задание.

S02 спрашивает работника перед постановкой задания и кладёт ответ в задание
(`params.analysis_execution`). Спрашивают процесс, который будет задание
исполнять: копия этой логики в Java разошлась бы с ним молча, а задание со
снимком, которого работник не понимает, падало бы только после захвата.

Снимок описывает **весь** исполняемый контракт прогона, а не только модели:
модель и калибровку каждого этапа, мерки кадров, режим, состав проверок и
блоков, редакцию справочника и отпечаток контура. Всё это читается во время
прогона слоем `src/service/execution_settings.py`, поэтому задание, поставленное
до перезапуска процесса на другом файле настроек, считается тем, чем его ставили.

Значения берутся из настроек графа, а не из окружения: на этом дереве у каждого
этапа своя настройка, и одна переменная (`LLM_MAX_IMAGES_PER_REQUEST`) читается
сразу несколькими узлами — по имени переменной снимок не собрать.

Форма — ядровая (`schema_version 1.0`): ключи `base`, `frame_filter`, `deep`,
`consolidate` ждёт S02, и у нас на них ложатся узлы, ведущие ту же работу.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

from src.integration.core.config import Settings
from src.integration.service.version import version
from src.pipeline.schemas.nodes import CXCriteriaModel
from src.pipeline.settings.api.graph import (
    BaseSummarizeSettings,
    GraphSettings,
)

SCHEMA_VERSION = "1.0"
MODE = "legacy"
"""Режим один: сравнение — админская ручка ядра, S02 сравнений не заводит."""

PHASES: Mapping[str, str] = {
    "base": "check_cx_criteria",
    "frame_filter": "scan_frame",
    "deep": "consolidate_findings",
    "consolidate": "summarize_journey",
}
"""Фаза ядра -> узел этого графа, ведущий ту же работу.

У ядра было четыре фазы со своей переменной модели (`EVAL_MODEL`,
`FRAME_FILTER_MODEL`, `EVAL_DEEP_MODEL`, `EVAL_CONSOLIDATE_MODEL`). Наш граф
другой, и один-к-одному фазе отвечает не одноимённый узел, а тот, что делает ту
же работу: скан кадров — фильтр кадров, сведение находок — глубокий разбор.
Имена фаз остаются ядровыми: по ним S02 и задание читают снимок.
"""

PHASE_VARIABLES: Mapping[str, str] = {
    "base": "LLM_CHECK_CX_CRITERIA",
    "frame_filter": "LLM_SCAN_JOURNEY_FRAME",
    "deep": "LLM_CONSOLIDATE_FINDINGS",
    "consolidate": "LLM_SUMMARIZE_JOURNEY",
}
"""Фаза снимка -> переменная модели её узла.

Имена переменных живут у полей настроек (`alias`), и здесь они нужны затем,
что узел спрашивает контур по своему имени, а снимок называет фазу. Соответствие
названо один раз и рядом с `PHASES`: разъехавшись, эти две таблицы означали бы,
что снимок закрепляет модель не того этапа.
"""

CALIBRATION: Mapping[str, str] = {
    "LLM_TEMPERATURE": "temperature",
    "LLM_TOP_P": "top_p",
    "LLM_MAX_TOKENS": "max_tokens",
    "LLM_REPETITION_PENALTY": "repetition_penalty",
    "LLM_PROFANITY_CHECK": "profanity_check",
    "LLM_REASONING_EFFORT": "reasoning_effort",
}
"""Калибровка этапа: имя настройки -> поле `BaseSummarizeSettings`.

Порядок и состав повторяют `chat_options` того же класса — второго списка
настроек нет, здесь названы только имена, под которыми они едут в снимке.
"""


class SnapshotRefused(Exception):
    """Снимок не собирается: в настройках нет модели этапа или узла."""


class SnapshotRejected(ValueError):
    """Снимок из задания непригоден: прогон на нём считать нельзя."""


def build_snapshot(
    settings: Settings, criteria: CXCriteriaModel
) -> dict[str, Any]:
    """Снимок исполнения на момент постановки задания.

    Справочник приходит готовым из базы и подкладывается в настройки этапа:
    файла нет, и снимок обязан назвать ту редакцию каталога, которой прогон
    будет считаться, а не ту, что лежала на диске у сборки.

    :param settings: собранные настройки процесса
    :param criteria: справочник из базы — что именно будет считаться
    :raises SnapshotRefused: фаза не нашла узел или узел не назвал модель
    :return: снимок в ядровой форме
    """
    graph = settings.graph
    graph.bind_criteria_model(criteria)
    stages = _phase_stages(graph)
    base = stages["base"]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "models": {
            phase: _model_of(phase, stage) for phase, stage in stages.items()
        },
        "model_sources": {
            phase: _source_of(stage) for phase, stage in stages.items()
        },
        "runtime_settings": _runtime_settings(base),
        "calibration": _calibration(base),
        "composition": _composition(graph),
        "catalog": _catalog(base),
        "transport": _transport(settings),
    }


def execution_settings(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Настройки исполнения в том виде, в каком их читает прогон.

    Узлы ищут значение по имени переменной этапа, а снимок называет фазы и поля —
    здесь одно переводится в другое. Модель фазы ложится под переменную своего
    узла (`PHASE_VARIABLES`), калибровка — под общее имя.

    :param snapshot: снимок из задания
    :return: имя настройки -> значение; то, что читает `execution_setting`
    """
    values: dict[str, Any] = {}
    for phase, name in PHASE_VARIABLES.items():
        model = (snapshot.get("models") or {}).get(phase)
        if model:
            values[name] = model
    values.update(snapshot.get("runtime_settings") or {})
    values.update(snapshot.get("calibration") or {})
    return {name: value for name, value in values.items() if name}


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Проверяет снимок из задания до прогона.

    Задание могло быть поставлено другой сборкой процесса: снимок едет через
    JSON и никем не проверялся до захвата. Неверный снимок — отказ задания, а не
    прогон на значениях по умолчанию: молча посчитать не тем хуже, чем не
    посчитать вовсе.

    :param snapshot: снимок из задания
    :raises SnapshotRejected: схема, фазы, модели или мерки не годятся
    """
    if not isinstance(snapshot, Mapping):
        raise SnapshotRejected("снимок исполнения не словарь")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotRejected(
            f"снимок исполнения версии {snapshot.get('schema_version')!r},"
            f" ожидается {SCHEMA_VERSION!r}"
        )
    if snapshot.get("mode") != MODE:
        raise SnapshotRejected(
            f"режим исполнения {snapshot.get('mode')!r} не поддерживается"
        )
    models = snapshot.get("models")
    if not isinstance(models, Mapping) or set(models) != set(PHASES):
        raise SnapshotRejected(
            f"в снимке не четыре фазы: {sorted(models or {})}"
        )
    empty = sorted(phase for phase, model in models.items() if not model)
    if empty:
        raise SnapshotRejected(f"в снимке нет модели фаз {empty}")
    limits = snapshot.get("runtime_settings") or {}
    if not isinstance(limits, Mapping):
        raise SnapshotRejected("мерки исполнения не словарь")
    for name, value in limits.items():
        if name not in _RUNTIME_ALLOWED:
            raise SnapshotRejected(f"мера {name} в снимке неизвестна")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SnapshotRejected(f"мера {name} в снимке не целое ≥ 0")


_RUNTIME_ALLOWED = frozenset(
    {"LLM_MAX_IMAGES_PER_REQUEST", "LLM_MIN_FRAMES_PER_REQUEST"}
)
"""Что прогон берёт из мерок: пределы кадровой нарезки.

Список закрыт намеренно: ключ вне его — расхождение сборок, и принимать его
молча значило бы исполнять задание не тем, чем его ставили.
"""


def _phase_stages(graph: GraphSettings) -> dict[str, BaseSummarizeSettings]:
    """Настройки каждого этапа снимка; этапы, которых нет, — отказ, а не пропуск.

    Скан кадров живёт внутри `journey_exploration` (`scan_frame`), и путь к нему
    назван здесь, а не угадывается по имени узла верхнего уровня.
    """
    resolved: dict[str, BaseSummarizeSettings] = {}
    missing: list[str] = []
    for phase, node in PHASES.items():
        stage = (
            graph.journey_exploration.scan_frame
            if node == "scan_frame"
            else getattr(graph, node, None)
        )
        if isinstance(stage, BaseSummarizeSettings):
            resolved[phase] = stage
        else:
            missing.append(f"{phase}={node}")
    if missing:
        raise SnapshotRefused(
            "у графа нет узла этапа: " + ", ".join(sorted(missing))
        )
    return resolved


def _model_of(phase: str, stage: BaseSummarizeSettings) -> str:
    """Модель этапа; пустое имя — отказ называет фазу, а не значение."""
    model = (stage.model or "").strip()
    if not model:
        raise SnapshotRefused(f"этап {phase} не назвал модель")
    return model


def _source_of(stage: BaseSummarizeSettings) -> str:
    """Откуда взялось значение модели: `env` или `default`.

    Строка конфига `model:` и подмена общей `LLM_MODEL` сюда не попадают:
    настройка не хранит, каким источником собрано поле, и выдавать одно за другое
    значило бы соврать о происхождении модели.
    """
    own = type(stage).model_fields["model"].alias
    if own and os.environ.get(own, "").strip():
        return "env"
    return "default"


def _runtime_settings(base: BaseSummarizeSettings) -> dict[str, int]:
    """Мерки прогона: пределы кадровой нарезки с этапа блоков."""
    return {
        "LLM_MAX_IMAGES_PER_REQUEST": base.max_images_per_request,
        "LLM_MIN_FRAMES_PER_REQUEST": base.min_frames_per_request,
    }


def _calibration(base: BaseSummarizeSettings) -> dict[str, Any]:
    """Калибровка этапа — значения процесса на момент постановки."""
    return {
        name: getattr(base, field) for name, field in CALIBRATION.items()
    }


def _composition(graph: GraphSettings) -> dict[str, Any]:
    """Состав прогона: какие проверки и блоки в нём были.

    Задание, поставленное на одном составе, а исполненное на другом, дало бы
    другие находки — и заметить это было бы нечем.
    """
    return {
        "checks": list(graph.journey_frictions.keys()),
        "blocks": [block.key for block in graph.check_cx_criteria.blocks()],
    }


def _catalog(base: BaseSummarizeSettings) -> dict[str, str]:
    """Редакция справочника, которым считается путь.

    Имени версии мало: под одним номером может лежать переписанный выпуск,
    поэтому в отпечаток идёт и версия, и идентификатор выпуска.
    """
    info = base.load_criteria_model().info
    return {
        "key": info.key,
        "version": info.version,
        "id": info.id or "",
    }


def _transport(settings: Settings) -> dict[str, str]:
    """Отпечаток контура и сборки: чем и каким кодом ходили.

    Секретов здесь нет: адрес, скоуп и версия процесса — то, что позволяет
    сверить снимок с тем, куда прогон реально ходил.
    """
    return {
        "base_url": settings.gigachat.base_url,
        "scope": str(settings.gigachat.scope or ""),
        "build": version(),
    }


def fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Короткий отпечаток снимка — им сверяют постановку и исполнение в журнале.

    :param snapshot: снимок исполнения
    :return: первые 16 символов SHA-256 канонического JSON
    """
    canonical = json.dumps(
        snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CALIBRATION",
    "MODE",
    "PHASES",
    "PHASE_VARIABLES",
    "SCHEMA_VERSION",
    "SnapshotRefused",
    "SnapshotRejected",
    "build_snapshot",
    "execution_settings",
    "fingerprint",
    "validate_snapshot",
]
