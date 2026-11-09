"""Контур модели: вход и модели всех этапов проверены до старта сервера.

Другого поставщика моделей внутри контура нет: адрес `gigachat` один, а модели
у него свои — `Qwen3.5-397b`, `GigaChat-3-Ultra` и всё остальное, что контур
назовёт в своём списке. Проверяется поэтому не фамилия модели, а другое: вход
ведёт в контур, и контур знает каждую модель из настроек этапов. Процесс,
который не может отдать ни один этап, не должен подниматься и притворяться
готовым; отказ до первого запроса стоит на стенде секунду, отказ после захвата
задания — потерянный прогон.

Повторяет поведению `overlay/src/gigachat_only.py`, на который работает образ на
ядре, — но читает наши настройки, а не переменные окружения ядра.

Здесь перенесено ровно то, что нужно запуску, — `prepare` и `probe`. Не
перенесено запретительное (`enforce`, `allows`, `refused_models`, снимок для
S02) и ручки генерации: это фаза 5 плана `docs/port-vendor-service-into-src.md`,
где отказ чужой модели становится частью контракта запроса, а не проверки старта.

Вход описан секцией `gigachat` конфига (`ssl`, `base_path`, `cert`, `key`, `ca`), а
способ его берёт из файла настроек: задан `GIGACHAT_CREDENTIALS` — вход по ключу,
нет — по файлам сертификата. Имена файлов разбираются относительно `base_path`, а
чем именно проверять сервер решает
`pipeline.aef.patterns.gigachat_provider.client_ssl_context`: при входе по клиентскому
сертификату он же собирает mTLS-контекст с потолком TLS 1.2, без которого контур
сертификат не принимает; при входе по ключу файлы секции не участвуют вовсе.
"""

import asyncio
import logging
import os
from collections.abc import Sequence

from src.integration.core.config import Settings
from src.integration.service.errors import detail_of
from src.integration.service.launch import stage_models
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider

logger = logging.getLogger(__name__)

probe_timeout_s = 30.0
"""Срок одной пробы контура: дольше ждать «ГигаЧат не отвечает» смысла нет."""


class GigachatStartRefused(Exception):
    """Старт отвергнут: настройки входа или моделей не ведут в контур."""


def login_mode(settings: Settings) -> str:
    """Как процесс входит в контур: по ключу, по сертификату или вовсе без входа.

    Ключ главнее файлов входа: его задаёт одна переменная файла настроек, а файлы
    описывают стенд в конфиге и остаются на месте при входе по ключу. Так стенд
    переключается правкой `.env`, а не правкой конфига.

    Без входа — обычный путь стенда: адрес контура указывает на прокси, который
    входит в ГигаЧат сам и может слушать простой `http://`.

    :param settings: собранные настройки процесса
    :return: `credentials`, `certificate` или `none (proxy)`
    """
    gigachat = settings.gigachat
    if gigachat.login_by_token:
        return "credentials"
    if gigachat.cert_path and gigachat.key_path:
        return "certificate"
    return "none (proxy)"


def prepare(settings: Settings) -> list[str]:
    """Проверяет настройки контура и собирает модели всех этапов для пробы.

    :param settings: собранные настройки процесса
    :raises GigachatStartRefused: сертификат задан без ключа или не читается,
        вход есть поверх простого `http://`, этап остался без модели
    :return: имена моделей этапов без повторов, отсортированные
    """
    gigachat = settings.gigachat
    problems: list[str] = []

    certificate = bool((gigachat.cert or "").strip())
    key = bool((gigachat.key or "").strip())
    if certificate != key:
        problems.append(
            "gigachat: cert и key задаются вместе или не задаются вовсе"
        )
    if gigachat.login_by_token:
        if certificate or key:
            logger.info(
                "[gigachat] вход по ключу: cert и key из конфига остались на месте,"
                " для этого входа они не читаются"
            )
    else:
        for label, path in (
            ("cert", gigachat.cert_path),
            ("key", gigachat.key_path),
            ("ca", gigachat.ca_path),
        ):
            if path and not _readable(path):
                problems.append(f"gigachat: файл {label} не читается")

    if not _scheme_allows_login(gigachat.base_url, login_mode(settings)):
        problems.append(
            "gigachat.base_url должен быть https://; http:// допускается только"
            " без входа — прокси стенда"
        )

    stages = stage_models(settings)
    unnamed = sorted(name for name, used in stages.items() if not used.strip())
    if unnamed:
        problems.append(f"этап без модели: {unnamed}")
    models = sorted({used.strip() for used in stages.values() if used.strip()})
    if not models:
        problems.append("ни один этап графа не назвал свою модель")

    if problems:
        raise GigachatStartRefused("; ".join(problems))

    logger.info(
        "[gigachat] contour: login=%s, verify_server=%s, models=%s",
        login_mode(settings),
        gigachat.ssl,
        models,
    )
    if not gigachat.ssl:
        logger.warning(
            "[gigachat] сертификат сервера НЕ проверяется (gigachat.ssl=false):"
            " вход и кадры идут по TLS, которому доверяют любой сервер; включите"
            " ssl и положите доверяющий набор в ca"
        )
    return models


def probe(settings: Settings, models: Sequence[str]) -> None:
    """Один раз спрашивает у контура список моделей и проверяет в нём каждый этап.

    Единственная попытка и свой срок: пробы, которая переспрашивает трижды по
    минуте, в критическом пути запуска быть не должно — контейнер к этому моменту
    ещё не успел стать готовым.

    :param settings: собранные настройки процесса
    :param models: имена моделей, которые вернул `prepare`
    :raises GigachatStartRefused: контур недоступен, отверг вход или не знает
        хотя бы одной модели из настроек
    """
    provider = GigaChatProvider(
        base_url=settings.gigachat.base_url,
        authorization=settings.gigachat.auth,
        retry_limit=1,
        request_timeout=int(probe_timeout_s),
    )
    try:
        models_of = asyncio.run(provider.list_models()).models
        offered = {model.name for model in models_of}
    except Exception as exc:
        logger.warning("проба ГигаЧата не прошла", exc_info=True)
        raise GigachatStartRefused(
            f"ГигаЧат не готов: {detail_of(exc)}"
        ) from exc

    missing = sorted(set(models) - offered)
    if missing:
        raise GigachatStartRefused(
            f"ГигаЧат не даёт модели {missing}; доступны: {sorted(offered)}"
        )
    logger.info("readiness ok: models=%s", sorted(set(models)))


def _scheme_allows_login(base_url: str, mode: str) -> bool:
    """Простой `http://` контура допустим ровно одному способу входа — без входа."""
    scheme = base_url.lower().split("://", 1)[0]
    if scheme == "https":
        return True
    return scheme == "http" and mode == "none (proxy)"


def _readable(path: str) -> bool:
    """Файл входа на месте и прочитан (путь в текст отказа не попадает)."""
    return os.path.isfile(path) and os.access(path, os.R_OK)
