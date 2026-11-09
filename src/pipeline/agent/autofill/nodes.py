"""Узел фокусного прохода автозаполнения: один вопрос про лишний ручной ввод.

Ветка кадров: кадры уходят в вопрос содержимым с картинками, как в соседних
ветках (`pipeline.agent.journey_frictions.nodes`, `pipeline.agent.criteria_model_frames`).
Отличие одно и то же везде: вопрос узкий — только про те поля, значение которых
банк уже знает, и задаётся он один раз на весь путь.

Узел делает то же, что узлы `pipeline.agent.nodes`: собирает вопрос по настройкам,
спрашивает контур и кладёт его ответ в состояние. Разбором ответа ведает
`pipeline.utils.autofill_findings`, а формой отчёта — DTO после графа.

Группы кадров (`batch_frames`) узел собирает сам и спрашивает их последовательно:
на десятках экранов форма тонет среди дублей карточки продукта, а одновременные
вопросы складываются в очередь контура (429). Размер группы — документированный
лимит контура, короткий хвост пути дотягивается кадрами предыдущей группы
(`group_slices`), и каждое обращение подписано номером батча
(`AutofillSettings.batch_note`) — по нему вызовы различают и журнал, и
лаборатория. `batch_frames = 0` возвращает один вопрос на весь путь, но не дальше
лимита изображений.
"""

import logging
from collections import Counter

from src.pipeline.aef.exceptions import MalformedAnswer
from src.pipeline.aef.models import ChatResponse
from src.pipeline.aef.patterns.gigachat_provider import GigaChatProvider
from src.pipeline.aef.patterns.retry import (
    failure_of,
    is_shape_failure,
    llm_retryable,
    repeat_recoverable,
)
from src.pipeline.aef.patterns.utils import node
from src.pipeline.schemas.nodes import (
    CxAutofillFindings,
    CXFrame,
    CXPinnedDeviation,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.api.graph import AutofillSettings, GraphSettings
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    NOT_ASKED_PREFIX,
    CXNodeStatus,
    CXZone,
)
from src.pipeline.utils.autofill_findings import (
    autofill_findings,
    autofill_update,
)

logger = logging.getLogger(__name__)


@node
async def check_autofill(
    state: AgentStateSchema,
    gigachat_provider: GigaChatProvider,
    settings: GraphSettings,
) -> dict:
    """Сколько полевого труда клиента банк мог взять на себя.

    Узел спрашивает все кадры пути: значение, которое банк уже знает, и поле,
    где его просят набрать заново, живут на разных экранах, и по половине
    маршрута такой повтор не виден вовсе.

    Гейт применимости решают настройки, а не слово модели: политика прогона
    запрещает ссылаться на то, чем путь является (основание, которое снимало с
    проверки целые блоки), поэтому список проверяемых типов путей пуст по
    умолчанию — проход идёт по пути любого типа и объясняет пустоту сам (`note`
    ответа). Пропуск узел пишет в `degraded` отметкой `not_asked:` — «не
    спрашивали» обязано отличаться от «спросили и чисто».

    Сбой делится как в соседних ветках: неразборчивый ответ и временный отказ
    контура поднимаются наверх политике повторов, остальное гасится здесь. Когда
    часть групп сорвана, узел оставляет разобранные находки и пишет счётчик в
    `degraded` — молчание прохода не должно читаться как «предзаполнять нечего».

    :param state: состояние прогона: материалы, кадры и атрибуция пути
    :param gigachat_provider: провайдер контура модели
    :param settings: настройки графа — из них берётся эта ветка
    :return: находки прохода в состояние прогона
    """
    autofill = settings.autofill
    label = LogLabel.CHECK_AUTOFILL
    frames_count = len(state.frames)
    asked = state.kept_frames

    skip = autofill.skip_reason(state.journey_details)
    if not asked:
        skip = skip or (
            "для анализа не осталось кадров"
            if frames_count
            else "кадров пути нет"
        )
    if skip:
        logger.info("%s: проход не запускается — %s", label, skip)
        return autofill_update(degraded=f"{NOT_ASKED_PREFIX}{skip}")

    groups = autofill.batches(asked)
    answers: list[tuple[CxAutofillFindings, str]] = []
    broken: list[Exception] = []
    for number, batch in enumerate(groups, start=1):
        try:
            answers.append(
                await _ask_batch(
                    autofill,
                    gigachat_provider,
                    batch,
                    state,
                    frames_count,
                    number,
                    len(groups),
                )
            )
        except Exception as error:
            broken.append(error)
            if not is_shape_failure(error) and not llm_retryable(error):
                logger.warning(
                    "%s: проход остановлен на батче %s из %s (%s)",
                    label,
                    number,
                    len(groups),
                    failure_of(error),
                )
                break

    failed = len(groups) - len(answers)
    if not answers:
        error = broken[0]
        if is_shape_failure(error) or llm_retryable(error):
            logger.warning(
                "%s: ни один из %s батчей не отвечен (%s); обращение к контуру "
                "повторится",
                label,
                len(groups),
                failure_of(error),
            )
            if is_shape_failure(error):
                raise MalformedAnswer(
                    f"проход автозаполния не отвечен: {failure_of(error)}"
                ) from error
            raise error
        logger.error(
            "%s: Failed to generate response check_autofill. Exception: %s",
            label,
            failure_of(error),
            exc_info=error,
        )
        return autofill_update(status=CXNodeStatus.ERROR)

    findings = _merge(label.value, [part for part, _ in answers])
    model = answers[0][1]
    degraded = f"batches_failed:{failed}/{len(groups)}" if failed else ""
    counts = Counter(item.zone for item in findings.deviations)
    zones = ", ".join(f"{zone}={counts.get(zone, 0)}" for zone in CXZone)
    why = findings.emptiness_note()
    logger.info(
        "%s: кадров %s, батчей %s (сорвано %s), экранов с лишним вводом %s (%s)%s%s,"
        " модель %s",
        label,
        f"{len(asked)} из {frames_count}",
        len(groups),
        failed,
        len(findings.deviations),
        zones,
        f"; {degraded}" if degraded else "",
        f" — {why}" if why else "",
        model,
    )
    return autofill_update(findings=findings, degraded=degraded)


async def _ask_batch(
    autofill: AutofillSettings,
    gigachat_provider: GigaChatProvider,
    batch: list[CXFrame],
    state: AgentStateSchema,
    frames_count: int,
    number: int = 1,
    count: int = 1,
) -> tuple[CxAutofillFindings, str]:
    """Одна группа кадров, спрашиваемая заново, пока сбой лечится повтором.

    Обращения идут последовательно, и политикой графа частичная потеря не
    прикрывается: узел перевызывает себя целиком только когда не отвечена ВСЯ
    группа. Пока одна группа отвечена, сорванная молча оставалась бы
    непросмотренной половиной путей ввода — её считает счётчик `degraded` узла.

    :param autofill: настройки ветки
    :param gigachat_provider: провайдер контура модели
    :param batch: кадры этой группы
    :param state: состояние прогона — заметка эксперта и атрибуция пути
    :param frames_count: число кадров всего пути — за его пределами экрана нет
    :param number: который по счёту батч
    :param count: сколько батчей легло на проход
    :return: находки группы и имя отвечавшей модели
    :raises Exception: группа осталась без ответа после всех попыток
    """
    return await repeat_recoverable(
        lambda: _ask_contour(
            autofill,
            gigachat_provider,
            batch,
            state,
            frames_count,
            number,
            count,
        ),
        subject=(
            f"{LogLabel.CHECK_AUTOFILL}: батч {number} из {count} "
            f"(кадры {batch[0].step}-{batch[-1].step})"
        ),
    )


async def _ask_contour(
    autofill: AutofillSettings,
    gigachat_provider: GigaChatProvider,
    batch: list[CXFrame],
    state: AgentStateSchema,
    frames_count: int,
    number: int = 1,
    count: int = 1,
) -> tuple[CxAutofillFindings, str]:
    """Одна группа кадров: обращение к контуру и его разбор.

    Разбор стоит наравне с обращением: ответ без JSON и ответ, где нет перечня
    `findings`, — сбой группы, а не «предзаполнять было нечего». Сделано это
    внутри задачи нарочно: сорванная группа не имеет права утянуть за собой
    остальные, их находки уже заплачены.

    :param autofill: настройки ветки
    :param gigachat_provider: провайдер контура модели
    :param batch: кадры этой группы
    :param state: состояние прогона — заметка эксперта и атрибуция пути
    :param frames_count: число кадров всего пути — за его пределами экрана нет
    :param number: который по счёту батч — им подписан вопрос
    :param count: сколько батчей легло на проход
    :return: находки группы и имя отвечавшей модели
    :raises MalformedAnswer: ответ есть, но он не разобран
    """
    response: ChatResponse = await gigachat_provider.chat(
        autofill.chat_request(
            state.material,
            batch,
            state.journey_details,
            total=frames_count,
            number=number,
            count=count,
        )
    )
    try:
        findings = autofill_findings(
            response.message.content or "", frames_count
        )
    except ValueError as ex:
        raise MalformedAnswer(
            f"ответ группы автозаполнения не разобран: {ex}"
        ) from ex
    return findings, response.used_model.name


def _merge(
    source_key: str, parts: list[CxAutofillFindings]
) -> CxAutofillFindings:
    """Ответы групп в один итог ветки.

    Номера экранов модель берёт из подписей `[кадр N]`, а подписывает их узел
    глобальными номерами кадров: складывать смещение группы, как на своих батчах
    делает ядро, здесь не нужно, и находки групп просто складываются.

    Перехлёст хвоста пути (`group_slices`) даёт одну и ту же находку в двух
    обращениях: с экрана, который спрашивали дважды. Повтор снимается по номеру
    экрана и тексту — иначе штраф за одно поле прогон получил бы дважды, а отчёт
    показал бы дубль двумя строками.

    :param source_key: ключ источника — метка ветки
    :param parts: итоги разобранных групп
    :return: общий итог прохода
    """
    deviations: list[CXPinnedDeviation] = []
    seen: set[tuple[int | None, str, str]] = set()
    notes: list[str] = []
    dropped: Counter[str] = Counter()
    for part in parts:
        for deviation in part.deviations:
            mark = (
                deviation.step,
                deviation.code,
                " ".join(deviation.text.lower().split())[:80],
            )
            if mark in seen:
                continue
            seen.add(mark)
            deviations.append(deviation)
        dropped.update(part.dropped)
        if part.note.strip():
            notes.append(part.note.strip())
    return CxAutofillFindings(
        source_key=source_key,
        deviations=deviations,
        note=" | ".join(notes),
        dropped=dict(dropped),
    )
