"""Находка в том виде, в каком она входит в балл.

Форма среза — `ScoringView` — остаётся в `src/schemas/scoring.py` вместе с
формой ответа прогона; здесь то, что находку делает числом: откуда берётся её
привязка и какие её поля считаются эффективными.
"""

from typing import Any, Optional

from src.pipeline.catalog.address import check_key_of
from src.pipeline.schemas.scoring import ScoringView
from src.pipeline.scoring.reading import field_of
from src.pipeline.settings.params.nodes import CXConfidence, CXSeverity, CXZone


def issue_check_key(issue: Any) -> Optional[str]:
    """Подкритерий, к которому привязана находка: своё поле или адрес дефекта.

    Формулу адреса задаёт `pipeline.catalog.address`, и привязка в нём есть всегда,
    когда разметка разобрала код. Читать только `check_key` значило бы молча
    не штрафовать блок по находке, пришедшей из ответа, где поля разведены, —
    при том, что ядро по тому же адресу привязку видит.
    """
    key = str(field_of(issue, "check_key") or "")
    if key:
        return key
    return check_key_of(field_of(issue, "defect_ref")) or None


def scoring_view(issue: Any) -> Optional[ScoringView]:
    """Находка для балла: эффективные зона, тяжесть и уверенность.

    Модерация меняет то, за что штрафуют, а не только счётчик мнений: названная
    проверкой тяжесть перевешивает зону справочника (зона берётся из самой мягкой
    зоны этой тяжести), пустая или посторонняя зона добирается из тяжести,
    принятая находка считается при полной уверенности. Без этого прогон мерил бы
    снятую модератором находку полным весом и писал бы `critical_count` по зоне,
    которую человек уже опроверг.

    :param issue: сведённая находка (`Issue`) или её дамп
    :return: срез для балла; None, если находка отклонена модерацией
    """
    review_status = str(field_of(issue, "review_status") or "")
    if review_status == "rejected":
        return None
    severity = CXSeverity.parse(field_of(issue, "severity"))
    validated = CXSeverity.parse(field_of(issue, "validated_severity"))
    zone = CXZone.parse(field_of(issue, "zone"))
    if validated is not None and validated != severity:
        zone = CXZone.of_severity(validated) or zone
    elif zone is None:
        zone = CXZone.of_severity(severity)
    confidence = (
        CXConfidence.HIGH
        # спор о формулировке закрыт: сомнение модели больше не сбрасывает штраф
        if review_status in ("approved", "accepted")
        else CXConfidence.parse(field_of(issue, "confidence"))
    )
    return ScoringView(
        zone=zone,
        severity=validated or severity,
        confidence=confidence,
        criterion=str(field_of(issue, "criterion") or ""),
        check_key=issue_check_key(issue),
    )
