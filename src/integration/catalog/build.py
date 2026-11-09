"""Сборка справочника из плоской выгрузки — вся или ничего.

Порядок проверок внутри сборки нарочный: сначала версия (без неё все прочие
причины ложные), затем строки её версии, затем ссылки и адреса. Основание —
цена молчания: блока, до которого справочник не доехал, просто нет в перечне, и
прогон печатает рейтинг по оставшимся.
"""

from src.integration.catalog.columns import code_of, ordered
from src.integration.catalog.failure import CatalogValidationError, listing
from src.integration.catalog.rules import (
    address_of,
    declared_counts,
    known_zones,
    shared_codes,
    sort_orders,
    unique_keys,
)
from src.integration.catalog.version import of_version, pick_version
from src.pipeline.schemas.criteria_export import CriteriaExport
from src.pipeline.schemas.nodes import (
    CXBlock,
    CXCriteria,
    CXCriteriaDefect,
    CXCriteriaModel,
    CXModelInfo,
)
from src.pipeline.settings.params.nodes import CXZone


def build_criteria_model(export: CriteriaExport) -> CXCriteriaModel:
    """Нормальная форма справочника из плоской выгрузки — вся или ничего.

    Справочник, из которого потерялся блок, отдал бы его вердикты чужому блоку,
    а потерянный дефект снял бы находку молча; молчание блок отвечал бы
    «чисто», и прогон получил бы баллы за то, чего не проверяли.
    """
    errors: list[str] = []
    version = pick_version(export.model, errors)
    if version is None:
        raise CatalogValidationError(errors)

    blocks = ordered(of_version(export.blocks, version.id))
    checks = ordered(of_version(export.checks, version.id))
    defects = ordered(of_version(export.defects, version.id))
    declared_counts(
        export,
        {"blocks": len(blocks), "checks": len(checks), "defects": len(defects)},
        errors,
    )
    for rows, what in (
        (blocks, "блоки"),
        (checks, "проверки"),
        (defects, "дефекты"),
    ):
        sort_orders(rows, what, errors)
    known_zones(checks, "проверки", errors)
    known_zones(defects, "дефекты", errors)
    unique_keys(blocks, "блок", errors)
    unique_keys(checks, "проверка", errors)

    block_ids = {row.id for row in blocks}
    checks_by_id = {row.id: row for row in checks}
    for check in checks:
        if check.block_id not in block_ids:
            errors.append(
                f"проверка {check.key}: блока {check.block_id} "
                "в выгрузке нет"
            )

    defects_by_check: dict[str, list[CXCriteriaDefect]] = {}
    taken: set[str] = set()
    codes: dict[str, set[str]] = {}
    zoneless: list[str] = []
    for defect in defects:
        check = checks_by_id.get(defect.check_id)
        if check is None:
            errors.append(
                f"дефект {defect.display_code}: проверки "
                f"{defect.check_id} в выгрузке нет"
            )
            continue
        address = address_of(defect, check, errors)
        if address in taken:
            errors.append(f"дефект {address}: адрес занят другим дефектом")
        taken.add(address)
        codes.setdefault(defect.display_code, set()).add(address)
        if not code_of(defect.zone):
            zoneless.append(address)
            continue
        zone = CXZone.parse(defect.zone)
        if zone is None:
            continue
        defects_by_check.setdefault(check.id, []).append(
            CXCriteriaDefect(
                ref=address,
                code=defect.display_code,
                defect=defect.name,
                symptom=defect.manifestation,
                zone=zone,
            )
        )
    if zoneless:
        errors.append(
            f"дефекты без зоны: {listing(zoneless)} — такой дефект нельзя "
            "ни оштрафовать, ни назвать в отчёте"
        )
    shared_codes(codes)

    criteria_by_block: dict[str, list[CXCriteria]] = {}
    for check in checks:
        criteria_by_block.setdefault(check.block_id, []).append(
            CXCriteria(
                key=check.key,
                text=check.name,
                essence=check.description,
                defects=defects_by_check.get(check.id, []),
            )
        )
    built: list[CXBlock] = []
    for block in blocks:
        criteria = criteria_by_block.get(block.id, [])
        if not criteria:
            errors.append(f"блока {block.key} проверок в выгрузке нет")
        built.append(
            CXBlock(
                key=block.key,
                name=block.name,
                essence=block.description,
                criteria=criteria,
            )
        )
    if not built:
        errors.append("в справочнике нет ни одного блока")
    elif not any(block.criteria for block in built):
        errors.append("в справочнике нет ни одной проверки")
    if not any(defects_by_check.values()):
        errors.append("в справочнике нет ни одного дефекта")
    if errors:
        raise CatalogValidationError(errors)

    return CXCriteriaModel(
        info=CXModelInfo(
            id=version.id,
            key=version.key,
            version=version.version,
            title=version.title,
            status=code_of(version.status),
            created_at=version.created_at,
            published_at=version.published_at,
            source_s3_key=version.source_s3_key,
            source_sha256=version.source_sha256,
        ),
        blocks=built,
    )
