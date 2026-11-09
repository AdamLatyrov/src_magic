from typing import Any, List, Optional

from pydantic import Field

from src.pipeline.schemas.core import CoreSchema
from src.pipeline.settings.params.nodes import (
    CXConfidence,
    CXItemStatus,
    CXSeverity,
    CXZone,
)


class DetectedEntity(CoreSchema):
    """Распознанная привязка пути к каталогу: org/app/product/journey."""

    org_id: Optional[str] = None
    org_name: Optional[str] = None
    app_id: Optional[str] = None
    app_name: Optional[str] = None
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    journey_id: Optional[str] = None
    journey_name: Optional[str] = None


class ScreenshotApp(CoreSchema):
    """Приложение одного кадра (выровнено по индексу со `screenshot_keys`)."""

    package_name: Optional[str] = None
    app_version: Optional[str] = None


class FrameTiming(CoreSchema):
    """Тайминги одного кадра (видео; выровнено по индексу со `screenshot_keys`)."""

    dwell_sec: float = 0.0
    load_sec_before: float = 0.0
    scroll_count: int = 0
    scrolled: bool = False
    is_return: bool = False
    return_to_index: int = -1
    source: str = "static"


class TokenPhase(CoreSchema):
    """Затраченные токены одной фазы анализа."""

    phase: str
    label: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    failed_calls: int = 0


class TokenUsage(CoreSchema):
    """Затраченные токены прогона по фазам."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    failed_calls: int = 0
    phases: List[TokenPhase] = []


class SubCriterionOut(CoreSchema):
    """Оценка по одному подкритерию блока."""

    key: str
    name: str
    status: CXItemStatus = CXItemStatus.NA
    note: str = ""
    improvement: str = ""
    zone: Optional[CXZone] = None


class CriterionOut(CoreSchema):
    """Оценка по одному блоку критериальной модели."""

    key: str
    name: str
    score: Optional[float] = None
    rating: Optional[int] = None
    note: str = ""
    hint: str = ""
    items: List[SubCriterionOut] = []
    zone: Optional[CXZone] = None


class ScreenFinding(CoreSchema):
    """Точечная находка на экране с координатами пина (x/y в процентах).

    Код пуст, пока находка не размечена: проход описывает экран, не зная
    справочника, а адрес ему даёт разметка (`pipeline.utils.screen_mapping`). По зоне
    читается та же тяжесть, что и в штрафе: заматченной находке разметка ставит
    зону справочника, незаматченной остаётся её собственная.
    """

    zone: Optional[CXZone] = None
    severity: CXSeverity = CXSeverity.INFO
    criterion: str = ""
    code: str = ""
    text: str = ""
    rec: str = ""
    evidence: str = ""
    x: float = 50.0
    y: float = 50.0


class Friction(CoreSchema):
    """Трение уровня пути. `step` — 0-based индекс кадра-улики, -1 без привязки.

    `criterion` и `code` — адрес, поставленный разметкой находок проверок пути
    (`map_journey_frictions`). У незаматченной находки код пуст, и в `criterion`
    остаётся ключ проверки, которая её нашла: внекаталожное трение обязано
    отличаться и от размеченного, и от находки блока.
    """

    step: int = -1
    zone: Optional[CXZone] = None
    severity: CXSeverity = CXSeverity.MID
    criterion: str = ""
    code: str = ""
    text: str = ""
    rec: str = ""
    evidence: str = ""


class ScreenOut(CoreSchema):
    """Находки по одному экрану (`step` — 0-based индекс кадра)."""

    step: int
    summary: str = ""
    findings: List[ScreenFinding] = []


class IssueEvidence(CoreSchema):
    """Улика находки: экран (1-based) + процитированный видимый текст с него."""

    screen: Optional[int] = None
    quote: str = ""


class IssuePin(CoreSchema):
    """Координата проблемы на экране-улике (x/y в процентах 0..100)."""

    step: int
    x: float = 50.0
    y: float = 50.0


class IssueValidationOut(CoreSchema):
    """Ход валидации находки: кто принял/отклонил и почему."""

    author: str = ""
    role: str = ""
    decision: str = ""
    comment: str = ""
    date: str = ""


class IssueCommentOut(CoreSchema):
    """Сообщение треда обсуждения находки."""

    author: str = ""
    role: str = ""
    body: str = ""
    date: str = ""


class ValidationRequestUserOut(CoreSchema):
    id: str
    name: str


class IssueValidationRequestOut(CoreSchema):
    id: str
    state: str
    requested_for_me: bool = False
    can_review_now: bool = False
    requested_user: Optional[ValidationRequestUserOut] = None
    can_cancel: bool = False
    can_reissue: bool = False
    created_at: str = ""


class Issue(CoreSchema):
    """Консолидированная уникальная проблема пути — выход узла сведения находок.

    Единица штрафа: дубли между блоками схлопнуты, и каждая проблема считается один раз.
    """

    severity: CXSeverity = CXSeverity.MID
    criterion: str = ""
    title: str = ""
    text: str = ""
    rec: str = ""
    steps: List[int] = []
    pins: List[IssuePin] = []
    evidence: List[IssueEvidence] = []
    confidence: Optional[CXConfidence] = None
    kind: Optional[str] = None
    defect_code: Optional[str] = None
    defect_ref: Optional[str] = None
    defect_name: Optional[str] = None
    check_key: Optional[str] = None
    zone: Optional[CXZone] = None
    id: Optional[str] = None
    verified_verdict: Optional[str] = None
    source: str = "agent"
    author: Optional[str] = None
    review_status: str = "pending"
    validations: List[IssueValidationOut] = []
    my_turn: bool = False
    comments: List[IssueCommentOut] = []
    comment_count: int = 0
    can_validate: bool = False
    can_request_validation: bool = False
    can_comment: bool = False
    can_finalize: bool = False
    is_closed: bool = False
    finalized_by: Optional[str] = None
    finalized_at: Optional[str] = None
    finalize_comment: Optional[str] = None
    validated_severity: Optional[str] = None
    validation_request: Optional[IssueValidationRequestOut] = None


class AnalyzeResult(CoreSchema):
    """Итог прогона: четыре списка + балл + обёртка деградации."""

    run_id: str
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    model: Optional[str] = None
    screenshot_keys: List[str]
    source_screenshot_keys: List[str] = []
    screenshot_urls: List[str] = []
    frame_mask_meta: List[Optional[dict]] = []
    score: Optional[float] = None
    base_score: Optional[float] = None
    friction_penalty: Optional[float] = None
    rating: Optional[int] = None
    critical_count: int = 0
    stop_flag: bool = False
    criteria: List[CriterionOut] = []
    screens: List[ScreenOut] = []
    frictions: List[Friction] = []
    issues: List[Issue] = Field(default_factory=list)
    applied_profiles: List[str] = []
    summary: Optional[str] = None
    feedback: Optional[str] = None
    report: str = ""
    analysis_status: str = "complete"
    is_degraded: bool = False
    degradation_reasons: List[str] = Field(default_factory=list)
    failed_blocks: List[str] = Field(default_factory=list)
    fallbacks_used: List[str] = Field(default_factory=list)
    detected: DetectedEntity = Field(default_factory=DetectedEntity)
    package_name: Optional[str] = None
    app_version: Optional[str] = None
    screenshot_apps: List[ScreenshotApp] = []
    timeline: List[FrameTiming] = []
    token_usage: Optional[TokenUsage] = None
    trace: Optional[dict] = None
    pipeline: dict[str, Any] = Field(default_factory=dict)
    eval_model_id: Optional[str] = None
    eval_model_version: Optional[str] = None
    is_journey: bool = True
    note: Optional[str] = None
