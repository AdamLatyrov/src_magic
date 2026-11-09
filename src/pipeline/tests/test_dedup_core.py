"""Real reconciliation algorithm with only external LLM/SDK imports stubbed."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from src.pipeline.schemas.result import Issue


@pytest.fixture
def reconcile_module(monkeypatch):
    # Only the external model-facing modules are replaced. Reconciliation
    # code itself comes from src/pipeline/dedup/reconcile.py in this tree.
    for name, names in (
        ("src.pipeline.aef.patterns.gigachat_provider", {"GigaChatProvider": object}),
        ("src.pipeline.llm.gateway", {"ask": lambda *a, **kw: None}),
        ("src.pipeline.settings.api.graph", {"DedupFindingsSettings": object}),
    ):
        mod = types.ModuleType(name)
        mod.__dict__.update(names)
        monkeypatch.setitem(sys.modules, name, mod)
    path = Path(__file__).resolve().parents[1] / "dedup" / "reconcile.py"
    spec = importlib.util.spec_from_file_location("reconcile_actual_test_target", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_wording_pending_uses_current_and_approved_uses_previous(reconcile_module):
    previous = Issue(id="id-old", title="OLD", text="OLD", rec="OLD",
                     defect_ref="ref", review_status="pending")
    current = Issue(id="id-new", title="NEW", text="NEW", rec="NEW",
                    defect_ref="ref", steps=[3], review_status="pending")
    pending = reconcile_module._wording_of(previous, current)
    approved = reconcile_module._wording_of(
        previous.model_copy(update={"review_status": "approved"}), current)
    assert pending.title == "NEW" and pending.id == "id-old"
    assert approved.title == "OLD" and approved.steps == [3]


@pytest.mark.asyncio
async def test_exact_match_does_not_invoke_model(reconcile_module):
    prev = [Issue(id="old", defect_ref="same", title="old", review_status="approved")]
    curr = [Issue(defect_ref="same", title="new", steps=[7])]
    result = await reconcile_module.dedup(prev, curr, settings=object(), provider=object())
    assert len(result.issues) == 1
    assert result.issues[0].title == "old"
    assert result.issues[0].steps == [7]
    assert result.merged == 1 and result.model_calls == 0


@pytest.mark.asyncio
async def test_rejected_legacy_verdict_is_not_restored(reconcile_module):
    prev = [Issue(defect_ref="a", verified_verdict="not_ok", title="removed")]
    result = await reconcile_module.dedup(prev, [], settings=object(), provider=object())
    assert result.issues == []
