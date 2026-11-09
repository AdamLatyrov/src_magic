"""Orchestration tests with stub dependencies; NOT real-provider/live LangGraph tests."""

import asyncio
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from src.pipeline.schemas.api import PipelineRequest, PreviousFinding, PipelineRunStatus
from src.pipeline.schemas.nodes import CXFrame
from src.pipeline.schemas.result import AnalyzeResult, Issue
from src.pipeline.status import StageName


def stub(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    return module


@pytest.fixture
def facade_module(monkeypatch):
    # Module-level external dependencies unavailable in the offline sandbox.
    import src.pipeline.scoring  # keep real scoring code
    import src.pipeline.schemas.state.agent
    @dataclass
    class DedupResult:
        issues: list = field(default_factory=list)
        degraded: bool = False
        restored: int = 0
        merged: int = 0
        added: int = 0
        model_calls: int = 0
    async def dedup(previous, current, **kwargs):
        return DedupResult(issues=[*previous, *current], restored=len(previous), added=len(current))
    def same_reference(a, b):
        return bool(a.defect_ref and b.defect_ref and a.defect_ref == b.defect_ref) or (
            bool(a.defect_code and b.defect_code and a.check_key and b.check_key)
            and (a.defect_code, a.check_key) == (b.defect_code, b.check_key))
    stubs = {
        "langgraph": stub("langgraph"),
        "langgraph.graph": stub("langgraph.graph"),
        "langgraph.graph.state": stub("langgraph.graph.state", CompiledStateGraph=type("CompiledStateGraph", (), {})),
        "src.pipeline.aef.patterns.gigachat_provider": stub(
            "src.pipeline.aef.patterns.gigachat_provider", GigaChatProvider=type("GigaChatProvider", (), {})),
        "src.pipeline.agent.graph": stub("src.pipeline.agent.graph", build_graph=lambda cfg: None),
        "src.pipeline.dedup": stub("src.pipeline.dedup", DedupResult=DedupResult, dedup=dedup),
        "src.pipeline.dedup.reconcile": stub("src.pipeline.dedup.reconcile", same_reference=same_reference),
        "src.pipeline.dto.agent.result": stub("src.pipeline.dto.agent.result", AgentResult=object),
        "src.pipeline.settings.api.graph": stub("src.pipeline.settings.api.graph", GraphSettings=object),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "main.py"
    spec = importlib.util.spec_from_file_location("pipeline_facade_test_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_pipeline(module, *, issues=None, degraded=False, is_journey=True,
                 gate=None, fail=False):
    pipeline = object.__new__(module.AnalysisPipeline)
    pipeline._run_states = {}
    pipeline._run_trackers = {}
    pipeline._tasks = {}
    pipeline._provider = object()
    pipeline._settings = types.SimpleNamespace(dedup_findings=object())
    pipeline._catalog = object()

    async def run_graph(frames, material, tracker, *, run_id, screenshot_keys):
        tracker.begin(StageName.SCAN_JOURNEY_FRAME)
        if gate:
            await gate.wait()
        if fail:
            raise RuntimeError("internal exception contains a secret")
        tracker.finish(StageName.SCAN_JOURNEY_FRAME, module.StageState.DONE)
        return AnalyzeResult(
            run_id=run_id, screenshot_keys=screenshot_keys,
            issues=issues or [], criteria=[],
            is_degraded=degraded, is_journey=is_journey,
            degradation_reasons=["model_missing"] if degraded else [],
            analysis_status="rejected" if not is_journey else ("partial" if degraded else "complete"),
        )
    pipeline.run_graph = run_graph
    return pipeline


def req(run_id="run-1", previous=None):
    return PipelineRequest(
        run_id=run_id, product_name="Bank A", product_type="mobile",
        frames=[CXFrame(step=1, data_url="data:image/png;base64,AA==")],
        screenshot_keys=["one"], previous=previous or [],
    )


@pytest.mark.asyncio
async def test_start_reserves_run_id_and_reports_final_result(facade_module):
    gate = asyncio.Event()
    pipeline = new_pipeline(facade_module, issues=[Issue(defect_ref="a")], gate=gate)
    events = []
    def callback(snapshot):
        # Mutations by a subscriber must not corrupt stored snapshots.
        events.append(snapshot.model_dump(mode="json"))
        snapshot.findings_count = 1000
    request = req()
    task = pipeline.start(request, on_event=callback)
    assert pipeline.get_state(request.run_id).status == "queued"
    with pytest.raises(facade_module.PipelineRefused):
        pipeline.start(request)
    await asyncio.sleep(0)
    assert pipeline.get_state(request.run_id).status == "running"
    gate.set()
    result = await task
    assert result.run_id == request.run_id
    assert result.product_name == "Bank A" and result.product_type == "mobile"
    assert result.analysis.product_name == "Bank A"
    assert result.state.status == "complete" and result.state.findings_count == 1
    assert len(result.findings) == len(result.analysis.issues) == 1
    assert events[-1]["status"] == "complete" and events[-1]["findings_count"] == 1
    assert pipeline.get_state(request.run_id).findings_count == 1


@pytest.mark.asyncio
async def test_failure_captured_without_exposing_exception(facade_module):
    pipeline = new_pipeline(facade_module, fail=True)
    events = []
    with pytest.raises(RuntimeError, match="internal exception"):
        await pipeline.start(req(), on_event=lambda s: events.append(s.model_dump(mode="json")))
    state = pipeline.get_state("run-1")
    assert state.status == "failed" and state.error_code == "pipeline_execution_error"
    assert "secret" not in state.model_dump_json()
    assert events[-1]["status"] == "failed"
    assert any(s.state == "failed" for s in state.stages)


@pytest.mark.asyncio
async def test_rejected_previous_not_resurfaced_by_reference(facade_module):
    issue = Issue(defect_ref="defect-1", check_key="abc", title="Old")
    pipeline = new_pipeline(facade_module, issues=[Issue(defect_ref="defect-1")])
    result = await pipeline.start(req(previous=[PreviousFinding(issue=issue, expert_status="rejected")]))
    assert result.findings == result.analysis.issues == []
    assert result.state.status == "complete"


@pytest.mark.asyncio
async def test_partial_flag_and_non_journey_does_not_restore_history(facade_module):
    previous = [PreviousFinding(issue=Issue(defect_ref="old"), expert_status="approved")]
    partial = new_pipeline(facade_module, degraded=True)
    result = await partial.start(req(previous=previous))
    assert result.state.status == "partial"
    assert result.analysis.analysis_status == "partial"
    non_journey = new_pipeline(facade_module, is_journey=False)
    result2 = await non_journey.start(req("run-2", previous=previous))
    assert result2.analysis.analysis_status == "rejected"
    assert result2.findings == [] and result2.analysis.score is None


@pytest.mark.asyncio
async def test_cancel_before_task_started_and_callback_errors_ignored(facade_module):
    pipeline = new_pipeline(facade_module)
    def bad_callback(snapshot):
        raise RuntimeError("subscriber is down")
    task = pipeline.start(req(), on_event=bad_callback)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert pipeline.get_state("run-1").status == "cancelled"


def test_provider_api_shape_check(facade_module):
    with pytest.raises(facade_module.PipelineRefused, match="upload_frame"):
        facade_module.AnalysisPipeline._validated_provider(types.SimpleNamespace())
    complete = types.SimpleNamespace(**{name: lambda *a, **k: None for name in (
        "chat", "chat_structured_output", "chat_with_structure_output_with_frames",
        "upload_frame", "delete_frames")})
    assert facade_module.AnalysisPipeline._validated_provider(complete) is complete

@pytest.mark.asyncio
async def test_reconciled_issues_reset_criterion_display_and_score(facade_module):
    from src.pipeline.schemas.result import CriterionOut, SubCriterionOut
    original = Issue(criterion="navigation", check_key="check-1",
                     zone="red", rec="Устаревший совет", defect_ref="ref-x")
    pipeline = new_pipeline(facade_module, issues=[original])
    async def run_graph(frames, material, tracker, *, run_id, screenshot_keys):
        return AnalyzeResult(
            run_id=run_id, screenshot_keys=screenshot_keys,
            issues=[original], criteria=[CriterionOut(key="navigation", name="Навигация", items=[
                SubCriterionOut(key="check-1", name="Переход", status="fail",
                                zone="red", improvement="Устаревший совет"),
            ], zone="red")],
        )
    pipeline.run_graph = run_graph
    result = await pipeline.start(req(previous=[PreviousFinding(
        issue=original, expert_status="rejected",
    )]))
    block = result.analysis.criteria[0]
    assert result.findings == []
    assert block.items[0].status == "ok"
    assert block.items[0].improvement == ""
    assert block.items[0].zone is None and block.zone is None
    assert result.analysis.score is not None
    assert result.analysis.rating is not None


@pytest.mark.asyncio
async def test_degraded_stage_produces_partial_even_when_analysis_did_not_mark_it(facade_module):
    pipeline = new_pipeline(facade_module)
    async def run_graph(frames, material, tracker, *, run_id, screenshot_keys):
        tracker.finish(StageName.SCREEN, facade_module.StageState.DEGRADED)
        return AnalyzeResult(run_id=run_id, screenshot_keys=screenshot_keys)
    pipeline.run_graph = run_graph
    result = await pipeline.start(req())
    assert result.state.status == "partial"
    assert result.analysis.is_degraded
    assert "stage_status_degraded" in result.analysis.degradation_reasons

@pytest.mark.asyncio
async def test_async_callback_rejected_before_scheduling(facade_module):
    pipeline = new_pipeline(facade_module)
    async def async_callback(snapshot):
        return None
    with pytest.raises(TypeError, match="синхронным"):
        pipeline.start(req(), on_event=async_callback)
    with pytest.raises(KeyError):
        pipeline.get_state("run-1")
