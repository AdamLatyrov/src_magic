"""Contract checks that do not require the graph or the model provider."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.pipeline.schemas.api import (
    PipelineRequest, PipelineRunStatus, PipelineStateSnapshot, PreviousFinding,
)
from src.pipeline.schemas.nodes import CXFrame
from src.pipeline.schemas.result import Issue
from src.pipeline.status import RunTracker, StageName, StageState


def request(**overrides):
    payload = dict(
        product_name="Bank A", product_type="mobile",
        frames=[CXFrame(step=1, data_url="data:image/png;base64,AA=="),
                CXFrame(step=2, data_url="data:image/png;base64,AA==")],
        screenshot_keys=["key1", "key2"],
        previous=[PreviousFinding(issue=Issue(defect_ref="ref-1"), expert_status="pending")],
    )
    payload.update(overrides)
    return PipelineRequest(**payload)


def test_public_request_and_json_snapshot():
    req = request()
    assert req.run_id and req.product_name == "Bank A" and req.product_type == "mobile"
    assert [f.step for f in req.frames] == [1, 2]
    assert req.screenshot_keys == ["key1", "key2"]
    assert req.previous[0].expert_status == "pending"
    snap = PipelineStateSnapshot(run_id=req.run_id, status=PipelineRunStatus.QUEUED)
    assert snap.model_dump(mode="json")["status"] == "queued"


@pytest.mark.parametrize("extra", [
    {"frames": []},
    {"frames": [CXFrame(step=2, data_url="x"), CXFrame(step=1, data_url="x")]},
    {"frames": [CXFrame(step=1, data_url="x"), CXFrame(step=1, data_url="x")]},
    {"frames": [CXFrame(step=0, data_url="x")]},
    {"screenshot_keys": ["only-one"]},
    {"product_name": "   "}, {"product_type": ""}, {"run_id": "   "},
    {"previous": [dict(issue=Issue(), expert_status="unknown")]},
])
def test_reject_invalid_request(extra):
    with pytest.raises(ValidationError):
        request(**extra)


def test_unnamed_keys_optional_and_step_numbers_preserved():
    req = request(screenshot_keys=[], frames=[CXFrame(step=3, data_url="x")])
    assert req.frames[0].step == 3
    assert req.screenshot_keys == []


def test_failed_tracker_marks_running_stage_failed():
    tracker = RunTracker()
    tracker.begin_run()
    tracker.begin(StageName.SCAN_JOURNEY_FRAME)
    tracker.finish_run(failed=True)
    assert tracker.state_of(StageName.SCAN_JOURNEY_FRAME) == StageState.FAILED
    assert tracker.finished_at is not None
    assert tracker.to_dict()["stages"][0]["state"] == "failed"


def test_snapshot_is_json_and_contains_no_frames():
    tracker = RunTracker()
    tracker.begin_run()
    tracker.begin(StageName.CRITERIA_MODEL_FRAMES, note="проверка блока")
    tracker.count(StageName.CRITERIA_MODEL_FRAMES, checked=3)
    tracker.finish_run()
    snapshot = PipelineStateSnapshot(
        run_id="run-1", status=PipelineRunStatus.COMPLETE,
        stages=[dict(**row) for row in tracker.to_dict()["stages"]],
    )
    payload = snapshot.model_dump_json()
    assert '"checked":3' in payload
    assert "data_url" not in payload
    assert '"frames":' not in payload
