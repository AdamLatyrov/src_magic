"""Real scoring implementation; no model connection required."""
from src.pipeline.scoring import score_run
from src.pipeline.scoring.finding import scoring_view
from src.pipeline.schemas.result import CriterionOut, SubCriterionOut, Issue


def test_score_changes_when_final_issue_set_changes():
    def criteria():
        return [CriterionOut(key="navigation", name="Навигация", items=[
            SubCriterionOut(key="check-1", name="Переходы", status="ok")
        ])]
    issue = Issue(criterion="navigation", check_key="check-1",
                  defect_ref="navigation/check-1/ERROR", zone="red")
    with_problem = criteria()
    marked = score_run(with_problem, [issue])
    without_problem = criteria()
    clean = score_run(without_problem, [])
    assert with_problem[0].items[0].status == "fail"
    assert without_problem[0].items[0].status == "ok"
    assert marked.score < clean.score
    assert marked.critical_count == 1
    assert clean.critical_count == 0


def test_approved_and_pending_have_different_scoring_confidence():
    pending = Issue(criterion="navigation", check_key="check-1", zone="yellow",
                    confidence="low", review_status="pending")
    approved = pending.model_copy(update={"review_status": "approved"})
    assert scoring_view(pending).confidence != scoring_view(approved).confidence
