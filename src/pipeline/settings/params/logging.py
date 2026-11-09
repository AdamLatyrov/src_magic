from enum import Enum


class LogLabel(Enum):
    SCAN_JOURNEY_FRAME = "scan_journey_frame"
    SUMMARIZE_JOURNEY = "summarize_journey"
    CHECK_CX_CRITERIA_FRAMES = "criteria_model_frames"
    CHECK_JOURNEY_FRICTIONS = "journey_frictions"
    MAP_JOURNEY_FRICTIONS = "map_journey_frictions"
    CHECK_AUTOFILL = "autofill"
    CHECK_SCREEN = "screen"
    SUMMARIZE_SCREEN = "summarize_screen"
    CONSOLIDATE_FINDINGS = "consolidate_findings"
    DEDUP_FINDINGS = "dedup_findings"
    RELEASE_FRAMES = "release_frames"

    def __str__(self):
        return self.value
