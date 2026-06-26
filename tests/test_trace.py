from __future__ import annotations

import pytest

from src.trace import SegmentEvent, activity_projections, build_event_log


def test_event_log_collapses_case_local_repeats() -> None:
    event_log = build_event_log(
        boundaries=[0, 5, 10, 15, 20],
        segment_labels=[0, 0, 1, 2],
        case_boundaries=[0, 10, 20],
    )

    assert event_log == [
        [SegmentEvent(activity=0, start=0, end=10)],
        [
            SegmentEvent(activity=1, start=10, end=15),
            SegmentEvent(activity=2, start=15, end=20),
        ],
    ]
    assert activity_projections(event_log) == [[0], [1, 2]]


def test_event_log_rejects_case_boundaries_inside_segments() -> None:
    with pytest.raises(ValueError):
        build_event_log(
            boundaries=[0, 10, 20],
            segment_labels=[0, 1],
            case_boundaries=[0, 5, 20],
        )
