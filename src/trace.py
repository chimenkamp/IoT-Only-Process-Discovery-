from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentEvent:
    """Event induced by one maximal block of equal segment labels."""

    activity: int
    start: int
    end: int


def build_event_log(
    boundaries: list[int],
    segment_labels: list[int],
    case_boundaries: list[int] | None = None,
) -> list[list[SegmentEvent]]:
    """Build a case-local event log from segment labels."""
    if len(segment_labels) != len(boundaries) - 1:
        raise ValueError("segment_labels must have length len(boundaries) - 1")
    cases = _case_boundaries_or_single_case(boundaries, case_boundaries)
    _validate_case_boundaries(boundaries, cases)

    event_log: list[list[SegmentEvent]] = []
    for case_start, case_end in zip(cases, cases[1:]):
        events: list[SegmentEvent] = []
        block_label: int | None = None
        block_start: int | None = None
        block_end: int | None = None

        for idx, label in enumerate(segment_labels):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            if end <= case_start or start >= case_end:
                continue

            clipped_start = max(start, case_start)
            clipped_end = min(end, case_end)
            if label == block_label:
                block_end = clipped_end
                continue

            if block_label is not None and block_start is not None:
                events.append(SegmentEvent(
                    block_label,
                    block_start,
                    block_end or block_start,
                ))
            block_label = label
            block_start = clipped_start
            block_end = clipped_end

        if block_label is not None and block_start is not None:
            events.append(SegmentEvent(
                block_label,
                block_start,
                block_end or block_start,
            ))
        event_log.append(events)

    return event_log


def activity_projections(
    event_log: list[list[SegmentEvent]],
) -> list[list[int]]:
    """Drop timestamps and keep only activity labels."""
    return [[event.activity for event in trace] for trace in event_log]


def _case_boundaries_or_single_case(
    boundaries: list[int],
    case_boundaries: list[int] | None,
) -> list[int]:
    if case_boundaries is None:
        return [boundaries[0], boundaries[-1]]
    return case_boundaries


def _validate_case_boundaries(
    boundaries: list[int],
    case_boundaries: list[int],
) -> None:
    if not case_boundaries:
        raise ValueError("case_boundaries must not be empty")
    if (
        case_boundaries[0] != boundaries[0]
        or case_boundaries[-1] != boundaries[-1]
    ):
        raise ValueError(
            "case_boundaries must share first and last segment boundaries"
        )
    if case_boundaries != sorted(set(case_boundaries)):
        raise ValueError("case_boundaries must be strictly increasing")
    missing = [
        boundary for boundary in case_boundaries
        if boundary not in boundaries
    ]
    if missing:
        raise ValueError(
            "case boundaries must be present in segment boundaries: "
            f"{missing}"
        )
