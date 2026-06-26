from __future__ import annotations

import numpy as np
import ruptures


def detect_changepoints(
    data: np.ndarray,
    penalty: float,
    min_size: int,
    model: str = "l2",
) -> list[int]:
    """Detect change points in a multivariate sensor log using PELT.

    Parameters
    ----------
    data : np.ndarray
        Sensor log of shape ``(N, n_vars)`` where *N* is the number of
        time points and *n_vars* is the number of sensor variables.
    penalty : float
        PELT penalty parameter.  Lower values produce more change
        points (more over-segmentation).
    min_size : int
        Minimum number of samples between two change points.
    model : str
        Cost model for PELT (e.g. ``"l2"``, ``"l1"``, ``"rbf"``).
        ``"l2"`` is O(N) per segment and recommended for large data.
        ``"rbf"`` is O(N²) and much slower.

    Returns
    -------
    list[int]
        Sorted boundary indices ``[0, cp1, cp2, ..., N]``.  The list
        always starts with ``0`` and ends with ``N``.
    """
    n = data.shape[0]

    algo = ruptures.Pelt(model=model, min_size=min_size).fit(data)
    bkps = algo.predict(pen=penalty)

    boundaries = sorted({0} | set(bkps))
    if boundaries[-1] != n:
        boundaries.append(n)
    return boundaries


def detect_changepoints_by_case(
    data: np.ndarray,
    case_boundaries: list[int],
    penalty: float,
    min_size: int,
    model: str = "l2",
) -> list[int]:
    """Detect change points independently inside each case interval."""
    _validate_case_boundaries_for_detection(data, case_boundaries)
    boundaries: set[int] = set(case_boundaries)

    for case_start, case_end in zip(case_boundaries, case_boundaries[1:]):
        case_length = case_end - case_start
        if case_length <= min_size:
            continue
        local = detect_changepoints(
            data[case_start:case_end],
            penalty,
            min_size,
            model=model,
        )
        boundaries.update(case_start + boundary for boundary in local)

    return sorted(boundaries)


def add_required_boundaries(
    boundaries: list[int],
    required_boundaries: list[int],
) -> list[int]:
    """Return boundaries with mandatory cut points inserted.

    This is useful when a continuous sensor stream contains natural
    cases, such as production cycles.  Change detection can still find
    data-driven cuts inside each case, while case boundaries are kept as
    hard trace boundaries for process discovery.
    """
    if not boundaries:
        return sorted(set(required_boundaries))

    start = boundaries[0]
    end = boundaries[-1]
    clipped_required = {
        boundary
        for boundary in required_boundaries
        if start <= boundary <= end
    }
    return sorted(set(boundaries) | clipped_required)


def merge_short_segments(
    boundaries: list[int],
    min_size: int,
    required_boundaries: list[int] | None = None,
) -> list[int]:
    """Merge segments shorter than ``min_size`` where boundaries are movable."""
    if min_size <= 1 or len(boundaries) < 3:
        return sorted(set(boundaries))

    merged = sorted(set(boundaries))
    required = set(required_boundaries or ())
    required.update({merged[0], merged[-1]})

    changed = True
    while changed:
        changed = False
        for idx in range(len(merged) - 1):
            left = merged[idx]
            right = merged[idx + 1]
            if right - left >= min_size:
                continue

            removable = _short_segment_boundary_to_remove(
                merged,
                idx,
                required,
            )
            if removable is None:
                continue
            merged.remove(removable)
            changed = True
            break

    return merged


def _short_segment_boundary_to_remove(
    boundaries: list[int],
    segment_idx: int,
    required: set[int],
) -> int | None:
    left = boundaries[segment_idx]
    right = boundaries[segment_idx + 1]
    can_remove_left = segment_idx > 0 and left not in required
    can_remove_right = (
        segment_idx + 1 < len(boundaries) - 1
        and right not in required
    )

    if can_remove_left and can_remove_right:
        left_neighbor = left - boundaries[segment_idx - 1]
        right_neighbor = boundaries[segment_idx + 2] - right
        return left if left_neighbor <= right_neighbor else right
    if can_remove_left:
        return left
    if can_remove_right:
        return right
    return None


def _validate_case_boundaries_for_detection(
    data: np.ndarray,
    case_boundaries: list[int],
) -> None:
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    if len(case_boundaries) < 2:
        raise ValueError("case_boundaries must contain at least start and end")
    if case_boundaries != sorted(set(case_boundaries)):
        raise ValueError("case_boundaries must be strictly increasing")
    if case_boundaries[0] != 0 or case_boundaries[-1] != data.shape[0]:
        raise ValueError("case_boundaries must span the data")
