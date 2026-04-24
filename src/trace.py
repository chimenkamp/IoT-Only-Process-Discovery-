from __future__ import annotations

import numpy as np

from src.synthesis import IntervalRule, evaluate_rule


def build_traces(
    data: np.ndarray,
    rules: list[IntervalRule],
) -> list[list[int]]:
    """Build an event log from a sensor log and synthesised rules.

    Each time point is mapped to the unique rule that covers it.
    Consecutive time points covered by the same rule are merged into
    a single event.  For a single sensor log the result is a list
    containing one trace.

    Parameters
    ----------
    data : np.ndarray
        Sensor log of shape ``(N, n_vars)``.
    rules : list[IntervalRule]
        Synthesised interval rules (one per equivalence class).

    Returns
    -------
    list[list[int]]
        Event log — a list containing one trace.  Each trace is a
        sequence of rule (equivalence-class) identifiers.
    """
    n = data.shape[0]
    trace: list[int] = []
    prev: int = -1
    for t in range(n):
        active = next(r.class_id for r in rules if evaluate_rule(r, data[t]))
        if active != prev:
            trace.append(active)
            prev = active
    return [trace]
