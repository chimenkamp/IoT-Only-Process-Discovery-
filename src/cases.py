from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CaseRun:
    """One contiguous run of the same case identifier."""

    case_id: int
    occurrence: int
    start_row: int
    end_row: int

    @property
    def n_samples(self) -> int:
        return self.end_row - self.start_row


def contiguous_case_runs(
    case_ids: np.ndarray,
    min_samples: int = 1,
    include_zero: bool = False,
) -> list[CaseRun]:
    """Return contiguous case-id runs without merging repeated IDs."""
    if case_ids.ndim != 1:
        raise ValueError("case_ids must be one-dimensional")
    if case_ids.size == 0:
        return []

    starts = np.r_[0, np.where(case_ids[1:] != case_ids[:-1])[0] + 1]
    ends = np.r_[starts[1:], case_ids.size]
    occurrences: dict[int, int] = {}
    runs: list[CaseRun] = []

    for start, end in zip(starts, ends):
        case_id = int(case_ids[start])
        occurrences[case_id] = occurrences.get(case_id, 0) + 1
        if case_id == 0 and not include_zero:
            continue
        if end - start < min_samples:
            continue
        runs.append(CaseRun(
            case_id=case_id,
            occurrence=occurrences[case_id],
            start_row=int(start),
            end_row=int(end),
        ))

    return runs


def select_contiguous_case_runs(
    df: pd.DataFrame,
    case_id_col: str,
    max_cases: int | None,
    min_samples: int = 1,
    include_zero: bool = False,
) -> list[CaseRun]:
    """Select contiguous case-id runs from a table."""
    if case_id_col not in df.columns:
        raise ValueError(f"{case_id_col} is missing from the dataset")
    runs = contiguous_case_runs(
        df[case_id_col].to_numpy(),
        min_samples=min_samples,
        include_zero=include_zero,
    )
    if max_cases is None:
        return runs
    return runs[:max_cases]


def stacked_case_boundaries(runs: list[CaseRun]) -> list[int]:
    """Return boundaries after vertically stacking selected case-run arrays."""
    boundaries = [0]
    for run in runs:
        boundaries.append(boundaries[-1] + run.n_samples)
    return boundaries
