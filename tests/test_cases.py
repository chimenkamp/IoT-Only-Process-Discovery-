from __future__ import annotations

import numpy as np
import pandas as pd

from src.cases import contiguous_case_runs, select_contiguous_case_runs


def test_contiguous_case_runs_do_not_merge_repeated_ids() -> None:
    case_ids = np.array([1, 1, 2, 2, 0, 1, 1, 1, 2, 2])

    runs = contiguous_case_runs(case_ids, min_samples=2)

    assert [(r.case_id, r.occurrence, r.start_row, r.end_row) for r in runs] == [
        (1, 1, 0, 2),
        (2, 1, 2, 4),
        (1, 2, 5, 8),
        (2, 2, 8, 10),
    ]


def test_select_contiguous_case_runs_filters_short_runs() -> None:
    df = pd.DataFrame({"case": [1, 1, 0, 2, 2, 2]})

    runs = select_contiguous_case_runs(
        df,
        case_id_col="case",
        max_cases=2,
        min_samples=3,
    )

    assert len(runs) == 1
    assert runs[0].case_id == 2
    assert runs[0].start_row == 3
    assert runs[0].end_row == 6
