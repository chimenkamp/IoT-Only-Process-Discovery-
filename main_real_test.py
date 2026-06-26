"""Future Factory entry point for the unsupervised discovery pipeline."""

from __future__ import annotations

import pickle
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.cases import CaseRun, select_contiguous_case_runs
from src.discovery import activity_name
from src.evaluation import (
    TuningCandidate,
    candidate_config,
    metrics_for_discovery,
    metrics_for_validation,
    split_cases,
    stack_case_arrays,
    tune_pipeline,
)
from src.pipeline import validate_sensor_log, run_pipeline


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PKL_PATH = Path("data/Future_Factory/combined_[1-6].pkl")
N_CASE_RUNS: int | None = None
MIN_CASE_SAMPLES = 500
TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2

TUNING_CANDIDATES = [
    TuningCandidate(
        penalty=6.0,
        min_segment_size=20,
        n_activity_clusters=10,
        signature_depth=1,
        rule_margin=0.25,
        max_profile_features=40,
        inductive_miner_noise_threshold=0.0,
        variant_coverage_threshold=0.0,
    ),
    TuningCandidate(
        penalty=8.0,
        min_segment_size=20,
        n_activity_clusters=12,
        signature_depth=1,
        rule_margin=0.35,
        max_profile_features=40,
        inductive_miner_noise_threshold=0.0,
        variant_coverage_threshold=0.0,
    ),
    TuningCandidate(
        penalty=10.0,
        min_segment_size=25,
        n_activity_clusters=14,
        signature_depth=1,
        rule_margin=0.45,
        max_profile_features=48,
        inductive_miner_noise_threshold=0.0,
        variant_coverage_threshold=0.0,
    ),
    TuningCandidate(
        penalty=8.0,
        min_segment_size=20,
        n_activity_clusters=12,
        signature_depth=2,
        rule_margin=0.35,
        max_profile_features=40,
        inductive_miner_noise_threshold=0.0,
        variant_coverage_threshold=0.0,
    ),
]

OUTPUT_PATH = "real_model.png"
SIGNATURE_DEBUG_PATH = "signature_debug.png"
ACTIVITY_LEGEND_PATH = "real_activity_legend.md"
MAX_PRINTED_TRACE_VARIANTS = 25

SENSOR_COLS = [
    "M_R01_SJointAngle_Degree",
    "M_R01_LJointAngle_Degree",
    "M_R01_UJointAngle_Degree",
    "M_R01_RJointAngle_Degree",
    "M_R01_BJointAngle_Degree",
    "M_R01_TJointAngle_Degree",
    "M_R02_SJointAngle_Degree",
    "M_R02_LJointAngle_Degree",
    "M_R02_UJointAngle_Degree",
    "M_R02_RJointAngle_Degree",
    "M_R02_BJointAngle_Degree",
    "M_R02_TJointAngle_Degree",
    "M_R03_SJointAngle_Degree",
    "M_R03_LJointAngle_Degree",
    "M_R03_UJointAngle_Degree",
    "M_R03_RJointAngle_Degree",
    "M_R03_BJointAngle_Degree",
    "M_R03_TJointAngle_Degree",
    "M_R04_SJointAngle_Degree",
    "M_R04_LJointAngle_Degree",
    "M_R04_UJointAngle_Degree",
    "M_R04_RJointAngle_Degree",
    "M_R04_BJointAngle_Degree",
    "M_R04_TJointAngle_Degree",
    "I_R01_Gripper_Pot",
    "I_R01_Gripper_Load",
    "I_R02_Gripper_Pot",
    "I_R02_Gripper_Load",
    "I_R03_Gripper_Pot",
    "I_R03_Gripper_Load",
    "I_R04_Gripper_Pot",
    "I_R04_Gripper_Load",
    "M_Conv1_Speed_mmps",
    "M_Conv2_Speed_mmps",
    "M_Conv3_Speed_mmps",
    "M_Conv4_Speed_mmps",
]


def load_dataset(pkl_path: Path) -> pd.DataFrame:
    """Load the local Future Factory pickle."""
    with open(pkl_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="numpy.core.numeric is deprecated",
                category=DeprecationWarning,
            )
            df: pd.DataFrame = pickle.load(f)
    return df


def load_cycle_arrays(
    df: pd.DataFrame,
    sensor_cols: list[str],
    n_case_runs: int | None,
    min_case_samples: int,
) -> tuple[list[np.ndarray], list[CaseRun]]:
    """Load complete-looking contiguous cycle runs and sensor arrays."""
    runs = select_contiguous_case_runs(
        df,
        case_id_col="Q_Cell_CycleCount",
        max_cases=n_case_runs,
        min_samples=min_case_samples,
    )
    cycles = [
        df.iloc[run.start_row:run.end_row][sensor_cols]
        .to_numpy(dtype=np.float64)
        for run in runs
    ]
    return cycles, runs


def main() -> None:
    print(f"Loading Future Factory data from {PKL_PATH} ...")
    df = load_dataset(PKL_PATH)
    cycles, runs = load_cycle_arrays(
        df,
        SENSOR_COLS,
        N_CASE_RUNS,
        MIN_CASE_SAMPLES,
    )

    print(f"Selected {len(runs)} contiguous case run(s).")
    _print_case_selection_summary(df, runs, cycles)

    train_cases, validation_cases, test_cases = split_cases(
        cycles,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
    )
    print(
        "\nSplit cases: "
        f"{len(train_cases)} train, "
        f"{len(validation_cases)} validation, "
        f"{len(test_cases)} test"
    )

    print("\nTuning generic pipeline candidates on held-out validation cases ...")
    tuning = tune_pipeline(
        train_cases,
        validation_cases,
        TUNING_CANDIDATES,
        SENSOR_COLS,
    )
    _print_tuning_results(tuning.evaluations)

    best = tuning.best.candidate
    print(f"\nSelected candidate: {_candidate_summary(best)}")

    final_train_cases = [*train_cases, *validation_cases]
    final_data, final_case_boundaries = stack_case_arrays(final_train_cases)
    result = run_pipeline(candidate_config(
        best,
        final_data,
        final_case_boundaries,
        SENSOR_COLS,
        output_path=OUTPUT_PATH,
        signature_debug_path=SIGNATURE_DEBUG_PATH,
        activity_legend_path=ACTIVITY_LEGEND_PATH,
    ))

    test_data, test_case_boundaries = stack_case_arrays(test_cases)
    test_validation = validate_sensor_log(
        test_data,
        result,
        case_boundaries=test_case_boundaries,
    )

    print("\nFinal training metrics:")
    _print_metrics(metrics_for_discovery(result))
    print("\nHeld-out test metrics:")
    _print_metrics(metrics_for_validation(test_validation))

    print("\nFinal generic trace variants:")
    _print_trace_variants(result.traces, result.n_classes)

    print(f"\nPetri net: {len(result.net.transitions)} transitions, "
          f"{len(result.net.places)} places")
    print(f"Model saved to {OUTPUT_PATH}")
    print(f"Activity legend saved to {ACTIVITY_LEGEND_PATH}")
    print(f"Signature debug view saved to {SIGNATURE_DEBUG_PATH}")


def _print_case_selection_summary(
    df: pd.DataFrame,
    runs: list[CaseRun],
    cycles: list[np.ndarray],
) -> None:
    if not runs:
        return
    first = runs[0]
    last = runs[-1]
    print(
        "  Time span: "
        f"{df['timestamp'].iloc[first.start_row]} -> "
        f"{df['timestamp'].iloc[last.end_row - 1]}"
    )
    lengths = np.array([cycle.shape[0] for cycle in cycles])
    print(
        "  Samples per case: "
        f"min={lengths.min()}, median={int(np.median(lengths))}, "
        f"max={lengths.max()}"
    )


def _print_tuning_results(evaluations: object) -> None:
    for idx, evaluation in enumerate(evaluations, start=1):
        metrics = evaluation.validation_metrics
        print(
            f"  Candidate {idx}: {_candidate_summary(evaluation.candidate)} | "
            f"coverage={metrics.coverage:.3f}, "
            f"fitness={metrics.model_fitness:.3f}, "
            f"uncovered={metrics.uncovered_segments}, "
            f"ambiguous={metrics.ambiguous_segments}, "
            f"variants={metrics.n_variants}"
        )


def _print_metrics(metrics: object) -> None:
    print(
        f"  cases={metrics.n_cases}, segments={metrics.n_segments}, "
        f"events={metrics.n_events}, variants={metrics.n_variants}, "
        f"activities={metrics.n_activities}"
    )
    print(
        f"  coverage={metrics.coverage:.3f}, "
        f"fitness={metrics.model_fitness:.3f}, "
        f"uncovered={metrics.uncovered_segments}, "
        f"ambiguous={metrics.ambiguous_segments}"
    )


def _print_trace_variants(traces: list[list[int]], n_activities: int) -> None:
    counts: dict[tuple[int, ...], int] = {}
    for trace in traces:
        key = tuple(trace)
        counts[key] = counts.get(key, 0) + 1

    variants = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    for idx, (variant, count) in enumerate(
        variants[:MAX_PRINTED_TRACE_VARIANTS],
        start=1,
    ):
        print(
            f"  Variant {idx} ({count} case(s)): "
            f"{_format_trace(list(variant), n_activities)}"
        )
    remaining = len(variants) - MAX_PRINTED_TRACE_VARIANTS
    if remaining > 0:
        print(f"  ... {remaining} additional variant(s) omitted")


def _candidate_summary(candidate: TuningCandidate) -> str:
    return (
        f"penalty={candidate.penalty}, "
        f"min_size={candidate.min_segment_size}, "
        f"clusters={candidate.n_activity_clusters}, "
        f"depth={candidate.signature_depth}, "
        f"margin={candidate.rule_margin}, "
        f"max_features={candidate.max_profile_features}"
    )


def _format_trace(trace: list[int], n_activities: int) -> list[str]:
    return [
        activity_name(activity, n_activities)
        if activity >= 0
        else "UNMAPPED"
        for activity in trace
    ]


if __name__ == "__main__":
    main()
