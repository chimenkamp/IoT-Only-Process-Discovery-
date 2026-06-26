"""Sensor2EventLog-style evaluation on the HACCP pasteurization dataset.

The pipeline remains knowledge agnostic: reference states are used only after
discovery to compute Sensor2EventLog's rule coverage and rule precision.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from src.cases import CaseRun, contiguous_case_runs
from src.evaluation import (
    metrics_for_discovery,
    metrics_for_validation,
    stack_case_arrays,
)
from src.pipeline import PipelineConfig, run_pipeline, validate_sensor_log
from src.rule_metrics import (
    aggregate_rule_metrics,
    best_rule_per_state,
    mapped_state_accuracy,
    paper_reported_summary,
    rule_performance_table,
    rule_state_mapping,
    timestamp_rule_ids_from_segments,
)


DATASET_DIR = Path("data/PM_HACCP_PASTEURIZATION-main")
OUTPUT_DIR = Path("evaluation_results/haccp_pasteurization")

# Change this for quick smoke tests versus larger benchmark runs.
SAMPLE_BATCHES: int | None = None

TRAIN_FRACTION = 0.60

STATE_ORDER = [
    "Idle",
    "Fill",
    "HeatUp",
    "Hold",
    "Cool",
    "Discharge",
]

SENSOR_COLUMNS = [
    "T",
    "pH",
    "Kappa",
    "Mu",
    "Tau",
    "Q_in",
    "Q_out",
    "P",
]

PIPELINE_PENALTY = 6.0
PIPELINE_MIN_SEGMENT_SIZE = 20
PIPELINE_SIGNATURE_DEPTH = 1
PIPELINE_INCLUDE_DERIVATIVES = True
PIPELINE_RULE_MARGIN = 2.0
PIPELINE_MAX_PROFILE_FEATURES = 40
PIPELINE_PROFILE_SCALING = "robust"
PIPELINE_RANDOM_STATE = 0

EXPORT_SIGNATURE_DEBUG = False

# Our method does not use Sensor2EventLog's teacher loop.
TEACHER_ITERATIONS = 0
DOMAIN_RULES = 0
RULE_ADDITIONS = 0
RULE_EDITS = 0


@dataclass(frozen=True)
class HaccpSample:
    """A complete-batch subset of the HACCP pasteurization data."""

    dataframe: pd.DataFrame
    runs: list[CaseRun]


@dataclass(frozen=True)
class CaseSplit:
    """Sensor and state cases split into train and test partitions."""

    train_sensor_cases: list[np.ndarray]
    train_state_cases: list[np.ndarray]
    test_sensor_cases: list[np.ndarray]
    test_state_cases: list[np.ndarray]


def main() -> None:
    sample = load_haccp_sample(DATASET_DIR, SAMPLE_BATCHES)
    split = split_cases(sample, TRAIN_FRACTION)

    train_data, train_boundaries = stack_case_arrays(split.train_sensor_cases)
    test_data, test_boundaries = stack_case_arrays(split.test_sensor_cases)

    output_paths = _artifact_paths(OUTPUT_DIR)
    discovery = run_pipeline(PipelineConfig(
        data=train_data,
        penalty=PIPELINE_PENALTY,
        min_segment_size=PIPELINE_MIN_SEGMENT_SIZE,
        changepoint_model="l2",
        segment_by_case=True,
        merge_short_segments=True,
        var_names=SENSOR_COLUMNS,
        case_boundaries=train_boundaries,
        signature_depth=PIPELINE_SIGNATURE_DEPTH,
        include_derivative_features=PIPELINE_INCLUDE_DERIVATIVES,
        activity_abstraction="clustered_interval",
        n_activity_clusters=len(STATE_ORDER),
        max_profile_features=PIPELINE_MAX_PROFILE_FEATURES,
        profile_scaling=PIPELINE_PROFILE_SCALING,
        profile_cluster_random_state=PIPELINE_RANDOM_STATE,
        rule_margin=PIPELINE_RULE_MARGIN,
        activity_label_style="generic",
        output_path=output_paths.get("model"),
        signature_debug_path=output_paths.get("signature_debug"),
        activity_legend_path=output_paths.get("activity_legend"),
    ))
    validation = validate_sensor_log(
        test_data,
        discovery,
        case_boundaries=test_boundaries,
    )

    train_true = np.concatenate(split.train_state_cases)
    test_true = np.concatenate(split.test_state_cases)
    train_rule_ids = timestamp_rule_ids_from_segments(
        discovery.boundaries,
        discovery.segment_labels,
    )
    test_rule_ids = timestamp_rule_ids_from_segments(
        validation.boundaries,
        validation.segment_labels,
    )

    rule_ids = [rule.class_id for rule in discovery.rules]
    train_rule_performance = rule_performance_table(
        train_true,
        train_rule_ids,
        rule_ids,
        STATE_ORDER,
    )
    test_rule_performance = rule_performance_table(
        test_true,
        test_rule_ids,
        rule_ids,
        STATE_ORDER,
    )
    train_best_rules = best_rule_per_state(train_rule_performance)
    test_best_rules = best_rule_per_state(test_rule_performance)
    train_rule_state_mapping = rule_state_mapping(train_rule_performance)
    train_rule_state_accuracy = mapped_state_accuracy(
        train_true,
        train_rule_ids,
        train_rule_state_mapping,
    )
    test_rule_state_accuracy = mapped_state_accuracy(
        test_true,
        test_rule_ids,
        train_rule_state_mapping,
    )
    summary = _summary_table(
        sample,
        split,
        discovery,
        validation,
        train_best_rules,
        test_best_rules,
        train_rule_state_accuracy,
        test_rule_state_accuracy,
    )
    comparison = _comparison_table(test_best_rules)

    write_outputs(
        OUTPUT_DIR,
        summary,
        comparison,
        train_rule_performance,
        test_rule_performance,
        train_best_rules,
        test_best_rules,
    )
    print_report(summary, comparison, test_best_rules, output_paths)


def load_haccp_sample(
    dataset_dir: Path,
    sample_batches: int | None,
) -> HaccpSample:
    """Load complete contiguous batches from the split CSV files."""
    files = _dataset_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No HACCP CSV files found in {dataset_dir}")
    if sample_batches is not None and sample_batches < 2:
        raise ValueError("SAMPLE_BATCHES must be at least 2")

    columns = ["timestamp", "batch_id", "state", *SENSOR_COLUMNS]
    frames: list[pd.DataFrame] = []
    for idx, path in enumerate(files):
        frames.append(pd.read_csv(path, usecols=columns))
        combined = pd.concat(frames, ignore_index=True)
        runs = contiguous_case_runs(combined["batch_id"].to_numpy())
        if sample_batches is None:
            continue
        if len(runs) >= sample_batches + 1 or idx == len(files) - 1:
            return _trim_to_complete_runs(combined, sample_batches)

    combined = pd.concat(frames, ignore_index=True)
    runs = contiguous_case_runs(combined["batch_id"].to_numpy())
    return HaccpSample(dataframe=combined, runs=runs)


def split_cases(sample: HaccpSample, train_fraction: float) -> CaseSplit:
    """Split complete cases into train/test partitions."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("TRAIN_FRACTION must be between 0 and 1")
    if len(sample.runs) < 2:
        raise ValueError("At least two complete batches are required")

    sensor_cases = [
        sample.dataframe.iloc[run.start_row:run.end_row][SENSOR_COLUMNS]
        .to_numpy(dtype=np.float64)
        for run in sample.runs
    ]
    state_cases = [
        sample.dataframe.iloc[run.start_row:run.end_row]["state"]
        .to_numpy(dtype=object)
        for run in sample.runs
    ]

    n_train = max(1, int(round(len(sensor_cases) * train_fraction)))
    if n_train >= len(sensor_cases):
        n_train = len(sensor_cases) - 1

    return CaseSplit(
        train_sensor_cases=sensor_cases[:n_train],
        train_state_cases=state_cases[:n_train],
        test_sensor_cases=sensor_cases[n_train:],
        test_state_cases=state_cases[n_train:],
    )


def write_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    train_rule_performance: pd.DataFrame,
    test_rule_performance: pd.DataFrame,
    train_best_rules: pd.DataFrame,
    test_best_rules: pd.DataFrame,
) -> None:
    """Write evaluation artifacts as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison.to_csv(output_dir / "paper_metric_comparison.csv", index=False)
    train_rule_performance.to_csv(
        output_dir / "train_rule_performance.csv",
        index=False,
    )
    test_rule_performance.to_csv(
        output_dir / "test_rule_performance.csv",
        index=False,
    )
    train_best_rules.to_csv(output_dir / "train_best_rules.csv", index=False)
    test_best_rules.to_csv(output_dir / "test_best_rules.csv", index=False)


def print_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    test_best_rules: pd.DataFrame,
    output_paths: dict[str, str],
) -> None:
    """Print the high-signal evaluation results."""
    float_format = lambda value: f"{value:.3f}"
    print("\nHACCP pasteurization evaluation")
    print(summary.to_string(index=False, float_format=float_format))
    print("\nHeld-out best anonymous rule per reference state:")
    print(test_best_rules.to_string(index=False, float_format=float_format))
    print("\nComparison with Sensor2EventLog reported rule metrics:")
    print(comparison.to_string(index=False, float_format=float_format))
    print(f"\nProcess model image: {output_paths['model']}")
    print(f"\nWrote CSV artifacts to {OUTPUT_DIR}")


def _trim_to_complete_runs(
    dataframe: pd.DataFrame,
    sample_batches: int,
) -> HaccpSample:
    runs = contiguous_case_runs(dataframe["batch_id"].to_numpy())
    if len(runs) > sample_batches:
        end_row = runs[sample_batches].start_row
    elif runs:
        end_row = runs[-1].end_row
    else:
        end_row = 0
    trimmed = dataframe.iloc[:end_row].reset_index(drop=True)
    trimmed_runs = contiguous_case_runs(trimmed["batch_id"].to_numpy())
    if len(trimmed_runs) > sample_batches:
        trimmed_runs = trimmed_runs[:sample_batches]
    if len(trimmed_runs) < sample_batches:
        print(
            f"Requested {sample_batches} batches, found "
            f"{len(trimmed_runs)} complete batches."
        )
    return HaccpSample(dataframe=trimmed, runs=trimmed_runs)


def _dataset_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        dataset_dir.glob("synthetic_pasteurization_with_cip_signals1000_part*.csv"),
        key=_part_number,
    )


def _part_number(path: Path) -> int:
    suffix = path.stem.rsplit("part", maxsplit=1)[-1]
    return int(suffix)


def _artifact_paths(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": str(output_dir / "process_model.png"),
        "signature_debug": None,
        "activity_legend": str(output_dir / "activity_legend.md"),
    }
    if EXPORT_SIGNATURE_DEBUG:
        paths["signature_debug"] = str(output_dir / "signature_debug.png")
    return paths


def _summary_table(
    sample: HaccpSample,
    split: CaseSplit,
    discovery: object,
    validation: object,
    train_best_rules: pd.DataFrame,
    test_best_rules: pd.DataFrame,
    train_rule_state_accuracy: float,
    test_rule_state_accuracy: float,
) -> pd.DataFrame:
    discovery_metrics = metrics_for_discovery(discovery)
    validation_metrics = metrics_for_validation(validation)
    train_aggregates = aggregate_rule_metrics(train_best_rules)
    test_aggregates = aggregate_rule_metrics(test_best_rules)
    return pd.DataFrame([{
        "approach": "ours_signature_interval_rules",
        "dataset": "PM_HACCP_PASTEURIZATION",
        "sample_batches": len(sample.runs),
        "train_batches": len(split.train_sensor_cases),
        "test_batches": len(split.test_sensor_cases),
        "sensor_columns": len(SENSOR_COLUMNS),
        "state_count": len(STATE_ORDER),
        "discovered_activities": discovery.n_classes,
        "train_segments": discovery_metrics.n_segments,
        "test_segments": validation_metrics.n_segments,
        "train_mean_rule_coverage": train_aggregates["mean_coverage"],
        "train_mean_rule_precision": train_aggregates["mean_precision"],
        "train_mean_rule_effectiveness": train_aggregates["mean_effectiveness"],
        "train_rule_state_accuracy": train_rule_state_accuracy,
        "test_mean_rule_coverage": test_aggregates["mean_coverage"],
        "test_mean_rule_precision": test_aggregates["mean_precision"],
        "test_mean_rule_effectiveness": test_aggregates["mean_effectiveness"],
        "test_rule_state_accuracy": test_rule_state_accuracy,
        "test_segment_rule_applicability": validation_metrics.coverage,
        "test_model_fitness": validation_metrics.model_fitness,
        "teacher_iterations": TEACHER_ITERATIONS,
        "domain_rules": DOMAIN_RULES,
        "rule_additions": RULE_ADDITIONS,
        "rule_edits": RULE_EDITS,
        "pipeline_penalty": PIPELINE_PENALTY,
        "pipeline_min_segment_size": PIPELINE_MIN_SEGMENT_SIZE,
        "pipeline_signature_depth": PIPELINE_SIGNATURE_DEPTH,
        "pipeline_rule_margin": PIPELINE_RULE_MARGIN,
        "pipeline_max_profile_features": PIPELINE_MAX_PROFILE_FEATURES,
    }])


def _comparison_table(test_best_rules: pd.DataFrame) -> pd.DataFrame:
    ours = aggregate_rule_metrics(test_best_rules)
    ours_row = pd.DataFrame([{
        "approach": "ours_knowledge_agnostic",
        "rules_compared": len(test_best_rules),
        "mean_coverage": ours["mean_coverage"],
        "mean_precision": ours["mean_precision"],
        "mean_effectiveness": ours["mean_effectiveness"],
        "teacher_iterations": TEACHER_ITERATIONS,
        "domain_rules": DOMAIN_RULES,
    }])
    return pd.concat([ours_row, paper_reported_summary()], ignore_index=True)


if __name__ == "__main__":
    main()
