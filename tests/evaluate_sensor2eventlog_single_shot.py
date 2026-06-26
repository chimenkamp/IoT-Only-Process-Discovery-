"""Single-shot benchmark against the Sensor2EventLog evaluation setup.

The benchmark intentionally does not add Sensor2EventLog's expert event rules
or iterative teacher feedback. Reference labels are used only after discovery
to compute the paper's rule coverage and precision metrics when labels are
available in the local dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

import tests.evaluate_haccp_pasteurization as haccp
from src.discovery import activity_name
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
    rule_performance_table,
    rule_state_mapping,
    timestamp_rule_ids_from_segments,
)


OUTPUT_DIR = Path("evaluation_results/sensor2eventlog_single_shot")
HACCP_SOURCE_DIR = Path("evaluation_results/haccp_pasteurization")
HACCP_OUTPUT_DIR = OUTPUT_DIR / "haccp_pasteurization"

SWAT_DIR = Path("data/SWaT.A4 & A5_Jul 2019")
SWAT_WORKBOOK = SWAT_DIR / "SWaT_dataset_Jul 19 v2.xlsx"
SWAT_OUTPUT_DIR = OUTPUT_DIR / "swat_p1"

SWAT_TIMESTAMP_COLUMN = "GMT +0"
SWAT_SENSOR_COLUMNS = [
    "FIT 101",
    "LIT 101",
    "MV 101",
    "P101 Status",
    "P102 Status",
]
SWAT_AUDIT_COLUMNS = ["P1_STATE"]
SWAT_STATE_COLUMN = "P1_STATE"

# The SWaT collection note says the workbook timestamps are GMT+0 and the
# normal run is 12:35-14:50 GMT+8, i.e. 04:35-06:50 GMT+0.
SWAT_NORMAL_START = pd.Timestamp("2019-07-20T04:35:00Z")
SWAT_NORMAL_END = pd.Timestamp("2019-07-20T06:50:00Z")

# The paper's SWaT DFG reports four cases, but the local workbook has no case
# identifier. These are transparent pseudo-batches used only for PM formatting.
SWAT_PSEUDO_BATCHES = 4
SWAT_TRAIN_FRACTION = 0.60
SWAT_TARGET_ACTIVITY_COUNT = 3

SWAT_PIPELINE_PENALTY = 4.0
SWAT_PIPELINE_MIN_SEGMENT_SIZE = 20
SWAT_PIPELINE_SIGNATURE_DEPTH = 1
SWAT_PIPELINE_INCLUDE_DERIVATIVES = True
SWAT_PIPELINE_RULE_MARGIN = 2.0
SWAT_PIPELINE_MAX_PROFILE_FEATURES = 30
SWAT_PIPELINE_PROFILE_SCALING = "robust"
SWAT_PIPELINE_RANDOM_STATE = 0


def main() -> None:
    """Run the full single-shot benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_haccp()
    swat_summary = run_swat_p1()

    haccp_summary = pd.read_csv(HACCP_OUTPUT_DIR / "summary.csv")
    haccp_summary["artifact_source"] = str(HACCP_OUTPUT_DIR)
    swat_summary["artifact_source"] = "generated_this_run"
    summaries = [haccp_summary, swat_summary]
    benchmark_summary = pd.concat(summaries, ignore_index=True, sort=False)
    benchmark_summary.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    write_methodology_note()

    print("\nSingle-shot Sensor2EventLog benchmark")
    print(benchmark_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote benchmark artifacts to {OUTPUT_DIR}")


def run_haccp() -> None:
    """Copy or regenerate the existing all-batch HACCP evaluation."""
    if os.environ.get("RERUN_HACCP", "0") == "1":
        haccp.OUTPUT_DIR = HACCP_OUTPUT_DIR
        haccp.main()
        return

    if not (HACCP_SOURCE_DIR / "summary.csv").exists():
        raise FileNotFoundError(
            "Existing HACCP summary is missing. Run "
            "`venv/bin/python evaluate_haccp_pasteurization.py` first, or set "
            "RERUN_HACCP=1 for this combined benchmark."
        )
    HACCP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in HACCP_SOURCE_DIR.iterdir():
        if path.is_file():
            shutil.copy2(path, HACCP_OUTPUT_DIR / path.name)


def run_swat_p1() -> pd.DataFrame:
    """Run the knowledge-agnostic discovery pipeline on SWaT P1."""
    SWAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_swat_p1_sample()
    split = split_swat_cases(sample)

    train_data, train_boundaries = stack_case_arrays(split["train_sensor_cases"])
    test_data, test_boundaries = stack_case_arrays(split["test_sensor_cases"])

    discovery = run_pipeline(PipelineConfig(
        data=train_data,
        penalty=SWAT_PIPELINE_PENALTY,
        min_segment_size=SWAT_PIPELINE_MIN_SEGMENT_SIZE,
        changepoint_model="l2",
        segment_by_case=True,
        merge_short_segments=True,
        var_names=SWAT_SENSOR_COLUMNS,
        case_boundaries=train_boundaries,
        signature_depth=SWAT_PIPELINE_SIGNATURE_DEPTH,
        include_derivative_features=SWAT_PIPELINE_INCLUDE_DERIVATIVES,
        activity_abstraction="clustered_interval",
        n_activity_clusters=SWAT_TARGET_ACTIVITY_COUNT,
        max_profile_features=SWAT_PIPELINE_MAX_PROFILE_FEATURES,
        profile_scaling=SWAT_PIPELINE_PROFILE_SCALING,
        profile_cluster_random_state=SWAT_PIPELINE_RANDOM_STATE,
        rule_margin=SWAT_PIPELINE_RULE_MARGIN,
        activity_label_style="generic",
        output_path=str(SWAT_OUTPUT_DIR / "process_model.png"),
        activity_legend_path=str(SWAT_OUTPUT_DIR / "activity_legend.md"),
    ))
    validation = validate_sensor_log(
        test_data,
        discovery,
        case_boundaries=test_boundaries,
    )

    train_true = np.concatenate(split["train_state_cases"])
    test_true = np.concatenate(split["test_state_cases"])
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
        observed_state_order(train_true),
    )
    test_rule_performance = rule_performance_table(
        test_true,
        test_rule_ids,
        rule_ids,
        observed_state_order(test_true),
    )
    train_best_rules = best_rule_per_state(train_rule_performance)
    test_best_rules = best_rule_per_state(test_rule_performance)
    train_aggregates = aggregate_rule_metrics(train_best_rules)
    test_aggregates = aggregate_rule_metrics(test_best_rules)
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
    discovery_metrics = metrics_for_discovery(discovery)
    validation_metrics = metrics_for_validation(validation)

    write_swat_outputs(
        sample,
        split,
        discovery,
        validation,
        train_rule_performance,
        test_rule_performance,
        train_best_rules,
        test_best_rules,
    )
    summary = pd.DataFrame([{
        "approach": "ours_signature_interval_rules",
        "dataset": "SWaT.A4_A5_Jul_2019_P1",
        "data_scope": "normal_operation_window_from_dataset_note",
        "sample_batches": SWAT_PSEUDO_BATCHES,
        "train_batches": len(split["train_sensor_cases"]),
        "test_batches": len(split["test_sensor_cases"]),
        "sensor_columns": len(SWAT_SENSOR_COLUMNS),
        "state_count": len(observed_state_order(np.concatenate([
            train_true,
            test_true,
        ]))),
        "target_activity_count": SWAT_TARGET_ACTIVITY_COUNT,
        "discovered_activities": discovery.n_classes,
        "train_segments": discovery_metrics.n_segments,
        "test_segments": validation_metrics.n_segments,
        "train_rule_applicability": discovery_metrics.coverage,
        "test_segment_rule_applicability": validation_metrics.coverage,
        "test_model_fitness": validation_metrics.model_fitness,
        "train_mean_rule_coverage": train_aggregates["mean_coverage"],
        "train_mean_rule_precision": train_aggregates["mean_precision"],
        "train_mean_rule_effectiveness": train_aggregates["mean_effectiveness"],
        "train_rule_state_accuracy": train_rule_state_accuracy,
        "test_mean_rule_coverage": test_aggregates["mean_coverage"],
        "test_mean_rule_precision": test_aggregates["mean_precision"],
        "test_mean_rule_effectiveness": test_aggregates["mean_effectiveness"],
        "test_rule_state_accuracy": test_rule_state_accuracy,
        "coverage_metric_status": "computed_from_workbook_p1_state",
        "reference_state_source": "P1_STATE_column_in_workbook",
        "teacher_iterations": 0,
        "domain_rules": 0,
        "rule_additions": 0,
        "rule_edits": 0,
        "pipeline_penalty": SWAT_PIPELINE_PENALTY,
        "pipeline_min_segment_size": SWAT_PIPELINE_MIN_SEGMENT_SIZE,
        "pipeline_signature_depth": SWAT_PIPELINE_SIGNATURE_DEPTH,
        "pipeline_rule_margin": SWAT_PIPELINE_RULE_MARGIN,
        "pipeline_max_profile_features": SWAT_PIPELINE_MAX_PROFILE_FEATURES,
        "paper_hmm_baseline_accuracy": 0.58,
        "paper_hmm_final_accuracy": 0.76,
        "paper_kmeans_baseline_accuracy": 0.70,
        "paper_kmeans_final_accuracy": 0.94,
        "paper_min_reported_state_explainability": 0.78,
    }])
    summary.to_csv(SWAT_OUTPUT_DIR / "summary.csv", index=False)
    write_swat_paper_comparison(summary)
    return summary


def load_swat_p1_sample() -> pd.DataFrame:
    """Load the SWaT workbook's P1 normal-operation window."""
    usecols = [
        SWAT_TIMESTAMP_COLUMN,
        *SWAT_SENSOR_COLUMNS,
        *SWAT_AUDIT_COLUMNS,
    ]
    df = pd.read_excel(
        SWAT_WORKBOOK,
        header=1,
        skiprows=[2],
        usecols=usecols,
    )
    df["timestamp"] = pd.to_datetime(
        df[SWAT_TIMESTAMP_COLUMN],
        format="mixed",
        utc=True,
    )
    df = df[
        (df["timestamp"] >= SWAT_NORMAL_START)
        & (df["timestamp"] < SWAT_NORMAL_END)
    ].copy()
    df = df.reset_index(drop=True)

    for column in SWAT_SENSOR_COLUMNS:
        df[column] = coerce_sensor_column(df[column])

    if df[SWAT_SENSOR_COLUMNS].isna().any().any():
        missing = df[SWAT_SENSOR_COLUMNS].isna().sum()
        raise ValueError(f"SWaT P1 sensor columns contain missing values: {missing}")
    return df


def split_swat_cases(sample: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """Create transparent pseudo-batches because SWaT has no local case IDs."""
    sensor_matrix = sample[SWAT_SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    state_values = swat_state_labels(sample[SWAT_STATE_COLUMN])
    timestamp_values = sample["timestamp"].to_numpy()
    indices = np.array_split(np.arange(len(sample)), SWAT_PSEUDO_BATCHES)
    sensor_cases = [sensor_matrix[index] for index in indices]
    state_cases = [state_values[index] for index in indices]
    timestamp_cases = [timestamp_values[index] for index in indices]

    n_train = max(1, int(round(len(sensor_cases) * SWAT_TRAIN_FRACTION)))
    if n_train >= len(sensor_cases):
        n_train = len(sensor_cases) - 1

    return {
        "train_sensor_cases": sensor_cases[:n_train],
        "train_state_cases": state_cases[:n_train],
        "test_sensor_cases": sensor_cases[n_train:],
        "test_state_cases": state_cases[n_train:],
        "train_timestamp_cases": timestamp_cases[:n_train],
        "test_timestamp_cases": timestamp_cases[n_train:],
    }


def coerce_sensor_column(series: pd.Series) -> pd.Series:
    """Convert numeric/status sensor values to floats without semantic mapping."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    status_map = {"Inactive": 0.0, "Active": 1.0}
    mapped = series.map(status_map)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(mapped).astype(float)


def swat_state_labels(series: pd.Series) -> np.ndarray:
    """Return readable P1 reference-state labels for evaluation only."""
    return series.map(lambda value: f"{SWAT_STATE_COLUMN}={value}").to_numpy(
        dtype=object,
    )


def observed_state_order(states: np.ndarray) -> list[object]:
    """Return deterministic reference-state order from observed labels."""
    return sorted(pd.unique(states).tolist(), key=str)


def write_swat_outputs(
    sample: pd.DataFrame,
    split: dict[str, list[np.ndarray]],
    discovery: Any,
    validation: Any,
    train_rule_performance: pd.DataFrame,
    test_rule_performance: pd.DataFrame,
    train_best_rules: pd.DataFrame,
    test_best_rules: pd.DataFrame,
) -> None:
    """Write SWaT artifacts for auditability."""
    sample_audit = sample[
        ["timestamp", *SWAT_SENSOR_COLUMNS, *SWAT_AUDIT_COLUMNS]
    ].copy()
    sample_audit.to_csv(SWAT_OUTPUT_DIR / "input_audit.csv", index=False)

    train_timestamps = np.concatenate(split["train_timestamp_cases"])
    test_timestamps = np.concatenate(split["test_timestamp_cases"])
    event_log_to_frame(
        discovery.event_log,
        train_timestamps,
        discovery.n_classes,
        split_name="train",
    ).to_csv(SWAT_OUTPUT_DIR / "train_event_log.csv", index=False)
    event_log_to_frame(
        validation.event_log,
        test_timestamps,
        discovery.n_classes,
        split_name="test",
    ).to_csv(SWAT_OUTPUT_DIR / "test_event_log.csv", index=False)
    train_rule_performance.to_csv(
        SWAT_OUTPUT_DIR / "train_rule_performance.csv",
        index=False,
    )
    test_rule_performance.to_csv(
        SWAT_OUTPUT_DIR / "test_rule_performance.csv",
        index=False,
    )
    train_best_rules.to_csv(SWAT_OUTPUT_DIR / "train_best_rules.csv", index=False)
    test_best_rules.to_csv(SWAT_OUTPUT_DIR / "test_best_rules.csv", index=False)


def event_log_to_frame(
    event_log: list[list[Any]],
    timestamps: np.ndarray,
    n_activities: int,
    split_name: str,
) -> pd.DataFrame:
    """Convert internal segment events to a readable CSV event log."""
    rows: list[dict[str, Any]] = []
    for case_idx, trace in enumerate(event_log):
        for event_idx, event in enumerate(trace):
            activity = (
                "UNASSIGNED"
                if event.activity < 0
                else activity_name(event.activity, n_activities)
            )
            end_index = max(event.start, event.end - 1)
            rows.append({
                "split": split_name,
                "case_id": f"{split_name}_{case_idx + 1}",
                "activity_sequence": event_idx + 1,
                "activity": activity,
                "start_index": event.start,
                "end_index": event.end,
                "start_timestamp": timestamps[event.start],
                "end_timestamp": timestamps[end_index],
                "duration_samples": event.end - event.start,
            })
    return pd.DataFrame(rows)


def write_swat_paper_comparison(summary: pd.DataFrame) -> None:
    """Write the paper comparison while keeping unavailable metrics explicit."""
    rows = [
        {
            "approach": "ours_knowledge_agnostic",
            "model": "signature_interval_rules",
            "accuracy": summary.loc[0, "test_rule_state_accuracy"],
            "accuracy_status": "computed_from_workbook_p1_state_after_discovery",
            "mean_coverage": summary.loc[0, "test_mean_rule_coverage"],
            "mean_precision": summary.loc[0, "test_mean_rule_precision"],
            "mean_effectiveness": summary.loc[0, "test_mean_rule_effectiveness"],
            "state_explainability": np.nan,
            "test_rule_applicability": summary.loc[
                0,
                "test_segment_rule_applicability",
            ],
            "teacher_iterations": 0,
            "domain_rules": 0,
        },
        {
            "approach": "Sensor2EventLog_reported",
            "model": "HMM",
            "accuracy": 0.76,
            "accuracy_status": "reported_by_paper_after_teacher_loop",
            "mean_coverage": np.nan,
            "mean_precision": np.nan,
            "mean_effectiveness": np.nan,
            "state_explainability": "all_states_above_0.78",
            "test_rule_applicability": np.nan,
            "teacher_iterations": 2,
            "domain_rules": 2,
        },
        {
            "approach": "Sensor2EventLog_reported",
            "model": "K-means",
            "accuracy": 0.94,
            "accuracy_status": "reported_by_paper_after_teacher_loop",
            "mean_coverage": np.nan,
            "mean_precision": np.nan,
            "mean_effectiveness": np.nan,
            "state_explainability": "all_states_above_0.78",
            "test_rule_applicability": np.nan,
            "teacher_iterations": 2,
            "domain_rules": 2,
        },
    ]
    pd.DataFrame(rows).to_csv(
        SWAT_OUTPUT_DIR / "paper_metric_comparison.csv",
        index=False,
    )


def write_methodology_note() -> None:
    """Document fairness decisions for the benchmark."""
    note = f"""# Single-shot Sensor2EventLog Benchmark

This benchmark compares against the evaluation protocol described in
`Paper/Sensor2EventLog.pdf`.

## Constraints Applied

- No Sensor2EventLog event-rule family was added to our method.
- No iterative planning/explaining/reviewing loop was used.
- No domain thresholds such as `Qin > tau_Q`, `T > 70`, `LIT101_diff_smooth < 0`,
  or `LIT101_stability > 0.8` were used by our pipeline.
- Reference labels are used only after discovery to compute the paper's rule
  coverage, precision, and effectiveness metrics.

## HACCP

The local HACCP dataset contains both `batch_id` and `state`, so the paper's
coverage metrics are computed directly on held-out batches. By default this
combined script copies the existing all-batch HACCP artifacts from
`evaluation_results/haccp_pasteurization`, because regenerating changepoints for
all 968 batches is slow. Set `RERUN_HACCP=1` to force regeneration.

## SWaT P1

The local SWaT workbook contains raw P1 sensor streams and controller/status
columns. It does not contain the prepared `batch_id` column used by the public
Sensor2EventLog implementation, but it does contain a `P1_STATE` column. The
benchmark therefore uses `P1_STATE` only after discovery to compute anonymous
rule coverage, precision, effectiveness, and a train-mapped timestamp accuracy.

For transparency, the SWaT run uses only the normal-operation window from the
provided data-collection note ({SWAT_NORMAL_START.isoformat()} to
{SWAT_NORMAL_END.isoformat()}, GMT+0) and four equal contiguous pseudo-batches
for process-mining formatting. These pseudo-batches are not domain batches, so
the reported SWaT accuracy is still a single-shot workbook-label diagnostic
rather than a reproduction of the paper's teacher-guided HMM/K-means results.
"""
    (OUTPUT_DIR / "methodology.md").write_text(note)


if __name__ == "__main__":
    main()
