from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pipeline import (
    PipelineConfig,
    PipelineResult,
    ValidationResult,
    run_pipeline,
    validate_sensor_log,
)
from src.discovery import token_replay_fitness


@dataclass(frozen=True)
class DiscoveryMetrics:
    """Compact quality summary for discovered or validated traces."""

    n_cases: int
    n_segments: int
    n_events: int
    n_variants: int
    n_activities: int
    uncovered_segments: int
    ambiguous_segments: int
    coverage: float
    model_fitness: float


@dataclass(frozen=True)
class TuningCandidate:
    """Domain-agnostic configuration values to compare on held-out cases."""

    penalty: float
    min_segment_size: int
    n_activity_clusters: int
    signature_depth: int = 1
    include_derivative_features: bool = True
    rule_margin: float = 0.25
    max_profile_features: int | None = 40
    profile_scaling: str = "robust"
    inductive_miner_noise_threshold: float = 0.0
    variant_coverage_threshold: float = 0.0


@dataclass(frozen=True)
class CandidateEvaluation:
    """Result of training one candidate and validating it on held-out cases."""

    candidate: TuningCandidate
    validation_metrics: DiscoveryMetrics
    discovery: PipelineResult


@dataclass(frozen=True)
class TuningResult:
    """All candidate evaluations plus the selected best candidate."""

    best: CandidateEvaluation
    evaluations: list[CandidateEvaluation]


def stack_case_arrays(cases: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Stack case arrays and return the matching case-boundary offsets."""
    if not cases:
        raise ValueError("at least one case is required")
    boundaries = [0]
    for case in cases:
        if case.ndim != 2:
            raise ValueError("each case must be a 2-D array")
        boundaries.append(boundaries[-1] + case.shape[0])
    return np.vstack(cases), boundaries


def split_cases(
    cases: list[np.ndarray],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split ordered cases into train/validation/test partitions."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave test cases")
    if len(cases) < 3:
        raise ValueError("at least three cases are required")

    n_cases = len(cases)
    n_train = max(1, int(round(n_cases * train_fraction)))
    n_validation = max(1, int(round(n_cases * validation_fraction)))
    if n_train + n_validation >= n_cases:
        n_validation = max(1, n_cases - n_train - 1)
    if n_train + n_validation >= n_cases:
        n_train = max(1, n_cases - n_validation - 1)

    return (
        cases[:n_train],
        cases[n_train:n_train + n_validation],
        cases[n_train + n_validation:],
    )


def metrics_for_discovery(discovery: PipelineResult) -> DiscoveryMetrics:
    """Return metrics for the training discovery result."""
    n_segments = len(discovery.segment_labels)
    model_fitness = token_replay_fitness(
        discovery.traces,
        discovery.rules,
        discovery.profile_names,
        discovery.net,
        discovery.initial_marking,
        discovery.final_marking,
        label_style=discovery.activity_label_style,
        activity_prefix=discovery.activity_label_prefix,
    )
    return DiscoveryMetrics(
        n_cases=len(discovery.traces),
        n_segments=n_segments,
        n_events=sum(len(trace) for trace in discovery.traces),
        n_variants=len({tuple(trace) for trace in discovery.traces}),
        n_activities=discovery.n_classes,
        uncovered_segments=len(discovery.uncovered_segments),
        ambiguous_segments=len(discovery.ambiguous_segments),
        coverage=_coverage(
            n_segments,
            len(discovery.uncovered_segments),
            len(discovery.ambiguous_segments),
        ),
        model_fitness=model_fitness,
    )


def metrics_for_validation(validation: ValidationResult) -> DiscoveryMetrics:
    """Return metrics for validation against a learned model."""
    n_segments = len(validation.segment_labels)
    return DiscoveryMetrics(
        n_cases=len(validation.traces),
        n_segments=n_segments,
        n_events=sum(len(trace) for trace in validation.traces),
        n_variants=len({tuple(trace) for trace in validation.traces}),
        n_activities=len({
            label for label in validation.segment_labels if label >= 0
        }),
        uncovered_segments=len(validation.uncovered_segments),
        ambiguous_segments=len(validation.ambiguous_segments),
        coverage=_coverage(
            n_segments,
            len(validation.uncovered_segments),
            len(validation.ambiguous_segments),
        ),
        model_fitness=validation.model_fitness,
    )


def candidate_config(
    candidate: TuningCandidate,
    data: np.ndarray,
    case_boundaries: list[int],
    var_names: list[str],
    output_path: str | None = None,
    signature_debug_path: str | None = None,
    activity_legend_path: str | None = None,
) -> PipelineConfig:
    """Build a pipeline config from one generic tuning candidate."""
    return PipelineConfig(
        data=data,
        penalty=candidate.penalty,
        min_segment_size=candidate.min_segment_size,
        changepoint_model="l2",
        segment_by_case=True,
        merge_short_segments=True,
        var_names=var_names,
        case_boundaries=case_boundaries,
        signature_depth=candidate.signature_depth,
        include_derivative_features=candidate.include_derivative_features,
        activity_abstraction="clustered_interval",
        n_activity_clusters=candidate.n_activity_clusters,
        max_profile_features=candidate.max_profile_features,
        profile_scaling=candidate.profile_scaling,
        rule_margin=candidate.rule_margin,
        inductive_miner_noise_threshold=(
            candidate.inductive_miner_noise_threshold
        ),
        variant_coverage_threshold=candidate.variant_coverage_threshold,
        activity_label_style="generic",
        activity_label_prefix="A",
        output_path=output_path,
        signature_debug_path=signature_debug_path,
        activity_legend_path=activity_legend_path,
    )


def tune_pipeline(
    train_cases: list[np.ndarray],
    validation_cases: list[np.ndarray],
    candidates: list[TuningCandidate],
    var_names: list[str],
) -> TuningResult:
    """Select a candidate by held-out coverage, fitness, and simplicity."""
    if not candidates:
        raise ValueError("at least one tuning candidate is required")

    train_data, train_boundaries = stack_case_arrays(train_cases)
    validation_data, validation_boundaries = stack_case_arrays(validation_cases)
    evaluations: list[CandidateEvaluation] = []

    for candidate in candidates:
        discovery = run_pipeline(candidate_config(
            candidate,
            train_data,
            train_boundaries,
            var_names,
        ))
        validation = validate_sensor_log(
            validation_data,
            discovery,
            case_boundaries=validation_boundaries,
        )
        evaluations.append(CandidateEvaluation(
            candidate=candidate,
            validation_metrics=metrics_for_validation(validation),
            discovery=discovery,
        ))

    best = max(evaluations, key=_candidate_score)
    return TuningResult(best=best, evaluations=evaluations)


def _candidate_score(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    metrics = evaluation.validation_metrics
    candidate = evaluation.candidate
    max_features = (
        candidate.max_profile_features
        if candidate.max_profile_features is not None
        else 10_000
    )
    return (
        metrics.coverage,
        metrics.model_fitness,
        -metrics.ambiguous_segments,
        -metrics.uncovered_segments,
        -metrics.n_variants,
        -candidate.n_activity_clusters,
        -candidate.signature_depth,
        -max_features,
    )


def _coverage(
    n_segments: int,
    uncovered_segments: int,
    ambiguous_segments: int,
) -> float:
    if n_segments == 0:
        return 0.0
    covered = n_segments - uncovered_segments - ambiguous_segments
    return max(0.0, covered / n_segments)
