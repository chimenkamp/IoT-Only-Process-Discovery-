from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from src.changepoint import (
    add_required_boundaries,
    detect_changepoints,
    detect_changepoints_by_case,
    merge_short_segments,
)
from src.discovery import (
    discover_model,
    save_activity_legend,
    save_model_visualization,
    token_replay_fitness,
)
from src.merging import (
    SegmentProfile,
    interval_equivalence_classes,
    merge_overlapping_classes,
)
from src.preprocessing import Normaliser
from src.signatures import compute_signature_profiles, save_signature_debug_image
from src.synthesis import IntervalRule, classify_profiles, synthesize_rules
from src.trace import SegmentEvent, activity_projections, build_event_log


@dataclass
class PipelineConfig:
    """Configuration for the paper-aligned discovery procedure."""

    data: np.ndarray
    penalty: float = 1.0
    min_segment_size: int = 2
    changepoint_model: str = "l2"
    segment_by_case: bool = False
    merge_short_segments: bool = False
    var_names: list[str] | None = None
    case_boundaries: list[int] | None = None
    signature_depth: int = 2
    include_derivative_features: bool = False
    signature_profile_radius: float = 0.0
    activity_abstraction: str = "interval"
    n_activity_clusters: int | None = None
    profile_variance_threshold: float = 1e-10
    max_profile_features: int | None = None
    profile_scaling: str = "standard"
    profile_cluster_random_state: int = 0
    rule_margin: float = 0.0
    inductive_miner_noise_threshold: float = 0.0
    variant_coverage_threshold: float = 0.0
    activity_label_style: str = "rule"
    activity_label_prefix: str = "A"
    output_path: str | None = None
    signature_debug_path: str | None = None
    activity_legend_path: str | None = None
    activity_legend_compact_parts: int | None = None


@dataclass
class PipelineResult:
    """Artifacts produced by discovery from raw sensor values."""

    traces: list[list[int]]
    event_log: list[list[SegmentEvent]]
    rules: list[IntervalRule]
    boundaries: list[int]
    segment_labels: list[int]
    profiles: list[SegmentProfile]
    profile_names: list[str]
    profile_feature_matrix: np.ndarray
    preprocessed_data: np.ndarray
    normaliser: Normaliser
    sensor_names: list[str]
    signature_depth: int
    include_derivative_features: bool
    signature_profile_radius: float
    penalty: float
    min_segment_size: int
    changepoint_model: str
    segment_by_case: bool
    merge_short_segments: bool
    activity_abstraction: str
    max_profile_features: int | None
    profile_scaling: str
    selected_profile_features: np.ndarray | None
    profile_feature_means: np.ndarray | None
    profile_feature_scales: np.ndarray | None
    rule_margin: float
    inductive_miner_noise_threshold: float
    variant_coverage_threshold: float
    activity_label_style: str
    activity_label_prefix: str
    uncovered_segments: list[int]
    ambiguous_segments: list[int]
    net: object
    initial_marking: object
    final_marking: object

    @property
    def n_classes(self) -> int:
        return len(self.rules)


@dataclass
class ValidationResult:
    """Validation artifacts for a new raw sensor log."""

    traces: list[list[int]]
    event_log: list[list[SegmentEvent]]
    boundaries: list[int]
    segment_labels: list[int]
    profiles: list[SegmentProfile]
    profile_feature_matrix: np.ndarray
    uncovered_segments: list[int]
    ambiguous_segments: list[int]
    model_fitness: float

    @property
    def conforming(self) -> bool:
        return (
            not self.uncovered_segments
            and not self.ambiguous_segments
            and self.model_fitness == 1.0
        )


@dataclass(frozen=True)
class RuleFeatureSpace:
    """Feature space used to synthesize and apply activity rules."""

    profiles: list[SegmentProfile]
    names: list[str]
    matrix: np.ndarray
    selected_indices: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the draft pipeline end to end."""
    if config.data.ndim != 2:
        raise ValueError("data must be a 2-D array")

    sensor_names = config.var_names or [
        f"v{k}" for k in range(config.data.shape[1])
    ]
    if len(sensor_names) != config.data.shape[1]:
        raise ValueError("var_names must match data.shape[1]")

    normaliser = Normaliser.fit(config.data)
    preprocessed = normaliser.transform(config.data)

    boundaries = _detect_boundaries(preprocessed, config)

    raw_profiles, raw_profile_names, raw_feature_matrix = compute_signature_profiles(
        preprocessed,
        boundaries,
        var_names=sensor_names,
        radius=config.signature_profile_radius,
        signature_depth=config.signature_depth,
        include_derivative_features=config.include_derivative_features,
    )
    rule_space = _build_rule_feature_space(
        raw_profiles,
        raw_profile_names,
        raw_feature_matrix,
        config,
    )
    classes = _activity_classes(rule_space, config)
    rules = synthesize_rules(
        classes,
        rule_space.profiles,
        len(rule_space.names),
        margin=config.rule_margin,
    )
    segment_labels, uncovered, ambiguous = classify_profiles(
        rule_space.profiles,
        rules,
    )

    if uncovered or ambiguous:
        raise RuntimeError(
            "Synthesized rules are not a well-formed rule alphabet "
            f"(uncovered={uncovered}, ambiguous={ambiguous})"
        )

    event_log = build_event_log(
        boundaries,
        segment_labels,
        case_boundaries=config.case_boundaries,
    )
    traces = activity_projections(event_log)

    if config.signature_debug_path is not None:
        save_signature_debug_image(
            preprocessed,
            boundaries,
            config.signature_debug_path,
            var_names=sensor_names,
            segment_labels=segment_labels,
            signature_depth=config.signature_depth,
            include_derivative_features=config.include_derivative_features,
        )

    net, initial_marking, final_marking = discover_model(
        traces,
        rules,
        rule_space.names,
        label_style=config.activity_label_style,
        activity_prefix=config.activity_label_prefix,
        inductive_miner_noise_threshold=(
            config.inductive_miner_noise_threshold
        ),
        variant_coverage_threshold=config.variant_coverage_threshold,
    )

    if config.output_path is not None:
        save_model_visualization(
            net,
            initial_marking,
            final_marking,
            config.output_path,
        )
    if config.activity_legend_path is not None:
        save_activity_legend(
            rules,
            rule_space.names,
            config.activity_legend_path,
            activity_prefix=config.activity_label_prefix,
            compact_parts=config.activity_legend_compact_parts,
        )

    return PipelineResult(
        traces=traces,
        event_log=event_log,
        rules=rules,
        boundaries=boundaries,
        segment_labels=segment_labels,
        profiles=rule_space.profiles,
        profile_names=rule_space.names,
        profile_feature_matrix=rule_space.matrix,
        preprocessed_data=preprocessed,
        normaliser=normaliser,
        sensor_names=sensor_names,
        signature_depth=config.signature_depth,
        include_derivative_features=config.include_derivative_features,
        signature_profile_radius=config.signature_profile_radius,
        penalty=config.penalty,
        min_segment_size=config.min_segment_size,
        changepoint_model=config.changepoint_model,
        segment_by_case=config.segment_by_case,
        merge_short_segments=config.merge_short_segments,
        activity_abstraction=config.activity_abstraction,
        max_profile_features=config.max_profile_features,
        profile_scaling=config.profile_scaling,
        selected_profile_features=rule_space.selected_indices,
        profile_feature_means=rule_space.means,
        profile_feature_scales=rule_space.scales,
        rule_margin=config.rule_margin,
        inductive_miner_noise_threshold=(
            config.inductive_miner_noise_threshold
        ),
        variant_coverage_threshold=config.variant_coverage_threshold,
        activity_label_style=config.activity_label_style,
        activity_label_prefix=config.activity_label_prefix,
        uncovered_segments=uncovered,
        ambiguous_segments=ambiguous,
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
    )


def validate_sensor_log(
    data: np.ndarray,
    discovery: PipelineResult,
    case_boundaries: list[int] | None = None,
    penalty: float | None = None,
    min_segment_size: int | None = None,
    changepoint_model: str | None = None,
) -> ValidationResult:
    """Apply learned rules and model to a new raw sensor log."""
    preprocessed = discovery.normaliser.transform(data)
    boundaries = _detect_validation_boundaries(
        preprocessed,
        discovery,
        case_boundaries=case_boundaries,
        penalty=penalty,
        min_segment_size=min_segment_size,
        changepoint_model=changepoint_model,
    )

    raw_profiles, _, raw_feature_matrix = compute_signature_profiles(
        preprocessed,
        boundaries,
        var_names=discovery.sensor_names,
        radius=discovery.signature_profile_radius,
        signature_depth=discovery.signature_depth,
        include_derivative_features=discovery.include_derivative_features,
    )
    rule_space = _apply_rule_feature_space(
        raw_profiles,
        raw_feature_matrix,
        discovery,
    )
    labels, uncovered, ambiguous = classify_profiles(
        rule_space.profiles,
        discovery.rules,
    )
    event_log = build_event_log(
        boundaries,
        labels,
        case_boundaries=case_boundaries,
    )
    traces = activity_projections(event_log)
    assigned_traces = _assigned_activity_projections(event_log)
    model_fitness = 0.0
    if any(assigned_traces):
        model_fitness = token_replay_fitness(
            assigned_traces,
            discovery.rules,
            discovery.profile_names,
            discovery.net,
            discovery.initial_marking,
            discovery.final_marking,
            label_style=discovery.activity_label_style,
            activity_prefix=discovery.activity_label_prefix,
        )

    return ValidationResult(
        traces=traces,
        event_log=event_log,
        boundaries=boundaries,
        segment_labels=labels,
        profiles=rule_space.profiles,
        profile_feature_matrix=rule_space.matrix,
        uncovered_segments=uncovered,
        ambiguous_segments=ambiguous,
        model_fitness=model_fitness,
    )


def _assigned_activity_projections(
    event_log: list[list[SegmentEvent]],
) -> list[list[int]]:
    """Project only activities from the discovered rule alphabet."""
    return [
        [event.activity for event in trace if event.activity >= 0]
        for trace in event_log
    ]


def _detect_boundaries(
    data: np.ndarray,
    config: PipelineConfig,
) -> list[int]:
    if config.segment_by_case:
        if config.case_boundaries is None:
            raise ValueError("segment_by_case requires case_boundaries")
        boundaries = detect_changepoints_by_case(
            data,
            config.case_boundaries,
            config.penalty,
            config.min_segment_size,
            model=config.changepoint_model,
        )
    else:
        boundaries = detect_changepoints(
            data,
            config.penalty,
            config.min_segment_size,
            model=config.changepoint_model,
        )
        if config.case_boundaries is not None:
            boundaries = add_required_boundaries(
                boundaries,
                config.case_boundaries,
            )

    if config.merge_short_segments:
        boundaries = merge_short_segments(
            boundaries,
            config.min_segment_size,
            required_boundaries=config.case_boundaries,
        )
    return boundaries


def _detect_validation_boundaries(
    data: np.ndarray,
    discovery: PipelineResult,
    case_boundaries: list[int] | None = None,
    penalty: float | None = None,
    min_segment_size: int | None = None,
    changepoint_model: str | None = None,
) -> list[int]:
    effective_penalty = penalty if penalty is not None else discovery.penalty
    effective_min_size = (
        min_segment_size
        if min_segment_size is not None
        else discovery.min_segment_size
    )
    effective_model = changepoint_model or discovery.changepoint_model

    if discovery.segment_by_case:
        if case_boundaries is None:
            raise ValueError("case_boundaries are required for validation")
        boundaries = detect_changepoints_by_case(
            data,
            case_boundaries,
            effective_penalty,
            effective_min_size,
            model=effective_model,
        )
    else:
        boundaries = detect_changepoints(
            data,
            effective_penalty,
            effective_min_size,
            model=effective_model,
        )
        if case_boundaries is not None:
            boundaries = add_required_boundaries(boundaries, case_boundaries)

    if discovery.merge_short_segments:
        boundaries = merge_short_segments(
            boundaries,
            effective_min_size,
            required_boundaries=case_boundaries,
        )
    return boundaries


def _build_rule_feature_space(
    raw_profiles: list[SegmentProfile],
    raw_profile_names: list[str],
    raw_feature_matrix: np.ndarray,
    config: PipelineConfig,
) -> RuleFeatureSpace:
    if config.activity_abstraction == "interval":
        return RuleFeatureSpace(
            profiles=raw_profiles,
            names=raw_profile_names,
            matrix=raw_feature_matrix,
        )
    if config.activity_abstraction != "clustered_interval":
        raise ValueError(
            "activity_abstraction must be 'interval' or 'clustered_interval'"
        )
    if config.n_activity_clusters is None:
        raise ValueError(
            "n_activity_clusters is required for clustered_interval"
        )
    if config.n_activity_clusters < 2:
        raise ValueError("n_activity_clusters must be at least 2")
    if raw_feature_matrix.shape[0] < config.n_activity_clusters:
        raise ValueError("n_activity_clusters cannot exceed the segment count")
    if config.max_profile_features is not None and config.max_profile_features < 1:
        raise ValueError("max_profile_features must be positive")

    variances = raw_feature_matrix.var(axis=0)
    selected = np.flatnonzero(variances > config.profile_variance_threshold)
    if selected.size == 0:
        selected = np.arange(raw_feature_matrix.shape[1])
    if (
        config.max_profile_features is not None
        and selected.size > config.max_profile_features
    ):
        selected = _select_high_variance_features(
            variances,
            selected,
            config.max_profile_features,
        )

    selected_matrix = raw_feature_matrix[:, selected]
    means, scales = _feature_scaling_parameters(
        selected_matrix,
        config.profile_scaling,
    )
    matrix = (selected_matrix - means) / scales
    profiles = [
        SegmentProfile(lo=matrix[idx], hi=matrix[idx])
        for idx in range(matrix.shape[0])
    ]
    names = [f"z({raw_profile_names[idx]})" for idx in selected]
    return RuleFeatureSpace(
        profiles=profiles,
        names=names,
        matrix=matrix,
        selected_indices=selected,
        means=means,
        scales=scales,
    )


def _apply_rule_feature_space(
    raw_profiles: list[SegmentProfile],
    raw_feature_matrix: np.ndarray,
    discovery: PipelineResult,
) -> RuleFeatureSpace:
    if discovery.activity_abstraction == "interval":
        return RuleFeatureSpace(
            profiles=raw_profiles,
            names=discovery.profile_names,
            matrix=raw_feature_matrix,
        )

    selected = discovery.selected_profile_features
    means = discovery.profile_feature_means
    scales = discovery.profile_feature_scales
    if selected is None or means is None or scales is None:
        raise ValueError("clustered feature-space parameters are missing")

    selected_matrix = raw_feature_matrix[:, selected]
    matrix = (selected_matrix - means) / scales
    profiles = [
        SegmentProfile(lo=matrix[idx], hi=matrix[idx])
        for idx in range(matrix.shape[0])
    ]
    return RuleFeatureSpace(
        profiles=profiles,
        names=discovery.profile_names,
        matrix=matrix,
        selected_indices=selected,
        means=means,
        scales=scales,
    )


def _select_high_variance_features(
    variances: np.ndarray,
    selected: np.ndarray,
    max_features: int,
) -> np.ndarray:
    order = np.argsort(variances[selected])[::-1]
    top = selected[order[:max_features]]
    return np.array(sorted(top.tolist()), dtype=int)


def _feature_scaling_parameters(
    matrix: np.ndarray,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray]:
    if scaling == "standard":
        centers = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
    elif scaling == "robust":
        centers = np.median(matrix, axis=0)
        q75 = np.percentile(matrix, 75, axis=0)
        q25 = np.percentile(matrix, 25, axis=0)
        scales = q75 - q25
        fallback = matrix.std(axis=0)
        scales[scales == 0.0] = fallback[scales == 0.0]
    else:
        raise ValueError("profile_scaling must be 'standard' or 'robust'")

    scales[scales == 0.0] = 1.0
    return centers, scales


def _activity_classes(
    rule_space: RuleFeatureSpace,
    config: PipelineConfig,
) -> list[list[int]]:
    if config.activity_abstraction == "interval":
        return interval_equivalence_classes(rule_space.profiles)

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    from sklearn.cluster import KMeans

    labels = KMeans(
        n_clusters=config.n_activity_clusters,
        random_state=config.profile_cluster_random_state,
        n_init=20,
    ).fit_predict(rule_space.matrix)
    classes = [
        np.flatnonzero(labels == class_id).astype(int).tolist()
        for class_id in range(config.n_activity_clusters or 0)
    ]
    return merge_overlapping_classes(rule_space.profiles, classes)
