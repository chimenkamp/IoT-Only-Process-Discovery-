from __future__ import annotations

import numpy as np

from src.discovery import build_label_map, rule_label
from src.pipeline import PipelineConfig, run_pipeline, validate_sensor_log
from src.synthesis import IntervalRule


def test_pipeline_builds_rule_log_and_model_from_signature_segments() -> None:
    data, case_boundaries = _paper_example_log(n_cases=2)

    result = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        var_names=["position", "pressure"],
        case_boundaries=case_boundaries,
        signature_profile_radius=0.0,
    ))

    assert result.n_classes == 3
    assert result.traces == [[0, 1, 2], [0, 1, 2]]
    assert result.uncovered_segments == []
    assert result.ambiguous_segments == []
    assert len(result.event_log) == 2
    assert result.event_log[0][0].start == 0
    assert result.event_log[0][-1].end == case_boundaries[1]
    assert any(
        name.startswith("signed_area(")
        for name in result.profile_names
    )


def test_case_boundaries_are_hard_segment_boundaries() -> None:
    data, case_boundaries = _paper_example_log(n_cases=2)

    result = run_pipeline(PipelineConfig(
        data=data,
        penalty=100.0,
        min_segment_size=10,
        case_boundaries=case_boundaries,
    ))

    assert set(case_boundaries).issubset(set(result.boundaries))
    assert len(result.traces) == 2


def test_validation_reuses_learned_feature_space_and_model() -> None:
    data, case_boundaries = _paper_example_log(n_cases=2)
    discovery = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        case_boundaries=case_boundaries,
    ))

    validation = validate_sensor_log(
        data,
        discovery,
        case_boundaries=case_boundaries,
    )

    assert validation.uncovered_segments == []
    assert validation.ambiguous_segments == []
    assert validation.conforming


def test_validation_fitness_uses_assigned_activity_alphabet() -> None:
    train_case = np.vstack([
        np.tile([0.0, 0.0], (20, 1)),
        np.tile([1.0, 0.0], (20, 1)),
        np.tile([2.0, 0.0], (20, 1)),
    ])
    validation_case = np.vstack([
        np.tile([0.0, 0.0], (20, 1)),
        np.tile([1.0, 0.0], (20, 1)),
        np.tile([3.0, 0.0], (20, 1)),
    ])
    discovery = run_pipeline(PipelineConfig(
        data=train_case,
        penalty=0.01,
        min_segment_size=10,
        case_boundaries=[0, len(train_case)],
        activity_abstraction="interval",
    ))

    validation = validate_sensor_log(
        validation_case,
        discovery,
        case_boundaries=[0, len(validation_case)],
    )

    assert validation.uncovered_segments
    assert validation.segment_labels == [0, 1, -1]
    assert validation.model_fitness > 0.0
    assert not validation.conforming


def test_clustered_interval_abstraction_reuses_rule_feature_space() -> None:
    data, case_boundaries = _paper_example_log(n_cases=3)
    discovery = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        case_boundaries=case_boundaries,
        signature_depth=1,
        activity_abstraction="clustered_interval",
        n_activity_clusters=3,
    ))

    validation = validate_sensor_log(
        data,
        discovery,
        case_boundaries=case_boundaries,
    )

    assert discovery.n_classes == 3
    assert discovery.selected_profile_features is not None
    assert validation.uncovered_segments == []
    assert validation.ambiguous_segments == []
    assert validation.conforming


def test_case_local_segmentation_and_short_segment_merge() -> None:
    case = np.vstack([
        np.tile([0.0, 0.0], (20, 1)),
        np.tile([1.0, 0.0], (20, 1)),
    ])
    data = np.vstack([case, case])
    case_boundaries = [0, len(case), len(data)]

    result = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.01,
        min_segment_size=10,
        case_boundaries=case_boundaries,
        segment_by_case=True,
        merge_short_segments=True,
    ))

    assert set(case_boundaries).issubset(result.boundaries)
    assert min(
        right - left
        for left, right in zip(result.boundaries, result.boundaries[1:])
    ) >= 10


def test_signature_debug_path_exports_image(tmp_path) -> None:
    data, case_boundaries = _paper_example_log(n_cases=1)
    output_path = tmp_path / "debug.png"

    run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        case_boundaries=case_boundaries,
        signature_debug_path=str(output_path),
    ))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_large_rule_labels_are_compacted_for_visualization() -> None:
    rule = IntervalRule(
        lo=[0.0] * 100,
        hi=[1.0] * 100,
        class_id=7,
    )
    feature_names = [f"feature_{idx}" for idx in range(100)]

    label = rule_label(rule, feature_names)

    assert label.startswith("R7:")
    assert "... (98 more)" in label
    assert len(label) < 220


def test_generic_activity_labels_and_legend_are_exported(tmp_path) -> None:
    data, case_boundaries = _paper_example_log(n_cases=2)
    legend_path = tmp_path / "legend.md"

    result = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        case_boundaries=case_boundaries,
        activity_label_style="generic",
        activity_legend_path=str(legend_path),
    ))
    label_map = build_label_map(
        result.rules,
        result.profile_names,
        label_style="generic",
    )

    assert label_map[0] == "A01"
    assert legend_path.exists()
    assert "## A01" in legend_path.read_text(encoding="utf-8")


def _paper_example_log(n_cases: int) -> tuple[np.ndarray, list[int]]:
    case = np.vstack([
        np.tile([0.0, 0.0], (40, 1)),
        np.tile([1.0, 0.0], (40, 1)),
        np.tile([1.0, 1.0], (40, 1)),
    ])
    data = np.vstack([case for _ in range(n_cases)])
    case_boundaries = [idx * len(case) for idx in range(n_cases + 1)]
    return data, case_boundaries
