from __future__ import annotations

import numpy as np

from src.pipeline import PipelineConfig, run_pipeline
from src.playout import (
    PlayoutConfig,
    build_playout_support,
    compare_case_area_distributions,
    compare_sensor_value_distributions,
    playout_sensor_log,
    sample_activity_features,
)


def test_playout_generates_sensor_log_from_discovery() -> None:
    data, case_boundaries = _paper_example_log(n_cases=3)
    discovery = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        var_names=["position", "pressure"],
        case_boundaries=case_boundaries,
    ))

    generated = playout_sensor_log(
        discovery,
        PlayoutConfig(n_cases=4, random_state=7),
    )

    assert len(generated.traces) == 4
    assert generated.data.shape[1] == 2
    assert generated.raw_data.shape == generated.data.shape
    assert generated.case_boundaries[0] == 0
    assert generated.case_boundaries[-1] == generated.data.shape[0]
    assert np.isfinite(generated.data).all()
    assert generated.data.min() >= 0.0
    assert generated.data.max() <= 1.0


def test_smt_feature_sampler_satisfies_rule_constraints() -> None:
    data, case_boundaries = _paper_example_log(n_cases=4)
    discovery = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        var_names=["position", "pressure"],
        case_boundaries=case_boundaries,
        activity_abstraction="clustered_interval",
        n_activity_clusters=3,
        max_profile_features=12,
        profile_scaling="robust",
    ))
    support = build_playout_support(discovery)
    activity = discovery.rules[0].class_id

    features = sample_activity_features(
        discovery,
        support,
        activity,
        np.random.default_rng(13),
        sampler="smt",
    )

    _assert_features_satisfy_rule(discovery, activity, features)
    for sensor in discovery.sensor_names:
        start = features[f"start({sensor})"]
        end = features[f"end({sensor})"]
        min_value = features[f"min({sensor})"]
        max_value = features[f"max({sensor})"]
        signature = features[f"sig({sensor})"]
        assert min_value <= start <= max_value
        assert min_value <= end <= max_value
        assert np.isclose(signature, end - start)


def test_playout_is_reproducible_with_fixed_seed() -> None:
    data, case_boundaries = _paper_example_log(n_cases=3)
    discovery = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        var_names=["position", "pressure"],
        case_boundaries=case_boundaries,
    ))

    config = PlayoutConfig(n_cases=4, random_state=7)
    first = playout_sensor_log(discovery, config)
    second = playout_sensor_log(discovery, config)

    assert first.traces == second.traces
    assert first.case_boundaries == second.case_boundaries
    np.testing.assert_allclose(first.data, second.data)


def test_playout_distribution_comparison_helpers() -> None:
    data, case_boundaries = _paper_example_log(n_cases=3)
    discovery = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        var_names=["position", "pressure"],
        case_boundaries=case_boundaries,
    ))
    generated = playout_sensor_log(
        discovery,
        PlayoutConfig(n_cases=3, random_state=3),
    )

    value_comparison = compare_sensor_value_distributions(
        discovery.preprocessed_data,
        generated.data,
        discovery.sensor_names,
    )
    area_comparison = compare_case_area_distributions(
        discovery.preprocessed_data,
        case_boundaries,
        generated.data,
        generated.case_boundaries,
        discovery.sensor_names,
    )

    assert set(value_comparison["sensor"]) == {"position", "pressure"}
    assert set(area_comparison["sensor"]) == {"position", "pressure"}
    assert (value_comparison["wasserstein_1d"] >= 0.0).all()
    assert (area_comparison["area_wasserstein_1d"] >= 0.0).all()


def _paper_example_log(n_cases: int) -> tuple[np.ndarray, list[int]]:
    case = np.vstack([
        np.tile([0.0, 0.0], (40, 1)),
        np.tile([1.0, 0.0], (40, 1)),
        np.tile([1.0, 1.0], (40, 1)),
    ])
    data = np.vstack([case for _ in range(n_cases)])
    case_boundaries = [idx * len(case) for idx in range(n_cases + 1)]
    return data, case_boundaries


def _assert_features_satisfy_rule(discovery, activity: int, features: dict[str, float]) -> None:
    rule = discovery.rules[activity]
    for idx, (lo, hi, name) in enumerate(
        zip(rule.lo, rule.hi, discovery.profile_names)
    ):
        raw_name = _unwrap_scaled_feature_name(name)
        value = features[raw_name]
        if (
            discovery.profile_feature_means is not None
            and discovery.profile_feature_scales is not None
        ):
            value = (
                (value - discovery.profile_feature_means[idx])
                / discovery.profile_feature_scales[idx]
            )
        assert lo - 1e-8 <= value <= hi + 1e-8


def _unwrap_scaled_feature_name(name: str) -> str:
    if name.startswith("z(") and name.endswith(")"):
        return name[2:-1]
    return name
