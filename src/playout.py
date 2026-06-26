from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
try:
    import z3
except ImportError:  # pragma: no cover - exercised only without optional dep.
    z3 = None

from src.discovery import build_label_map
from src.pipeline import PipelineResult
from src.signatures import segment_signature_feature_matrix


@dataclass(frozen=True)
class PlayoutConfig:
    """Configuration for model playout and sensor-log generation."""

    n_cases: int = 10
    max_trace_length: int = 80
    min_segment_length: int = 3
    max_segment_length: int = 300
    noise_scale: float = 0.04
    random_state: int | None = 0
    feature_sampler: str = "smt"
    smt_timeout_ms: int = 500


@dataclass(frozen=True)
class GeneratedSegment:
    """Metadata for one generated sensor segment."""

    case_id: int
    activity: int
    start: int
    end: int
    sampled_length: int


@dataclass(frozen=True)
class GeneratedSensorLog:
    """Generated sensor log from a discovered process model."""

    data: np.ndarray
    raw_data: np.ndarray
    case_boundaries: list[int]
    traces: list[list[int]]
    segments: list[GeneratedSegment]
    sensor_names: list[str]


@dataclass(frozen=True)
class PlayoutSupport:
    """Precomputed empirical and feature-space support for generation."""

    raw_feature_names: list[str]
    raw_feature_matrix: np.ndarray
    empirical_feature_rows: dict[int, np.ndarray]
    empirical_lengths: dict[int, np.ndarray]
    label_to_activity: dict[str, int]


def playout_sensor_log(
    discovery: PipelineResult,
    config: PlayoutConfig | None = None,
) -> GeneratedSensorLog:
    """Generate a sensor log by replaying the Petri net and sampling rules.

    The generated data is not meant to reproduce the original trace point by
    point. It samples feature vectors inside the learned rule alphabet and
    reconstructs approximate segment paths that respect envelope constraints
    and first-order signature directions where those features are available.
    """
    cfg = config or PlayoutConfig()
    rng = np.random.default_rng(cfg.random_state)
    support = build_playout_support(discovery)
    traces = playout_activity_traces(
        discovery,
        n_cases=cfg.n_cases,
        max_trace_length=cfg.max_trace_length,
        rng=rng,
        label_to_activity=support.label_to_activity,
    )

    generated_segments: list[np.ndarray] = []
    case_boundaries = [0]
    segment_metadata: list[GeneratedSegment] = []

    for case_id, trace in enumerate(traces):
        for activity in trace:
            features = sample_activity_features(
                discovery,
                support,
                activity,
                rng,
                sampler=cfg.feature_sampler,
                smt_timeout_ms=cfg.smt_timeout_ms,
            )
            length = sample_segment_length(
                support,
                activity,
                cfg.min_segment_length,
                cfg.max_segment_length,
                rng,
            )
            start = sum(segment.shape[0] for segment in generated_segments)
            segment = synthesize_segment_path(
                features,
                discovery.sensor_names,
                length,
                rng,
                noise_scale=cfg.noise_scale,
            )
            generated_segments.append(segment)
            segment_metadata.append(GeneratedSegment(
                case_id=case_id,
                activity=activity,
                start=start,
                end=start + segment.shape[0],
                sampled_length=length,
            ))
        case_boundaries.append(
            sum(segment.shape[0] for segment in generated_segments)
        )

    data = (
        np.vstack(generated_segments)
        if generated_segments
        else np.empty((0, len(discovery.sensor_names)), dtype=float)
    )
    raw_data = inverse_normalise(data, discovery)
    return GeneratedSensorLog(
        data=data,
        raw_data=raw_data,
        case_boundaries=case_boundaries,
        traces=traces,
        segments=segment_metadata,
        sensor_names=discovery.sensor_names,
    )


def build_playout_support(discovery: PipelineResult) -> PlayoutSupport:
    """Build feature-space and empirical support used during playout."""
    raw_features = segment_signature_feature_matrix(
        discovery.preprocessed_data,
        discovery.boundaries,
        var_names=discovery.sensor_names,
        signature_depth=discovery.signature_depth,
        include_derivative_features=discovery.include_derivative_features,
    )

    labels = np.array(discovery.segment_labels, dtype=int)
    empirical_feature_rows = {
        activity: raw_features.matrix[np.flatnonzero(labels == activity)]
        for activity in sorted(set(discovery.segment_labels))
        if activity >= 0
    }
    empirical_lengths: dict[int, list[int]] = {
        activity: [] for activity in empirical_feature_rows
    }
    for trace in discovery.event_log:
        for event in trace:
            empirical_lengths.setdefault(event.activity, []).append(
                event.end - event.start
            )

    label_map = build_label_map(
        discovery.rules,
        discovery.profile_names,
        label_style=discovery.activity_label_style,
        activity_prefix=discovery.activity_label_prefix,
    )
    return PlayoutSupport(
        raw_feature_names=raw_features.names,
        raw_feature_matrix=raw_features.matrix,
        empirical_feature_rows=empirical_feature_rows,
        empirical_lengths={
            activity: np.array(lengths, dtype=int)
            for activity, lengths in empirical_lengths.items()
        },
        label_to_activity={label: activity for activity, label in label_map.items()},
    )


def playout_activity_traces(
    discovery: PipelineResult,
    n_cases: int,
    max_trace_length: int,
    rng: np.random.Generator,
    label_to_activity: dict[str, int] | None = None,
) -> list[list[int]]:
    """Replay a Petri net stochastically and return visible activity traces."""
    support_labels = label_to_activity or build_playout_support(
        discovery,
    ).label_to_activity
    traces: list[list[int]] = []
    for _ in range(n_cases):
        marking = _marking_dict(discovery.initial_marking)
        trace: list[int] = []

        for _ in range(max_trace_length):
            if _marking_reached(marking, discovery.final_marking):
                break
            enabled = _enabled_transitions(discovery.net, marking)
            if not enabled:
                break
            transition = enabled[int(rng.integers(0, len(enabled)))]
            _fire_transition(marking, transition)
            if transition.label in support_labels:
                trace.append(support_labels[transition.label])

        traces.append(trace)
    return traces


def sample_activity_features(
    discovery: PipelineResult,
    support: PlayoutSupport,
    activity: int,
    rng: np.random.Generator,
    sampler: str = "smt",
    smt_timeout_ms: int = 500,
) -> dict[str, float]:
    """Sample one raw segment-feature dictionary for an activity rule."""
    if sampler == "smt":
        return sample_activity_features_smt(
            discovery,
            support,
            activity,
            rng,
            timeout_ms=smt_timeout_ms,
        )
    if sampler == "interval":
        return sample_activity_features_interval(
            discovery,
            support,
            activity,
            rng,
        )
    raise ValueError("feature_sampler must be 'smt' or 'interval'")


def sample_activity_features_interval(
    discovery: PipelineResult,
    support: PlayoutSupport,
    activity: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample features with independent uniform draws inside rule intervals."""
    empirical_rows = support.empirical_feature_rows.get(activity)
    if empirical_rows is None or empirical_rows.size == 0:
        base = np.zeros(len(support.raw_feature_names), dtype=float)
    else:
        base = empirical_rows[int(rng.integers(0, empirical_rows.shape[0]))].copy()

    raw_index = {
        name: idx for idx, name in enumerate(support.raw_feature_names)
    }
    rule = discovery.rules[activity]
    for idx, (lo, hi, name) in enumerate(
        zip(rule.lo, rule.hi, discovery.profile_names)
    ):
        sampled = _sample_interval(lo, hi, rng)
        raw_name = _unwrap_scaled_feature_name(name)
        if (
            discovery.profile_feature_means is not None
            and discovery.profile_feature_scales is not None
        ):
            sampled = (
                sampled * discovery.profile_feature_scales[idx]
                + discovery.profile_feature_means[idx]
            )
        if raw_name in raw_index:
            base[raw_index[raw_name]] = sampled

    return {
        name: float(value)
        for name, value in zip(support.raw_feature_names, base)
    }


def sample_activity_features_smt(
    discovery: PipelineResult,
    support: PlayoutSupport,
    activity: int,
    rng: np.random.Generator,
    timeout_ms: int = 500,
) -> dict[str, float]:
    """Sample features by solving rule and consistency constraints with Z3."""
    if z3 is None:
        raise RuntimeError(
            "SMT playout requires z3-solver. Install dependencies with "
            "`pip install -r requirements.txt`."
        )

    empirical_rows = support.empirical_feature_rows.get(activity)
    if empirical_rows is None or empirical_rows.size == 0:
        base = np.zeros(len(support.raw_feature_names), dtype=float)
    else:
        base = empirical_rows[int(rng.integers(0, empirical_rows.shape[0]))].copy()

    raw_index = {
        name: idx for idx, name in enumerate(support.raw_feature_names)
    }
    rule = discovery.rules[activity]
    rule_intervals = _rule_raw_intervals(discovery, rule)
    feature_names = _smt_feature_names(
        support.raw_feature_names,
        discovery.sensor_names,
        set(rule_intervals),
    )
    if not feature_names:
        return {
            name: float(value)
            for name, value in zip(support.raw_feature_names, base)
        }

    bounds = _activity_feature_bounds(support, activity, base)
    solver = z3.Optimize()
    solver.set(timeout=max(1, int(timeout_ms)))
    variables = {
        name: z3.Real(f"x_{idx}")
        for idx, name in enumerate(feature_names)
    }
    distances = []

    for idx, name in enumerate(feature_names):
        raw_pos = raw_index[name]
        lo, hi = bounds[raw_pos]
        if name in rule_intervals:
            rule_lo, rule_hi = rule_intervals[name]
            lo = max(lo, rule_lo)
            hi = min(hi, rule_hi)
            if hi < lo:
                lo, hi = rule_lo, rule_hi
        lo, hi = _stable_bounds(lo, hi, base[raw_pos])

        var = variables[name]
        solver.add(var >= _z3_real(lo))
        solver.add(var <= _z3_real(hi))

        if name in rule_intervals and hi > lo:
            target = float(rng.uniform(lo, hi))
        else:
            target = float(np.clip(base[raw_pos], lo, hi))
        distance = z3.Real(f"d_{idx}")
        solver.add(distance >= 0)
        solver.add(distance >= var - _z3_real(target))
        solver.add(distance >= _z3_real(target) - var)
        distances.append(distance)

    _add_path_consistency_constraints(
        solver,
        variables,
        discovery.sensor_names,
    )
    if distances:
        solver.minimize(z3.Sum(distances))

    result = solver.check()
    if result != z3.sat:
        raise RuntimeError(
            f"SMT feature sampling failed for activity {activity}: {result}"
        )

    model = solver.model()
    for name, var in variables.items():
        raw_pos = raw_index[name]
        base[raw_pos] = _z3_value_to_float(model.eval(var, model_completion=True))

    return {
        name: float(value)
        for name, value in zip(support.raw_feature_names, base)
    }


def sample_segment_length(
    support: PlayoutSupport,
    activity: int,
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> int:
    """Sample a segment length from empirical activity durations."""
    lengths = support.empirical_lengths.get(activity)
    if lengths is None or lengths.size == 0:
        return max(min_length, 3)
    length = int(lengths[int(rng.integers(0, lengths.size))])
    return int(np.clip(length, min_length, max_length))


def synthesize_segment_path(
    features: dict[str, float],
    sensor_names: list[str],
    length: int,
    rng: np.random.Generator,
    noise_scale: float = 0.04,
) -> np.ndarray:
    """Reconstruct an approximate normalized sensor path from features."""
    if length < 1:
        raise ValueError("length must be positive")

    segment = np.zeros((length, len(sensor_names)), dtype=float)
    for sensor_idx, sensor in enumerate(sensor_names):
        start = features.get(f"start({sensor})", 0.5)
        signature_delta = features.get(f"sig({sensor})")
        end = features.get(
            f"end({sensor})",
            start + signature_delta if signature_delta is not None else start,
        )
        min_value = features.get(f"min({sensor})", min(start, end))
        max_value = features.get(f"max({sensor})", max(start, end))
        min_value, max_value = sorted((min_value, max_value))
        min_value = float(np.clip(min_value, 0.0, 1.0))
        max_value = float(np.clip(max_value, 0.0, 1.0))
        if min_value == max_value:
            max_value = min(1.0, min_value + 1e-6)

        start = float(np.clip(start, min_value, max_value))
        end = float(np.clip(end, min_value, max_value))
        if signature_delta is not None:
            end = _align_end_with_signature_direction(
                start,
                end,
                signature_delta,
                min_value,
                max_value,
            )

        path = np.linspace(start, end, length)
        if length > 2 and max_value > min_value and noise_scale > 0.0:
            amplitude = (max_value - min_value) * noise_scale
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            wiggle = amplitude * np.sin(
                np.linspace(0.0, 2.0 * np.pi, length) + phase
            )
            path = np.clip(path + wiggle, min_value, max_value)
        segment[:, sensor_idx] = path

    return np.clip(segment, 0.0, 1.0)


def inverse_normalise(
    data: np.ndarray,
    discovery: PipelineResult,
) -> np.ndarray:
    """Map generated normalized data back to the discovery data scale."""
    return data * discovery.normaliser.span + discovery.normaliser.lo


def case_area_distribution(
    data: np.ndarray,
    case_boundaries: list[int],
    sensor_names: list[str],
) -> pd.DataFrame:
    """Return per-case area-under-curve values for each sensor."""
    rows: list[dict[str, float | int | str]] = []
    for case_id, (start, end) in enumerate(
        zip(case_boundaries, case_boundaries[1:])
    ):
        case = data[start:end]
        for sensor_idx, sensor in enumerate(sensor_names):
            rows.append({
                "case_id": case_id,
                "sensor": sensor,
                "area": float(np.trapezoid(case[:, sensor_idx]))
                if case.shape[0] > 1
                else float(case[:, sensor_idx].sum()),
                "mean": float(case[:, sensor_idx].mean())
                if case.shape[0]
                else 0.0,
            })
    return pd.DataFrame(rows)


def compare_sensor_value_distributions(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    sensor_names: list[str],
) -> pd.DataFrame:
    """Compare point-value distributions between real and generated sensors."""
    rows = []
    for sensor_idx, sensor in enumerate(sensor_names):
        real = real_data[:, sensor_idx]
        generated = generated_data[:, sensor_idx]
        rows.append({
            "sensor": sensor,
            "real_mean": float(real.mean()),
            "generated_mean": float(generated.mean()),
            "real_std": float(real.std()),
            "generated_std": float(generated.std()),
            "wasserstein_1d": wasserstein_1d(real, generated),
        })
    return pd.DataFrame(rows)


def compare_case_area_distributions(
    real_data: np.ndarray,
    real_case_boundaries: list[int],
    generated_data: np.ndarray,
    generated_case_boundaries: list[int],
    sensor_names: list[str],
) -> pd.DataFrame:
    """Compare case-level sensor-area distributions."""
    real_areas = case_area_distribution(
        real_data,
        real_case_boundaries,
        sensor_names,
    )
    generated_areas = case_area_distribution(
        generated_data,
        generated_case_boundaries,
        sensor_names,
    )
    rows = []
    for sensor in sensor_names:
        real = real_areas.loc[real_areas["sensor"] == sensor, "area"].to_numpy()
        generated = generated_areas.loc[
            generated_areas["sensor"] == sensor,
            "area",
        ].to_numpy()
        rows.append({
            "sensor": sensor,
            "real_area_mean": float(real.mean()),
            "generated_area_mean": float(generated.mean()),
            "real_area_std": float(real.std()),
            "generated_area_std": float(generated.std()),
            "area_wasserstein_1d": wasserstein_1d(real, generated),
        })
    return pd.DataFrame(rows)


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Return a simple empirical 1-D Wasserstein distance."""
    if left.size == 0 and right.size == 0:
        return 0.0
    if left.size == 0 or right.size == 0:
        return float("inf")
    n_quantiles = max(left.size, right.size)
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    left_q = np.quantile(left, quantiles)
    right_q = np.quantile(right, quantiles)
    return float(np.mean(np.abs(left_q - right_q)))


def _sample_interval(
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> float:
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Rule intervals must be finite for playout")
    if hi < lo:
        raise ValueError("Rule interval upper bound is below lower bound")
    if hi == lo:
        return float(lo)
    return float(rng.uniform(lo, hi))


def _unwrap_scaled_feature_name(name: str) -> str:
    if name.startswith("z(") and name.endswith(")"):
        return name[2:-1]
    return name


def _rule_raw_intervals(
    discovery: PipelineResult,
    rule: Any,
) -> dict[str, tuple[float, float]]:
    intervals: dict[str, tuple[float, float]] = {}
    for idx, (lo, hi, name) in enumerate(
        zip(rule.lo, rule.hi, discovery.profile_names)
    ):
        raw_name = _unwrap_scaled_feature_name(name)
        raw_lo = float(lo)
        raw_hi = float(hi)
        if (
            discovery.profile_feature_means is not None
            and discovery.profile_feature_scales is not None
        ):
            mean = float(discovery.profile_feature_means[idx])
            scale = float(discovery.profile_feature_scales[idx])
            if abs(scale) < 1e-12:
                raw_lo = mean
                raw_hi = mean
            else:
                endpoints = [raw_lo * scale + mean, raw_hi * scale + mean]
                raw_lo = min(endpoints)
                raw_hi = max(endpoints)

        if raw_name in intervals:
            prev_lo, prev_hi = intervals[raw_name]
            intervals[raw_name] = (max(prev_lo, raw_lo), min(prev_hi, raw_hi))
        else:
            intervals[raw_name] = (raw_lo, raw_hi)
    return intervals


def _smt_feature_names(
    raw_feature_names: list[str],
    sensor_names: list[str],
    rule_feature_names: set[str],
) -> list[str]:
    touched_sensors = {
        sensor
        for name in rule_feature_names
        for sensor in [_path_feature_sensor(name, sensor_names)]
        if sensor is not None
    }
    closure = set(rule_feature_names)
    for sensor in touched_sensors:
        closure.update({
            f"start({sensor})",
            f"end({sensor})",
            f"min({sensor})",
            f"max({sensor})",
            f"sig({sensor})",
        })
    return [name for name in raw_feature_names if name in closure]


def _path_feature_sensor(
    feature_name: str,
    sensor_names: list[str],
) -> str | None:
    for prefix in ("start(", "end(", "min(", "max(", "sig("):
        if feature_name.startswith(prefix) and feature_name.endswith(")"):
            inner = feature_name[len(prefix):-1]
            if "," not in inner and inner in sensor_names:
                return inner
    return None


def _activity_feature_bounds(
    support: PlayoutSupport,
    activity: int,
    base: np.ndarray,
) -> np.ndarray:
    rows = support.empirical_feature_rows.get(activity)
    if rows is None or rows.size == 0:
        rows = support.raw_feature_matrix
    if rows.size == 0:
        rows = base.reshape(1, -1)

    lo = np.nanmin(rows, axis=0).astype(float)
    hi = np.nanmax(rows, axis=0).astype(float)
    global_lo = np.nanmin(support.raw_feature_matrix, axis=0).astype(float)
    global_hi = np.nanmax(support.raw_feature_matrix, axis=0).astype(float)

    invalid = ~np.isfinite(lo) | ~np.isfinite(hi)
    lo[invalid] = global_lo[invalid]
    hi[invalid] = global_hi[invalid]
    invalid = ~np.isfinite(lo) | ~np.isfinite(hi)
    lo[invalid] = base[invalid]
    hi[invalid] = base[invalid]

    pad = np.maximum((hi - lo) * 1e-9, 1e-9)
    return np.column_stack([lo - pad, hi + pad])


def _stable_bounds(
    lo: float,
    hi: float,
    fallback: float,
) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        value = float(fallback) if np.isfinite(fallback) else 0.0
        return value, value
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _add_path_consistency_constraints(
    solver: Any,
    variables: dict[str, Any],
    sensor_names: list[str],
) -> None:
    for sensor in sensor_names:
        start = variables.get(f"start({sensor})")
        end = variables.get(f"end({sensor})")
        min_value = variables.get(f"min({sensor})")
        max_value = variables.get(f"max({sensor})")
        signature = variables.get(f"sig({sensor})")

        if min_value is not None and max_value is not None:
            solver.add(min_value <= max_value)
        if min_value is not None and start is not None:
            solver.add(min_value <= start)
        if max_value is not None and start is not None:
            solver.add(start <= max_value)
        if min_value is not None and end is not None:
            solver.add(min_value <= end)
        if max_value is not None and end is not None:
            solver.add(end <= max_value)
        if signature is not None and start is not None and end is not None:
            solver.add(signature == end - start)


def _z3_real(value: float) -> Any:
    return z3.RealVal(repr(float(value)))


def _z3_value_to_float(value: Any) -> float:
    if hasattr(value, "numerator_as_long") and hasattr(value, "denominator_as_long"):
        return value.numerator_as_long() / value.denominator_as_long()
    text = value.as_decimal(20) if hasattr(value, "as_decimal") else str(value)
    if text.endswith("?"):
        text = text[:-1]
    return float(text)


def _align_end_with_signature_direction(
    start: float,
    end: float,
    signature_delta: float,
    min_value: float,
    max_value: float,
) -> float:
    direction = np.sign(signature_delta)
    if direction == 0.0 or np.sign(end - start) == direction:
        return end
    distance = abs(end - start)
    if distance < 1e-6:
        distance = max((max_value - min_value) * 0.35, 1e-4)
    adjusted = start + direction * distance
    return float(np.clip(adjusted, min_value, max_value))


def _marking_dict(marking: Any) -> dict[Any, int]:
    return {
        place: int(tokens)
        for place, tokens in dict(marking).items()
        if int(tokens) > 0
    }


def _marking_reached(marking: dict[Any, int], final_marking: Any) -> bool:
    final = _marking_dict(final_marking)
    return all(marking.get(place, 0) >= tokens for place, tokens in final.items())


def _enabled_transitions(net: Any, marking: dict[Any, int]) -> list[Any]:
    enabled = [
        transition for transition in net.transitions
        if _transition_enabled(transition, marking)
    ]
    return sorted(enabled, key=_transition_sort_key)


def _transition_sort_key(transition: Any) -> tuple[Any, ...]:
    label = getattr(transition, "label", None)
    name = getattr(transition, "name", "")
    in_places = tuple(sorted(
        _node_name(arc.source) for arc in getattr(transition, "in_arcs", [])
    ))
    out_places = tuple(sorted(
        _node_name(arc.target) for arc in getattr(transition, "out_arcs", [])
    ))
    return (label is None, str(label or ""), str(name), in_places, out_places)


def _node_name(node: Any) -> str:
    return str(getattr(node, "name", node))


def _transition_enabled(transition: Any, marking: dict[Any, int]) -> bool:
    for arc in transition.in_arcs:
        if marking.get(arc.source, 0) < getattr(arc, "weight", 1):
            return False
    return True


def _fire_transition(marking: dict[Any, int], transition: Any) -> None:
    for arc in transition.in_arcs:
        place = arc.source
        marking[place] = marking.get(place, 0) - getattr(arc, "weight", 1)
        if marking[place] <= 0:
            marking.pop(place, None)
    for arc in transition.out_arcs:
        place = arc.target
        marking[place] = marking.get(place, 0) + getattr(arc, "weight", 1)
