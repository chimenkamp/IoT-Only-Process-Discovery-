"""Sensor stream process discovery pipeline.

Exposes ``PipelineConfig`` and ``run_pipeline`` as the single entry
point for the three-phase synthesis procedure: change point detection,
abductive segment merging, and SyGuS-based rule synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.changepoint import detect_changepoints
from src.merging import (
    SegmentProfile,
    build_compatibility_graph,
    compute_profiles,
    minimum_clique_cover,
)
from src.discovery import (
    DiscoveryAlgorithm,
    discover_model,
    save_model_visualization,
)
from src.smt import get_solver
from src.synthesis import IntervalRule, synthesize_rules
from src.trace import build_traces


@dataclass
class PipelineConfig:
    """Configuration for the sensor stream process discovery pipeline.

    Parameters
    ----------
    data : np.ndarray
        Sensor log of shape ``(N, n_sensors)``.
    seed : int
        Random seed for reproducibility.
    n_sensors : int
        Number of sensor variables (must match ``data.shape[1]``).
    penalty : float
        PELT penalty parameter.  Lower values produce more change
        points (more over-segmentation).
    min_segment_size : int
        Minimum number of samples between two change points.
    smt_timeout_ms : int | None
        SMT solver timeout in milliseconds.  ``None`` means no timeout.
    var_names : list[str] | None
        Optional sensor variable names.  When ``None``, names are
        generated as ``v0, v1, ...``.
    discovery_algorithm : DiscoveryAlgorithm
        Process discovery algorithm to apply (default: Inductive Miner).
    output_path : str | None
        When set, save a Petri net visualisation to this file path
        (e.g. ``"model.png"``).
    """

    data: np.ndarray
    seed: int
    n_sensors: int
    penalty: float = 1.0
    min_segment_size: int = 2
    smt_timeout_ms: int | None = None
    var_names: list[str] | None = None
    discovery_algorithm: DiscoveryAlgorithm = DiscoveryAlgorithm.INDUCTIVE
    output_path: str | None = None


@dataclass
class PipelineResult:
    """Result of the sensor stream process discovery pipeline.

    Parameters
    ----------
    traces : list[list[int]]
        Event log — list of traces over the rule alphabet.
    rules : list[IntervalRule]
        Synthesised interval rules (one per equivalence class).
    boundaries : list[int]
        Change point boundaries from Phase 1.
    n_classes : int
        Number of equivalence classes.
    segment_labels : list[int]
        Equivalence-class assignment for each segment.
    profiles : list[SegmentProfile]
        Per-segment value profiles.
    net : object
        Discovered Petri net (pm4py ``PetriNet``).
    initial_marking : object
        Initial marking of the Petri net.
    final_marking : object
        Final marking of the Petri net.
    """

    traces: list[list[int]]
    rules: list[IntervalRule]
    boundaries: list[int]
    n_classes: int
    segment_labels: list[int]
    profiles: list[SegmentProfile]
    net: object = None
    initial_marking: object = None
    final_marking: object = None


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the three-phase synthesis pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration including the sensor log and all
        tuneable parameters.

    Returns
    -------
    PipelineResult
        Synthesised rules, event log, and intermediate results.

    Raises
    ------
    ValueError
        If ``config.data.shape[1] != config.n_sensors``.
    RuntimeError
        If discriminator synthesis fails for any class pair.
    """
    if config.data.shape[1] != config.n_sensors:
        raise ValueError(
            f"data has {config.data.shape[1]} columns but "
            f"n_sensors is {config.n_sensors}"
        )

    boundaries = detect_changepoints(
        config.data, config.penalty, config.min_segment_size,
    )

    profiles = compute_profiles(config.data, boundaries)
    adj = build_compatibility_graph(profiles)
    classes = minimum_clique_cover(adj, len(profiles))

    smt = get_solver()
    rules = synthesize_rules(
        smt, classes, profiles, config.n_sensors, config.smt_timeout_ms,
    )

    traces = build_traces(config.data, rules)

    segment_labels = [0] * len(profiles)
    for class_idx, members in enumerate(classes):
        for seg_idx in members:
            segment_labels[seg_idx] = class_idx

    net, im, fm = discover_model(
        traces, rules, config.discovery_algorithm, config.var_names,
    )

    if config.output_path is not None:
        save_model_visualization(net, im, fm, config.output_path)

    return PipelineResult(
        traces=traces,
        rules=rules,
        boundaries=boundaries,
        n_classes=len(classes),
        segment_labels=segment_labels,
        profiles=profiles,
        net=net,
        initial_marking=im,
        final_marking=fm,
    )

if __name__ == "__main__":
    from tests.datagen import generate_synthetic_data

    regimes = [
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([5.0, 5.0]), np.array([6.0, 6.0])),
        (np.array([10.0, 10.0]), np.array([11.0, 11.0])),
    ]
    sd = generate_synthetic_data(
        seed=42,
        n_vars=2,
        regimes=regimes,
        regime_sequence=[0, 1, 2, 0, 1],
        points_per_segment=200,
    )

    config = PipelineConfig(
        data=sd.data,
        seed=42,
        n_sensors=2,
        penalty=1.0,
        min_segment_size=2,
        discovery_algorithm=DiscoveryAlgorithm.INDUCTIVE,
        output_path="model.png",
    )
    result = run_pipeline(config)

    print(f"Detected {len(result.boundaries) - 1} segments "
          f"from {len(result.boundaries)} boundaries")
    print(f"Merged into {result.n_classes} equivalence classes")
    print()
    for rule in result.rules:
        lo = ", ".join(f"{v:.3f}" for v in rule.lo)
        hi = ", ".join(f"{v:.3f}" for v in rule.hi)
        print(f"  Rule {rule.class_id}: [{lo}] – [{hi}]")
    print()
    print(f"Event log ({len(result.traces)} trace(s)):")
    for i, trace in enumerate(result.traces):
        print(f"  Trace {i}: {trace}")
    print()
    print(f"Discovered Petri net: {len(result.net.transitions)} transitions, "
          f"{len(result.net.places)} places")
    if config.output_path:
        print(f"Model saved to {config.output_path}")
