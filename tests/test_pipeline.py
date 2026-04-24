"""Test suite for the sensor stream process discovery pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from main import PipelineConfig, PipelineResult, run_pipeline
from src.synthesis import evaluate_rule
from tests.datagen import generate_synthetic_data


def _well_separated_regimes() -> (
    tuple[list[tuple[np.ndarray, np.ndarray]], int]
):
    """Return three well-separated 2-D regimes and n_vars.

    Returns
    -------
    tuple[list[tuple[np.ndarray, np.ndarray]], int]
        ``(regimes, n_vars)``
    """
    regimes = [
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([5.0, 5.0]), np.array([6.0, 6.0])),
        (np.array([10.0, 10.0]), np.array([11.0, 11.0])),
    ]
    return regimes, 2


def _run_clean(seed: int = 42) -> tuple[PipelineResult, int]:
    """Run the pipeline on clean, well-separated synthetic data.

    Returns
    -------
    tuple[PipelineResult, int]
        ``(result, n_true_regimes)``
    """
    regimes, n_vars = _well_separated_regimes()
    sd = generate_synthetic_data(
        seed=seed,
        n_vars=n_vars,
        regimes=regimes,
        regime_sequence=[0, 1, 2, 0, 1],
        points_per_segment=200,
    )
    config = PipelineConfig(
        data=sd.data,
        seed=seed,
        n_sensors=n_vars,
        penalty=1.0,
        min_segment_size=2,
    )
    return run_pipeline(config), len(regimes)


class TestRoundTrip:
    """Round-trip correctness on clean, well-separated data."""

    def test_correct_number_of_regimes(self) -> None:
        """Pipeline recovers the correct number of regimes."""
        result, n_true = _run_clean()
        assert result.n_classes == n_true

    def test_boundary_accuracy(self) -> None:
        """Detected boundaries are within tolerance of ground truth."""
        regimes, n_vars = _well_separated_regimes()
        sd = generate_synthetic_data(
            seed=42,
            n_vars=n_vars,
            regimes=regimes,
            regime_sequence=[0, 1, 2, 0, 1],
            points_per_segment=200,
        )
        config = PipelineConfig(
            data=sd.data, seed=42, n_sensors=n_vars, penalty=1.0,
        )
        result = run_pipeline(config)
        true_boundaries = set(sd.boundaries)
        detected = set(result.boundaries)
        tolerance = 5
        for tb in true_boundaries:
            assert any(
                abs(tb - db) <= tolerance for db in detected
            ), f"True boundary {tb} not found near any detected boundary"

    def test_rules_consistent_with_ground_truth(self) -> None:
        """Each synthesised rule covers data from exactly one true regime."""
        regimes, n_vars = _well_separated_regimes()
        sd = generate_synthetic_data(
            seed=42,
            n_vars=n_vars,
            regimes=regimes,
            regime_sequence=[0, 1, 2, 0, 1],
            points_per_segment=200,
        )
        config = PipelineConfig(
            data=sd.data, seed=42, n_sensors=n_vars, penalty=1.0,
        )
        result = run_pipeline(config)

        for rule in result.rules:
            covered_regimes: set[int] = set()
            for seg_idx, label in enumerate(sd.labels):
                start = sd.boundaries[seg_idx]
                end = sd.boundaries[seg_idx + 1]
                for t in range(start, end):
                    if evaluate_rule(rule, sd.data[t]):
                        covered_regimes.add(label)
                        break
            assert len(covered_regimes) == 1, (
                f"Rule {rule.class_id} covers data from "
                f"{len(covered_regimes)} true regimes (expected 1)"
            )


class TestOverSegmentationAbsorption:
    """Phase 2 merges spurious segments from over-segmentation."""

    def test_spurious_segments_merged(self) -> None:
        """Over-segmented data yields the correct regime count."""
        regimes = [
            (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            (np.array([5.0, 5.0]), np.array([6.0, 6.0])),
        ]
        sd = generate_synthetic_data(
            seed=7,
            n_vars=2,
            regimes=regimes,
            regime_sequence=[0, 1, 0],
            points_per_segment=200,
        )
        config = PipelineConfig(
            data=sd.data,
            seed=7,
            n_sensors=2,
            penalty=0.1,
            min_segment_size=20,
        )
        result = run_pipeline(config)
        assert len(result.boundaries) > len(sd.boundaries), (
            "Expected PELT to over-segment but it did not"
        )
        assert result.n_classes == len(regimes), (
            f"Expected {len(regimes)} classes but got {result.n_classes}"
        )


class TestMutualExclusivity:
    """No two rules fire on the same data point."""

    def test_no_overlap(self) -> None:
        """Every data point satisfies at most one rule."""
        result, _ = _run_clean()
        regimes, n_vars = _well_separated_regimes()
        sd = generate_synthetic_data(
            seed=42,
            n_vars=n_vars,
            regimes=regimes,
            regime_sequence=[0, 1, 2, 0, 1],
            points_per_segment=200,
        )
        for t in range(sd.data.shape[0]):
            count = sum(
                1 for r in result.rules if evaluate_rule(r, sd.data[t])
            )
            assert count <= 1, (
                f"Time point {t} covered by {count} rules (expected <= 1)"
            )


class TestCoverage:
    """Every data point is covered by exactly one rule."""

    def test_full_coverage(self) -> None:
        """Every data point satisfies exactly one rule."""
        result, _ = _run_clean()
        regimes, n_vars = _well_separated_regimes()
        sd = generate_synthetic_data(
            seed=42,
            n_vars=n_vars,
            regimes=regimes,
            regime_sequence=[0, 1, 2, 0, 1],
            points_per_segment=200,
        )
        for t in range(sd.data.shape[0]):
            count = sum(
                1 for r in result.rules if evaluate_rule(r, sd.data[t])
            )
            assert count == 1, (
                f"Time point {t} covered by {count} rules (expected 1)"
            )


class TestDeterminism:
    """Pipeline produces identical results across runs with the same seed."""

    def test_identical_results(self) -> None:
        """Two runs with the same config yield equal traces and rules."""
        result1, _ = _run_clean(seed=99)
        result2, _ = _run_clean(seed=99)

        assert result1.traces == result2.traces
        assert result1.n_classes == result2.n_classes
        for r1, r2 in zip(result1.rules, result2.rules):
            assert r1.lo == r2.lo
            assert r1.hi == r2.hi
            assert r1.class_id == r2.class_id
