"""Deterministic synthetic data generator for pipeline tests.

Generates multivariate sensor logs with known ground-truth regime
structure for testing the process discovery pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic sensor log with ground-truth annotations.

    Parameters
    ----------
    data : np.ndarray
        Sensor log of shape ``(N, n_vars)``.
    boundaries : list[int]
        Ground-truth segment boundaries ``[0, b1, b2, ..., N]``.
    labels : list[int]
        Ground-truth regime label for each segment.
    regimes : list[tuple[np.ndarray, np.ndarray]]
        ``(lo, hi)`` bounds defining each regime's hyperrectangle.
    """

    data: np.ndarray
    boundaries: list[int]
    labels: list[int]
    regimes: list[tuple[np.ndarray, np.ndarray]]


def generate_synthetic_data(
    seed: int,
    n_vars: int,
    regimes: list[tuple[np.ndarray, np.ndarray]],
    regime_sequence: list[int],
    points_per_segment: int,
) -> SyntheticData:
    """Generate a synthetic multivariate sensor log.

    Samples sensor values uniformly within each regime's hyper-
    rectangular region according to the given regime sequence.

    Parameters
    ----------
    seed : int
        Seed for ``numpy.random.Generator`` (full reproducibility).
    n_vars : int
        Number of sensor variables.
    regimes : list[tuple[np.ndarray, np.ndarray]]
        Each element is ``(lo, hi)`` with arrays of shape
        ``(n_vars,)`` defining the hyper-rectangular region for
        one regime.
    regime_sequence : list[int]
        Sequence of regime indices defining the temporal order of
        segments.  Each entry indexes into *regimes*.
    points_per_segment : int
        Number of time points per segment.

    Returns
    -------
    SyntheticData
        Generated sensor log together with ground-truth annotations.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    segments: list[np.ndarray] = []
    boundaries: list[int] = [0]
    labels: list[int] = []

    for regime_idx in regime_sequence:
        lo, hi = regimes[regime_idx]
        segment = rng.uniform(lo, hi, size=(points_per_segment, n_vars))
        segments.append(segment)
        boundaries.append(boundaries[-1] + points_per_segment)
        labels.append(regime_idx)

    data = np.vstack(segments)
    return SyntheticData(
        data=data,
        boundaries=boundaries,
        labels=labels,
        regimes=regimes,
    )
