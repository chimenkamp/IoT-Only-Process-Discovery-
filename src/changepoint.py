from __future__ import annotations

import numpy as np
import ruptures


def detect_changepoints(
    data: np.ndarray,
    penalty: float,
    min_size: int,
    model: str = "l2",
    downsample: int = 1,
) -> list[int]:
    """Detect change points in a multivariate sensor log using PELT.

    Parameters
    ----------
    data : np.ndarray
        Sensor log of shape ``(N, n_vars)`` where *N* is the number of
        time points and *n_vars* is the number of sensor variables.
    penalty : float
        PELT penalty parameter.  Lower values produce more change
        points (more over-segmentation).
    min_size : int
        Minimum number of samples between two change points.
    model : str
        Cost model for PELT (e.g. ``"l2"``, ``"l1"``, ``"rbf"``).
        ``"l2"`` is O(N) per segment and recommended for large data.
        ``"rbf"`` is O(N²) and much slower.
    downsample : int
        Take every *downsample*-th sample before running PELT.
        Boundaries are mapped back to the original indices.
        Use values > 1 to speed up detection on very long series.

    Returns
    -------
    list[int]
        Sorted boundary indices ``[0, cp1, cp2, ..., N]``.  The list
        always starts with ``0`` and ends with ``N``.
    """
    n = data.shape[0]

    if downsample > 1:
        data_ds = data[::downsample]
        min_size_ds = max(2, min_size // downsample)
    else:
        data_ds = data
        min_size_ds = min_size

    algo = ruptures.Pelt(model=model, min_size=min_size_ds).fit(data_ds)
    bkps = algo.predict(pen=penalty)

    if downsample > 1:
        # Map breakpoints back to original indices
        bkps = [min(b * downsample, n) for b in bkps]

    boundaries = sorted({0} | set(bkps))
    if boundaries[-1] != n:
        boundaries.append(n)
    return boundaries
