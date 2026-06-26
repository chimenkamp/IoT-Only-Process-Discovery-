from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normaliser:
    """Min-max preprocessing fitted once on a raw sensor log."""

    lo: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normaliser":
        if data.ndim != 2:
            raise ValueError("data must be a 2-D array")
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        span = hi - lo
        span[span == 0.0] = 1.0
        return cls(lo=lo, span=span)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] != self.lo.shape[0]:
            raise ValueError("data column count does not match normaliser")
        return (data - self.lo) / self.span
