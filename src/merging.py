from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentProfile:
    """Interval abstraction of one segment feature vector."""

    lo: np.ndarray
    hi: np.ndarray


def interval_equivalence_classes(
    profiles: list[SegmentProfile],
) -> list[list[int]]:
    """Find the finest partition separable by interval predicates."""
    n = len(profiles)
    if n == 0:
        return []

    lo_arr = np.array([p.lo for p in profiles])
    hi_arr = np.array([p.hi for p in profiles])
    classes = [[idx] for idx in range(n)]

    changed = True
    while changed:
        changed = False
        parent = list(range(len(classes)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            nonlocal changed
            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                return
            parent[root_right] = root_left
            changed = True

        class_lo = np.array([
            np.min(lo_arr[members], axis=0)
            for members in classes
        ])
        class_hi = np.array([
            np.max(hi_arr[members], axis=0)
            for members in classes
        ])

        hits = []
        for lo, hi in zip(class_lo, class_hi):
            hits.append((
                np.maximum(lo, lo_arr) <= np.minimum(hi, hi_arr)
            ).all(axis=1))

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                if hits[i][classes[j]].any() or hits[j][classes[i]].any():
                    union(i, j)

        if changed:
            grouped: dict[int, list[int]] = defaultdict(list)
            for idx, members in enumerate(classes):
                grouped[find(idx)].extend(members)
            classes = [
                sorted(members)
                for _, members in sorted(grouped.items())
            ]

    return classes


def merge_overlapping_classes(
    profiles: list[SegmentProfile],
    classes: list[list[int]],
) -> list[list[int]]:
    """Merge pre-clustered classes until interval predicates are separable."""
    if not classes:
        return []
    if any(not members for members in classes):
        raise ValueError("classes must not be empty")

    lo_arr = np.array([p.lo for p in profiles])
    hi_arr = np.array([p.hi for p in profiles])
    merged = [sorted(members) for members in classes]

    changed = True
    while changed:
        changed = False
        parent = list(range(len(merged)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            nonlocal changed
            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                return
            parent[root_right] = root_left
            changed = True

        class_lo = np.array([
            np.min(lo_arr[members], axis=0)
            for members in merged
        ])
        class_hi = np.array([
            np.max(hi_arr[members], axis=0)
            for members in merged
        ])

        hits = []
        for lo, hi in zip(class_lo, class_hi):
            hits.append((
                np.maximum(lo, lo_arr) <= np.minimum(hi, hi_arr)
            ).all(axis=1))

        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if hits[i][merged[j]].any() or hits[j][merged[i]].any():
                    union(i, j)

        if changed:
            grouped: dict[int, list[int]] = defaultdict(list)
            for idx, members in enumerate(merged):
                grouped[find(idx)].extend(members)
            merged = [
                sorted(members)
                for _, members in sorted(grouped.items())
            ]

    return merged
