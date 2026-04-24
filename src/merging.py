from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_MAX_CLIQUE_COVER_NODES = 500


@dataclass(frozen=True)
class SegmentProfile:
    """Per-variable min/max interval for a temporal segment.

    Parameters
    ----------
    lo : np.ndarray
        Lower bounds, shape ``(n_vars,)``.
    hi : np.ndarray
        Upper bounds, shape ``(n_vars,)``.
    """

    lo: np.ndarray
    hi: np.ndarray


def compute_profiles(
    data: np.ndarray,
    boundaries: list[int],
) -> list[SegmentProfile]:
    """Compute empirical value profiles for each segment.

    Parameters
    ----------
    data : np.ndarray
        Sensor log of shape ``(N, n_vars)``.
    boundaries : list[int]
        Sorted boundary indices ``[0, cp1, ..., N]``.

    Returns
    -------
    list[SegmentProfile]
        One profile per segment (``len(boundaries) - 1`` profiles).
    """
    profiles: list[SegmentProfile] = []
    for i in range(len(boundaries) - 1):
        segment = data[boundaries[i]:boundaries[i + 1]]
        profiles.append(SegmentProfile(
            lo=segment.min(axis=0),
            hi=segment.max(axis=0),
        ))
    return profiles


def build_compatibility_graph(profiles: list[SegmentProfile]) -> list[list[bool]]:
    """Build a compatibility graph over segment profiles.

    Two segments are compatible (connected by an edge) if and only if
    their per-variable intervals overlap on **every** dimension.  This
    means a single hyperrectangular predicate in the grammar can cover
    both segments simultaneously.

    Parameters
    ----------
    profiles : list[SegmentProfile]
        The segment profiles to compare.

    Returns
    -------
    list[list[bool]]
        Symmetric adjacency matrix where ``adj[a][b]`` is ``True``
        when segments *a* and *b* are compatible.
    """
    n = len(profiles)
    # Vectorised: stack all lo/hi into (n, d) arrays
    lo_arr = np.array([p.lo for p in profiles])  # (n, d)
    hi_arr = np.array([p.hi for p in profiles])  # (n, d)
    # Intervals overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b) for all dims
    # Broadcasting: (n,1,d) vs (1,n,d) → (n,n,d)
    overlap = (np.maximum(lo_arr[:, None, :], lo_arr[None, :, :])
               <= np.minimum(hi_arr[:, None, :], hi_arr[None, :, :]))
    compat = overlap.all(axis=2)  # (n, n) bool
    np.fill_diagonal(compat, True)
    return compat.tolist()


def minimum_clique_cover(
    adj: list[list[bool]],
    n: int,
) -> list[list[int]]:
    """Compute exact minimum clique cover of an undirected graph.

    Minimum clique cover of *G* equals the chromatic number of the
    complement graph.  This function builds the complement and solves
    exact graph coloring via backtracking with branch-and-bound,
    seeded by a greedy upper bound.

    Parameters
    ----------
    adj : list[list[bool]]
        Symmetric adjacency matrix of the compatibility graph (size
        ``n × n``).
    n : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Each inner list is an equivalence class (clique in the original
        graph) containing segment indices.

    Raises
    ------
    ValueError
        If *n* exceeds the hard-coded node limit.
    """
    if n > _MAX_CLIQUE_COVER_NODES:
        raise ValueError(
            f"Minimum clique cover requested for {n} nodes, "
            f"exceeding the limit of {_MAX_CLIQUE_COVER_NODES}."
        )
    if n == 0:
        return []

    comp = [
        [not adj[i][j] and i != j for j in range(n)]
        for i in range(n)
    ]

    greedy_colors = _greedy_coloring(comp, n)
    upper = max(greedy_colors) + 1

    best_k = [upper]
    best_coloring = greedy_colors[:]
    coloring = [-1] * n

    def _backtrack(idx: int, num_colors: int) -> None:
        """Assign color to vertex *idx*, pruning when hopeless."""
        if num_colors >= best_k[0]:
            return
        if idx == n:
            best_k[0] = num_colors
            best_coloring[:] = coloring[:]
            return

        used: set[int] = set()
        for u in range(idx):
            if comp[idx][u] and coloring[u] != -1:
                used.add(coloring[u])

        for c in range(num_colors):
            if c not in used:
                coloring[idx] = c
                _backtrack(idx + 1, num_colors)
                coloring[idx] = -1

        if num_colors + 1 < best_k[0]:
            coloring[idx] = num_colors
            _backtrack(idx + 1, num_colors + 1)
            coloring[idx] = -1

    _backtrack(0, 0)

    k = best_k[0]
    classes: list[list[int]] = [[] for _ in range(k)]
    for v in range(n):
        classes[best_coloring[v]].append(v)
    return [c for c in classes if c]


def _greedy_coloring(comp: list[list[bool]], n: int) -> list[int]:
    """Greedy sequential coloring on the complement graph.

    Parameters
    ----------
    comp : list[list[bool]]
        Adjacency matrix of the complement graph.
    n : int
        Number of vertices.

    Returns
    -------
    list[int]
        Color assignment (0-indexed) for each vertex.
    """
    colors = [-1] * n
    for v in range(n):
        used: set[int] = set()
        for u in range(v):
            if comp[v][u] and colors[u] != -1:
                used.add(colors[u])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors
