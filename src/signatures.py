from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import esig
import numpy as np
import roughpy as rp

from src.merging import SegmentProfile


@dataclass(frozen=True)
class SignatureFeatureResult:
    """Segment-level feature matrix and generated feature names."""

    matrix: np.ndarray
    names: list[str]


def truncated_signature_features(
    path: np.ndarray,
    names: list[str] | None = None,
    depth: int = 2,
    include_constant: bool = False,
    include_signed_areas: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute truncated path-signature features with ``esig``."""
    if depth < 1:
        raise ValueError("signature depth must be at least 1")
    if path.ndim != 2:
        raise ValueError("path must be a 2-D array")
    if path.shape[0] == 0:
        raise ValueError("path must contain at least one point")
    if not np.isfinite(path).all():
        raise ValueError("path must contain only finite values")

    clean_path = np.asarray(path, dtype=float)
    coord_names = _coordinate_names(clean_path, names)
    words = _signature_words(clean_path.shape[1], depth)
    values = _stable_stream2sig(clean_path, depth)
    if values.shape[0] != len(words):
        raise RuntimeError("esig signature keys do not match values")

    selected_values: list[float] = []
    selected_names: list[str] = []
    for value, word in zip(values, words):
        if word or include_constant:
            selected_values.append(float(value))
            selected_names.append(_format_signature_name(word, coord_names))

    if include_signed_areas and depth >= 2:
        value_by_word = dict(zip(words, values))
        for i, j in combinations(range(len(coord_names)), 2):
            selected_values.append(
                float(0.5 * (
                    value_by_word[(i, j)] - value_by_word[(j, i)]
                ))
            )
            selected_names.append(
                f"signed_area({coord_names[i]},{coord_names[j]})"
            )

    return np.array(selected_values, dtype=float), selected_names


def segment_signature_feature_matrix(
    data: np.ndarray,
    boundaries: list[int],
    var_names: list[str] | None = None,
    signature_depth: int = 2,
    include_derivative_features: bool = False,
) -> SignatureFeatureResult:
    """Map each segment to envelope and truncated-signature features."""
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    if len(boundaries) < 2:
        return SignatureFeatureResult(
            matrix=np.empty((0, 0), dtype=float),
            names=[],
        )

    n_vars = data.shape[1]
    names = var_names or [f"v{k}" for k in range(n_vars)]
    if len(names) != n_vars:
        raise ValueError("var_names must match data.shape[1]")

    rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    total_length = max(boundaries[-1] - boundaries[0], 1)

    for start, end in zip(boundaries, boundaries[1:]):
        segment = data[start:end]
        envelope_parts = _segment_envelope_features(
            segment,
            start,
            end,
            total_length,
        )
        derivative_parts = (
            _segment_derivative_features(segment)
            if include_derivative_features
            else []
        )
        path, path_names = _time_augmented_path(segment, names)
        signature_values, signature_names = truncated_signature_features(
            path,
            path_names,
            depth=signature_depth,
        )
        row_names = [
            *_segment_envelope_names(names),
            *(
                _segment_derivative_names(names)
                if include_derivative_features
                else []
            ),
            *signature_names,
        ]
        rows.append(np.concatenate([
            *envelope_parts,
            *derivative_parts,
            signature_values,
        ]))
        if feature_names is None:
            feature_names = row_names

    return SignatureFeatureResult(
        matrix=np.vstack(rows),
        names=feature_names or [],
    )


def compute_signature_profiles(
    data: np.ndarray,
    boundaries: list[int],
    var_names: list[str] | None = None,
    radius: float = 0.0,
    signature_depth: int = 2,
    include_derivative_features: bool = False,
) -> tuple[list[SegmentProfile], list[str], np.ndarray]:
    """Return interval profiles over paper-aligned segment features."""
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    result = segment_signature_feature_matrix(
        data,
        boundaries,
        var_names=var_names,
        signature_depth=signature_depth,
        include_derivative_features=include_derivative_features,
    )
    if result.matrix.size == 0:
        return [], result.names, result.matrix

    lo = result.matrix - radius
    hi = result.matrix + radius
    profiles = [
        SegmentProfile(lo=lo[idx], hi=hi[idx])
        for idx in range(result.matrix.shape[0])
    ]
    return profiles, result.names, result.matrix


def minmax_scale_columns(matrix: np.ndarray) -> np.ndarray:
    """Scale columns to [0, 1] for visualization only."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if matrix.size == 0:
        return matrix.copy()

    col_min = matrix.min(axis=0)
    span = matrix.max(axis=0) - col_min
    scaled = np.zeros_like(matrix, dtype=float)
    non_constant = span > 0.0
    scaled[:, non_constant] = (
        (matrix[:, non_constant] - col_min[non_constant])
        / span[non_constant]
    )
    return scaled


def save_signature_debug_image(
    data: np.ndarray,
    boundaries: list[int],
    output_path: str | Path,
    var_names: list[str] | None = None,
    segment_labels: list[int] | None = None,
    signature_depth: int = 2,
    include_derivative_features: bool = False,
) -> None:
    """Export visual diagnostics for the segment signatures."""
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    if len(boundaries) < 2:
        raise ValueError("Cannot plot signature debug view without segments")

    _configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    features = segment_signature_feature_matrix(
        data,
        boundaries,
        var_names=var_names,
        signature_depth=signature_depth,
        include_derivative_features=include_derivative_features,
    )
    matrix = minmax_scale_columns(features.matrix)
    labels = segment_labels or list(range(matrix.shape[0]))
    if len(labels) != matrix.shape[0]:
        raise ValueError("segment_labels must match the number of segments")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        constrained_layout=True,
    )
    fig.suptitle("Signature Debug View")

    names = var_names or [f"v{k}" for k in range(data.shape[1])]
    _plot_segment_paths(axes[0, 0], data, boundaries, labels, names)

    if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(matrix)
    else:
        coords = np.zeros((matrix.shape[0], 2), dtype=float)
    _plot_signature_pca(axes[0, 1], coords, labels)
    _plot_signature_heatmap(fig, axes[1, 0], matrix, features.names)
    _plot_signature_feature_magnitude(axes[1, 1], matrix, features.names)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _coordinate_names(path: np.ndarray, names: list[str] | None) -> list[str]:
    dim = path.shape[1]
    coord_names = names or [f"x{k}" for k in range(dim)]
    if len(coord_names) != dim:
        raise ValueError("names must match path.shape[1]")
    return coord_names


def _signature_words(dim: int, depth: int) -> list[tuple[int, ...]]:
    keys = esig.sigkeys(dim, depth).strip().split()
    return [_parse_signature_key(key) for key in keys]


def _stable_stream2sig(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute signatures without esig's rounded ``np.arange`` indices."""
    n_samples, width = path.shape
    if depth == 1:
        increments = (
            np.sum(np.diff(path, axis=0), axis=0)
            if n_samples > 1
            else np.zeros(width, dtype=float)
        )
        return np.concatenate([[1.0], increments.astype(float, copy=False)])

    context = rp.get_context(width, depth, rp.DPReal)
    if n_samples == 1:
        values = np.zeros(context.tensor_size(depth), dtype=np.float64)
        values[0] = 1.0
        return values

    increments = np.diff(path, axis=0)
    indices = np.linspace(0.0, 1.0, n_samples - 1, endpoint=False)
    stream = rp.LieIncrementStream.from_increments(
        increments,
        indices=indices,
        ctx=context,
    )
    return np.array(
        stream.signature(rp.RealInterval(0.0, 1.0)),
        copy=True,
    ).astype(float, copy=False)


def _parse_signature_key(key: str) -> tuple[int, ...]:
    if key == "()":
        return ()
    body = key[1:-1]
    return tuple(int(part) - 1 for part in body.split(","))


def _format_signature_name(
    word: tuple[int, ...],
    coord_names: list[str],
) -> str:
    if not word:
        return "sig()"
    return "sig(" + ",".join(coord_names[idx] for idx in word) + ")"


def _time_augmented_path(
    segment: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if segment.shape[0] <= 1:
        time = np.zeros((segment.shape[0], 1), dtype=float)
    else:
        time = np.linspace(0.0, 1.0, segment.shape[0]).reshape(-1, 1)
    return np.hstack([time, segment]), ["time", *names]


def _segment_envelope_features(
    segment: np.ndarray,
    start: int,
    end: int,
    total_length: int,
) -> list[np.ndarray]:
    if segment.shape[0] == 0:
        raise ValueError("empty segments are not supported")

    duration = np.array([(end - start) / total_length], dtype=float)
    return [
        duration,
        segment[0].astype(float, copy=False),
        segment[-1].astype(float, copy=False),
        segment.min(axis=0),
        segment.max(axis=0),
    ]


def _segment_envelope_names(names: list[str]) -> list[str]:
    return (
        ["duration"]
        + [f"start({name})" for name in names]
        + [f"end({name})" for name in names]
        + [f"min({name})" for name in names]
        + [f"max({name})" for name in names]
    )


def _segment_derivative_features(segment: np.ndarray) -> list[np.ndarray]:
    if segment.shape[0] <= 1:
        deltas = np.zeros((1, segment.shape[1]), dtype=float)
    else:
        deltas = np.diff(segment, axis=0)
    return [
        deltas.mean(axis=0).astype(float, copy=False),
        deltas.std(axis=0).astype(float, copy=False),
        deltas.min(axis=0).astype(float, copy=False),
        deltas.max(axis=0).astype(float, copy=False),
    ]


def _segment_derivative_names(names: list[str]) -> list[str]:
    return (
        [f"mean_delta({name})" for name in names]
        + [f"std_delta({name})" for name in names]
        + [f"min_delta({name})" for name in names]
        + [f"max_delta({name})" for name in names]
    )


def _configure_matplotlib_cache() -> None:
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
    )


def _plot_segment_paths(
    ax: object,
    data: np.ndarray,
    boundaries: list[int],
    labels: list[int],
    var_names: list[str],
) -> None:
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    colors = {label: cmap(idx) for idx, label in enumerate(unique_labels)}

    for seg_idx in _sample_indices(len(boundaries) - 1, limit=80):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]
        segment = data[start:end]
        color = colors[labels[seg_idx]]
        if data.shape[1] >= 2:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                alpha=0.42,
                linewidth=1.1,
            )
            ax.scatter(segment[0, 0], segment[0, 1], color=color, s=10)
        else:
            local_time = (
                np.linspace(0.0, 1.0, segment.shape[0])
                if segment.shape[0] > 1
                else np.zeros(segment.shape[0])
            )
            ax.plot(local_time, segment[:, 0], color=color, alpha=0.42)

    if data.shape[1] >= 2:
        ax.set_xlabel(var_names[0])
        ax.set_ylabel(var_names[1])
        ax.set_title("Segment paths")
    else:
        ax.set_xlabel("normalized local time")
        ax.set_ylabel(var_names[0])
        ax.set_title("Segment paths")
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)


def _plot_signature_pca(
    ax: object,
    coords: np.ndarray,
    labels: list[int],
) -> None:
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    for idx, label in enumerate(unique_labels):
        rows = [row for row, row_label in enumerate(labels) if row_label == label]
        ax.scatter(
            coords[rows, 0],
            coords[rows, 1],
            s=38,
            alpha=0.82,
            color=cmap(idx),
            edgecolors="white",
            linewidths=0.4,
            label=f"R{label}",
        )
    ax.set_title("PCA over signature features")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(title="Rule", loc="best", fontsize="small")


def _plot_signature_heatmap(
    fig: object,
    ax: object,
    matrix: np.ndarray,
    names: list[str],
) -> None:
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Signature feature matrix")
    ax.set_xlabel("feature")
    ax.set_ylabel("segment")
    ticks = _sample_indices(len(names), limit=16)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [_shorten_name(names[idx]) for idx in ticks],
        rotation=55,
        ha="right",
        fontsize="x-small",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_signature_feature_magnitude(
    ax: object,
    matrix: np.ndarray,
    names: list[str],
) -> None:
    if matrix.shape[1] == 0:
        ax.set_axis_off()
        return

    scores = matrix.std(axis=0)
    top = np.argsort(scores)[-min(12, len(scores)):]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, scores[top], color="#2563eb", alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [_shorten_name(names[idx]) for idx in top],
        fontsize="small",
    )
    ax.set_xlabel("standard deviation")
    ax.set_title("Most separating coordinates")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)


def _sample_indices(n: int, limit: int) -> list[int]:
    if n <= limit:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, limit, dtype=int).tolist()))


def _shorten_name(name: str, limit: int = 28) -> str:
    if len(name) <= limit:
        return name
    return name[:limit - 1] + "..."
