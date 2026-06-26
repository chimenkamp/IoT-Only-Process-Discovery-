from __future__ import annotations

import numpy as np

from src.signatures import (
    compute_signature_profiles,
    save_signature_debug_image,
    truncated_signature_features,
)


def test_truncated_signature_features_use_esig_key_order() -> None:
    path = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    features, names = truncated_signature_features(path, ["x", "y"], depth=2)

    assert names[:6] == [
        "sig(x)",
        "sig(y)",
        "sig(x,x)",
        "sig(x,y)",
        "sig(y,x)",
        "sig(y,y)",
    ]
    assert features[names.index("sig(x)")] == 1.0
    assert features[names.index("sig(y)")] == 1.0


def test_truncated_signature_features_handles_arange_rounding_lengths() -> None:
    path = np.column_stack([
        np.linspace(0.0, 1.0, 50),
        np.linspace(0.0, 1.0, 50) ** 2,
    ])

    features, names = truncated_signature_features(path, ["x", "y"], depth=2)

    assert len(features) == len(names)
    assert np.isfinite(features).all()
    assert features[names.index("sig(x)")] == 1.0


def test_signed_area_separates_path_orientation() -> None:
    clockwise = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    counter_clockwise = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ])

    cw_features, names = truncated_signature_features(
        clockwise,
        ["x", "y"],
        depth=2,
    )
    ccw_features, _ = truncated_signature_features(
        counter_clockwise,
        ["x", "y"],
        depth=2,
    )

    area_idx = names.index("signed_area(x,y)")
    assert cw_features[area_idx] == -ccw_features[area_idx]
    assert cw_features[area_idx] != 0.0


def test_signature_profiles_keep_feature_space_unscaled() -> None:
    data = np.vstack([
        np.column_stack([
            np.linspace(0.0, 2.0, 10),
            np.zeros(10),
        ]),
        np.column_stack([
            np.linspace(0.0, 1.0, 10),
            np.zeros(10),
        ]),
    ])

    _, names, matrix = compute_signature_profiles(
        data,
        [0, 10, 20],
        var_names=["x", "y"],
        radius=0.0,
    )

    assert matrix[0, names.index("sig(x)")] == 2.0
    assert matrix[1, names.index("sig(x)")] == 1.0


def test_derivative_features_can_be_included() -> None:
    data = np.column_stack([
        np.array([0.0, 1.0, 3.0, 6.0]),
        np.zeros(4),
    ])

    _, names, matrix = compute_signature_profiles(
        data,
        [0, 4],
        var_names=["x", "y"],
        include_derivative_features=True,
    )

    assert "mean_delta(x)" in names
    assert "max_delta(x)" in names
    assert matrix[0, names.index("mean_delta(x)")] == 2.0
    assert matrix[0, names.index("max_delta(x)")] == 3.0


def test_signature_debug_image_is_exported(tmp_path) -> None:
    data = np.vstack([
        np.column_stack([
            np.linspace(0.0, 1.0, 20),
            np.zeros(20),
        ]),
        np.column_stack([
            np.ones(20),
            np.linspace(0.0, 1.0, 20),
        ]),
    ])
    output_path = tmp_path / "signature_debug.png"

    save_signature_debug_image(
        data,
        [0, 20, 40],
        output_path,
        var_names=["x", "y"],
        segment_labels=[0, 1],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
