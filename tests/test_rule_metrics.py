from __future__ import annotations

import numpy as np

from src.rule_metrics import (
    best_rule_per_state,
    mapped_state_accuracy,
    rule_performance_table,
    rule_state_mapping,
    timestamp_rule_ids_from_segments,
)


def test_timestamp_rule_ids_from_segments_expands_boundaries() -> None:
    rule_ids = timestamp_rule_ids_from_segments(
        boundaries=[0, 2, 5],
        segment_rule_ids=[4, 7],
    )

    assert rule_ids.tolist() == [4, 4, 7, 7, 7]


def test_rule_performance_uses_coverage_and_precision_formulas() -> None:
    reference_states = np.array(["Idle", "Idle", "Fill", "Fill"])
    predicted_rule_ids = np.array([0, 1, 1, 1])

    performance = rule_performance_table(
        reference_states,
        predicted_rule_ids,
        rule_ids=[0, 1],
        state_order=["Idle", "Fill"],
    )
    best = best_rule_per_state(performance)
    fill_row = best.loc[best["reference_state"] == "Fill"].iloc[0]
    idle_row = best.loc[best["reference_state"] == "Idle"].iloc[0]

    assert fill_row["best_rule_id"] == 1
    assert fill_row["coverage"] == 1.0
    assert fill_row["precision"] == 2 / 3
    assert idle_row["best_rule_id"] == 0
    assert idle_row["coverage"] == 0.5
    assert idle_row["precision"] == 1.0


def test_rule_state_mapping_scores_timestamp_accuracy() -> None:
    reference_states = np.array(["Fill", "Fill", "Hold", "Hold", "Hold"])
    predicted_rule_ids = np.array([0, 0, 1, 1, -1])
    performance = rule_performance_table(
        reference_states,
        predicted_rule_ids,
        rule_ids=[0, 1],
        state_order=["Fill", "Hold"],
    )

    mapping = rule_state_mapping(performance)
    accuracy = mapped_state_accuracy(
        reference_states,
        predicted_rule_ids,
        mapping,
    )

    assert mapping == {0: "Fill", 1: "Hold"}
    assert accuracy == 4 / 5
