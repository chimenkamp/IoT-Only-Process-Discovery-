from __future__ import annotations

import numpy as np
import pandas as pd


def timestamp_rule_ids_from_segments(
    boundaries: list[int],
    segment_rule_ids: list[int],
) -> np.ndarray:
    """Expand segment-level rule IDs to one rule ID per timestamp."""
    if len(segment_rule_ids) != len(boundaries) - 1:
        raise ValueError("segment_rule_ids must have length len(boundaries) - 1")
    if not boundaries:
        return np.array([], dtype=int)

    rule_ids = np.empty(boundaries[-1] - boundaries[0], dtype=int)
    offset = boundaries[0]
    for rule_id, start, end in zip(
        segment_rule_ids,
        boundaries,
        boundaries[1:],
    ):
        rule_ids[start - offset:end - offset] = rule_id
    return rule_ids


def rule_performance_table(
    reference_states: np.ndarray,
    predicted_rule_ids: np.ndarray,
    rule_ids: list[int],
    state_order: list[str],
) -> pd.DataFrame:
    """Compute Sensor2EventLog rule coverage and precision for all pairs.

    Ground-truth states are used only as reference classes for evaluation.
    The discovered rule IDs remain anonymous.
    """
    if reference_states.shape[0] != predicted_rule_ids.shape[0]:
        raise ValueError("reference states and rule IDs must have equal length")

    rows: list[dict[str, object]] = []
    for state in state_order:
        state_mask = reference_states == state
        state_support = int(state_mask.sum())
        for rule_id in rule_ids:
            rule_mask = predicted_rule_ids == rule_id
            rule_support = int(rule_mask.sum())
            overlap = int((state_mask & rule_mask).sum())
            coverage = _safe_ratio(overlap, state_support)
            precision = _safe_ratio(overlap, rule_support)
            rows.append({
                "reference_state": state,
                "rule_id": rule_id,
                "state_support": state_support,
                "rule_support": rule_support,
                "overlap": overlap,
                "coverage": coverage,
                "precision": precision,
                "effectiveness": float(np.sqrt(coverage * precision)),
            })
    return pd.DataFrame(rows)


def best_rule_per_state(
    rule_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Select the strongest anonymous rule for each reference state."""
    rows: list[pd.Series] = []
    for _, group in rule_performance.groupby("reference_state", sort=False):
        ordered = group.sort_values(
            ["effectiveness", "coverage", "precision", "overlap"],
            ascending=False,
        )
        rows.append(ordered.iloc[0])
    result = pd.DataFrame(rows).reset_index(drop=True)
    return result.rename(columns={"rule_id": "best_rule_id"})


def aggregate_rule_metrics(best_rules: pd.DataFrame) -> dict[str, float]:
    """Summarize best-rule coverage and precision over reference states."""
    if best_rules.empty:
        return {
            "mean_coverage": 0.0,
            "mean_precision": 0.0,
            "mean_effectiveness": 0.0,
        }
    return {
        "mean_coverage": float(best_rules["coverage"].mean()),
        "mean_precision": float(best_rules["precision"].mean()),
        "mean_effectiveness": float(best_rules["effectiveness"].mean()),
    }


def rule_state_mapping(rule_performance: pd.DataFrame) -> dict[int, object]:
    """Map each anonymous rule to its strongest reference state.

    The mapping is an evaluation-only decoder. It is learned from a split's
    reference labels after discovery and never feeds back into rule synthesis.
    """
    mapping: dict[int, object] = {}
    if rule_performance.empty:
        return mapping

    for rule_id, group in rule_performance.groupby("rule_id", sort=False):
        supported = group[group["rule_support"] > 0]
        if supported.empty:
            continue
        ordered = supported.sort_values(
            ["precision", "coverage", "overlap"],
            ascending=False,
        )
        mapping[int(rule_id)] = ordered.iloc[0]["reference_state"]
    return mapping


def mapped_state_accuracy(
    reference_states: np.ndarray,
    predicted_rule_ids: np.ndarray,
    mapping: dict[int, object],
) -> float:
    """Return timestamp accuracy after mapping rules to reference states."""
    if reference_states.shape[0] != predicted_rule_ids.shape[0]:
        raise ValueError("reference states and rule IDs must have equal length")
    if reference_states.shape[0] == 0:
        return 0.0

    correct = 0
    for reference_state, rule_id in zip(reference_states, predicted_rule_ids):
        predicted_state = mapping.get(int(rule_id))
        if predicted_state == reference_state:
            correct += 1
    return float(correct / reference_states.shape[0])


def paper_reported_rule_metrics() -> pd.DataFrame:
    """Rule metrics reported for Sensor2EventLog's pasteurization example."""
    rows = [
        {
            "approach": "Sensor2EventLog_reported",
            "reference_state": "Fill",
            "rule": "Qin > tau_Q",
            "coverage": 1.000,
            "precision": 1.000,
        },
        {
            "approach": "Sensor2EventLog_reported",
            "reference_state": "Hold",
            "rule": "(T > 70) & stability",
            "coverage": 1.000,
            "precision": 0.951,
        },
        {
            "approach": "Sensor2EventLog_reported",
            "reference_state": "HeatUp",
            "rule": "T smooth gradient > 1",
            "coverage": 0.930,
            "precision": 0.953,
        },
        {
            "approach": "Sensor2EventLog_reported",
            "reference_state": "Cool",
            "rule": "T smooth gradient < -1",
            "coverage": 0.922,
            "precision": 0.976,
        },
    ]
    table = pd.DataFrame(rows)
    table["effectiveness"] = np.sqrt(table["coverage"] * table["precision"])
    return table


def paper_reported_summary() -> pd.DataFrame:
    """Aggregate the rule metrics reported in Sensor2EventLog."""
    table = paper_reported_rule_metrics()
    return pd.DataFrame([{
        "approach": "Sensor2EventLog_reported",
        "rules_compared": len(table),
        "mean_coverage": float(table["coverage"].mean()),
        "mean_precision": float(table["precision"].mean()),
        "mean_effectiveness": float(table["effectiveness"].mean()),
        "teacher_iterations": 3,
        "domain_rules": 4,
    }])


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
