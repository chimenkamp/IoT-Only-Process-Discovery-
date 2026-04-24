from __future__ import annotations

from enum import Enum, auto
from typing import Any

import pandas as pd
import pm4py

from src.synthesis import IntervalRule


class DiscoveryAlgorithm(Enum):
    """Supported process discovery algorithms."""

    ALPHA = auto()
    ALPHA_PLUS = auto()
    HEURISTICS = auto()
    INDUCTIVE = auto()
    ILP = auto()


def rule_label(rule: IntervalRule, var_names: list[str] | None = None) -> str:
    """Format an interval rule as a human-readable predicate string.

    Parameters
    ----------
    rule : IntervalRule
        The hyperrectangular predicate.
    var_names : list[str] | None
        Sensor variable names.  When ``None``, uses ``v0, v1, ...``.

    Returns
    -------
    str
        A string such as ``"v0∈[0.00,1.00] ∧ v1∈[5.00,6.00]"``.
    """
    parts: list[str] = []
    for k in range(len(rule.lo)):
        name = var_names[k] if var_names else f"v{k}"
        parts.append(f"{name}\u2208[{rule.lo[k]:.2f},{rule.hi[k]:.2f}]")
    return " \u2227 ".join(parts)


def build_label_map(
    rules: list[IntervalRule],
    var_names: list[str] | None = None,
) -> dict[int, str]:
    """Build a mapping from class ids to rule predicate labels.

    Parameters
    ----------
    rules : list[IntervalRule]
        Synthesised interval rules.
    var_names : list[str] | None
        Sensor variable names.

    Returns
    -------
    dict[int, str]
        Mapping ``{class_id: label_string}``.
    """
    return {r.class_id: rule_label(r, var_names) for r in rules}


def traces_to_dataframe(
    traces: list[list[int]],
    label_map: dict[int, str],
) -> pd.DataFrame:
    """Convert rule traces into a pm4py-compatible event log DataFrame.

    Parameters
    ----------
    traces : list[list[int]]
        Event log — list of traces, each a sequence of class ids.
    label_map : dict[int, str]
        Mapping from class ids to activity label strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``case:concept:name``,
        ``concept:name``, and ``time:timestamp``.
    """
    rows: list[dict[str, Any]] = []
    for case_idx, trace in enumerate(traces):
        for event_idx, class_id in enumerate(trace):
            rows.append({
                "case:concept:name": str(case_idx),
                "concept:name": label_map[class_id],
                "time:timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=event_idx),
            })
    df = pd.DataFrame(rows)
    df = pm4py.format_dataframe(df, case_id="case:concept:name",
                                activity_key="concept:name",
                                timestamp_key="time:timestamp")
    return df


def discover_model(
    traces: list[list[int]],
    rules: list[IntervalRule],
    algorithm: DiscoveryAlgorithm = DiscoveryAlgorithm.INDUCTIVE,
    var_names: list[str] | None = None,
) -> tuple[Any, Any, Any]:
    """Discover a Petri net process model from rule traces.

    Parameters
    ----------
    traces : list[list[int]]
        Event log — list of traces over the rule alphabet.
    rules : list[IntervalRule]
        Synthesised interval rules (used to generate transition labels).
    algorithm : DiscoveryAlgorithm
        The discovery algorithm to apply.
    var_names : list[str] | None
        Sensor variable names for labelling.

    Returns
    -------
    tuple[Any, Any, Any]
        ``(net, initial_marking, final_marking)`` — a pm4py Petri net
        with rule predicates as transition labels.
    """
    label_map = build_label_map(rules, var_names)
    df = traces_to_dataframe(traces, label_map)

    discoverers = {
        DiscoveryAlgorithm.ALPHA: pm4py.discover_petri_net_alpha,
        DiscoveryAlgorithm.ALPHA_PLUS: pm4py.discover_petri_net_alpha_plus,
        DiscoveryAlgorithm.HEURISTICS: pm4py.discover_petri_net_heuristics,
        DiscoveryAlgorithm.INDUCTIVE: pm4py.discover_petri_net_inductive,
        DiscoveryAlgorithm.ILP: pm4py.discover_petri_net_ilp,
    }
    discover_fn = discoverers[algorithm]
    net, im, fm = discover_fn(df)
    return net, im, fm


def save_model_visualization(
    net: Any,
    im: Any,
    fm: Any,
    output_path: str,
) -> None:
    """Save a Petri net visualisation to a file.

    Parameters
    ----------
    net : PetriNet
        The discovered Petri net.
    im : Marking
        Initial marking.
    fm : Marking
        Final marking.
    output_path : str
        File path for the output image (e.g. ``"model.png"``
        or ``"model.svg"``).
    """
    pm4py.save_vis_petri_net(net, im, fm, output_path)
