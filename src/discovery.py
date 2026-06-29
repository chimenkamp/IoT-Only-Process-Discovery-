from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

import pandas as pd
import pm4py

from src.synthesis import IntervalAtom, IntervalRule, active_rule_atoms


def rule_label(
    rule: IntervalRule,
    var_names: list[str],
    max_parts: int = 2,
    max_chars: int = 220,
) -> str:
    """Format a SyGuS rule as a readable segment-feature predicate."""
    return compact_rule_predicate(
        rule,
        var_names,
        max_parts=max_parts,
        max_chars=max_chars,
        prefix=f"R{rule.class_id}: ",
    )


def full_rule_predicate(rule: IntervalRule, var_names: list[str]) -> str:
    """Format a full SyGuS rule predicate without visualization truncation."""
    parts = _rule_predicate_parts(rule, var_names)
    return " and ".join(parts)


def compact_rule_predicate(
    rule: IntervalRule,
    var_names: list[str],
    max_parts: int = 2,
    max_chars: int = 220,
    prefix: str = "",
) -> str:
    """Format a compact SyGuS rule predicate."""
    parts = _rule_predicate_parts(rule, var_names)
    full_label = " and ".join(parts)
    if len(parts) <= max_parts and len(full_label) <= max_chars:
        return f"{prefix}{full_label}"

    shown: list[str] = []
    for part in parts:
        remaining = len(parts) - len(shown) - 1
        suffix = f" and ... ({remaining} more)" if remaining else ""
        candidate = " and ".join([*shown, part])
        if len(shown) >= max_parts or len(candidate) + len(suffix) > max_chars:
            break
        shown.append(part)

    if not shown:
        return f"R{rule.class_id} ({len(parts)} feature intervals)"

    remaining = len(parts) - len(shown)
    suffix = f" and ... ({remaining} more)" if remaining else ""
    return f"{prefix}{' and '.join(shown)}{suffix}"


def _rule_predicate_parts(rule: IntervalRule, var_names: list[str]) -> list[str]:
    atoms = active_rule_atoms(rule)
    if not atoms:
        return ["true"]
    return [_format_atom(atom, var_names) for atom in atoms]


def _format_atom(atom: IntervalAtom, var_names: list[str]) -> str:
    name = var_names[atom.feature]
    if atom.lo is not None and atom.hi is not None:
        return f"{name} in [{atom.lo:.3g},{atom.hi:.3g}]"
    if atom.lo is not None:
        return f"{name} >= {atom.lo:.3g}"
    if atom.hi is not None:
        return f"{name} <= {atom.hi:.3g}"
    return "true"


def activity_name(
    class_id: int,
    n_activities: int,
    prefix: str = "A",
) -> str:
    """Return a stable generic activity name such as ``A01``."""
    width = max(2, len(str(max(n_activities, 1))))
    return f"{prefix}{class_id + 1:0{width}d}"


def build_label_map(
    rules: list[IntervalRule],
    var_names: list[str],
    label_style: str = "rule",
    activity_prefix: str = "A",
) -> dict[int, str]:
    """Build activity labels from synthesized rule predicates."""
    if label_style == "rule":
        return {rule.class_id: rule_label(rule, var_names) for rule in rules}
    if label_style == "generic":
        return {
            rule.class_id: activity_name(
                rule.class_id,
                len(rules),
                prefix=activity_prefix,
            )
            for rule in rules
        }
    raise ValueError("label_style must be 'rule' or 'generic'")


def traces_to_dataframe(
    traces: list[list[int]],
    label_map: dict[int, str],
) -> pd.DataFrame:
    """Convert activity projections into a pm4py event log DataFrame."""
    rows: list[dict[str, Any]] = []
    for case_idx, trace in enumerate(traces):
        for event_idx, class_id in enumerate(trace):
            rows.append({
                "case:concept:name": str(case_idx),
                "concept:name": label_map[class_id],
                "time:timestamp": (
                    pd.Timestamp("2026-01-01")
                    + pd.Timedelta(seconds=event_idx)
                ),
            })
    df = pd.DataFrame(rows)
    return pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )


def discover_model(
    traces: list[list[int]],
    rules: list[IntervalRule],
    var_names: list[str],
    label_style: str = "rule",
    activity_prefix: str = "A",
    inductive_miner_noise_threshold: float = 0.0,
    variant_coverage_threshold: float = 0.0,
) -> tuple[Any, Any, Any]:
    """Discover a Petri net with the Inductive Miner."""
    label_map = build_label_map(
        rules,
        var_names,
        label_style=label_style,
        activity_prefix=activity_prefix,
    )
    return discover_labeled_model(
        traces,
        label_map,
        inductive_miner_noise_threshold=inductive_miner_noise_threshold,
        variant_coverage_threshold=variant_coverage_threshold,
    )


def discover_labeled_model(
    traces: list[list[int]],
    label_map: dict[int, str],
    inductive_miner_noise_threshold: float = 0.0,
    variant_coverage_threshold: float = 0.0,
) -> tuple[Any, Any, Any]:
    """Discover a Petri net from traces with explicit activity labels."""
    if inductive_miner_noise_threshold < 0.0:
        raise ValueError("inductive_miner_noise_threshold must be non-negative")
    if variant_coverage_threshold < 0.0:
        raise ValueError("variant_coverage_threshold must be non-negative")
    df = traces_to_dataframe(traces, label_map)
    if variant_coverage_threshold > 0.0:
        df = pm4py.filter_variants_by_coverage_percentage(
            df,
            variant_coverage_threshold,
        )
    return pm4py.discover_petri_net_inductive(
        df,
        noise_threshold=inductive_miner_noise_threshold,
    )


def token_replay_fitness(
    traces: list[list[int]],
    rules: list[IntervalRule],
    var_names: list[str],
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    label_style: str = "rule",
    activity_prefix: str = "A",
) -> float:
    """Return token-based replay fitness for activity projections."""
    label_map = build_label_map(
        rules,
        var_names,
        label_style=label_style,
        activity_prefix=activity_prefix,
    )
    return token_replay_fitness_labeled(
        traces,
        label_map,
        net,
        initial_marking,
        final_marking,
    )


def token_replay_fitness_labeled(
    traces: list[list[int]],
    label_map: dict[int, str],
    net: Any,
    initial_marking: Any,
    final_marking: Any,
) -> float:
    """Return token-based replay fitness for explicitly labeled traces."""
    df = traces_to_dataframe(traces, label_map)
    result = pm4py.fitness_token_based_replay(
        df,
        net,
        initial_marking,
        final_marking,
    )
    return float(result.get("log_fitness", 0.0))


def save_model_visualization(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    output_path: str,
) -> None:
    """Save the discovered Petri net visualization."""
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, output_path)


def save_activity_legend(
    rules: list[IntervalRule],
    var_names: list[str],
    output_path: str,
    activity_prefix: str = "A",
    compact_parts: int | None = None,
) -> None:
    """Write a Markdown legend from generic activity labels to SyGuS rules."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Activity Legend",
        "",
        "Generic activity labels are domain-agnostic identifiers. "
        "Each rule is a CVC5 SyGuS interval predicate over the discovered "
        "segment feature space.",
        "",
    ]
    for rule in sorted(rules, key=lambda item: item.class_id):
        label = activity_name(
            rule.class_id,
            len(rules),
            prefix=activity_prefix,
        )
        if compact_parts is None:
            predicate = full_rule_predicate(rule, var_names)
        else:
            predicate = compact_rule_predicate(
                rule,
                var_names,
                max_parts=compact_parts,
                max_chars=100_000,
            )
        lines.extend([
            f"## {label}",
            "",
            "```text",
            predicate,
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
