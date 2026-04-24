"""Real-data test: Future Factory sensor log.

Loads a small sample of production cycles from the Future Factory
dataset, normalises the data to [0, 1] per sensor, runs the
three-phase synthesis pipeline to learn rules, builds per-cycle
traces, and discovers a process model with pm4py.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.changepoint import detect_changepoints
from src.discovery import (
    DiscoveryAlgorithm,
    discover_model,
    save_model_visualization,
)
from src.merging import (
    SegmentProfile,
    build_compatibility_graph,
    compute_profiles,
    minimum_clique_cover,
)
from src.smt import get_solver
from src.synthesis import IntervalRule, synthesize_rules, evaluate_rule
from src.trace import build_traces

# ── configuration ────────────────────────────────────────────────────
PKL_PATH = Path("data/Future_Factory/combined_[1-6].pkl")

# Production cycles to use (cycle 0 is tiny / startup, skip it)
CYCLE_IDS = [1, 2, 3, 4, 5]

# Sensor columns — Robot 1 joint angles + gripper (8 continuous vars)
SENSOR_COLS = [
    "M_R01_SJointAngle_Degree",
    "M_R01_LJointAngle_Degree",
    "M_R01_UJointAngle_Degree",
    "M_R01_RJointAngle_Degree",
    "M_R01_BJointAngle_Degree",
    "M_R01_TJointAngle_Degree",
    "I_R01_Gripper_Pot",
    "I_R01_Gripper_Load",
]

PENALTY = 10.0          # PELT penalty (higher = fewer changepoints)
MIN_SEGMENT_SIZE = 10   # minimum samples between two changepoints
SMT_TIMEOUT_MS = 10000  # 10 s per SMT query
ALGORITHM = DiscoveryAlgorithm.INDUCTIVE
OUTPUT_PATH = "real_model.png"


# ── helpers ──────────────────────────────────────────────────────────
@dataclass
class Normaliser:
    """Min-max normaliser fitted on training data."""
    lo: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normaliser":
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        span = hi - lo
        span[span == 0] = 1.0
        return cls(lo=lo, span=span)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.lo) / self.span

    def label(self, col: int, val: float) -> float:
        """Map a normalised value back to original scale."""
        return val * self.span[col] + self.lo[col]


def load_cycles(
    pkl_path: Path,
    cycle_ids: list[int],
    sensor_cols: list[str],
) -> list[np.ndarray]:
    """Load selected cycles and sensor columns from the pickle file."""
    with open(pkl_path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)

    cycles: list[np.ndarray] = []
    for cid in cycle_ids:
        cycle_df = df.loc[df["Q_Cell_CycleCount"] == cid, sensor_cols]
        arr = cycle_df.to_numpy(dtype=np.float64)
        cycles.append(arr)
    return cycles


def per_cycle_traces(
    cycles: list[np.ndarray],
    rules: list[IntervalRule],
) -> list[list[int]]:
    """Apply shared rules to each cycle independently → one trace per cycle."""
    all_traces: list[list[int]] = []
    for cycle_data in cycles:
        traces = build_traces_nearest(cycle_data, rules)
        all_traces.extend(traces)
    return all_traces


def bounding_box_rules(
    classes: list[list[int]],
    profiles: list[SegmentProfile],
    n_vars: int,
) -> list[IntervalRule]:
    """Create one bounding-box rule per class (no SMT, always succeeds)."""
    rules: list[IntervalRule] = []
    for i, members in enumerate(classes):
        class_profiles = [profiles[idx] for idx in members]
        lo = np.min([p.lo for p in class_profiles], axis=0).tolist()
        hi = np.max([p.hi for p in class_profiles], axis=0).tolist()
        rules.append(IntervalRule(lo=lo, hi=hi, class_id=i))
    return rules


def try_synthesize_rules(
    classes: list[list[int]],
    profiles: list[SegmentProfile],
    n_vars: int,
    timeout_ms: int | None,
) -> tuple[list[IntervalRule], bool]:
    """Try exact SMT synthesis; fall back to bounding boxes on UNSAT."""
    smt = get_solver()
    try:
        rules = synthesize_rules(smt, classes, profiles, n_vars, timeout_ms)
        return rules, True
    except RuntimeError:
        rules = bounding_box_rules(classes, profiles, n_vars)
        return rules, False


def build_traces_nearest(
    data: np.ndarray,
    rules: list[IntervalRule],
) -> list[list[int]]:
    """Build traces using nearest-center matching for overlapping rules.

    For each time point, if exactly one rule covers it, use that rule.
    If multiple or none cover it, pick the rule whose bounding-box
    center is closest (Euclidean distance).
    """
    centers = np.array([
        [(r.lo[k] + r.hi[k]) / 2 for k in range(len(r.lo))]
        for r in rules
    ])
    n = data.shape[0]
    trace: list[int] = []
    prev = -1
    for t in range(n):
        pt = data[t]
        matching = [r for r in rules if evaluate_rule(r, pt)]
        if len(matching) == 1:
            active = matching[0].class_id
        else:
            dists = np.linalg.norm(centers - pt, axis=1)
            active = rules[int(np.argmin(dists))].class_id
        if active != prev:
            trace.append(active)
            prev = active
    return [trace]


def format_rule(rule: IntervalRule, norm: Normaliser, names: list[str]) -> str:
    """Pretty-print a rule with denormalised (original-scale) bounds."""
    parts: list[str] = []
    for k in range(len(rule.lo)):
        lo_orig = norm.label(k, rule.lo[k])
        hi_orig = norm.label(k, rule.hi[k])
        parts.append(f"{names[k]}∈[{lo_orig:.1f},{hi_orig:.1f}]")
    return " ∧ ".join(parts)


def main() -> None:
    # ── 1. Load data ─────────────────────────────────────────────────
    print(f"Loading cycles {CYCLE_IDS} from {PKL_PATH} ...")
    cycles_raw = load_cycles(PKL_PATH, CYCLE_IDS, SENSOR_COLS)
    for i, cid in enumerate(CYCLE_IDS):
        print(f"  Cycle {cid}: {cycles_raw[i].shape[0]} samples × "
              f"{cycles_raw[i].shape[1]} sensors")

    # ── 2. Normalise to [0, 1] ───────────────────────────────────────
    concat_raw = np.vstack(cycles_raw)
    norm = Normaliser.fit(concat_raw)
    cycles = [norm.transform(c) for c in cycles_raw]
    concat = np.vstack(cycles)
    n_sensors = concat.shape[1]
    print(f"\nNormalised concatenation: {concat.shape[0]} samples × "
          f"{n_sensors} sensors")

    # ── 3. Phase 1 — change point detection ──────────────────────────
    print(f"\nPhase 1: PELT (penalty={PENALTY}, min_size={MIN_SEGMENT_SIZE})")
    boundaries = detect_changepoints(concat, PENALTY, MIN_SEGMENT_SIZE)
    n_segments = len(boundaries) - 1
    print(f"  → {n_segments} segments from {len(boundaries)} boundaries")

    # ── 4. Phase 2 — segment merging ─────────────────────────────────
    print("\nPhase 2: Segment profiling + clique cover ...")
    profiles = compute_profiles(concat, boundaries)
    adj = build_compatibility_graph(profiles)
    classes = minimum_clique_cover(adj, len(profiles))
    n_classes = len(classes)
    print(f"  → {n_segments} segments merged into {n_classes} equivalence "
          f"classes")

    # ── 5. Phase 3 — rule synthesis ──────────────────────────────────
    print(f"\nPhase 3: Rule synthesis ({n_classes} classes) ...")
    rules, exact = try_synthesize_rules(
        classes, profiles, n_sensors, SMT_TIMEOUT_MS,
    )
    if exact:
        print("  (exact SMT discrimination)")
    else:
        print("  (bounding-box fallback — exact discrimination UNSAT)")
    print()
    for rule in rules:
        print(f"  Rule {rule.class_id}: {format_rule(rule, norm, SENSOR_COLS)}")

    # ── 6. Per-cycle trace construction ──────────────────────────────
    print(f"\nBuilding per-cycle traces ({len(cycles)} cycles) ...")
    traces = per_cycle_traces(cycles, rules)
    print(f"  → {len(traces)} trace(s)")
    for i, trace in enumerate(traces):
        label = f"Cycle {CYCLE_IDS[i]}" if i < len(CYCLE_IDS) else f"Trace {i}"
        print(f"  {label}: {len(trace)} events → {trace}")

    # ── 7. Process discovery ─────────────────────────────────────────
    print(f"\nProcess discovery ({ALGORITHM.name}) ...")
    net, im, fm = discover_model(traces, rules, ALGORITHM, SENSOR_COLS)
    print(f"  → Petri net: {len(net.transitions)} transitions, "
          f"{len(net.places)} places")

    # ── 8. Visualise ─────────────────────────────────────────────────
    save_model_visualization(net, im, fm, OUTPUT_PATH)
    print(f"\nModel saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
