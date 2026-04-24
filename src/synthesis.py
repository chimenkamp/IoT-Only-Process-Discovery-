from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.merging import SegmentProfile
from src.smt import SMTSolver, SatResult


@dataclass(frozen=True)
class IntervalRule:
    """Axis-aligned hyperrectangular predicate over sensor variables.

    Represents the conjunction ``v_k >= lo[k] AND v_k <= hi[k]``
    for each dimension *k*.

    Parameters
    ----------
    lo : list[float]
        Lower bounds per dimension.
    hi : list[float]
        Upper bounds per dimension.
    class_id : int
        Equivalence-class identifier this rule characterises.
    """

    lo: list[float]
    hi: list[float]
    class_id: int


def synthesize_discriminator(
    smt: SMTSolver,
    positive_profiles: list[SegmentProfile],
    negative_profiles: list[SegmentProfile],
    n_vars: int,
    timeout_ms: int | None = None,
) -> tuple[list[float], list[float]]:
    """Synthesise a discriminating hyperrectangular predicate via SMT.

    Finds bounds ``[lo_k, hi_k]`` for each dimension *k* such that
    every positive profile is fully contained within the hyperrectangle
    and every negative profile is fully excluded.

    Parameters
    ----------
    smt : SMTSolver
        Abstract SMT solver instance.
    positive_profiles : list[SegmentProfile]
        Segment profiles that the predicate must cover.
    negative_profiles : list[SegmentProfile]
        Segment profiles that the predicate must exclude.
    n_vars : int
        Number of sensor variables (dimensions).
    timeout_ms : int | None
        SMT solver timeout in milliseconds.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(lo, hi)`` — lower and upper bounds per dimension.

    Raises
    ------
    RuntimeError
        If the solver returns UNSAT or UNKNOWN.
    """
    lo_vars = [smt.Real(f"lo_{k}") for k in range(n_vars)]
    hi_vars = [smt.Real(f"hi_{k}") for k in range(n_vars)]

    with smt.create_context(timeout_ms=timeout_ms) as ctx:
        for profile in positive_profiles:
            for k in range(n_vars):
                ctx.add(smt.LE(lo_vars[k], smt.RealVal(float(profile.lo[k]))))
                ctx.add(smt.LE(smt.RealVal(float(profile.hi[k])), hi_vars[k]))

        for profile in negative_profiles:
            disjuncts = []
            for k in range(n_vars):
                disjuncts.append(
                    smt.GT(lo_vars[k], smt.RealVal(float(profile.hi[k])))
                )
                disjuncts.append(
                    smt.GT(smt.RealVal(float(profile.lo[k])), hi_vars[k])
                )
            ctx.add(smt.Or(*disjuncts))

        result = ctx.check()
        if result != SatResult.SAT:
            raise RuntimeError(
                f"Discriminator synthesis failed: {result.name}"
            )

        m = ctx.model()
        lo_vals = [
            smt.to_real_float(m.evaluate(lo_vars[k], model_completion=True))
            for k in range(n_vars)
        ]
        hi_vals = [
            smt.to_real_float(m.evaluate(hi_vars[k], model_completion=True))
            for k in range(n_vars)
        ]
    return lo_vals, hi_vals


def synthesize_rules(
    smt: SMTSolver,
    classes: list[list[int]],
    profiles: list[SegmentProfile],
    n_vars: int,
    timeout_ms: int | None = None,
) -> list[IntervalRule]:
    """Synthesise one interval rule per equivalence class.

    For each class *E_i*, pairwise discriminators ``d_{ij}`` are
    synthesised against every other class *E_j*.  The final rule is
    the intersection (tightest bounds) of all pairwise discriminators.

    Parameters
    ----------
    smt : SMTSolver
        Abstract SMT solver instance.
    classes : list[list[int]]
        Equivalence classes — each inner list holds segment indices.
    profiles : list[SegmentProfile]
        Segment profiles (one per segment).
    n_vars : int
        Number of sensor variables.
    timeout_ms : int | None
        SMT solver timeout in milliseconds.

    Returns
    -------
    list[IntervalRule]
        One rule per equivalence class.

    Raises
    ------
    RuntimeError
        If any pairwise discriminator synthesis fails.
    """
    if len(classes) == 1:
        class_profiles = [profiles[idx] for idx in classes[0]]
        lo = np.min([p.lo for p in class_profiles], axis=0).tolist()
        hi = np.max([p.hi for p in class_profiles], axis=0).tolist()
        return [IntervalRule(lo=lo, hi=hi, class_id=0)]

    rules: list[IntervalRule] = []
    for i, class_i in enumerate(classes):
        pos_profiles = [profiles[idx] for idx in class_i]
        bounds_per_j: list[tuple[list[float], list[float]]] = []
        for j, class_j in enumerate(classes):
            if i == j:
                continue
            neg_profiles = [profiles[idx] for idx in class_j]
            lo, hi = synthesize_discriminator(
                smt, pos_profiles, neg_profiles, n_vars, timeout_ms,
            )
            bounds_per_j.append((lo, hi))

        final_lo = [
            max(b[0][k] for b in bounds_per_j)
            for k in range(n_vars)
        ]
        final_hi = [
            min(b[1][k] for b in bounds_per_j)
            for k in range(n_vars)
        ]
        rules.append(IntervalRule(lo=final_lo, hi=final_hi, class_id=i))
    return rules


def evaluate_rule(rule: IntervalRule, point: np.ndarray) -> bool:
    """Evaluate whether a data point satisfies an interval rule.

    Parameters
    ----------
    rule : IntervalRule
        The hyperrectangular predicate.
    point : np.ndarray
        A single data point, shape ``(n_vars,)``.

    Returns
    -------
    bool
        ``True`` if the point falls within the rule's hyperrectangle.
    """
    for k in range(len(rule.lo)):
        if point[k] < rule.lo[k] or point[k] > rule.hi[k]:
            return False
    return True
