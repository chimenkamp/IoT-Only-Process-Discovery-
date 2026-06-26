from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.merging import SegmentProfile


@dataclass(frozen=True)
class IntervalRule:
    """Interval predicate over segment features."""

    lo: list[float]
    hi: list[float]
    class_id: int


def synthesize_rules(
    classes: list[list[int]],
    profiles: list[SegmentProfile],
    n_features: int,
    margin: float = 0.0,
) -> list[IntervalRule]:
    """Materialize one interval-grammar rule per segment class."""
    rules: list[IntervalRule] = []
    for class_id, members in enumerate(classes):
        positives = [profiles[idx] for idx in members]
        negatives = [
            profiles[idx]
            for other_id, other_members in enumerate(classes)
            if other_id != class_id
            for idx in other_members
        ]
        lo, hi = synthesize_discriminator(
            positives,
            negatives,
            n_features,
            margin=margin,
        )
        rules.append(IntervalRule(lo=lo, hi=hi, class_id=class_id))
    return rules


def synthesize_discriminator(
    positive_profiles: list[SegmentProfile],
    negative_profiles: list[SegmentProfile],
    n_features: int,
    margin: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Solve the interval SyGuS fragment by its exact constructive form."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if not positive_profiles:
        raise ValueError("positive_profiles must not be empty")
    for profile in [*positive_profiles, *negative_profiles]:
        if profile.lo.shape[0] != n_features or profile.hi.shape[0] != n_features:
            raise ValueError("profile dimension does not match n_features")

    lo_vals = np.min([profile.lo for profile in positive_profiles], axis=0)
    hi_vals = np.max([profile.hi for profile in positive_profiles], axis=0)

    for profile in negative_profiles:
        if _boxes_intersect(lo_vals, hi_vals, profile.lo, profile.hi):
            raise RuntimeError("Rule synthesis failed: interval grammar is UNSAT")

    if margin > 0.0 and negative_profiles:
        lo_vals, hi_vals = _expand_without_negative_intersections(
            lo_vals,
            hi_vals,
            negative_profiles,
            margin,
        )
    elif margin > 0.0:
        lo_vals = lo_vals - margin
        hi_vals = hi_vals + margin

    return lo_vals.astype(float).tolist(), hi_vals.astype(float).tolist()


def evaluate_rule(rule: IntervalRule, feature_vector: np.ndarray) -> bool:
    """Return whether one segment feature vector satisfies a rule."""
    if feature_vector.shape[0] != len(rule.lo):
        raise ValueError("feature dimension does not match rule")
    return bool(np.all(
        (feature_vector >= np.array(rule.lo))
        & (feature_vector <= np.array(rule.hi))
    ))


def rule_covers_profile(rule: IntervalRule, profile: SegmentProfile) -> bool:
    """Return whether a rule covers the whole interval profile."""
    return bool(np.all(
        (profile.lo >= np.array(rule.lo))
        & (profile.hi <= np.array(rule.hi))
    ))


def classify_profiles(
    profiles: list[SegmentProfile],
    rules: list[IntervalRule],
) -> tuple[list[int], list[int], list[int]]:
    """Classify segment profiles and report coverage violations."""
    labels: list[int] = []
    uncovered: list[int] = []
    ambiguous: list[int] = []

    for idx, profile in enumerate(profiles):
        active = [
            rule.class_id
            for rule in rules
            if rule_covers_profile(rule, profile)
        ]
        if len(active) == 1:
            labels.append(active[0])
        elif not active:
            labels.append(-1)
            uncovered.append(idx)
        else:
            labels.append(-1)
            ambiguous.append(idx)

    return labels, uncovered, ambiguous


def _boxes_intersect(
    lo_a: np.ndarray,
    hi_a: np.ndarray,
    lo_b: np.ndarray,
    hi_b: np.ndarray,
) -> bool:
    return bool(np.all(np.maximum(lo_a, lo_b) <= np.minimum(hi_a, hi_b)))


def _expand_without_negative_intersections(
    lo_vals: np.ndarray,
    hi_vals: np.ndarray,
    negative_profiles: list[SegmentProfile],
    requested_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    def valid(margin: float) -> bool:
        lo = lo_vals - margin
        hi = hi_vals + margin
        return not any(
            _boxes_intersect(lo, hi, profile.lo, profile.hi)
            for profile in negative_profiles
        )

    if valid(requested_margin):
        return lo_vals - requested_margin, hi_vals + requested_margin

    low = 0.0
    high = requested_margin
    for _ in range(32):
        mid = (low + high) / 2.0
        if valid(mid):
            low = mid
        else:
            high = mid

    return lo_vals - low, hi_vals + low
