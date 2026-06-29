from __future__ import annotations

import numpy as np
import pytest

from src.merging import SegmentProfile
from src.synthesis import active_rule_atoms, classify_profiles, synthesize_rules


def test_sygus_synthesizes_concise_separating_predicates() -> None:
    profiles = [
        _point_profile([1.0, 0.0, 10.0]),
        _point_profile([2.0, 0.0, 20.0]),
        _point_profile([1.0, 1.0, 10.0]),
        _point_profile([2.0, 1.0, 20.0]),
    ]

    rules = synthesize_rules([[0, 1], [2, 3]], profiles, n_features=3)

    assert all(
        rule.certificate.backend.startswith("cvc5-python")
        and "synth-fun R" in rule.certificate.grammar
        for rule in rules
    )
    assert [len(active_rule_atoms(rule)) for rule in rules] == [1, 1]
    assert active_rule_atoms(rules[0])[0].feature == 1
    assert active_rule_atoms(rules[1])[0].feature == 1
    labels, uncovered, ambiguous = classify_profiles(profiles, rules)
    assert labels == [0, 0, 1, 1]
    assert uncovered == []
    assert ambiguous == []


def test_sygus_respects_bounded_predicate_depth() -> None:
    profiles = [
        _point_profile([0.0, 0.0]),
        _point_profile([-1.0, 0.0]),
        _point_profile([0.0, 1.0]),
    ]

    with pytest.raises(RuntimeError, match="UNSAT"):
        synthesize_rules(
            [[0], [1, 2]],
            profiles,
            n_features=2,
            max_predicates=1,
        )


def _point_profile(values: list[float]) -> SegmentProfile:
    vector = np.array(values, dtype=float)
    return SegmentProfile(lo=vector, hi=vector)
