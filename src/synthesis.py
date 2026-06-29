from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from src.merging import SegmentProfile


@dataclass(frozen=True)
class IntervalAtom:
    """One grammar atom over a named segment feature."""

    feature: int
    lo: float | None = None
    hi: float | None = None


@dataclass(frozen=True)
class SygusCertificate:
    """Transparent metadata for a CVC5 SyGuS synthesis result."""

    backend: str
    grammar: str
    max_predicates: int
    effective_margin: float
    solution: str
    requested_max_predicates: int | None = None
    requested_margin: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntervalRule:
    """CVC5 SyGuS predicate over segment features."""

    class_id: int
    n_features: int
    atoms: tuple[IntervalAtom, ...]
    certificate: SygusCertificate


def synthesize_rules(
    classes: list[list[int]],
    profiles: list[SegmentProfile],
    n_features: int,
    margin: float = 0.0,
    max_predicates: int | None = None,
) -> list[IntervalRule]:
    """Materialize one bounded SyGuS rule per segment class."""
    rules: list[IntervalRule] = []
    for class_id, members in enumerate(classes):
        positives = [profiles[idx] for idx in members]
        negatives = [
            profiles[idx]
            for other_id, other_members in enumerate(classes)
            if other_id != class_id
            for idx in other_members
        ]
        atoms, certificate = _synthesize_rule_expression(
            positives,
            negatives,
            n_features,
            margin=margin,
            max_predicates=max_predicates,
        )
        rules.append(IntervalRule(
            class_id=class_id,
            n_features=n_features,
            atoms=atoms,
            certificate=certificate,
        ))
    return rules


def active_rule_atoms(rule: IntervalRule) -> tuple[IntervalAtom, ...]:
    """Return the predicate atoms that define a rule."""
    return rule.atoms


def evaluate_rule(rule: IntervalRule, feature_vector: np.ndarray) -> bool:
    """Return whether one segment feature vector satisfies a rule."""
    if feature_vector.shape[0] != rule.n_features:
        raise ValueError("feature dimension does not match rule")
    return all(
        _atom_contains_value(atom, feature_vector[atom.feature])
        for atom in active_rule_atoms(rule)
    )


def rule_covers_profile(rule: IntervalRule, profile: SegmentProfile) -> bool:
    """Return whether a rule covers the whole interval profile."""
    if (
        profile.lo.shape[0] != rule.n_features
        or profile.hi.shape[0] != rule.n_features
    ):
        raise ValueError("profile dimension does not match rule")
    return all(
        _atom_contains_profile(atom, profile)
        for atom in active_rule_atoms(rule)
    )


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


def _synthesize_rule_expression(
    positive_profiles: list[SegmentProfile],
    negative_profiles: list[SegmentProfile],
    n_features: int,
    margin: float,
    max_predicates: int | None,
) -> tuple[tuple[IntervalAtom, ...], SygusCertificate]:
    """Solve the paper's bounded interval SyGuS fragment with CVC5."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if max_predicates is not None and max_predicates < 0:
        raise ValueError("max_predicates must be non-negative")
    if not positive_profiles:
        raise ValueError("positive_profiles must not be empty")
    for profile in [*positive_profiles, *negative_profiles]:
        if (
            profile.lo.shape[0] != n_features
            or profile.hi.shape[0] != n_features
        ):
            raise ValueError("profile dimension does not match n_features")

    result = _synthesize_atoms_cvc5(
        positive_profiles,
        negative_profiles,
        n_features,
        margin=margin,
        max_predicates=max_predicates,
    )
    if result is None:
        raise RuntimeError("Rule synthesis failed: interval grammar is UNSAT")
    return result


def _synthesize_atoms_cvc5(
    positive_profiles: list[SegmentProfile],
    negative_profiles: list[SegmentProfile],
    n_features: int,
    margin: float,
    max_predicates: int | None,
) -> tuple[tuple[IntervalAtom, ...], SygusCertificate] | None:
    cvc5, Kind = _import_cvc5()
    return _solve_sygus_with_cvc5(
        cvc5,
        Kind,
        positive_profiles,
        negative_profiles,
        n_features,
        margin=margin,
        predicate_bound=_max_predicate_bound(n_features, max_predicates),
        requested_max_predicates=max_predicates,
    )


def _solve_sygus_with_cvc5(
    cvc5: Any,
    Kind: Any,
    positive_profiles: list[SegmentProfile],
    negative_profiles: list[SegmentProfile],
    n_features: int,
    margin: float,
    predicate_bound: int,
    requested_max_predicates: int | None,
) -> tuple[tuple[IntervalAtom, ...], SygusCertificate] | None:
    solver = cvc5.Solver()
    solver.setOption("sygus", "true")
    solver.setLogic("LRA")

    real_sort = solver.getRealSort()
    bool_sort = solver.getBooleanSort()
    variables = [
        solver.declareSygusVar(f"x{feature}", real_sort)
        for feature in range(n_features)
    ]
    grammar, grammar_description = _make_interval_grammar(
        solver,
        Kind,
        variables,
        bool_sort,
        _positive_hull_constants(positive_profiles, n_features, margin),
        predicate_bound,
    )
    function = solver.synthFun("R", variables, bool_sort, grammar)

    for profile in positive_profiles:
        lo, hi = _expanded_bounds(profile, margin)
        solver.addSygusConstraint(
            solver.mkTerm(
                Kind.IMPLIES,
                _box_term(solver, Kind, variables, lo, hi),
                _apply_rule(solver, Kind, function, variables),
            )
        )

    for profile in negative_profiles:
        solver.addSygusConstraint(
            solver.mkTerm(
                Kind.IMPLIES,
                _box_term(solver, Kind, variables, profile.lo, profile.hi),
                solver.mkTerm(
                    Kind.NOT,
                    _apply_rule(solver, Kind, function, variables),
                ),
            )
        )

    result = solver.checkSynth()
    if result.hasNoSolution():
        return None
    if result.isUnknown():
        raise RuntimeError(
            "Rule synthesis failed: CVC5 returned UNKNOWN for the SyGuS "
            f"problem with max_predicates={predicate_bound}"
        )
    if not result.hasSolution():
        raise RuntimeError(
            "Rule synthesis failed: CVC5 returned an unsupported SyGuS result "
            f"{result}"
        )

    solution = solver.getSynthSolution(function)
    atoms = _parse_solution_atoms(solution, Kind)
    _verify_sygus_atoms(atoms, positive_profiles, negative_profiles, margin)
    certificate = SygusCertificate(
        backend=f"cvc5-python {getattr(cvc5, '__version__', 'unknown')}",
        grammar=grammar_description,
        max_predicates=predicate_bound,
        effective_margin=margin,
        solution=str(solution),
        requested_max_predicates=requested_max_predicates,
        requested_margin=margin,
        notes=(
            "CVC5 solved a bounded SyGuS grammar over interval atoms",
            "candidate constants are the expanded positive-class hull bounds",
            "max_predicates is the configured grammar predicate bound",
        ),
    )
    return atoms, certificate


def _import_cvc5() -> tuple[Any, Any]:
    try:
        import cvc5
        from cvc5 import Kind
    except ImportError as exc:
        raise RuntimeError(
            "Rule synthesis requires the CVC5 SyGuS backend."
        ) from exc
    return cvc5, Kind


def _max_predicate_bound(
    n_features: int,
    max_predicates: int | None,
) -> int:
    if max_predicates is not None:
        return max_predicates
    return max(1, 2 * n_features)


def _positive_hull_constants(
    positive_profiles: list[SegmentProfile],
    n_features: int,
    margin: float,
) -> list[list[float]]:
    lo_vals = np.min([profile.lo for profile in positive_profiles], axis=0)
    hi_vals = np.max([profile.hi for profile in positive_profiles], axis=0)
    constants: list[list[float]] = []
    for feature in range(n_features):
        feature_constants = {
            float(lo_vals[feature] - margin),
            float(hi_vals[feature] + margin),
        }
        constants.append(sorted(feature_constants))
    return constants


def _make_interval_grammar(
    solver: Any,
    Kind: Any,
    variables: list[Any],
    bool_sort: Any,
    constants_by_feature: list[list[float]],
    predicate_bound: int,
) -> tuple[Any, str]:
    atom_nonterminal = solver.mkVar(bool_sort, "Atom")
    if predicate_bound == 0:
        start = solver.mkVar(bool_sort, "Rule0")
        grammar = solver.mkGrammar(variables, [start, atom_nonterminal])
        grammar.addRule(start, solver.mkBoolean(True))
    else:
        levels = [
            solver.mkVar(bool_sort, f"Rule{idx}")
            for idx in range(predicate_bound)
        ]
        start = levels[-1]
        grammar = solver.mkGrammar(variables, [start, *levels[:-1], atom_nonterminal])
        grammar.addRule(levels[0], solver.mkBoolean(True))
        grammar.addRule(levels[0], atom_nonterminal)
        for idx in range(1, predicate_bound):
            grammar.addRule(levels[idx], solver.mkBoolean(True))
            grammar.addRule(levels[idx], atom_nonterminal)
            grammar.addRule(
                levels[idx],
                solver.mkTerm(Kind.AND, atom_nonterminal, levels[idx - 1]),
            )

    for feature, constants in enumerate(constants_by_feature):
        variable = variables[feature]
        for value in constants:
            constant = _mk_cvc5_real(solver, value)
            grammar.addRule(
                atom_nonterminal,
                solver.mkTerm(Kind.LEQ, constant, variable),
            )
            grammar.addRule(
                atom_nonterminal,
                solver.mkTerm(Kind.LEQ, variable, constant),
            )

    constants_text = "; ".join(
        f"x{feature}: {', '.join(_format_real(value) for value in values)}"
        for feature, values in enumerate(constants_by_feature)
    )
    description = (
        "(synth-fun R ((x0 Real) ... (xn Real)) Bool "
        "((Rule Bool) (Atom Bool)) "
        "(Rule ::= true | Atom | (and Atom Rule)) "
        "(Atom ::= (<= c xi) | (<= xi c))) "
        f"max_predicates={predicate_bound}; constants={{ {constants_text} }}"
    )
    return grammar, description


def _expanded_bounds(
    profile: SegmentProfile,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    return profile.lo - margin, profile.hi + margin


def _box_term(
    solver: Any,
    Kind: Any,
    variables: list[Any],
    lo: np.ndarray,
    hi: np.ndarray,
) -> Any:
    constraints: list[Any] = []
    for feature, variable in enumerate(variables):
        lower = float(lo[feature])
        upper = float(hi[feature])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("profile bounds must be finite")
        if lower > upper:
            raise ValueError("profile lower bound exceeds upper bound")
        constraints.append(
            solver.mkTerm(Kind.LEQ, _mk_cvc5_real(solver, lower), variable)
        )
        constraints.append(
            solver.mkTerm(Kind.LEQ, variable, _mk_cvc5_real(solver, upper))
        )
    return _and_terms(solver, Kind, constraints)


def _apply_rule(
    solver: Any,
    Kind: Any,
    function: Any,
    variables: list[Any],
) -> Any:
    return solver.mkTerm(Kind.APPLY_UF, function, *variables)


def _and_terms(solver: Any, Kind: Any, terms: list[Any]) -> Any:
    if not terms:
        return solver.mkBoolean(True)
    if len(terms) == 1:
        return terms[0]
    return solver.mkTerm(Kind.AND, *terms)


def _mk_cvc5_real(solver: Any, value: float) -> Any:
    if not np.isfinite(value):
        raise ValueError("SyGuS constants must be finite real numbers")
    rational = Fraction(float(value)).limit_denominator(10**12)
    return solver.mkReal(rational.numerator, rational.denominator)


def _parse_solution_atoms(solution: Any, Kind: Any) -> tuple[IntervalAtom, ...]:
    children = list(solution)
    if solution.getKind() == Kind.LAMBDA and len(children) == 2:
        body = children[1]
    else:
        body = solution
    raw_atoms = _parse_boolean_term_atoms(body, Kind)
    return _merge_atoms(raw_atoms)


def _parse_boolean_term_atoms(term: Any, Kind: Any) -> list[IntervalAtom]:
    if term.isBooleanValue():
        if term.getBooleanValue():
            return []
        raise RuntimeError("CVC5 returned the unsatisfiable predicate false")

    kind = term.getKind()
    if kind == Kind.AND:
        atoms: list[IntervalAtom] = []
        for child in term:
            atoms.extend(_parse_boolean_term_atoms(child, Kind))
        return atoms
    if kind == Kind.LEQ:
        return [_parse_leq_atom(term)]

    raise RuntimeError(f"CVC5 returned a predicate outside the interval grammar: {term}")


def _parse_leq_atom(term: Any) -> IntervalAtom:
    left, right = list(term)
    left_symbol = _term_symbol(left)
    right_symbol = _term_symbol(right)
    left_real = _term_real(left)
    right_real = _term_real(right)

    if left_symbol is not None and right_real is not None:
        return IntervalAtom(
            feature=_feature_index(left_symbol),
            hi=right_real,
        )
    if left_real is not None and right_symbol is not None:
        return IntervalAtom(
            feature=_feature_index(right_symbol),
            lo=left_real,
        )
    if left_real is not None and right_real is not None:
        if left_real <= right_real:
            return IntervalAtom(feature=0)
        raise RuntimeError("CVC5 returned a contradictory constant comparison")
    raise RuntimeError(f"CVC5 returned an unsupported interval atom: {term}")


def _merge_atoms(atoms: list[IntervalAtom]) -> tuple[IntervalAtom, ...]:
    merged: dict[int, tuple[float | None, float | None]] = {}
    for atom in atoms:
        lo, hi = merged.get(atom.feature, (None, None))
        if atom.lo is not None:
            lo = atom.lo if lo is None else max(lo, atom.lo)
        if atom.hi is not None:
            hi = atom.hi if hi is None else min(hi, atom.hi)
        if lo is not None and hi is not None and lo > hi:
            raise RuntimeError("CVC5 returned an empty interval predicate")
        merged[atom.feature] = (lo, hi)

    normalized = [
        IntervalAtom(feature=feature, lo=lo, hi=hi)
        for feature, (lo, hi) in sorted(merged.items())
        if lo is not None or hi is not None
    ]
    return tuple(normalized)


def _term_symbol(term: Any) -> str | None:
    if not term.hasSymbol():
        return None
    return str(term.getSymbol())


def _term_real(term: Any) -> float | None:
    if not term.isRealValue():
        return None
    return float(term.getRealValue())


def _feature_index(symbol: str) -> int:
    if not symbol.startswith("x"):
        raise RuntimeError(f"unexpected SyGuS variable symbol {symbol!r}")
    return int(symbol[1:])


def _verify_sygus_atoms(
    atoms: tuple[IntervalAtom, ...],
    positive_profiles: list[SegmentProfile],
    negative_profiles: list[SegmentProfile],
    margin: float,
) -> None:
    for profile in positive_profiles:
        expanded = SegmentProfile(*_expanded_bounds(profile, margin))
        if not all(_atom_contains_profile(atom, expanded) for atom in atoms):
            raise RuntimeError(
                "CVC5 SyGuS solution failed local positive-profile verification"
            )
    for profile in negative_profiles:
        if all(_atom_intersects_profile(atom, profile) for atom in atoms):
            raise RuntimeError(
                "CVC5 SyGuS solution failed local negative-profile verification"
            )


def _format_real(value: float) -> str:
    return f"{value:.12g}"


def _atom_contains_value(atom: IntervalAtom, value: float) -> bool:
    return (
        (atom.lo is None or value >= atom.lo)
        and (atom.hi is None or value <= atom.hi)
    )


def _atom_contains_profile(atom: IntervalAtom, profile: SegmentProfile) -> bool:
    return (
        (atom.lo is None or profile.lo[atom.feature] >= atom.lo)
        and (atom.hi is None or profile.hi[atom.feature] <= atom.hi)
    )


def _atom_intersects_profile(atom: IntervalAtom, profile: SegmentProfile) -> bool:
    return (
        (atom.lo is None or profile.hi[atom.feature] >= atom.lo)
        and (atom.hi is None or profile.lo[atom.feature] <= atom.hi)
    )
