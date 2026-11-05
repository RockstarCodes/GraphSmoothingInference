"""Discrete factor tables used by the exact-inference validators.

A :class:`Factor` stores a nonnegative table over an ordered scope of discrete
variables. Axis ``k`` of ``values`` is indexed by the state of ``scope[k]``.
For the binary node-label model every variable has cardinality 2 and state
index 0 means ``Y=0`` and index 1 means ``Y=1``.

These routines are written for clarity and for checking the specialised forest
sum-product implementation; they are exponential in the factor width and are
only used on small fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass
class Factor:
    scope: Tuple[int, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        self.scope = tuple(int(v) for v in self.scope)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != len(self.scope):
            raise ValueError(f"factor table has {self.values.ndim} axes but scope {self.scope}")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError(f"duplicate variable in scope {self.scope}")
        if np.any(self.values < 0) or not np.all(np.isfinite(self.values)):
            raise ValueError("factor values must be finite and nonnegative")

    @property
    def card(self) -> Dict[int, int]:
        return {v: self.values.shape[k] for k, v in enumerate(self.scope)}

    # ------------------------------------------------------------------ algebra
    def _expand(self, scope: Sequence[int], card: Dict[int, int]) -> np.ndarray:
        """Return values broadcastable to ``scope`` (a superset of self.scope)."""
        perm = [self.scope.index(v) for v in scope if v in self.scope]
        arr = np.transpose(self.values, perm)
        shape = [card[v] if v in self.scope else 1 for v in scope]
        return arr.reshape(shape)

    def product(self, other: "Factor") -> "Factor":
        scope = list(self.scope) + [v for v in other.scope if v not in self.scope]
        card = {**self.card, **other.card}
        for v in set(self.scope) & set(other.scope):
            if self.card[v] != other.card[v]:
                raise ValueError(f"cardinality mismatch for variable {v}")
        vals = self._expand(scope, card) * other._expand(scope, card)
        return Factor(tuple(scope), np.broadcast_to(vals, [card[v] for v in scope]).copy())

    def marginalize(self, variables: Iterable[int]) -> "Factor":
        variables = [v for v in variables if v in self.scope]
        axes = tuple(self.scope.index(v) for v in variables)
        keep = tuple(v for v in self.scope if v not in variables)
        return Factor(keep, self.values.sum(axis=axes) if axes else self.values.copy())

    def reduce(self, evidence: Dict[int, int]) -> "Factor":
        """Condition on evidence by slicing (removes evidence variables from scope)."""
        idx = []
        keep = []
        for v in self.scope:
            if v in evidence:
                idx.append(int(evidence[v]))
            else:
                idx.append(slice(None))
                keep.append(v)
        return Factor(tuple(keep), np.asarray(self.values[tuple(idx)]))

    def normalize(self) -> "Factor":
        z = self.values.sum()
        if z <= 0:
            raise ZeroDivisionError("cannot normalise a factor with zero total mass")
        return Factor(self.scope, self.values / z)

    def value(self, assignment: Dict[int, int]) -> float:
        return float(self.values[tuple(int(assignment[v]) for v in self.scope)])


def unit_factor(scope: Sequence[int] = ()) -> Factor:
    return Factor(tuple(scope), np.ones([2] * len(scope)))


def product_all(factors: Sequence[Factor]) -> Factor:
    out = Factor((), np.array(1.0))
    for f in factors:
        out = out.product(f)
    return out


def indicator_factor(var: int, state: int, card: int = 2) -> Factor:
    """Evidence factor delta(Y_var = state)."""
    vals = np.zeros(card)
    vals[int(state)] = 1.0
    return Factor((var,), vals)
