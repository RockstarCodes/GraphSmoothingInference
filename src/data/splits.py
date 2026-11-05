"""Transductive train/validation/test label splits (repeated, seeded).

Stratified by (Y, gender) so every split keeps both classes in both gender
groups. Proportions follow FairGNN's 50/25/25 partition of labelled nodes.
Optional ``train_fraction`` keeps only that share of the training labels
(robustness: reduced label availability); the dropped training nodes become
unlabelled (latent) at inference and are excluded from evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def role(self, n: int) -> np.ndarray:
        r = np.full(n, "unlabelled", dtype=object)
        r[self.train] = "train"; r[self.val] = "val"; r[self.test] = "test"
        return r


def stratified_split(y: np.ndarray, strata: np.ndarray, seed: int, frac=(0.5, 0.25, 0.25),
                     train_fraction: float = 1.0) -> Split:
    if not np.isclose(sum(frac), 1.0):
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    key = np.array([f"{a}|{b}" for a, b in zip(y, strata)])
    tr, va, te = [], [], []
    for k in sorted(set(key)):
        idx = np.flatnonzero(key == k)
        rng.shuffle(idx)
        n_tr = int(round(frac[0] * len(idx))); n_va = int(round(frac[1] * len(idx)))
        tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va]); te.append(idx[n_tr + n_va:])
    train = np.sort(np.concatenate(tr))
    if train_fraction < 1.0:
        keep = rng.random(len(train)) < train_fraction
        train = train[keep]
    s = Split(train=train, val=np.sort(np.concatenate(va)), test=np.sort(np.concatenate(te)), seed=seed)
    check_disjoint(s)
    return s


def check_disjoint(s: Split) -> None:
    a, b, c = set(s.train.tolist()), set(s.val.tolist()), set(s.test.tolist())
    if a & b or a & c or b & c:
        raise AssertionError("train/val/test overlap")
