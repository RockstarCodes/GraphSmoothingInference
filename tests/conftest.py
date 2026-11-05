import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graphical_model.pairwise_model import PairwiseCMN  # noqa: E402


def random_model(n, edges, seed=0, evidence=None, lam_scale=1.5):
    rng = np.random.default_rng(seed)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    return PairwiseCMN(n=n, unary_p=rng.uniform(0.05, 0.95, n), edges=edges,
                       lambdas=rng.uniform(0, lam_scale, len(edges)), evidence=evidence or {})


def random_tree(n, seed):
    rng = np.random.default_rng(seed)
    return np.array([(int(rng.integers(0, i)), i) for i in range(1, n)], dtype=int).reshape(-1, 2)


@pytest.fixture
def tree_fixture():
    # two tree components + one isolated node (node 9)
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (5, 6), (6, 7), (6, 8)]
    return random_model(10, edges, seed=3, evidence={2: 1, 7: 0})
