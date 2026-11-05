"""Optional loopy belief propagation on the original (cyclic) graph.

Approximate. Used only as an exploratory reference for "what is lost by the
forest projection"; never as the core inference algorithm.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from .pairwise_model import PairwiseCMN


def loopy_bp(model: PairwiseCMN, max_iter: int = 200, damping: float = 0.5,
             tol: float = 1e-8) -> Tuple[np.ndarray, bool, int]:
    n = model.n
    e = model.edges
    m = len(e)
    lnode = np.stack([np.log1p(-model.unary_p), np.log(model.unary_p)], axis=1)
    for i, y in model.evidence.items():
        lnode[i, 1 - y] = -1e30
    # directed edge d: 0..m-1 is a->b, m..2m-1 is b->a
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    lam = np.concatenate([model.lambdas, model.lambdas])
    rev = np.concatenate([np.arange(m, 2 * m), np.arange(m)])
    msg = np.zeros((2 * m, 2))
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        inc = lnode.copy()
        np.add.at(inc, dst, msg)
        cav = inc[src] - msg[rev]                       # exclude message from target
        new0 = np.logaddexp(cav[:, 0] + lam, cav[:, 1])
        new1 = np.logaddexp(cav[:, 0], cav[:, 1] + lam)
        new = np.stack([new0, new1], axis=1)
        new -= logsumexp(new, axis=1, keepdims=True)
        upd = damping * msg + (1 - damping) * new
        delta = np.max(np.abs(upd - msg)) if len(msg) else 0.0
        msg = upd
        if delta < tol:
            converged = True
            break
    inc = lnode.copy()
    np.add.at(inc, dst, msg)
    p = np.exp(inc - logsumexp(inc, axis=1, keepdims=True))
    return p, converged, it
