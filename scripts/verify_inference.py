"""Inference-correctness evidence: enumeration vs forest sum-product vs variable elimination
vs clique tree on random tiny models (forests and, for VE/clique tree/enumeration, loopy graphs),
with evidence, isolated nodes and multiple components; elimination-order invariance; Markov-blanket
and joint conditional-independence checks; plus a Markov-blanket demonstration on the real forest.

Usage: python scripts/verify_inference.py --config configs/model.yaml
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.graphical_model.enumeration import enumerate_joint, enumerate_marginals
from src.graphical_model.forest import build_forest
from src.graphical_model.junction_tree import CliqueTree
from src.graphical_model.pairwise_model import PairwiseCMN
from src.graphical_model.sum_product import forest_sum_product
from src.graphical_model.variable_elimination import ve_all_marginals, ve_marginal
from src.utils import ROOT, dump_json, load_config, record, setup_logging


def random_instance(rng, max_nodes, forest_only):
    n = int(rng.integers(2, max_nodes + 1))
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    edges = [e for e in edges if rng.random() > 0.25]                    # several components / isolates
    if not forest_only:
        for _ in range(int(rng.integers(1, 4))):
            a, b = sorted(rng.choice(n, 2, replace=False)) if n > 2 else (0, 1)
            if (a, b) not in edges and a != b:
                edges.append((int(a), int(b)))
    obs = rng.choice(n, size=int(rng.integers(0, n // 2 + 1)), replace=False)
    ev = {int(i): int(rng.integers(0, 2)) for i in obs}
    return PairwiseCMN(n, rng.uniform(0.02, 0.98, n), np.array(edges, dtype=int).reshape(-1, 2),
                       rng.uniform(0, 3, len(edges)), ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model.yaml")
    args = ap.parse_args()
    log = setup_logging("verify_inference")
    cfg = load_config(args.config)
    v = cfg["verification"]
    tol = float(cfg["inference"]["verification_tolerance"])
    rng = np.random.default_rng(v["seed"])
    t = time.time()
    rows = []
    for k in range(int(v["n_random_fixtures"])):
        forest_only = k % 2 == 0
        m = random_instance(rng, int(v["max_nodes"]), forest_only)
        en, logz = enumerate_marginals(m)
        fs = m.factors()
        ve = ve_all_marginals(fs, m.n)
        ct_obj = CliqueTree(fs)
        ct = ct_obj.all_marginals(m.n)
        order = [int(x) for x in rng.permutation(m.n)]
        q = int(rng.integers(m.n))
        ve_perm = ve_marginal(fs, q, [x for x in order if x != q])
        row = {"fixture": k, "structure": "forest" if forest_only else "loopy", "n": m.n, "edges": len(m.edges),
               "evidence": len(m.evidence), "max_abs_ve_vs_enum": float(np.max(np.abs(ve - en))),
               "max_abs_ct_vs_enum": float(np.max(np.abs(ct - en))),
               "max_abs_ve_vs_ct": float(np.max(np.abs(ve - ct))),
               "clique_tree_calibration": ct_obj.check_calibration(),
               "elimination_order_gap": float(np.max(np.abs(ve_perm - ve[q]))),
               "max_abs_sp_vs_enum": np.nan, "logz_sp_vs_enum": np.nan}
        if forest_only:
            sp = forest_sum_product(m)
            row["max_abs_sp_vs_enum"] = float(np.max(np.abs(sp.marginals - en)))
            row["logz_sp_vs_enum"] = float(abs(sp.log_z - logz))
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results/metrics"
    df.to_csv(out_dir / "inference_verification.csv", index=False)
    cols = ["max_abs_sp_vs_enum", "max_abs_ve_vs_enum", "max_abs_ct_vs_enum", "max_abs_ve_vs_ct",
            "elimination_order_gap", "clique_tree_calibration", "logz_sp_vs_enum"]
    summary = {c: float(df[c].max(skipna=True)) for c in cols}
    summary.update({"tolerance": tol, "fixtures": len(df), "forest_fixtures": int((df.structure == "forest").sum()),
                    "loopy_fixtures": int((df.structure == "loopy").sum()),
                    "all_within_tolerance": bool(all(summary[c] <= tol for c in cols))})
    # Markov-blanket demonstration on the real Pokec-z forest (structure only; no labels used)
    try:
        from src.data.prepared import Prepared
        prep = Prepared.load(ROOT / "data/processed/pokec_z_core.npz")
        forest = build_forest(prep.n, prep.edges, cfg["inference"]["forest_method"])
        deg = np.bincount(forest.ravel(), minlength=prep.n)
        node = int(np.argmax(deg))
        mb = sorted(set(forest[forest[:, 0] == node, 1].tolist()) | set(forest[forest[:, 1] == node, 0].tolist()))
        lam = 0.5
        demo = []
        for k_pos in range(len(mb) + 1):
            # closed form P(Y_i = 1 | k of its |MB| neighbours labelled 1), unary p_i = 0.5
            s1 = k_pos * lam; s0 = (len(mb) - k_pos) * lam
            demo.append({"positives_in_blanket": k_pos, "p_y1_given_blanket": float(np.exp(s1) / (np.exp(s1) + np.exp(s0)))})
        summary["markov_blanket_demo"] = {"node_local_index": node, "forest_degree": int(deg[node]),
                                          "graph_degree": int(np.bincount(prep.edges.ravel(), minlength=prep.n)[node]),
                                          "lambda": lam, "unary_p": 0.5, "table": demo}
    except FileNotFoundError:
        summary["markov_blanket_demo"] = "skipped: prepared data not available"
    dump_json(summary, "results/metrics/inference_verification_summary.json")
    log.info("verification: %s", {k: summary[k] for k in cols + ["all_within_tolerance"]})
    record("results/metrics/inference_verification.csv", "python scripts/verify_inference.py --config " + args.config,
           cfg["_config_path"], seed=v["seed"], runtime_s=time.time() - t,
           status="ok" if summary["all_within_tolerance"] else "FAILED_TOLERANCE")
    if not summary["all_within_tolerance"]:
        raise SystemExit("inference verification failed tolerance")


if __name__ == "__main__":
    main()
