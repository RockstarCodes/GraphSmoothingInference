"""Mandatory experiment 4 (SIMULATED graphs): vary demographic homophily, degree imbalance,
label homophily and training-label availability; plus a small Node2Vec/FairWalk check.

Usage: python scripts/run_synthetic.py --config configs/synthetic.yaml [--smoke]
"""
import argparse
import itertools
import time
from dataclasses import replace
from multiprocessing import get_context

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.synthetic.experiment import one_realisation
from src.synthetic.generator import SyntheticConfig
from src.utils import ROOT, dump_json, load_config, record, setup_logging


def _base(cfg):
    b = dict(cfg["base"])
    return SyntheticConfig(n_nodes=b["n_nodes"], group_share=b["group_share"], prevalence=tuple(b["prevalence"]),
                           mean_degree=b["mean_degree"], group_homophily=b["group_homophily"],
                           label_homophily=b["label_homophily"], degree_ratio=b["degree_ratio"],
                           edge_corruption=b["edge_corruption"]), b


def _task(args):
    scen, sc, seed, tf, lam, eta_grid, strength, noise = args
    r = one_realisation(sc, seed, tf, lam, eta_grid, strength, noise)
    return {**scen, **r}


def _emb_task(args):
    """Node2Vec vs FairWalk on a synthetic graph (smaller, 3 realisations)."""
    from src.baselines.classifiers import fit_lr, sensitive_predictability
    from src.data.splits import stratified_split
    from src.embeddings.random_walks import WalkConfig, train_embeddings
    from src.evaluation.decisions import best_threshold
    from src.fairness.metrics import evaluate_all
    from src.graphical_model.attenuation import infer
    from src.graphical_model.forest import build_forest
    from src.synthetic.generator import generate
    sc, seed, lam = args
    g = generate(sc, seed)
    sp = stratified_split(g.y, g.a, seed)
    ev = {int(i): int(g.y[i]) for i in sp.train}
    forest = build_forest(g.n, g.edges)
    rows = []
    for name, grp in (("n2v", None), ("fw", g.a)):
        H = train_embeddings(g.n, g.edges, WalkConfig(dimensions=32, walks_per_node=5), seed, grp)["embedding"]
        sens = sensitive_predictability(H, g.a, sp.train, sp.test, seed)
        p = fit_lr(H, g.y, sp.train, sp.val, [0.01, 0.1, 1, 10], seed)["p"]
        for stage, q in (("unary", p), ("uniform", infer(g.n, p, forest, np.full(len(forest), lam), ev, 1e-6))):
            t = best_threshold(g.y[sp.val], q[sp.val])
            m = evaluate_all(g.y[sp.test], q[sp.test], (q[sp.test] >= t).astype(int), g.a[sp.test],
                             np.zeros(len(sp.test), int))
            rows.append({"group_homophily": sc.group_homophily, "seed": seed, "embedding": name, "stage": stage,
                         "sens_auroc": sens["auroc"], "auroc": m["auroc"], "macro_f1": m["macro_f1"],
                         "dEOpp": m["dEOpp_gender"], "dDP": m["dDP_gender"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny run for the smoke test")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    log = setup_logging("run_synthetic")
    cfg = load_config(args.config)
    base, b = _base(cfg)
    R = 2 if args.smoke else int(cfg["realizations"])
    out = ROOT / (cfg["output_dir"] + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    gr = cfg["grid"]
    if args.smoke:
        base = replace(base, n_nodes=300)
        gr = {"group_homophily": [0.5, 0.9], "degree_ratio": [1.0], "label_homophily": [0.7], "train_fraction": [0.5]}
    tasks = []
    # (i) 2-D grid: demographic homophily x degree imbalance (other factors at base)
    for h, dr in itertools.product(gr["group_homophily"], gr["degree_ratio"]):
        sc = replace(base, group_homophily=h, degree_ratio=dr)
        for r in range(R):
            tasks.append(({"design": "homophily_x_degree", "group_homophily": h, "degree_ratio": dr,
                           "label_homophily": base.label_homophily, "train_fraction": b["train_fraction"]},
                          sc, 1000 + r, b["train_fraction"], cfg["lambda"], cfg["eta_grid"],
                          b["unary_strength"], b["unary_noise"]))
    # (ii) one-factor-at-a-time: label homophily, training-label availability, at group_homophily 0.8
    for lh in gr["label_homophily"]:
        sc = replace(base, group_homophily=0.8, label_homophily=lh)
        for r in range(R):
            tasks.append(({"design": "label_homophily", "group_homophily": 0.8, "degree_ratio": 1.0,
                           "label_homophily": lh, "train_fraction": b["train_fraction"]}, sc, 2000 + r,
                          b["train_fraction"], cfg["lambda"], cfg["eta_grid"], b["unary_strength"], b["unary_noise"]))
    for tf in gr["train_fraction"]:
        sc = replace(base, group_homophily=0.8)
        for r in range(R):
            tasks.append(({"design": "train_fraction", "group_homophily": 0.8, "degree_ratio": 1.0,
                           "label_homophily": base.label_homophily, "train_fraction": tf}, sc, 3000 + r, tf,
                          cfg["lambda"], cfg["eta_grid"], b["unary_strength"], b["unary_noise"]))
    # (iii) prevalence gap off (control): no base-rate difference
    for h in (0.5, 0.9):
        sc = replace(base, group_homophily=h, prevalence=(0.5, 0.5))
        for r in range(R):
            tasks.append(({"design": "no_base_rate_gap", "group_homophily": h, "degree_ratio": 1.0,
                           "label_homophily": base.label_homophily, "train_fraction": b["train_fraction"]}, sc,
                          4000 + r, b["train_fraction"], cfg["lambda"], cfg["eta_grid"], b["unary_strength"],
                          b["unary_noise"]))
    t = time.time()
    with get_context("fork").Pool(args.n_jobs) as pool:
        rows = pool.map(_task, tasks, chunksize=2)
    df = pd.DataFrame(rows)
    df.to_csv(out / "synthetic_realisations.csv", index=False)
    log.info("synthetic grid: %d realisations in %.1fs", len(df), time.time() - t)
    record(str((out / "synthetic_realisations.csv").relative_to(ROOT)),
           "python scripts/run_synthetic.py --config " + args.config + (" --smoke" if args.smoke else ""),
           cfg["_config_path"], seed="1000+r..4000+r", runtime_s=time.time() - t, data_ver="synthetic")
    # embedding check
    ec = cfg["embedding_check"]
    etasks = [(replace(base, n_nodes=(200 if args.smoke else ec["n_nodes"]), group_homophily=h), 5000 + r, cfg["lambda"])
              for h in ec["group_homophily"] for r in range(1 if args.smoke else ec["realizations"])]
    t = time.time()
    with get_context("fork").Pool(args.n_jobs) as pool:
        erows = [x for rs in pool.map(_emb_task, etasks, chunksize=1) for x in rs]
    pd.DataFrame(erows).to_csv(out / "synthetic_embedding_check.csv", index=False)
    record(str((out / "synthetic_embedding_check.csv").relative_to(ROOT)),
           "python scripts/run_synthetic.py --config " + args.config + (" --smoke" if args.smoke else ""),
           cfg["_config_path"], seed="5000+r", runtime_s=time.time() - t, data_ver="synthetic")
    log.info("embedding check done in %.1fs", time.time() - t)


if __name__ == "__main__":
    main()
