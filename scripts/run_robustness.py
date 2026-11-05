"""Mandatory experiment 5 and the five robustness tests on Pokec-z (5 seeds each):
edge dropout, edge corruption, degree-preserving rewiring, forest constructions,
embedding dimensions, reduced training-label availability.

Usage: python scripts/run_robustness.py --config configs/robustness.yaml [--only forest dims ...]
"""
import argparse
import time
from functools import partial

import pandas as pd

import _bootstrap  # noqa: F401
from src.experiments.perturb import degree_preserving_rewire, edge_corruption, edge_dropout
from src.experiments.runner import load_cfg, run_jobs
from src.utils import ROOT, load_config, record, setup_logging


def _mod(kind, frac):
    f = {"dropout": edge_dropout, "corruption": edge_corruption, "rewire": degree_preserving_rewire}[kind]
    return partial(_apply, f=f, frac=frac)


def _apply(g, seed, f, frac):
    return f(g, frac, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robustness.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    log = setup_logging("run_robustness")
    rc = load_config(args.config)
    ec = load_config(rc["base"])
    cfg = load_cfg(ec)
    out = ROOT / rc["output_dir"]
    seeds = rc["seeds"]
    common = {"include_feature_baselines": False, "curves": False, "region_track": False}
    plan = []
    for fr in rc["edge_dropout"]:
        plan.append(("dropout", f"dropout{fr}", _mod("dropout", fr), dict(common), 1.0))
    for fr in rc["edge_corruption"]:
        plan.append(("corruption", f"corrupt{fr}", _mod("corruption", fr), dict(common), 1.0))
    for fr in rc["degree_preserving_rewiring"]:
        plan.append(("rewire", f"rewire{fr}", _mod("rewire", fr), dict(common), 1.0))
    for fm in rc["forest_methods"]:
        if fm != "max_jaccard":
            plan.append(("forest", f"forest_{fm}", None, dict(common, forest_method=fm, include_lbp=False), 1.0))
    for d in rc["embedding_dimensions"]:
        if d != 64:
            plan.append(("dims", f"dims{d}", None, dict(common, dims=d, include_lbp=False), 1.0))
    for tf in rc["train_label_fraction"]:
        if tf != 1.0:
            plan.append(("labels", f"labels{tf}", None, dict(common, include_lbp=False), tf))
    if args.only:
        plan = [p for p in plan if p[0] in args.only]
    jobs = [(f"{tag}_seed{s}", s, str(out), cfg, mod, kw, tf) for (_, tag, mod, kw, tf) in plan for s in seeds]
    t = time.time()
    res = run_jobs(jobs, args.n_jobs)
    for r in res:
        log.info("%s", r)
        record(f"{rc['output_dir']}/{r['tag']}_metrics.csv", "python scripts/run_robustness.py --config " + args.config,
               rc["_config_path"], seed=r["seed"], status=r["status"], runtime_s=r.get("runtime_s"),
               warnings=r.get("error", ""))
    frames = []
    for p in sorted(out.glob("*_seed*_metrics.csv")):
        df = pd.read_csv(p)
        df["condition"] = p.name.rsplit("_seed", 1)[0]
        frames.append(df)
    if frames:
        pd.concat(frames).to_csv(out / "all_metrics.csv", index=False)
    log.info("robustness done in %.1fs", time.time() - t)


if __name__ == "__main__":
    main()
