"""Mandatory experiments 1-3 and ablations A-C on the fixed Pokec-z subgraph, all seeds.

Usage: python scripts/run_experiments.py --config configs/experiments.yaml [--seeds 11 22]
Outputs: results/metrics/main/seed<k>_{predictions,embeddings}.npz, _info.json, _metrics.csv, _curves.csv
"""
import argparse
import time

import pandas as pd

import _bootstrap  # noqa: F401
from src.experiments.runner import load_cfg, run_jobs
from src.utils import ROOT, load_config, record, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--n-jobs", type=int)
    args = ap.parse_args()
    log = setup_logging("run_experiments")
    ec = load_config(args.config)
    cfg = load_cfg(ec)
    seeds = args.seeds or ec["seeds"]
    out = ROOT / ec["output_dir"]
    t = time.time()
    jobs = [(f"seed{s}", s, str(out), cfg, None, {}, 1.0) for s in seeds]
    res = run_jobs(jobs, args.n_jobs or ec["n_jobs"])
    for r in res:
        log.info("%s", r)
        record(f"{ec['output_dir']}/{r['tag']}_metrics.csv", "python scripts/run_experiments.py --config " + args.config,
               ec["_config_path"], seed=r["seed"], status=r["status"], runtime_s=r.get("runtime_s"),
               warnings=r.get("error", ""))
    frames = [pd.read_csv(out / f"seed{s}_metrics.csv") for s in seeds if (out / f"seed{s}_metrics.csv").exists()]
    if frames:
        pd.concat(frames).to_csv(out / "all_metrics.csv", index=False)
    log.info("done in %.1fs", time.time() - t)


if __name__ == "__main__":
    main()
