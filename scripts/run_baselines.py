"""Baseline summary (feature LR, SGC, Node2Vec+LR, FairWalk+LR, threshold-adjusted Node2Vec).

Design decision (documented deviation from the suggested CLI): baselines are *fitted* inside
`scripts/run_experiments.py`, in the same process and on the same split and embeddings as the
smoothed models, so every baseline-vs-smoothing comparison is exactly paired. Re-fitting them in a
separate script would duplicate the embedding training and risk unpaired comparisons.
This script therefore reads the saved per-seed results and writes the baseline table; with
`--rerun` it first re-executes the main experiment for the requested seeds.

Usage: python scripts/run_baselines.py --config configs/experiments.yaml [--rerun --seeds 11]
Output: results/tables/table_baselines_per_seed.csv
"""
import argparse
import subprocess
import sys

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import ROOT, load_config, record, setup_logging

BASELINES = ["feat_lr", "sgc", "n2v_lr", "fw_lr", "n2v_thr"]
COLS = ["auroc", "macro_f1", "balanced_acc", "brier", "log_loss", "ece", "dDP_gender", "dEOpp_gender",
        "dEOdds_gender", "R_sel_gender", "worst_group_acc_gender"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    log = setup_logging("run_baselines")
    ec = load_config(args.config)
    seeds = args.seeds or ec["seeds"]
    if args.rerun:
        subprocess.run([sys.executable, str(ROOT / "scripts/run_experiments.py"), "--config", args.config,
                        "--seeds", *map(str, seeds)], check=True)
    out = ROOT / ec["output_dir"]
    rows = []
    for s in seeds:
        p = out / f"seed{s}_metrics.csv"
        if not p.exists():
            log.warning("missing %s (run scripts/run_experiments.py first)", p)
            continue
        df = pd.read_csv(p)
        df = df[(df.method.isin(BASELINES)) & (df.part == "test") & (df.operating_point == "default")]
        rows.append(df[["method"] + COLS].assign(seed=s))
    if not rows:
        raise SystemExit("no saved main results found")
    tab = pd.concat(rows)
    dest = ROOT / "results/tables/table_baselines_per_seed.csv"
    tab.to_csv(dest, index=False)
    log.info("baselines (mean over seeds):\n%s", tab.groupby("method")[COLS].mean().round(3).to_string())
    record(str(dest.relative_to(ROOT)), "python scripts/run_baselines.py --config " + args.config, args.config)


if __name__ == "__main__":
    main()
