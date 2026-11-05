"""Statistical evaluation of the saved Pokec-z runs.

* Training uncertainty: mean, sd, min, max of each test metric across the 5 refits (seeds).
* Fixed-model evaluation uncertainty: graph-block bootstrap (Louvain blocks on the fixed
  subgraph, 1,000 replicates). In every replicate the same blocks are drawn for all methods and
  all seeds; the paired effect a - b is averaged over seeds. Percentile 95% intervals.
* Confirmatory family (declared in configs/evaluation.yaml before test inspection): two-sided
  bootstrap p-values, Holm-adjusted. Everything else is exploratory (unadjusted, labelled so).
* Group tables with denominators (pooled over seeds, per seed in the CSV).

Usage: python scripts/evaluate.py --config configs/evaluation.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.prepared import Prepared
from src.fairness.metrics import group_rates
from src.statistics.bootstrap import bootstrap_p_value, cohens_dz, holm, louvain_blocks, percentile_ci
from src.utils import ROOT, dump_json, load_config, record, setup_logging

KEY_METHODS = ["feat_lr", "sgc", "n2v_lr", "fw_lr", "n2v_thr", "n2v_nopair", "fw_nopair", "n2v_uniform", "fw_uniform",
               "n2v_global", "fw_global", "n2v_hca", "fw_hca", "n2v_lbp_uniform", "fw_lbp_uniform",
               "n2v_hca_region", "fw_hca_region", "fwr_lr", "fwr_uniform", "fwr_hca_region", "fwr_global_region",
               "n2v_global_region", "fw_global_region"]
METRICS = ["dEOpp_g", "dDP_g", "dEOdds_g", "dEOpp_r", "dDP_r", "macro_f1", "balanced_acc", "brier", "log_loss"]

EXPLORATORY = [
    ("E1", "fw_lr", "n2v_lr", "embedding-only mitigation (FairWalk vs Node2Vec), no smoothing"),
    ("E2", "n2v_thr", "n2v_lr", "decision-only mitigation (group thresholds) vs Node2Vec"),
    ("E3", "n2v_hca", "n2v_uniform", "HCA vs uniform, Node2Vec embeddings (ablation C)"),
    ("E4", "n2v_hca", "n2v_global", "HCA vs global scalar, Node2Vec (ablation B)"),
    ("E5", "fw_nopair", "fw_lr", "no-pairwise ablation equals unary (ablation A)"),
    ("E6", "fw_hca", "n2v_thr", "proposed vs decision-only mitigation"),
    ("E7", "fw_hca", "fw_lr", "proposed vs embedding-only mitigation"),
    ("E8", "fw_uniform", "fw_lbp_uniform", "forest exact vs loopy BP on full graph (FairWalk)"),
    ("E9", "fwr_uniform", "fwr_lr", "uniform smoothing on region-FairWalk (region track)"),
    ("E10", "fwr_hca_region", "fwr_uniform", "region-HCA vs uniform (region track)"),
    ("E11", "fwr_lr", "n2v_lr", "region-FairWalk vs Node2Vec (region track)"),
    ("E12", "n2v_uniform", "n2v_lr", "uniform smoothing, Node2Vec (all metrics)"),
    ("E13", "fwr_hca_region", "fwr_global_region", "region-HCA vs global scalar (region track)"),
]


def fast_metrics(y, p, d, a, r):
    out = {}
    for tag, grp in (("g", a), ("r", r)):
        s, t, f = [], [], []
        for v in (0, 1):
            m = grp == v
            s.append(d[m].mean() if m.any() else np.nan)
            mp, mn = m & (y == 1), m & (y == 0)
            t.append(d[mp].mean() if mp.any() else np.nan)
            f.append(d[mn].mean() if mn.any() else np.nan)
        out[f"dDP_{tag}"] = abs(s[0] - s[1])
        out[f"dEOpp_{tag}"] = abs(t[0] - t[1])
        out[f"dEOdds_{tag}"] = max(abs(t[0] - t[1]), abs(f[0] - f[1]))
    tp = np.sum((d == 1) & (y == 1)); fp = np.sum((d == 1) & (y == 0)); fn = np.sum((d == 0) & (y == 1))
    tn = np.sum((d == 0) & (y == 0))
    f1p = 2 * tp / max(2 * tp + fp + fn, 1); f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    out["macro_f1"] = (f1p + f1n) / 2
    out["balanced_acc"] = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["log_loss"] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--replicates", type=int)
    args = ap.parse_args()
    log = setup_logging("evaluate")
    cfg = load_config(args.config)
    t0 = time.time()
    main_dir = ROOT / cfg["main_dir"]
    tables = ROOT / cfg["tables_dir"]; tables.mkdir(parents=True, exist_ok=True)
    allm = pd.read_csv(main_dir / "all_metrics.csv")
    test = allm[allm.part == "test"]
    num = test.select_dtypes("number").columns.drop("seed")
    summ = test.groupby(["operating_point", "method"])[num].agg(["mean", "std", "min", "max"])
    summ.columns = [f"{a}__{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(tables / "main_summary_training_uncertainty.csv", index=False)

    prep = Prepared.load(ROOT / "data/processed/pokec_z_core.npz")
    bs = cfg["bootstrap"]
    B = args.replicates or int(bs["replicates"])
    blocks = louvain_blocks(prep.n, prep.edges, seed=int(bs["louvain_seed"]))
    nb = int(blocks.max()) + 1
    seeds = sorted(test.seed.unique())
    preds = {s: np.load(main_dir / f"seed{s}_predictions.npz") for s in seeds}
    y, a, r = prep.y, prep.gender, prep.region
    # point estimates per seed (decisions at both operating points)
    ops = {"default": "dec__", "matched_utility": "decmu__"}
    rng = np.random.default_rng(int(bs["seed"]))
    draws = [rng.integers(0, nb, nb) for _ in range(B)]
    boot = {op: {m: {k: np.full((B, len(seeds)), np.nan) for k in METRICS} for m in KEY_METHODS} for op in ops}
    point = {op: {m: {k: np.full(len(seeds), np.nan) for k in METRICS} for m in KEY_METHODS} for op in ops}
    for si, s in enumerate(seeds):
        z = preds[s]
        te = z["test"]
        members = {b: te[blocks[te] == b] for b in range(nb)}
        idx_list = [np.concatenate([members[b] for b in dr]) for dr in draws]
        for op, pre in ops.items():
            for m in KEY_METHODS:
                if f"score__{m}" not in z.files or f"{pre}{m}" not in z.files:
                    continue
                p, d = z[f"score__{m}"], z[f"{pre}{m}"]
                for k, v in fast_metrics(y[te], p[te], d[te], a[te], r[te]).items():
                    point[op][m][k][si] = v
                for bi, idx in enumerate(idx_list):
                    for k, v in fast_metrics(y[idx], p[idx], d[idx], a[idx], r[idx]).items():
                        boot[op][m][k][bi, si] = v
    log.info("bootstrap computed in %.1fs", time.time() - t0)

    def compare(op, ma, mb, metric):
        diff_b = np.nanmean(boot[op][ma][metric] - boot[op][mb][metric], axis=1)
        diff_seeds = point[op][ma][metric] - point[op][mb][metric]
        lo, hi = percentile_ci(diff_b, bs["level"])
        return {"operating_point": op, "a": ma, "b": mb, "metric": metric,
                "estimate_mean_over_seeds": float(np.nanmean(diff_seeds)), "ci95_low": lo, "ci95_high": hi,
                "p_unadjusted": bootstrap_p_value(diff_b),
                "a_mean": float(np.nanmean(point[op][ma][metric])), "b_mean": float(np.nanmean(point[op][mb][metric])),
                "seed_diff_sd": float(np.nanstd(diff_seeds, ddof=1)), "seed_diff_min": float(np.nanmin(diff_seeds)),
                "seed_diff_max": float(np.nanmax(diff_seeds)), "dz_over_seeds": cohens_dz(diff_seeds),
                "per_seed": json.dumps([round(float(x), 5) for x in diff_seeds])}

    conf = []
    for h in cfg["confirmatory_family"]:
        row = compare("default", h["a"], h["b"], h["metric"].replace("_gender", "_g"))
        conf.append({"id": h["id"], "hypothesis": h["alternative"], **row})
    cdf = pd.DataFrame(conf)
    cdf["p_holm"] = holm(cdf.p_unadjusted.values)
    cdf.to_csv(tables / "confirmatory_tests.csv", index=False)
    expl = []
    for eid, ma, mb, desc in EXPLORATORY:
        for op in ops:
            for metric in METRICS:
                if np.all(np.isnan(point[op][ma][metric])) or np.all(np.isnan(point[op][mb][metric])):
                    continue
                expl.append({"id": eid, "description": desc, **compare(op, ma, mb, metric)})
    for h in cfg["confirmatory_family"]:   # all metrics of the confirmatory pairs, as exploratory context
        for op in ops:
            for metric in METRICS:
                expl.append({"id": h["id"] + "-ctx", "description": h["alternative"],
                             **compare(op, h["a"], h["b"], metric)})
    pd.DataFrame(expl).to_csv(tables / "exploratory_paired_effects.csv", index=False)
    # per-method bootstrap intervals of levels (seed-averaged)
    lv = []
    for op in ops:
        for m in KEY_METHODS:
            for k in METRICS:
                b = np.nanmean(boot[op][m][k], axis=1)
                if np.all(np.isnan(b)):
                    continue
                lo, hi = percentile_ci(b, bs["level"])
                lv.append({"operating_point": op, "method": m, "metric": k,
                           "mean_over_seeds": float(np.nanmean(point[op][m][k])), "ci95_low": lo, "ci95_high": hi,
                           "seed_sd": float(np.nanstd(point[op][m][k], ddof=1))})
    pd.DataFrame(lv).to_csv(tables / "method_levels_block_bootstrap.csv", index=False)
    # group tables with denominators (pooled over seeds; per-seed rows kept)
    grows = []
    inter = np.array([f"g{x}_r{w}" for x, w in zip(a, r)])
    for s in seeds:
        z = preds[s]; te = z["test"]
        for m in KEY_METHODS:
            if f"dec__{m}" not in z.files:
                continue
            for gname, grp in (("gender", a), ("region", r), ("gender_x_region", inter)):
                t = group_rates(y[te], z[f"dec__{m}"][te], grp[te])
                t.insert(0, "attribute", gname); t.insert(0, "method", m); t.insert(0, "seed", s)
                grows.append(t)
    gdf = pd.concat(grows)
    gdf.to_csv(tables / "group_rates_per_seed.csv", index=False)
    dump_json({"n_blocks": nb, "block_sizes": np.bincount(blocks).tolist(), "replicates": B,
               "inter_block_edge_share": float(np.mean(blocks[prep.edges[:, 0]] != blocks[prep.edges[:, 1]])),
               "seeds": [int(s) for s in seeds]}, "results/metrics/bootstrap_design.json")
    for f in ("confirmatory_tests.csv", "exploratory_paired_effects.csv", "method_levels_block_bootstrap.csv",
              "group_rates_per_seed.csv", "main_summary_training_uncertainty.csv"):
        record(f"{cfg['tables_dir']}/{f}", "python scripts/evaluate.py --config " + args.config, cfg["_config_path"],
               seed=f"bootstrap {bs['seed']}; seeds {seeds}", runtime_s=time.time() - t0)
    log.info("confirmatory:\n%s", cdf[["id", "estimate_mean_over_seeds", "ci95_low", "ci95_high", "p_unadjusted",
                                         "p_holm"]].to_string())


if __name__ == "__main__":
    main()
