"""Schema and population audit of the released Pokec-z files and of the fixed core subgraph.
Writes counts only (no individual records).

Usage: python scripts/audit_dataset.py --config configs/data.yaml
"""
import argparse
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.loading import load_raw
from src.data.prepared import Prepared
from src.data.protected import group_table, intersectional_groups
from src.data.schema import audit, validate_schema
from src.data.target import TARGET_COLUMN, target_crosswalk, transform_target
from src.graphical_model.forest import build_forest, homophily, n_components, projection_stats
from src.utils import ROOT, dump_json, load_config, record, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    log = setup_logging("audit_dataset")
    cfg = load_config(args.config)
    d = cfg["dataset"]
    t = time.time()
    raw = load_raw(ROOT / d["raw_dir"], d["node_file"], d["edge_file"])
    validate_schema(raw)
    rep = {"full_release": audit(raw)}
    df = raw.nodes
    y = transform_target(df[TARGET_COLUMN])
    cw = target_crosswalk(df[TARGET_COLUMN])
    cw.to_csv(ROOT / "results/tables/target_crosswalk_full_release.csv", index=False)
    lab = y >= 0
    full_groups = group_table(intersectional_groups(df.gender.astype(int)[lab], df.region.astype(int)[lab]), y[lab])
    full_groups.to_csv(ROOT / "results/tables/groups_full_release_labelled.csv", index=False)
    prep = Prepared.load(ROOT / d["processed_path"])
    deg = np.bincount(prep.edges.ravel(), minlength=prep.n)
    inter = intersectional_groups(prep.gender, prep.region)
    core_groups = group_table(inter, prep.y)
    for name, grp in (("gender", prep.gender), ("region", prep.region)):
        gt = group_table(grp, prep.y)
        gt["mean_degree"] = [deg[grp == g].mean() for g in gt.group]
        gt["median_degree"] = [np.median(deg[grp == g]) for g in gt.group]
        gt.to_csv(ROOT / f"results/tables/core_groups_{name}.csv", index=False)
    core_groups["mean_degree"] = [deg[inter == g].mean() for g in core_groups.group]
    core_groups.to_csv(ROOT / "results/tables/core_groups_gender_x_region.csv", index=False)
    # group-pair edge mixing on the core graph
    mix = pd.crosstab(prep.gender[prep.edges[:, 0]], prep.gender[prep.edges[:, 1]])
    expected_same = (prep.gender.mean() ** 2 + (1 - prep.gender.mean()) ** 2)
    rep["core"] = {
        "nodes": int(prep.n), "edges": int(len(prep.edges)), "components": n_components(prep.n, prep.edges),
        "degree_mean": float(deg.mean()), "degree_median": float(np.median(deg)), "degree_max": int(deg.max()),
        "isolated": int((deg == 0).sum()), "positive_rate": float(prep.y.mean()),
        "edge_homophily_gender": homophily(prep.edges, prep.gender),
        "edge_homophily_gender_if_random_mixing": float(expected_same),
        "edge_homophily_region": homophily(prep.edges, prep.region),
        "edge_homophily_region_if_random_mixing": float(prep.region.mean() ** 2 + (1 - prep.region.mean()) ** 2),
        "edge_homophily_label": homophily(prep.edges, prep.y),
        "edge_homophily_label_if_random_mixing": float(prep.y.mean() ** 2 + (1 - prep.y.mean()) ** 2),
        "n_features": len(prep.feature_names), "sampling": prep.meta,
        "forest_projections": {m: projection_stats(prep.n, prep.edges, build_forest(prep.n, prep.edges, m),
                                                   {"gender": prep.gender, "region": prep.region, "label": prep.y})
                               for m in ("max_jaccard", "bfs", "low_degree", "random")},
    }
    rep["core"]["degree_hist_by_gender"] = {int(g): np.bincount(deg[prep.gender == g]).tolist() for g in (0, 1)}
    dump_json(rep, "results/metrics/dataset_audit.json")
    log.info("core: %s", {k: v for k, v in rep["core"].items() if not isinstance(v, dict)})
    for f in ("results/metrics/dataset_audit.json", "results/tables/target_crosswalk_full_release.csv",
              "results/tables/core_groups_gender_x_region.csv"):
        record(f, "python scripts/audit_dataset.py --config " + args.config, cfg["_config_path"],
               seed=cfg["subgraph"]["sampling_seed"], runtime_s=time.time() - t)


if __name__ == "__main__":
    main()
