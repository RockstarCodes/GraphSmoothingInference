"""Build the fixed ~2,000-node Pokec-z core subgraph (label-value blind).

Usage: python scripts/prepare_data.py --config configs/data.yaml
"""
import argparse
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.integrity import check_prepared
from src.data.loading import load_raw
from src.data.prepared import Prepared
from src.data.protected import recorded_attribute
from src.data.schema import validate_schema
from src.data.subgraph import bfs_sample, labelled_lcc
from src.data.target import TARGET_COLUMN, transform_target
from src.utils import ROOT, dump_json, load_config, record, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    log = setup_logging("prepare_data")
    cfg = load_config(args.config)
    d, sg = cfg["dataset"], cfg["subgraph"]
    t = time.time()
    raw = load_raw(ROOT / d["raw_dir"], d["node_file"], d["edge_file"])
    validate_schema(raw)
    df = raw.nodes
    y_all = transform_target(df[TARGET_COLUMN])
    lcc = labelled_lcc(df["user_id"].to_numpy(), y_all >= 0, raw.edges_raw)
    log.info("labelled LCC: %d nodes, %d edges", lcc.number_of_nodes(), lcc.number_of_edges())
    sub = bfs_sample(lcc, int(sg["target_size"]), int(sg["sampling_seed"]))
    rows = df.set_index("user_id").loc[sub.user_ids]
    y = transform_target(rows[TARGET_COLUMN])
    assert np.all(y >= 0)
    gender = recorded_attribute(rows.reset_index(), "gender")
    region = recorded_attribute(rows.reset_index(), "region")
    feat_cols = [c for c in rows.columns if c not in ("gender", "region", TARGET_COLUMN)]
    X = rows[feat_cols].to_numpy(dtype=float)
    checks = check_prepared(len(y), sub.edges, y, gender, X)
    meta = {"root_user_id": sub.root_user_id, "sampling_seed": sg["sampling_seed"], "target_size": sg["target_size"],
            "labelled_lcc_nodes": lcc.number_of_nodes(), "labelled_lcc_edges": lcc.number_of_edges(),
            "integrity": checks}
    prep = Prepared(sub.user_ids, sub.edges, y, gender, region, X, feat_cols, meta)
    out = ROOT / d["processed_path"]
    prep.save(out)
    # selection summary: subgraph vs labelled population (counts only, no records)
    pop = pd.DataFrame({"gender": df["gender"].astype(int), "region": df["region"].astype(int), "y": y_all})
    pop = pop[pop.y >= 0]
    summ = {"subgraph_nodes": int(len(y)), "subgraph_edges": int(len(sub.edges)), **meta,
            "subgraph_positive_rate": float(y.mean()), "population_positive_rate": float(pop.y.mean()),
            "subgraph_gender1_share": float(gender.mean()), "population_gender1_share": float(pop.gender.mean()),
            "subgraph_region1_share": float(region.mean()), "population_region1_share": float(pop.region.mean())}
    dump_json(summ, "results/metrics/subgraph_selection.json")
    log.info("saved %s (%d nodes, %d edges)", out, len(y), len(sub.edges))
    record(d["processed_path"], "python scripts/prepare_data.py --config " + args.config, cfg["_config_path"],
           seed=sg["sampling_seed"], runtime_s=time.time() - t)
    record("results/metrics/subgraph_selection.json", "python scripts/prepare_data.py --config " + args.config,
           cfg["_config_path"], seed=sg["sampling_seed"])


if __name__ == "__main__":
    main()
