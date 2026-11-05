"""Smoke test on a small SIMULATED graph: exercises data structures, graph construction,
Node2Vec/FairWalk, unaries, pairwise factors, exact inference, fairness metrics and output
writing in about a minute. Its numbers are NOT research results.

Usage: python scripts/smoke_test.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.splits import stratified_split
from src.experiments.pipeline import GraphData, evaluate_methods, run_one
from src.synthetic.generator import SyntheticConfig, generate
from src.utils import ROOT, dump_json, load_config, setup_logging


def main() -> None:
    log = setup_logging("smoke_test")
    t = time.time()
    sg = generate(SyntheticConfig(n_nodes=300, group_homophily=0.8, mean_degree=5), seed=1)
    rng = np.random.default_rng(1)
    region = rng.integers(0, 2, sg.n)
    X = np.c_[sg.y + rng.normal(0, 2, sg.n), rng.normal(size=(sg.n, 4))]
    g = GraphData(sg.n, sg.edges, sg.y, sg.a, region, X)
    split = stratified_split(g.y, g.gender, 1)
    cfg = {k: load_config(f"configs/{k}.yaml") for k in ("embeddings", "model", "fairness")}
    cfg["embeddings"]["embedding"].update({"dimensions": 16, "walks_per_node": 3, "walk_length": 15, "epochs": 2})
    run = run_one(g, split, cfg, seed=1, curves=False)
    rows = evaluate_methods(g, split, run, "test")
    out = ROOT / "results/smoke"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "smoke_metrics.csv", index=False)
    info = {k: v for k, v in run.info.items() if k != "embeddings"}
    dump_json({"SIMULATED_SMOKE_TEST_NOT_RESULTS": True, "runtime_s": time.time() - t, **info}, out / "smoke_info.json")
    assert np.allclose(run.scores["n2v_nopair"][split.test], np.clip(run.scores["n2v_lr"][split.test], 1e-6, 1 - 1e-6))
    assert all(np.all((v >= 0) & (v <= 1)) for v in run.scores.values())
    log.info("smoke test passed in %.1fs; %d method rows written to %s", time.time() - t, len(rows), out.relative_to(ROOT))


if __name__ == "__main__":
    main()
