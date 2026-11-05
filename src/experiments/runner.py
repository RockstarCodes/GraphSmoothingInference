"""Run the pipeline over seeds and persist every artifact needed for evaluation."""
from __future__ import annotations

import json
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.prepared import Prepared
from ..data.splits import stratified_split
from ..utils import ROOT, dump_json, load_config
from .pipeline import GraphData, evaluate_methods, run_one


def load_cfg(exp_cfg: Dict) -> Dict:
    return {k: load_config(exp_cfg[f"{k}_config"]) for k in ("data", "embeddings", "model", "fairness")}


def graph_from_prepared(prep: Prepared) -> GraphData:
    return GraphData(prep.n, prep.edges, prep.y, prep.gender, prep.region, prep.features)


def save_run(out_dir: Path, tag: str, split, run, rows_test, rows_val) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{tag}_predictions.npz",
                        train=split.train, val=split.val, test=split.test,
                        **{f"score__{k}": v for k, v in run.scores.items()},
                        **{f"dec__{k}": v for k, v in run.decisions.items()},
                        **{f"decmu__{k}": v for k, v in run.decisions_mu.items()})
    emb = run.info.pop("embeddings", None)
    if emb is not None:
        np.savez_compressed(out_dir / f"{tag}_embeddings.npz", **emb)
    dump_json(run.info, out_dir / f"{tag}_info.json")
    pd.DataFrame(rows_test + rows_val).to_csv(out_dir / f"{tag}_metrics.csv", index=False)
    if run.curves:
        pd.DataFrame(run.curves).to_csv(out_dir / f"{tag}_curves.csv", index=False)


def _job(args):
    (tag, seed, out_dir, cfg, graph_mod, kwargs, train_fraction) = args[:7]
    skip_existing = args[7] if len(args) > 7 else False
    if skip_existing and (Path(out_dir) / f"{tag}_metrics.csv").exists():
        return {"tag": tag, "seed": seed, "status": "ok", "runtime_s": None, "reused": True}
    prep = Prepared.load(ROOT / cfg["data"]["dataset"]["processed_path"])
    g = graph_from_prepared(prep)
    split = stratified_split(g.y, g.gender, seed, tuple(cfg["data"]["split"]["fractions"]),
                             train_fraction=train_fraction)
    if graph_mod is not None:
        g = graph_mod(g, seed)
    t = time.time()
    try:
        run = run_one(g, split, cfg, seed, **kwargs)
    except Exception as exc:  # preserve failures as artifacts
        dump_json({"tag": tag, "seed": seed, "status": "failed", "error": repr(exc)}, Path(out_dir) / f"{tag}_FAILED.json")
        return {"tag": tag, "seed": seed, "status": "failed", "error": repr(exc)}
    rows_t = [dict(r, seed=seed, tag=tag) for r in evaluate_methods(g, split, run, "test")]
    rows_v = [dict(r, seed=seed, tag=tag) for r in evaluate_methods(g, split, run, "val")]
    save_run(Path(out_dir), tag, split, run, rows_t, rows_v)
    return {"tag": tag, "seed": seed, "status": "ok", "runtime_s": time.time() - t}


def run_jobs(jobs: List[tuple], n_jobs: int) -> List[dict]:
    if n_jobs <= 1:
        return [_job(j) for j in jobs]
    with get_context("fork").Pool(n_jobs) as pool:
        return pool.map(_job, jobs, chunksize=1)
