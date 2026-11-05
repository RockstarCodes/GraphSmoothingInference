"""One complete run (one seed) of every method on a given graph.

Leakage controls (tested in tests/test_leakage.py):
  * classifiers are fitted on training labels; C chosen on validation log loss;
  * Markov-network evidence = training labels only (``training_evidence``);
  * lambda fitted on training folds; eta / global scale / thresholds chosen on validation;
  * test labels are only touched by ``evaluate_methods``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..baselines.classifiers import crossfit_train_probs, fit_lr, sensitive_predictability, sgc_features
from ..data.splits import Split, stratified_split
from ..embeddings.random_walks import WalkConfig, train_embeddings
from ..evaluation.decisions import (apply_group_thresholds, best_threshold, group_thresholds, threshold_grid,
                                    val_scores)
from ..fairness.metrics import evaluate_all, fairness_gaps
from ..graphical_model.attenuation import fit_lambda_training_folds, hca_lambdas, infer, select_on_validation
from ..graphical_model.forest import build_forest, projection_stats
from ..graphical_model.loopy_bp import loopy_bp
from ..graphical_model.pairwise_model import PairwiseCMN


@dataclass
class GraphData:
    n: int
    edges: np.ndarray
    y: np.ndarray
    gender: np.ndarray
    region: np.ndarray
    features: Optional[np.ndarray] = None


def training_evidence(split: Split, y: np.ndarray) -> Dict[int, int]:
    ev = {int(i): int(y[i]) for i in split.train}
    forbidden = set(split.val.tolist()) | set(split.test.tolist())
    if forbidden & set(ev):
        raise AssertionError("validation/test label used as evidence")
    return ev


@dataclass
class RunOutput:
    scores: Dict[str, np.ndarray] = field(default_factory=dict)        # P(Y=1) per node
    decisions: Dict[str, np.ndarray] = field(default_factory=dict)     # default operating point
    decisions_mu: Dict[str, np.ndarray] = field(default_factory=dict)  # matched-utility operating point
    info: Dict[str, object] = field(default_factory=dict)
    curves: List[dict] = field(default_factory=list)


def run_one(g: GraphData, split: Split, cfg: Dict, seed: int, forest_method: str = "max_jaccard",
            dims: Optional[int] = None, include_feature_baselines: bool = True,
            include_lbp: bool = True, embeddings: Optional[Dict[str, np.ndarray]] = None,
            curves: bool = True, region_track: bool = True) -> RunOutput:
    ecfg, mcfg, fcfg = cfg["embeddings"], cfg["model"], cfg["fairness"]
    out = RunOutput()
    t0 = time.time()
    y = g.y
    eps = float(mcfg["unary"]["clip_eps"])
    loss = float(fcfg["acceptable_utility_loss"])
    C_grid = ecfg["classifier"]["C_grid"]
    ev = training_evidence(split, y)
    tr, va, te = split.train, split.val, split.test

    # ------------------------------------------------------------ embeddings
    wc = WalkConfig(**{k: v for k, v in ecfg["embedding"].items() if k in WalkConfig.__dataclass_fields__})
    if dims is not None:
        wc.dimensions = int(dims)
    emb = embeddings or {}
    emb_specs = [("n2v", None), ("fw", g.gender)] + ([("fwr", g.region)] if region_track else [])
    for name, grp in emb_specs:
        if name not in emb:
            r = train_embeddings(g.n, g.edges, wc, seed, grp)
            emb[name] = r["embedding"]
            out.info[f"{name}_walk_time_s"] = r["walk_time_s"]
            out.info[f"{name}_train_time_s"] = r["train_time_s"]
            out.info[f"{name}_missing_embeddings"] = len(r["missing"])
    out.info["embeddings"] = emb

    # ------------------------------------------------------------ unary classifiers
    unary = {}
    for name, _ in emb_specs:
        r = fit_lr(emb[name], y, tr, va, C_grid, seed)
        unary[name] = r
        out.scores[f"{name}_lr"] = r["p"]
        out.info[f"{name}_C"] = r["C"]
        out.info[f"{name}_sens_gender"] = sensitive_predictability(emb[name], g.gender, tr, te, seed)
        out.info[f"{name}_sens_region"] = sensitive_predictability(emb[name], g.region, tr, te, seed)
    if include_feature_baselines and g.features is not None:
        r = fit_lr(g.features, y, tr, va, C_grid, seed)
        out.scores["feat_lr"] = r["p"]
        Z = sgc_features(g.features, g.n, g.edges, int(ecfg["sgc"]["hops"]))
        out.scores["sgc"] = fit_lr(Z, y, tr, va, C_grid, seed)["p"]

    # ------------------------------------------------------------ forest + pairwise
    forest = build_forest(g.n, g.edges, forest_method, seed=int(mcfg["inference"]["forest_seed"]))
    out.info["forest_stats"] = projection_stats(g.n, g.edges, forest, {"gender": g.gender, "region": g.region})
    grid = mcfg["pairwise"]["lambda_grid"]
    folds = int(mcfg["unary"]["crossfit_folds"])
    ref_floor = None
    tracks = [("n2v", "gender"), ("fw", "gender")] + ([("fwr", "region")] if region_track else [])
    attrs = {"gender": g.gender, "region": g.region}
    for name, track_attr in tracks:
        cf = crossfit_train_probs(emb[name], y, tr, unary[name]["C"], folds, seed)
        fit = fit_lambda_training_folds(g.n, unary[name]["p"], cf["p"], cf["fold"], tr, y, forest, grid, eps)
        lam = fit["lambda"]
        out.info[f"{name}_lambda"] = lam
        out.info[f"{name}_lambda_curve"] = fit["curve"]
        p_un = unary[name]["p"]
        out.scores[f"{name}_nopair"] = infer(g.n, p_un, forest, np.zeros(len(forest)), ev, eps)
        q_uni = infer(g.n, p_un, forest, np.full(len(forest), lam), ev, eps)
        out.scores[f"{name}_uniform"] = q_uni
        glob_post = lambda c: infer(g.n, p_un, forest, np.full(len(forest), c * lam), ev, eps)  # noqa: E731
        for att in ((["gender", "region"] if region_track else ["gender"]) if name != "fwr" else ["region"]):
            A = attrs[att]
            score_fn = lambda q, A=A: val_scores(y[va], q[va], A[va])  # noqa: E731
            floor = score_fn(q_uni)["macro_f1"] - loss
            suffix = "" if att == "gender" else "_region"
            hca_post = lambda e, A=A: infer(g.n, p_un, forest, hca_lambdas(forest, A, lam, e), ev, eps)  # noqa: E731
            sel = select_on_validation(mcfg["attenuation"]["eta_grid"], hca_post, score_fn, floor)
            out.info[f"{name}_eta{suffix}"] = sel["selected"]
            out.scores[f"{name}_hca{suffix}"] = hca_post(sel["selected"])
            # ablation B: global scalar with identical budget and selection rule
            selg = select_on_validation(mcfg["attenuation"]["global_scale_grid"], glob_post, score_fn, floor)
            out.info[f"{name}_global_scale{suffix}"] = selg["selected"]
            out.scores[f"{name}_global{suffix}"] = glob_post(selg["selected"])
            # descriptive curves over eta / global scale (selection above uses validation only)
            for kind, vals, fn in ((("hca", mcfg["attenuation"]["eta_grid"], hca_post),
                                    ("global", mcfg["attenuation"]["global_scale_grid"], glob_post)) if curves else ()):
                for v in vals:
                    q = fn(v)
                    t = best_threshold(y[va], q[va])
                    for part, idx in (("val", va), ("test", te)):
                        d = (q[idx] >= t).astype(int)
                        met = evaluate_all(y[idx], q[idx], d, g.gender[idx], g.region[idx])
                        out.curves.append({"emb": name, "attr": att, "kind": kind, "value": float(v), "part": part,
                                           "lambda": lam,
                                           **{k: met[k] for k in ("macro_f1", "auroc", "balanced_acc", "log_loss",
                                                                  "brier", "ece", "dEOpp_gender", "dDP_gender",
                                                                  "dEOpp_region", "dDP_region",
                                                                  "dEOpp_inter", "dDP_inter")}})
        if include_lbp:
            m = PairwiseCMN(g.n, p_un, g.edges, np.full(len(g.edges), lam), ev, eps=eps)
            q_lbp, conv, it = loopy_bp(m, max_iter=300, damping=0.5, tol=1e-7)
            out.scores[f"{name}_lbp_uniform"] = q_lbp[:, 1]
            out.info[f"{name}_lbp_converged"] = bool(conv)
            out.info[f"{name}_lbp_iters"] = int(it)
    # ------------------------------------------------------------ decisions
    ref_floor = val_scores(y[va], out.scores["n2v_lr"][va], g.gender[va])["macro_f1"] - loss
    out.info["matched_utility_floor"] = ref_floor
    for k, p in out.scores.items():
        t = best_threshold(y[va], p[va])
        out.decisions[k] = (p >= t).astype(int)
        out.info[f"thr_{k}"] = t
        # matched-utility operating point: min val dEOpp among thresholds with val macro-F1 >= common floor
        best = None
        from sklearn.metrics import f1_score
        for tt in np.r_[threshold_grid(p[va]), t]:
            d = (p[va] >= tt).astype(int)
            f = f1_score(y[va], d, average="macro", zero_division=0)
            if f < ref_floor - 1e-12:
                continue
            e = fairness_gaps(y[va], d, g.gender[va])["dEOpp"]
            key = (round(e, 12), -f)
            if best is None or key < best[0]:
                best = (key, tt)
        tt = best[1] if best is not None else t
        out.decisions_mu[k] = (p >= tt).astype(int)
        out.info[f"thr_mu_{k}"] = float(tt)
        out.info[f"mu_feasible_{k}"] = best is not None
    # decision-only mitigation: group thresholds on Node2Vec scores
    thr, _ = group_thresholds(y[va], out.scores["n2v_lr"][va], g.gender[va], ref_floor)
    out.scores["n2v_thr"] = out.scores["n2v_lr"]
    out.decisions["n2v_thr"] = apply_group_thresholds(out.scores["n2v_lr"], g.gender, thr)
    out.decisions_mu["n2v_thr"] = out.decisions["n2v_thr"]
    out.info["n2v_thr_thresholds"] = {str(k): v for k, v in thr.items()}
    out.info["runtime_s"] = time.time() - t0
    return out


def evaluate_methods(g: GraphData, split: Split, run: RunOutput, part: str = "test") -> List[dict]:
    idx = {"test": split.test, "val": split.val}[part]
    rows = []
    for op, dec in (("default", run.decisions), ("matched_utility", run.decisions_mu)):
        for k in run.scores:
            if k not in dec:
                continue
            met = evaluate_all(g.y[idx], run.scores[k][idx], dec[k][idx], g.gender[idx], g.region[idx])
            rows.append({"method": k, "operating_point": op, "part": part, **met})
    return rows
