"""Generate every figure and table from saved outputs (no numbers typed by hand).

Usage: python scripts/make_results.py --config configs/evaluation.yaml
Missing inputs (e.g. robustness not yet run) are reported and the affected figures skipped.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402
from src.data.prepared import Prepared  # noqa: E402
from src.fairness.metrics import reliability_table  # noqa: E402
from src.graphical_model.attenuation import hca_lambdas  # noqa: E402
from src.graphical_model.forest import build_forest  # noqa: E402
from src.utils import ROOT, load_config, record, setup_logging  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 110, "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
C = {"n2v": "#1f5f8b", "fw": "#d1495b", "fwr": "#2a9d8f", "feat": "#6c757d", "sgc": "#8d6a9f", "thr": "#e9a03b"}
LABEL = {"feat_lr": "Feature LR", "sgc": "SGC (simple GNN)", "n2v_lr": "Node2Vec+LR", "fw_lr": "FairWalk+LR",
         "n2v_thr": "Node2Vec+group thr.", "n2v_uniform": "Node2Vec+uniform CMN", "fw_uniform": "FairWalk+uniform CMN",
         "n2v_global": "Node2Vec+global scalar", "fw_global": "FairWalk+global scalar", "n2v_hca": "Node2Vec+HCA",
         "fw_hca": "FairWalk+HCA", "n2v_nopair": "Node2Vec CMN, no pairwise", "fw_nopair": "FairWalk CMN, no pairwise",
         "n2v_lbp_uniform": "Node2Vec+loopy BP (full graph)", "fw_lbp_uniform": "FairWalk+loopy BP (full graph)",
         "fwr_lr": "FairWalk(region)+LR", "fwr_uniform": "FairWalk(region)+uniform", "fwr_hca_region": "FairWalk(region)+HCA(region)",
         "fwr_global_region": "FairWalk(region)+global", "n2v_hca_region": "Node2Vec+HCA(region)",
         "fw_hca_region": "FairWalk+HCA(region)"}
CAPTIONS = {}


def save(fig, name, caption):
    out = ROOT / "results/figures" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[name] = caption


def tag(ax, text="REAL DATA: Pokec-z core subgraph"):
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555")


def fmt_ci(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    args = ap.parse_args()
    log = setup_logging("make_results")
    cfg = load_config(args.config)
    t0 = time.time()
    T = ROOT / cfg["tables_dir"]; T.mkdir(parents=True, exist_ok=True)
    prep = Prepared.load(ROOT / "data/processed/pokec_z_core.npz")
    audit = json.loads((ROOT / "results/metrics/dataset_audit.json").read_text())
    deg = np.bincount(prep.edges.ravel(), minlength=prep.n)
    missing = []

    # ---------------------------------------------------------------- 1 graph overview
    G = nx.Graph(); G.add_nodes_from(range(prep.n)); G.add_edges_from(map(tuple, prep.edges))
    pos = nx.spring_layout(G, seed=1, iterations=60)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, attr, nm in ((axs[0], prep.gender, "recorded gender"), (axs[1], prep.region, "recorded region")):
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.15, alpha=0.25)
        ax.scatter(*np.array([pos[i] for i in range(prep.n)]).T, c=np.where(attr == 1, C["fw"], C["n2v"]), s=3)
        ax.set_title(f"Core subgraph coloured by {nm} (blue=0, red=1)"); ax.axis("off"); tag(ax)
    save(fig, "fig01_graph_overview", "Pokec-z core subgraph (2,000 nodes, 5,351 edges, spring layout, seed 1). Left: recorded gender codes; right: recorded region codes. Region clusters strongly; gender does not.")

    # ---------------------------------------------------------------- 2 + 3 distributions
    cg = pd.read_csv(T / "core_groups_gender_x_region.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3))
    x = np.arange(len(cg))
    ax.bar(x, cg.negatives, color="#adb5bd", label="Y=0"); ax.bar(x, cg.positives, bottom=cg.negatives, color=C["n2v"], label="Y=1")
    ax.set_xticks(x, [g.replace("g", "gender ").replace("_r", ", region ") for g in cg.group], rotation=15)
    ax.set_ylabel("nodes"); ax.set_title("Group sizes and labels (gender × region)"); ax.legend(); tag(ax)
    save(fig, "fig02_group_distribution", "Core subgraph group sizes by recorded gender × region with label counts (Y=1: raw working-field code ≥ 1).")
    cw = pd.read_csv(T / "target_crosswalk_full_release.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3))
    ax.bar(cw.raw_code.astype(str), cw["count"], color=[("#adb5bd" if y < 0 else C["n2v"] if y == 1 else C["fw"]) for y in cw.Y])
    ax.set_yscale("log"); ax.set_xlabel("raw I_am_working_in_field code"); ax.set_ylabel("nodes (log)")
    ax.set_title("Working-field label transformation (full release)"); tag(ax, "REAL DATA: full Pokec-z release")
    for i, (c, y) in enumerate(zip(cw["count"], cw.Y)):
        ax.text(i, c * 1.1, {-1: "unlabelled", 0: "Y=0", 1: "Y=1"}[y], ha="center", fontsize=7)
    save(fig, "fig03_label_distribution", "Raw working-field codes in the full Pokec-z release and their FairGNN mapping: −1 unlabelled (excluded), 0 → Y=0, 1–4 → Y=1.")

    # ---------------------------------------------------------------- 4 degree distribution
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for ax, attr, nm in ((axs[0], prep.gender, "gender"), (axs[1], prep.region, "region")):
        bins = np.arange(0, deg.max() + 2) - 0.5
        for v, col in ((0, C["n2v"]), (1, C["fw"])):
            ax.hist(deg[attr == v], bins=bins, alpha=0.55, color=col, density=True,
                    label=f"{nm}={v} (mean {deg[attr == v].mean():.2f})")
        ax.set_xlabel("degree in core subgraph"); ax.set_ylabel("share of nodes"); ax.set_xlim(0, 30)
        ax.set_title(f"Degree distribution by {nm}"); ax.legend(); tag(ax)
    save(fig, "fig04_degree_by_group", "Degree distributions by recorded gender (balanced) and region (region 1 has higher degree).")

    # ---------------------------------------------------------------- 5 homophily
    core = audit["core"]
    fp = core["forest_projections"]["max_jaccard"]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    names = ["gender", "region", "label"]
    obs = [core["edge_homophily_gender"], core["edge_homophily_region"], core["edge_homophily_label"]]
    rnd = [core["edge_homophily_gender_if_random_mixing"], core["edge_homophily_region_if_random_mixing"],
           core["edge_homophily_label_if_random_mixing"]]
    fst = [fp["homophily_gender_forest"], fp["homophily_region_forest"], fp["homophily_label_forest"]]
    x = np.arange(3); w = 0.27
    ax.bar(x - w, rnd, w, color="#ced4da", label="random mixing"); ax.bar(x, obs, w, color=C["n2v"], label="graph")
    ax.bar(x + w, fst, w, color=C["fw"], label="max-Jaccard forest")
    ax.set_xticks(x, names); ax.set_ylabel("edge homophily (share same-value edges)"); ax.set_ylim(0, 1.05)
    ax.set_title("Edge homophily: graph vs forest vs random mixing"); ax.legend(loc="upper left"); tag(ax)
    save(fig, "fig05_homophily", "Edge homophily of the core graph and of the max-Jaccard forest against the random-mixing expectation. Gender is mildly heterophilous; region is strongly homophilous; label homophily is weak.")

    # ---------------------------------------------------------------- main results
    md = ROOT / cfg["main_dir"]
    allm = pd.read_csv(md / "all_metrics.csv")
    test = allm[allm.part == "test"]
    lv = pd.read_csv(T / "method_levels_block_bootstrap.csv")
    infos = {s: json.loads((md / f"seed{s}_info.json").read_text()) for s in sorted(test.seed.unique())}

    # 6 embedding diagnostics
    rows = []
    for s, inf in infos.items():
        for e in ("n2v", "fw", "fwr"):
            for att in ("gender", "region"):
                rows.append({"seed": s, "embedding": e, "attribute": att, **inf[f"{e}_sens_{att}"]})
    sens = pd.DataFrame(rows)
    st = sens.groupby(["embedding", "attribute"])[["auroc", "balanced_acc", "macro_f1", "acc", "majority_rate"]].agg(["mean", "std"]).round(3)
    st.columns = [f"{a}_{b}" for a, b in st.columns]
    st.reset_index().to_csv(T / "table_embedding_diagnostics.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for k, (e, nm) in enumerate((("n2v", "Node2Vec"), ("fw", "FairWalk (gender)"), ("fwr", "FairWalk (region)"))):
        sub = sens[sens.embedding == e].groupby("attribute").auroc
        ax.bar(np.arange(2) + (k - 1) * 0.27, sub.mean().values, 0.27, yerr=sub.std().values, capsize=2,
               color=[C["n2v"], C["fw"], C["fwr"]][k], label=nm)
    ax.axhline(0.5, ls="--", c="k", lw=0.7); ax.set_xticks([0, 1], ["predict gender", "predict region"])
    ax.set_ylabel("test AUROC (mean ± SD, 5 seeds)"); ax.set_ylim(0.4, 1.02)
    ax.set_title("Sensitive-attribute predictability from embeddings"); ax.legend(loc="upper left"); tag(ax)
    save(fig, "fig06_embedding_diagnostics", "Representation diagnostic (not a fairness definition): AUROC of an L2 logistic regression predicting the recorded attribute from embeddings (fit on training nodes, scored on test nodes), mean ± SD over 5 seeds.")

    # 7 fairness before/after smoothing (with block-bootstrap CIs)
    d = lv[lv.operating_point == "default"].set_index(["method", "metric"])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, metric, ttl in ((axs[0], "dEOpp_g", "ΔEOpp (gender)"), (axs[1], "dDP_g", "ΔDP (gender)")):
        ms = ["n2v_lr", "n2v_uniform", "n2v_global", "n2v_hca", "fw_lr", "fw_uniform", "fw_global", "fw_hca"]
        vals = [d.loc[(m, metric)] for m in ms]
        y = [v.mean_over_seeds for v in vals]
        err = np.array([[v.mean_over_seeds - v.ci95_low for v in vals], [v.ci95_high - v.mean_over_seeds for v in vals]])
        ax.bar(range(len(ms)), y, yerr=err, capsize=2, color=[C["n2v"]] * 4 + [C["fw"]] * 4,
               alpha=0.9, edgecolor="k", linewidth=0.3)
        ax.set_xticks(range(len(ms)), ["LR", "uniform", "global", "HCA"] * 2, rotation=35, fontsize=8)
        ax.set_title(f"{ttl}: before/after smoothing"); ax.set_ylabel(f"{ttl} on test (lower = fairer)")
        ax.text(1.5, ax.get_ylim()[1] * 0.97, "Node2Vec", ha="center", color=C["n2v"])
        ax.text(5.5, ax.get_ylim()[1] * 0.97, "FairWalk", ha="center", color=C["fw"]); tag(ax)
    save(fig, "fig07_fairness_before_after_smoothing", "Gender ΔEOpp and ΔDP before (LR) and after Markov smoothing (uniform, global scalar, HCA), mean over 5 seeds with 95% graph-block bootstrap intervals (1,000 replicates; fixed-model evaluation uncertainty).")

    # 8 predictive utility
    ms = ["feat_lr", "sgc", "n2v_lr", "n2v_thr", "n2v_uniform", "n2v_hca", "fw_lr", "fw_uniform", "fw_hca",
          "n2v_lbp_uniform", "fw_lbp_uniform"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, metric, ttl in ((axs[0], "macro_f1", "macro-F1"), (axs[1], "balanced_acc", "balanced accuracy"),
                            (axs[2], "brier", "Brier score (lower better)")):
        v = [d.loc[(m, metric)] for m in ms]
        ax.barh(range(len(ms)), [x.mean_over_seeds for x in v],
                xerr=np.array([[x.mean_over_seeds - x.ci95_low for x in v], [x.ci95_high - x.mean_over_seeds for x in v]]),
                capsize=2, color="#6c8ebf")
        ax.set_yticks(range(len(ms)), [LABEL[m] for m in ms] if ax is axs[0] else [""] * len(ms))
        ax.set_title(ttl); lo = min(x.ci95_low for x in v); ax.set_xlim(lo - 0.02, max(x.ci95_high for x in v) + 0.01)
        tag(ax)
    save(fig, "fig08_predictive_utility", "Predictive and probabilistic quality on test nodes, mean over 5 seeds with 95% graph-block bootstrap intervals.")
    # AUROC table from per-seed metrics
    au = test[test.operating_point == "default"].groupby("method")[["auroc", "macro_f1", "balanced_acc", "log_loss", "brier", "ece", "group_ece_gap"]].agg(["mean", "std"]).round(3)
    au.columns = [f"{a}_{b}" for a, b in au.columns]
    au.reset_index().to_csv(T / "table_predictive_training_uncertainty.csv", index=False)

    # 9 fairness-utility curves
    curves = pd.concat([pd.read_csv(p).assign(seed=int(p.name[4:6])) for p in sorted(md.glob("seed*_curves.csv"))])
    cu = curves[(curves.part == "test") & (curves.attr == "gender") & (curves.emb.isin(["n2v", "fw"]))]
    cg = cu.groupby(["emb", "kind", "value"])[["macro_f1", "dEOpp_gender", "dDP_gender", "log_loss", "brier"]].mean().reset_index()
    cg.to_csv(T / "table_attenuation_sensitivity.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, emb in zip(axs, ("n2v", "fw")):
        for kind, mk, nm in (("hca", "o-", "HCA: η = 0…1"), ("global", "s--", "global scalar: c = 1…0")):
            s = cg[(cg.emb == emb) & (cg.kind == kind)]
            ax.plot(s.macro_f1, s.dEOpp_gender, mk, ms=3.5, label=nm, color=C["fw"] if kind == "hca" else C["n2v"])
            for _, r in s.iloc[[0, -1]].iterrows():
                ax.annotate(f"{r.value:.1f}", (r.macro_f1, r.dEOpp_gender), fontsize=6.5)
        lr = d.loc[(f"{emb}_lr", "dEOpp_g")].mean_over_seeds; lrf = d.loc[(f"{emb}_lr", "macro_f1")].mean_over_seeds
        ax.plot([lrf], [lr], "k*", ms=8, label="no smoothing (LR)")
        ax.set_xlabel("test macro-F1 (mean over 5 seeds)"); ax.set_ylabel("test ΔEOpp gender")
        ax.set_title(f"{'Node2Vec' if emb == 'n2v' else 'FairWalk'}: fairness–utility as attenuation varies"); ax.legend(); tag(ax)
    save(fig, "fig09_fairness_utility_curves", "Descriptive fairness–utility curves over the full η grid (HCA) and global-scale grid (selection itself used validation only). Each point is the mean over 5 seeds of test macro-F1 and gender ΔEOpp at the validation-selected threshold; λ fixed at its training-fold estimate.")

    # 10 attenuation distribution
    forest = build_forest(prep.n, prep.edges, "max_jaccard")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    etas = pd.DataFrame([{"seed": s, "emb": e, "eta": inf[f"{e}_eta"], "global": inf[f"{e}_global_scale"],
                          "lambda": inf[f"{e}_lambda"]} for s, inf in infos.items() for e in ("n2v", "fw")])
    etas.to_csv(T / "table_selected_parameters.csv", index=False)
    for e, col, off in (("n2v", C["n2v"], -0.15), ("fw", C["fw"], 0.15)):
        sub = etas[etas.emb == e]
        axs[0].scatter(np.arange(len(sub)) + off, sub.eta, color=col, marker="o", label=f"{e}: η (HCA)")
        axs[0].scatter(np.arange(len(sub)) + off, sub["global"], color=col, marker="x", label=f"{e}: c (global)")
    axs[0].set_xticks(range(len(infos)), [f"seed {s}" for s in infos]); axs[0].set_ylabel("selected value")
    axs[0].set_title("Validation-selected attenuation per seed"); axs[0].legend(fontsize=6.5, ncol=2); tag(axs[0])
    same = prep.gender[forest[:, 0]] == prep.gender[forest[:, 1]]
    for e, col in (("fw", C["fw"]),):
        lam = etas[etas.emb == e]["lambda"].iloc[0]
        for s, inf in infos.items():
            lij = hca_lambdas(forest, prep.gender, lam, inf[f"{e}_eta"])
            axs[1].hist(lij, bins=np.linspace(0, lam * 1.05, 22), histtype="step", lw=1, label=f"seed {s} (η={inf[f'{e}_eta']:.1f})")
    axs[1].set_xlabel("λ_ij on forest edges (FairWalk HCA)"); axs[1].set_ylabel("edges")
    axs[1].set_title(f"Edge strength distribution ({same.mean():.0%} forest edges same-gender)"); axs[1].legend(fontsize=6.5); tag(axs[1])
    save(fig, "fig10_attenuation_distribution", "Left: η (HCA) and c (global scalar) selected on validation in each seed. Right: resulting distribution of pairwise strengths λ_ij over the 1,999 forest edges for FairWalk+HCA.")

    # 15 group calibration
    z = np.load(md / f"seed{min(infos)}_predictions.npz")
    te = z["test"]
    rows = []
    for s in infos:
        zz = np.load(md / f"seed{s}_predictions.npz"); tt = zz["test"]
        for m in ("fw_lr", "fw_uniform", "fw_hca"):
            r = reliability_table(prep.y[tt], zz[f"score__{m}"][tt], prep.gender[tt], n_bins=10)
            r["method"] = m; r["seed"] = s; rows.append(r)
    rel = pd.concat(rows)
    rel.to_csv(T / "table_group_reliability.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, m in zip(axs, ("fw_lr", "fw_uniform", "fw_hca")):
        for gv, col in ((0, C["n2v"]), (1, C["fw"])):
            sub = rel[(rel.method == m) & (rel.group == gv)]
            agg = sub.groupby("bin").apply(lambda q: pd.Series({"p": np.average(q.mean_p, weights=q.n),
                                                                "f": np.average(q.frac_pos, weights=q.n), "n": q.n.sum()}),
                                           include_groups=False)
            ax.plot(agg.p, agg.f, "o-", color=col, ms=3, label=f"gender={gv}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6); ax.set_title(LABEL[m]); ax.set_xlabel("mean predicted P(Y=1) in bin")
        tag(ax)
    axs[0].set_ylabel("observed share Y=1"); axs[0].legend()
    save(fig, "fig15_group_calibration", "Reliability diagrams by recorded gender for FairWalk before smoothing, after uniform smoothing and after HCA (10 equal-width bins, test nodes pooled over 5 seeds, bins weighted by count).")

    # ---------------------------------------------------------------- tables: baseline, groups, before/after, proposed, ablations
    def lvrow(m, op="default"):
        dd = lv[(lv.operating_point == op) & (lv.method == m)].set_index("metric")
        return {k: fmt_ci(dd.loc[k].mean_over_seeds, dd.loc[k].ci95_low, dd.loc[k].ci95_high) for k in
                ("macro_f1", "balanced_acc", "brier", "log_loss", "dEOpp_g", "dDP_g", "dEOdds_g", "dEOpp_r", "dDP_r")}
    order = ["feat_lr", "sgc", "n2v_lr", "fw_lr", "n2v_thr", "n2v_nopair", "fw_nopair", "n2v_uniform", "fw_uniform",
             "n2v_global", "fw_global", "n2v_hca", "fw_hca", "n2v_lbp_uniform", "fw_lbp_uniform",
             "fwr_lr", "fwr_uniform", "fwr_global_region", "fwr_hca_region"]
    for op in ("default", "matched_utility"):
        tb = pd.DataFrame([{"method": LABEL.get(m, m), "key": m, **lvrow(m, op)} for m in order])
        tb.to_csv(T / f"table_methods_{op}.csv", index=False)
    au_mean = test[test.operating_point == "default"].groupby("method").auroc.agg(["mean", "std"])
    au_mean.round(3).to_csv(T / "table_auroc_by_method.csv")

    gr = pd.read_csv(T / "group_rates_per_seed.csv")
    gp = gr[gr.method.isin(["n2v_lr", "fw_lr", "fw_uniform", "fw_hca", "n2v_thr"])].groupby(["method", "attribute", "group"]).agg(
        n=("n", "sum"), positives=("positives", "sum"), negatives=("negatives", "sum"), selected=("selected", "sum"),
        selection_rate=("selection_rate", "mean"), tpr=("tpr", "mean"), fpr=("fpr", "mean"), accuracy=("accuracy", "mean"),
        min_n_per_seed=("n", "min"), min_pos_per_seed=("positives", "min"), min_neg_per_seed=("negatives", "min")).reset_index()
    gp["exploratory_lt30"] = (gp[["min_n_per_seed", "min_pos_per_seed", "min_neg_per_seed"]].min(axis=1) < 30)
    gp.round(4).to_csv(T / "table_group_rates_pooled.csv", index=False)

    conf = pd.read_csv(T / "confirmatory_tests.csv")
    conf_t = conf[["id", "a", "b", "metric", "a_mean", "b_mean", "estimate_mean_over_seeds", "ci95_low", "ci95_high",
                   "p_unadjusted", "p_holm", "seed_diff_min", "seed_diff_max", "dz_over_seeds"]].round(4)
    conf_t.to_csv(T / "table_statistical_confirmatory.csv", index=False)

    ex = pd.read_csv(T / "exploratory_paired_effects.csv")
    smooth = ex[(ex.operating_point == "default") & (ex.id.isin(["H1-ctx", "H2-ctx", "E9"]))][
        ["id", "a", "b", "metric", "a_mean", "b_mean", "estimate_mean_over_seeds", "ci95_low", "ci95_high", "p_unadjusted"]]
    smooth.round(4).to_csv(T / "table_before_after_smoothing.csv", index=False)
    abl = ex[(ex.operating_point == "default") & (ex.id.isin(["E3", "E4", "E5", "H3-ctx", "H4-ctx", "E6", "E7", "E1", "E2"]))][
        ["id", "description", "a", "b", "metric", "estimate_mean_over_seeds", "ci95_low", "ci95_high", "p_unadjusted"]]
    abl.round(4).to_csv(T / "table_ablations_and_comparisons.csv", index=False)

    # 16 inference agreement
    iv = pd.read_csv(ROOT / "results/metrics/inference_verification.csv")
    ivs = json.loads((ROOT / "results/metrics/inference_verification_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3))
    cols = [("max_abs_sp_vs_enum", "sum-product vs enumeration"), ("max_abs_ve_vs_enum", "VE vs enumeration"),
            ("max_abs_ct_vs_enum", "clique tree vs enumeration"), ("max_abs_ve_vs_ct", "VE vs clique tree"),
            ("elimination_order_gap", "VE: random vs min-fill order")]
    data = [np.log10(np.maximum(iv[c].dropna().values, 1e-18)) for c, _ in cols]
    ax.boxplot(data, orientation="horizontal", tick_labels=[n for _, n in cols])
    ax.axvline(np.log10(ivs["tolerance"]), color="r", ls="--", lw=0.8, label="tolerance 1e-8")
    ax.set_xlabel("log10 max |marginal difference| per fixture"); ax.legend(loc="lower right")
    ax.set_title(f"Exact-inference agreement ({ivs['fixtures']} random fixtures)"); tag(ax, "SIMULATED fixtures (≤ 12 nodes)")
    save(fig, "fig16_inference_agreement", "Agreement of exact methods on 200 random tiny pairwise models (100 forests incl. isolated nodes/components, 100 loopy graphs; random evidence). Sum-product is only defined on forests.")
    pd.DataFrame([{"check": n, "max_abs_difference": iv[c].max(), "fixtures": int(iv[c].notna().sum()),
                   "tolerance": ivs["tolerance"], "pass": bool(iv[c].max() <= ivs["tolerance"])} for c, n in cols] +
                 [{"check": "clique-tree sepset calibration", "max_abs_difference": iv.clique_tree_calibration.max(),
                   "fixtures": len(iv), "tolerance": ivs["tolerance"], "pass": bool(iv.clique_tree_calibration.max() <= ivs["tolerance"])},
                  {"check": "|log Z| sum-product vs enumeration", "max_abs_difference": iv.logz_sp_vs_enum.max(),
                   "fixtures": int(iv.logz_sp_vs_enum.notna().sum()), "tolerance": ivs["tolerance"],
                   "pass": bool(iv.logz_sp_vs_enum.max() <= ivs["tolerance"])}]).to_csv(T / "table_inference_correctness.csv", index=False)

    # ---------------------------------------------------------------- synthetic (11, 12)
    sd = ROOT / cfg["synthetic_dir"] / "synthetic_realisations.csv"
    if sd.exists():
        syn = pd.read_csv(sd)
        hx = syn[syn.design == "homophily_x_degree"]
        agg = hx.groupby(["group_homophily", "degree_ratio"]).agg(
            n_real=("seed", "size"), real_hA=("real_homophily_A", "mean"),
            unary_dEOpp=("unary_dEOpp", "mean"), uniform_dEOpp=("uniform_dEOpp", "mean"), hca_dEOpp=("hca_dEOpp", "mean"),
            global_dEOpp=("global_dEOpp", "mean"),
            smooth_dEOpp_mean=("smoothing_change_dEOpp", "mean"), smooth_dEOpp_sd=("smoothing_change_dEOpp", "std"),
            smooth_dDP_mean=("smoothing_change_dDP", "mean"), smooth_dDP_sd=("smoothing_change_dDP", "std"),
            smooth_f1_mean=("smoothing_change_macro_f1", "mean"),
            hca_vs_uniform_dEOpp=("hca_vs_uniform_dEOpp", "mean"), hca_vs_global_dEOpp=("hca_vs_global_dEOpp", "mean"),
            hca_vs_global_f1=("hca_vs_global_macro_f1", "mean")).reset_index()
        agg.round(4).to_csv(T / "table_synthetic_homophily_degree.csv", index=False)
        okeys = ["design", "group_homophily", "label_homophily", "train_fraction"] + \
            [c for c in ("group_share", "prevalence0", "prevalence1") if c in syn.columns]
        oth = syn[syn.design != "homophily_x_degree"].groupby(okeys, dropna=False).agg(
            n_real=("seed", "size"), smooth_dEOpp_mean=("smoothing_change_dEOpp", "mean"),
            smooth_dEOpp_sd=("smoothing_change_dEOpp", "std"), smooth_dDP_mean=("smoothing_change_dDP", "mean"),
            smooth_dDP_sd=("smoothing_change_dDP", "std"), hca_vs_global_dEOpp=("hca_vs_global_dEOpp", "mean"),
            hca_vs_global_dDP=("hca_vs_global_dDP", "mean")).reset_index()
        oth.round(4).to_csv(T / "table_synthetic_other_factors.csv", index=False)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
        for ax, met, ttl in ((axs[0], "dEOpp", "ΔEOpp"), (axs[1], "dDP", "ΔDP")):
            for dr, col in zip(sorted(hx.degree_ratio.unique()), ("#264653", "#e9c46a", "#e76f51")):
                s = hx[hx.degree_ratio == dr].groupby("group_homophily")[f"smoothing_change_{met}"]
                ax.errorbar(s.mean().index, s.mean(), yerr=s.std(), capsize=2, marker="o", color=col,
                            label=f"degree ratio {dr:g}")
            ax.axhline(0, c="k", lw=0.6); ax.set_xlabel("target demographic edge homophily")
            ax.set_ylabel(f"{ttl}(uniform) − {ttl}(unary)"); ax.set_title(f"Smoothing-induced change in {ttl}")
            ax.legend(); tag(ax, "SIMULATED graphs, 10 realisations/cell (mean ± SD)")
        save(fig, "fig11_synthetic_homophily", "SIMULATED: change in the group gap caused by uniform smoothing (λ=1) of a unary score that is fair by construction, versus demographic homophily, for three group degree ratios. Error bars: SD over 10 graph realisations (realisation variability, not a confidence interval).")
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
        for ax, met in ((axs[0], "dEOpp"), (axs[1], "dDP")):
            for h, col in zip((0.5, 0.7, 0.9), ("#264653", "#e9c46a", "#e76f51")):
                s = hx[hx.group_homophily == h].groupby("degree_ratio")
                ax.errorbar(s[f"smoothing_change_{met}"].mean().index, s[f"smoothing_change_{met}"].mean(),
                            yerr=s[f"smoothing_change_{met}"].std(), marker="o", capsize=2, color=col, label=f"smoothing, h_A={h}")
                ax.plot(s[f"hca_vs_global_{met}"].mean().index, s[f"hca_vs_global_{met}"].mean(), "x--", color=col,
                        label=f"HCA − global, h_A={h}")
            ax.axhline(0, c="k", lw=0.6); ax.set_xscale("log", base=2); ax.set_xlabel("group degree ratio (group 1 / group 0)")
            ax.set_ylabel(f"change in {met}"); ax.set_title(f"Degree imbalance vs {met} change"); ax.legend(fontsize=6)
            tag(ax, "SIMULATED graphs, 10 realisations/cell")
        save(fig, "fig12_synthetic_degree_imbalance", "SIMULATED: smoothing-induced gap change (solid, mean ± SD over realisations) and HCA-minus-global-scalar gap difference (dashed) versus group degree imbalance at three homophily levels.")
    else:
        missing.append("synthetic")

    # ---------------------------------------------------------------- robustness (13, 14)
    rd = ROOT / cfg["robustness_dir"] / "all_metrics.csv"
    if rd.exists():
        rob = pd.read_csv(rd)
        rob = rob[(rob.part == "test") & (rob.operating_point == "default")]
        base = test[test.operating_point == "default"].assign(condition="baseline")
        both = pd.concat([base[rob.columns.intersection(base.columns)], rob], ignore_index=True)
        keep = ["n2v_lr", "fw_lr", "n2v_uniform", "fw_uniform", "n2v_hca", "fw_hca", "n2v_global", "fw_global",
                "n2v_lbp_uniform", "fw_lbp_uniform", "n2v_thr"]
        both = both[both.method.isin(keep)]
        rt = both.groupby(["condition", "method"]).agg(n_seeds=("seed", "nunique"), macro_f1=("macro_f1", "mean"),
                                                        macro_f1_sd=("macro_f1", "std"), auroc=("auroc", "mean"),
                                                        brier=("brier", "mean"), ece=("ece", "mean"),
                                                        dEOpp_gender=("dEOpp_gender", "mean"), dEOpp_sd=("dEOpp_gender", "std"),
                                                        dDP_gender=("dDP_gender", "mean")).reset_index()
        rt.round(4).to_csv(T / "table_robustness.csv", index=False)
        # smoothing effect per condition (paired over seeds)
        piv = both.pivot_table(index=["condition", "seed"], columns="method", values=["dEOpp_gender", "dDP_gender", "macro_f1", "brier"])
        eff = []
        for (cond, seed), r in piv.iterrows():
            for e in ("n2v", "fw"):
                for met in ("dEOpp_gender", "dDP_gender", "macro_f1", "brier"):
                    eff.append({"condition": cond, "seed": seed, "embedding": e, "metric": met,
                                "uniform_minus_lr": r[(met, f"{e}_uniform")] - r[(met, f"{e}_lr")],
                                "hca_minus_global": r[(met, f"{e}_hca")] - r[(met, f"{e}_global")]})
        eff = pd.DataFrame(eff)
        es = eff.groupby(["condition", "embedding", "metric"]).agg(
            uniform_minus_lr_mean=("uniform_minus_lr", "mean"), uniform_minus_lr_sd=("uniform_minus_lr", "std"),
            hca_minus_global_mean=("hca_minus_global", "mean"), hca_minus_global_sd=("hca_minus_global", "std"),
            n_seeds=("seed", "nunique")).reset_index()
        es.round(4).to_csv(T / "table_robustness_paired_effects.csv", index=False)
        # inference stability and forest-projection loss per condition (Experiment 5)
        srows = []
        for f in sorted((ROOT / cfg["robustness_dir"]).glob("*_info.json")):
            inf = json.loads(f.read_text())
            fs = inf.get("forest_stats", {})
            srows.append({"condition": f.name.split("_seed")[0], "graph_edges": fs.get("graph_edges"),
                          "forest_edges": fs.get("forest_edges"), "fraction_edges_kept": fs.get("fraction_edges_kept"),
                          "graph_components": fs.get("graph_components"), "forest_isolated_nodes": fs.get("forest_isolated_nodes"),
                          "hom_gender_graph": fs.get("homophily_gender_graph"), "hom_gender_forest": fs.get("homophily_gender_forest"),
                          "fw_lambda": inf.get("fw_lambda"), "fw_eta": inf.get("fw_eta"), "fw_global_scale": inf.get("fw_global_scale"),
                          "fw_lbp_converged": inf.get("fw_lbp_converged"), "fw_lbp_iters": inf.get("fw_lbp_iters"),
                          "n2v_lbp_converged": inf.get("n2v_lbp_converged")})
        if srows:
            st = pd.DataFrame(srows).groupby("condition").agg(
                n_seeds=("graph_edges", "size"), graph_edges=("graph_edges", "mean"), forest_edges=("forest_edges", "mean"),
                fraction_edges_kept=("fraction_edges_kept", "mean"), graph_components=("graph_components", "mean"),
                forest_isolated_nodes=("forest_isolated_nodes", "mean"), hom_gender_graph=("hom_gender_graph", "mean"),
                hom_gender_forest=("hom_gender_forest", "mean"), fw_lambda_values=("fw_lambda", lambda x: sorted(set(x))),
                fw_eta_sd=("fw_eta", "std"), fw_global_sd=("fw_global_scale", "std"),
                lbp_converged_runs=("fw_lbp_converged", "sum"), n2v_lbp_converged_runs=("n2v_lbp_converged", "sum"),
                fw_lbp_iters_max=("fw_lbp_iters", "max")).reset_index()
            st.round(4).to_csv(T / "table_robustness_inference_stability.csv", index=False)
        conds = ["baseline", "dropout0.1", "dropout0.3", "dropout0.5", "corrupt0.1", "corrupt0.3", "rewire0.1", "rewire0.3"]
        conds = [c for c in conds if c in set(rt.condition)]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, met, ttl in ((axs[0], "macro_f1", "test macro-F1"), (axs[1], "dEOpp_gender", "test ΔEOpp gender"),
                             (axs[2], "brier", "test Brier")):
            for m, col, ls in (("fw_lr", C["fw"], ":"), ("fw_uniform", C["fw"], "-"), ("fw_hca", "#7b2cbf", "-"),
                               ("n2v_lr", C["n2v"], ":"), ("n2v_uniform", C["n2v"], "-")):
                s = rt[rt.method == m].set_index("condition").reindex(conds)
                ax.errorbar(range(len(conds)), s[met], yerr=s.get(f"{met.split('_')[0]}_sd") if met != "brier" else None,
                            ls=ls, marker="o", ms=3, color=col, label=LABEL[m], capsize=2)
            ax.set_xticks(range(len(conds)), conds, rotation=40, fontsize=7); ax.set_title(ttl); tag(ax)
        axs[0].legend(fontsize=6.5)
        save(fig, "fig13_edge_perturbation_robustness", "Mandatory experiment 5: test metrics (mean ± SD over 5 seeds) under edge dropout, random edge corruption and degree-preserving rewiring. Embeddings, forest and λ are re-fitted on each perturbed graph; labels and splits unchanged.")
        fconds = [c for c in ["baseline", "forest_bfs", "forest_low_degree", "forest_random"] if c in set(rt.condition)]
        fs = pd.DataFrame([{"forest": m, **{k: v for k, v in audit["core"]["forest_projections"][m].items()
                                            if k.startswith(("homophily", "forest_max", "degree_spear", "forest_edges"))}}
                           for m in ("max_jaccard", "bfs", "low_degree", "random")])
        fs.round(4).to_csv(T / "table_forest_projection_stats.csv", index=False)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
        x = np.arange(len(fconds))
        for k, (m, col) in enumerate((("fw_lr", "#adb5bd"), ("fw_uniform", C["fw"]), ("fw_hca", "#7b2cbf"), ("fw_lbp_uniform", "#264653"))):
            s = rt[rt.method == m].set_index("condition").reindex(fconds)
            axs[0].bar(x + (k - 1.5) * 0.2, s.dEOpp_gender, 0.2, yerr=s.dEOpp_sd, capsize=2, color=col, label=LABEL[m])
            axs[1].bar(x + (k - 1.5) * 0.2, s.macro_f1, 0.2, yerr=s.macro_f1_sd, capsize=2, color=col)
        for ax, ttl in ((axs[0], "test ΔEOpp gender"), (axs[1], "test macro-F1")):
            ax.set_xticks(x, [c.replace("baseline", "max-Jaccard").replace("forest_", "") for c in fconds]); ax.set_title(ttl); tag(ax)
        axs[1].set_ylim(0.6, 0.72)
        fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
        save(fig, "fig14_forest_projection_comparison", "Robustness to the forest construction (FairWalk pipeline, mean ± SD over 5 seeds). Loopy BP runs on the full graph (no forest), so it is shown once, at the max-Jaccard baseline, as the non-forest reference.")
    else:
        missing.append("robustness")

    # synthetic embedding check table
    se = ROOT / cfg["synthetic_dir"] / "synthetic_embedding_check.csv"
    if se.exists():
        e = pd.read_csv(se).groupby(["group_homophily", "embedding", "stage"])[["sens_auroc", "auroc", "macro_f1", "dEOpp", "dDP"]].agg(["mean", "std"]).round(3)
        e.columns = [f"{a}_{b}" for a, b in e.columns]
        e.reset_index().to_csv(T / "table_synthetic_embedding_check.csv", index=False)

    (ROOT / "results/figures/CAPTIONS.md").write_text("# Figure captions (generated by scripts/make_results.py)\n\n" +
                                                       "\n".join(f"* **{k}** — {v}" for k, v in sorted(CAPTIONS.items())) + "\n")
    for name in CAPTIONS:
        record(f"results/figures/{name}.png", "python scripts/make_results.py --config " + args.config, cfg["_config_path"],
               runtime_s=time.time() - t0)
    for p in sorted(T.glob("table_*.csv")):
        record(f"results/tables/{p.name}", "python scripts/make_results.py --config " + args.config, cfg["_config_path"],
               runtime_s=time.time() - t0)
    log.info("made %d figures; missing inputs: %s", len(CAPTIONS), missing or "none")


if __name__ == "__main__":
    main()
