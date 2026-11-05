"""Write LaTeX macros with every number quoted in the report, manuscript and slides,
read from saved result files, so no result is typed by hand.

Usage: python scripts/make_macros.py   ->   results/tables/results_macros.tex
"""
import json
import re

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import ROOT, record

T = ROOT / "results/tables"
M = ROOT / "results/metrics"
out = {}


def name(s):
    s = re.sub(r"[^A-Za-z]", "", s.title().replace("_", ""))
    return s


def put(key, val, d=3):
    if isinstance(val, (float, np.floating)):
        val = "n/a" if not np.isfinite(val) else f"{val:.{d}f}"
    out[name(key)] = str(val)


def p_fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# data
audit = json.loads((M / "dataset_audit.json").read_text())["core"]
put("hom gender", audit["edge_homophily_gender"]); put("hom gender rand", audit["edge_homophily_gender_if_random_mixing"])
put("hom region", audit["edge_homophily_region"]); put("hom region rand", audit["edge_homophily_region_if_random_mixing"])
put("hom label", audit["edge_homophily_label"]); put("hom label rand", audit["edge_homophily_label_if_random_mixing"])
fp = audit["forest_projections"]["max_jaccard"]
put("forest edges", fp["forest_edges"]); put("forest kept", 100 * fp["fraction_edges_kept"], 1)
put("forest hom gender", fp["homophily_gender_forest"]); put("forest hom region", fp["homophily_region_forest"])
put("forest hom label", fp["homophily_label_forest"])
bd = json.loads((M / "bootstrap_design.json").read_text())
put("n blocks", bd["n_blocks"]); put("interblock share", bd["inter_block_edge_share"])
# inference
iv = json.loads((M / "inference_verification_summary.json").read_text())
put("inf max err", "{:.1e}".format(max(iv[k] for k in ("max_abs_sp_vs_enum", "max_abs_ve_vs_enum", "max_abs_ct_vs_enum", "max_abs_ve_vs_ct"))))
put("inf fixtures", iv["fixtures"])
# embedding diagnostics
ed = pd.read_csv(T / "table_embedding_diagnostics.csv").set_index(["embedding", "attribute"])
for e in ("n2v", "fw", "fwr"):
    for a in ("gender", "region"):
        put(f"sens {e} {a}", ed.loc[(e, a), "auroc_mean"]); put(f"sens {e} {a} sd", ed.loc[(e, a), "auroc_std"])
# method levels
lv = pd.read_csv(T / "method_levels_block_bootstrap.csv")
for _, r in lv[lv.operating_point == "default"].iterrows():
    k = f"{r.method} {r.metric}"
    put(k, r.mean_over_seeds); put(k + " lo", r.ci95_low); put(k + " hi", r.ci95_high)
for _, r in lv[lv.operating_point == "matched_utility"].iterrows():
    k = f"mu {r.method} {r.metric}"
    put(k, r.mean_over_seeds); put(k + " lo", r.ci95_low); put(k + " hi", r.ci95_high)
au = pd.read_csv(T / "table_auroc_by_method.csv").set_index("method")
for m, r in au.iterrows():
    put(f"auroc {m}", r["mean"]); put(f"auroc {m} sd", r["std"])
# confirmatory
cf = pd.read_csv(T / "confirmatory_tests.csv")
for _, r in cf.iterrows():
    k = {"H1": "hone", "H2": "htwo", "H3": "hthree", "H4": "hfour"}[r.id]
    put(k + " est", r.estimate_mean_over_seeds); put(k + " lo", r.ci95_low); put(k + " hi", r.ci95_high)
    put(k + " p", p_fmt(r.p_unadjusted)); put(k + " pholm", p_fmt(r.p_holm))
    put(k + " smin", r.seed_diff_min); put(k + " smax", r.seed_diff_max)
ex = pd.read_csv(T / "exploratory_paired_effects.csv")
ex = ex[ex.operating_point == "default"]
words = {"E1": "eone", "E2": "etwo", "E3": "ethree", "E4": "efour", "E5": "efive", "E6": "esix", "E7": "eseven",
         "E8": "eeight", "E9": "enine", "E10": "eten", "E11": "eeleven", "E12": "etwelve", "E13": "ethirteen",
         "H1-ctx": "honectx", "H2-ctx": "htwoctx", "H3-ctx": "hthreectx", "H4-ctx": "hfourctx"}
for _, r in ex.iterrows():
    k = f"{words[r.id]} {r.metric}"
    put(k, r.estimate_mean_over_seeds); put(k + " lo", r.ci95_low); put(k + " hi", r.ci95_high); put(k + " p", p_fmt(r.p_unadjusted))
# selected parameters
sp = pd.read_csv(T / "table_selected_parameters.csv")
put("lambda values", ", ".join(sorted({f"{x:g}" for x in sp["lambda"]})))
put("eta fw list", ", ".join(f"{x:g}" for x in sp[sp.emb == "fw"].eta))
put("glob fw list", ", ".join(f"{x:g}" for x in sp[sp.emb == "fw"]["global"]))
put("eta nv list", ", ".join(f"{x:g}" for x in sp[sp.emb == "n2v"].eta))
# group rates pooled (FairWalk LR / uniform)
gp = pd.read_csv(T / "table_group_rates_pooled.csv")
put("min test cell", int(gp[["min_n_per_seed", "min_pos_per_seed", "min_neg_per_seed"]].min().min()), 0)
for m in ("fw_lr", "fw_uniform"):
    for g in ("0", "1"):
        r = gp[(gp.method == m) & (gp.attribute == "gender") & (gp.group.astype(str) == g)].iloc[0]
        put(f"tpr {m} g{'zero' if g == '0' else 'one'}", r.tpr); put(f"sel {m} g{'zero' if g == '0' else 'one'}", r.selection_rate)
# synthetic
sy = T / "table_synthetic_homophily_degree.csv"
if sy.exists():
    s = pd.read_csv(sy)
    for h in (0.5, 0.7, 0.9):
        for dr in (1.0, 4.0):
            r = s[(s.group_homophily == h) & (s.degree_ratio == dr)].iloc[0]
            k = f"syn h{int(h * 10)} d{int(dr)}"
            k = k.replace("h5", "hfive").replace("h7", "hseven").replace("h9", "hnine").replace("d1", "done").replace("d4", "dfour")
            for c in ("smooth_dEOpp_mean", "smooth_dEOpp_sd", "smooth_dDP_mean", "smooth_dDP_sd", "hca_vs_global_dEOpp",
                      "hca_vs_uniform_dEOpp", "smooth_f1_mean", "real_hA", "unary_dEOpp", "uniform_dEOpp", "hca_dEOpp", "global_dEOpp"):
                put(f"{k} {c}", r[c])
    syn = pd.read_csv(ROOT / "results/metrics/synthetic/synthetic_realisations.csv")
    put("syn n real", len(syn), 0)
    hx = syn[syn.design == "homophily_x_degree"]
    from scipy.stats import spearmanr
    rho = spearmanr(hx.exposure_gap_train, hx.smoothing_change_dDP).statistic
    put("syn diag rho", rho)
    o = pd.read_csv(T / "table_synthetic_other_factors.csv")
    for _, r in o[o.design == "no_base_rate_gap"].iterrows():
        put(f"syn nogap h{'five' if r.group_homophily == 0.5 else 'nine'} dDP", r.smooth_dDP_mean)
        put(f"syn nogap h{'five' if r.group_homophily == 0.5 else 'nine'} dDP sd", r.smooth_dDP_sd)
    for _, r in o[o.design == "train_fraction"].iterrows():
        tfw = {0.1: "ten", 0.25: "twentyfive", 0.5: "fifty"}.get(round(float(r.train_fraction), 2), "other")
        put(f"syn tf {tfw} dDP", r.smooth_dDP_mean)
    for _, r in o[o.design == "group_share"].iterrows():
        gw = {0.2: "twenty", 0.35: "thirtyfive", 0.5: "fifty"}.get(round(float(r.group_share), 2), "other")
        put(f"syn share {gw} dDP", r.smooth_dDP_mean); put(f"syn share {gw} dDP sd", r.smooth_dDP_sd)
    for _, r in o[o.design == "prevalence"].iterrows():
        pw = {0.2: "low", 0.4: "mid", 0.6: "high"}.get(round(float(r.prevalence0), 2), "other")
        put(f"syn prev {pw} dDP", r.smooth_dDP_mean); put(f"syn prev {pw} dDP sd", r.smooth_dDP_sd)
    put("syn hcaglob min", hx.groupby(["group_homophily", "degree_ratio"]).hca_vs_global_dEOpp.mean().min())
    put("syn hcaglob max", hx.groupby(["group_homophily", "degree_ratio"]).hca_vs_global_dEOpp.mean().max())
# robustness
rp = T / "table_robustness_paired_effects.csv"
if rp.exists():
    r = pd.read_csv(rp)
    for _, x in r[(r.embedding == "fw") & (r.metric.isin(["dEOpp_gender", "macro_f1"]))].iterrows():
        k = f"rob {x.condition} {x.metric}"
        k = k.replace("0.1", "pone").replace("0.3", "pthree").replace("0.5", "pfive").replace("0.25", "ptwofive").replace("32", "thirtytwo").replace("128", "onetwoeight")
        put(k + " smooth", x.uniform_minus_lr_mean); put(k + " smoothsd", x.uniform_minus_lr_sd)
        put(k + " hcaglob", x.hca_minus_global_mean)
    rt = pd.read_csv(T / "table_robustness.csv")
    put("rob conditions", rt.condition.nunique() - 1, 0)

lines = ["% AUTO-GENERATED by scripts/make_macros.py from results/ -- do not edit by hand"]
lines += [f"\\newcommand{{\\R{k}}}{{{v}}}" for k, v in sorted(out.items())]
(T / "results_macros.tex").write_text("\n".join(lines) + "\n")
record("results/tables/results_macros.tex", "python scripts/make_macros.py", "configs/evaluation.yaml")
print(f"wrote {len(out)} macros")
