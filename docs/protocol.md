# Evaluation protocol (frozen before test evaluation)

All settings live in `configs/`; this file explains them. Dates: written 2026-11-05 before `scripts/evaluate.py` first computed test metrics (see `docs/proposal.md` for the one retrospective, exploratory addition).

## Population, graph and split
* **Data:** Pokec-z (`region_job.csv`, `region_job_relationship.txt`) from the FairGNN repository, SHA-256 in `configs/data.yaml`.
* **Core subgraph:** induced graph on labelled nodes → largest connected component (7,659 nodes) → BFS from a root drawn with `sampling_seed = 20261105` → first 2,000 visited nodes (5,351 edges, connected). The procedure never sees label values, attributes or model outputs and was run once.
* **Protocol: transductive.** The whole core topology (and, for FairWalk, recorded attributes) is available to unsupervised embedding training and to the forest projection. Labels are partitioned per seed by a stratified (Y × gender) 50/25/25 train/validation/test split (FairGNN uses 50/25/25 of labelled nodes). Why this is not label leakage: topology and recorded attributes are inputs available for every node under the declared protocol; no validation or test *label* enters embedding training, classifier fitting, λ fitting, Markov-network evidence, η/scale/threshold selection. `tests/test_leakage.py` shows that permuting validation/test labels leaves every fitted λ and posterior unchanged.
* **Seeds:** 11, 22, 33, 44, 55. Each seed redraws the label split and the walk/skip-gram seed. Seeds measure algorithmic and split variability, not independent populations.

## Models and selection
| Component | Fitted on | Selected on | Grid / rule |
|---|---|---|---|
| Node2Vec / FairWalk embeddings | topology (+ gender for FairWalk), no labels | — | d=64, walk 40, 10 walks/node, p=1, q=0.5, window 5, 5 negatives, 5 epochs |
| Logistic regression (unary, feature-only, SGC) | training labels | validation log loss | C ∈ {0.01, 0.1, 1, 10} |
| Pairwise strength λ | training folds (5-fold, cross-fitted unaries, held-out fold latent) | — | λ ∈ {0, .25, .5, .75, 1, 1.5, 2, 3}, min held-out log loss |
| HCA η | — | validation | η ∈ {0, .1, …, 1}: min val ΔEOpp(gender) s.t. val macro-F1 ≥ uniform − 0.02; ties → smaller η |
| Global scale c (ablation B) | — | validation | same 11-point grid and same rule |
| Decision threshold | — | validation | max val macro-F1 over 41 score quantiles (+0.5) |
| Group thresholds (decision-only mitigation) | — | validation | 21×21 per-gender quantile grid; min val ΔEOpp s.t. val macro-F1 ≥ Node2Vec − 0.02 |
| Matched-utility operating point | — | validation | per method, min val ΔEOpp among thresholds with val macro-F1 ≥ (Node2Vec+LR val macro-F1 − 0.02) |

Inference: exact sum-product on the `max_jaccard` maximum spanning forest (1,999 of 5,351 edges); evidence = training labels only; probabilities clipped to [1e-6, 1−1e-6].

## Metrics
Primary fairness: ΔEOpp(gender). Secondary: ΔDP, ΔEOdds, R_sel, worst-group accuracy (gender, region, gender×region). Primary utility: macro-F1; also AUROC, balanced accuracy. Probabilistic: log loss, Brier, ECE (10 equal-width bins), group ECE gap, reliability tables. Groups with < 30 relevant observations are flagged exploratory, never dropped (all gender/region/intersectional test cells have ≥ 30 nodes; see `results/tables/group_rates_per_seed.csv` for TPR/FPR denominators).

## Uncertainty (three sources reported separately)
1. **Fixed-model evaluation uncertainty:** graph-block bootstrap. Blocks = Louvain communities of the core topology (seed 0, label-blind). 1,000 replicates; each draws blocks with replacement and takes all test nodes of drawn blocks; the same draw is applied to all methods and seeds; the paired effect is averaged over seeds within a replicate. 95% percentile intervals; two-sided p = 2·min(P(Δ≤0), P(Δ≥0)) with +1 smoothing. Limitation: inter-block edges remain, so dependence is reduced, not removed; absolute-range metrics such as ΔEOpp are non-smooth near 0 and percentile intervals can be asymmetric.
2. **Training uncertainty:** per-seed estimates, their SD/min/max, and d_z = mean/SD of per-seed paired differences.
3. **Synthetic realisation variability:** across 10 independent graph realisations per scenario, reported only in the synthetic tables.

## Confirmatory family (Holm-corrected, default operating point, metric ΔEOpp gender)
* H1: FairWalk + uniform smoothing vs FairWalk + LR.
* H2: Node2Vec + uniform smoothing vs Node2Vec + LR.
* H3: FairWalk + HCA vs FairWalk + uniform.
* H4: FairWalk + HCA vs FairWalk + global scalar (same tuning budget).
All other comparisons (ablations A/C, decision-only, region track, robustness, synthetic) are exploratory and reported unadjusted.

## Inference correctness
Tolerance 1e-8 (max absolute marginal difference) for double-precision fixtures of ≤ 12 nodes. Compared: enumeration, forest sum-product (forest fixtures), variable elimination (min-fill and random orders), clique tree (Shafer–Shenoy). `scripts/verify_inference.py` (200 fixtures) and `tests/`.
