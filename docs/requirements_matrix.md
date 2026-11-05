# Requirements matrix

Sources: Problem 4 in `AFGML_Capstone_Problems.pdf` (authoritative for experiments, baselines, ablations, robustness, graphical model, inference, safeguards), the common capstone instructions (novelty, literature, statistics, fairness definitions, leakage, uncertainty, ethics, deliverables), `AFGML_Repository_Guide.md` (repository, evidence matrix, reproducibility) and `AFGML_Report_Template.tex` / `AFGML_Presentation_Template.tex`.

**Report-length discrepancy (resolved):** Problem 4 asks for an 8–10 page manuscript; the newer repository guide and report template ask for a 2–3 page evaluation report. We follow the newer guide for the graded report (`report/AFGML_Report.pdf`) and keep the longer supporting manuscript (`manuscript/Full_Research_Manuscript.pdf`) so neither requirement is dropped.

Statuses are only: **Complete**, **Partial**, **Failed**, **Missing**, **N/A** (with justification). "Complete" means the step was executed and its output inspected, not merely that a script exists. Report = `report/AFGML_Report.tex`, MS = manuscript, Slides = `slides/AFGML_Presentation.tex`. All commands run from the repository root with the pinned environment.

## R — Research

| ID | Requirement | Implementation | Verification command | Evidence | Status | Report/slide |
|---|---|---|---|---|---|---|
| R1 | ≥ 8 verified papers | `docs/related_work.md` (12 papers) | read file; DOIs/arXiv ids listed | 12 entries with venue, year, identifier; three entries flagged "to confirm" (node2vec venue/DOI, FMP AAAI-2024 listing, FairPSL DOI) | Partial — 9 fully verified, 3 flagged for the students to confirm | Report §1, MS §2 |
| R2 | Related-work comparison table | `docs/related_work.md` table | — | assumptions, data, model, fairness, inference, limitation per paper | Complete | MS Table 1, Slide 3 |
| R3 | Closest method identified | `docs/related_work.md` "closest methods" | — | C&S (pipeline), FairDrop (edge modification), FairPSL (fair pairwise model); FairWalk as reproduced baseline | Complete | Report §1 |
| R4 | Closest-method reproduction or documented restricted reproduction | `src/embeddings/random_walks.py` (FairWalk re-implementation) | `pytest tests/test_embeddings.py tests/test_data.py` | FairWalk re-implemented from the paper's description, group-balancing verified by test; not the authors' code; FairGNN not reproduced (pins torch 1.2.0 / DGL 0.4.3; uses region as its sensitive attribute) | Partial — restricted reproduction, documented | Report §1, MS §2 |
| R5 | Research gap analysis (known / not investigated / why / how different / falsification) | `docs/research_gap.md`, `docs/related_work.md` | — | five items answered | Complete | MS §1 |
| R6 | Falsifiable contribution statement | `docs/proposal.md` | — | H1–H4 with falsification rule declared before test evaluation | Complete | Report §1 |
| R7 | Dated proposal amendments | `docs/proposal.md` | — | pre-registration and the post-hoc region track are dated and labelled | Complete | MS §5 |

## D — Data

| ID | Requirement | Implementation | Verification command | Evidence | Status | Report/slide |
|---|---|---|---|---|---|---|
| D1 | Official Pokec-z source, checksums, provenance | `scripts/acquire_data.py`, `src/data/acquisition.py` | `python scripts/acquire_data.py --config configs/data.yaml` | `data/raw/PROVENANCE.json`, SHA-256 in `configs/data.yaml`; FairGNN commit hash not readable from this environment (github API blocked) — students must record it | Partial — hashes complete, upstream commit id missing | Report §2 |
| D2 | Schema audit | `src/data/schema.py`, `scripts/audit_dataset.py` | `python scripts/audit_dataset.py --config configs/data.yaml` | `results/metrics/dataset_audit.json`: 67,796 × 279, no missing, 617,958 undirected edges | Complete | MS §5 |
| D3 | Target transformation reproduced | `src/data/target.py` | `pytest tests/test_data.py::test_target_transformation_matches_fairgnn` | code −1 unlabelled, 0→0, 1–4→1 (FairGNN `labels[labels>1]=1`) | Complete | Report §2 |
| D4 | Gender and region coding documented, nothing inferred | `src/data/protected.py`, `docs/data_card.md` | `pytest tests/test_data.py::test_protected_attribute_not_imputed` | recorded 0/1 only; SNAP "1 - man" stated as unverified at record level | Complete | Report §2 |
| D5 | Fixed subgraph ≈ 1,000–3,000 nodes | `src/data/subgraph.py`, `scripts/prepare_data.py` | `python scripts/prepare_data.py --config configs/data.yaml` | 2,000 nodes, 5,351 edges, one component, seed 20261105, root 513797 | Complete | Report §2 |
| D6 | Leakage-safe splits | `src/data/splits.py`, `tests/test_leakage.py` | `pytest tests/test_leakage.py` | 50/25/25 stratified by Y×gender; val/test labels never in evidence, λ fit or thresholds (permutation test) | Complete | Report §2 |
| D7 | Data card | `docs/data_card.md` | — | population, label, groups, exclusions, sampling unit, licence | Complete | — |
| D8 | Pokec-n replication | — | — | not run | N/A — optional in the specification; time spent on the mandatory grid | MS Limitations |

## M — Models and baselines

| ID | Requirement | Implementation | Verification command | Evidence | Status | Report/slide |
|---|---|---|---|---|---|---|
| M1 | Feature-only logistic regression | `src/baselines/classifiers.py` (`feat_lr`) | `python scripts/run_baselines.py` | macro-F1 0.681, ΔEOpp 0.065, ΔDP 0.162 (5-seed means) | Complete | MS Table 2 |
| M2 | Node2Vec | `src/embeddings/random_walks.py` | `pytest tests/test_embeddings.py` | d=64, p=1, q=0.5, deterministic seeds (test) | Complete | Slide 5 |
| M3 | FairWalk | same file, `groups=` argument | `pytest tests/test_embeddings.py::test_fairwalk_balances_groups` | minority neighbour chosen ≈ 50% vs ≈ 10% for Node2Vec | Complete (re-implementation) | Slide 5 |
| M4 | Threshold-adjusted Node2Vec (decision-only mitigation) | `src/evaluation/decisions.py` (`n2v_thr`) | `python scripts/run_baselines.py` | ΔEOpp 0.013 but ΔDP 0.098 | Complete | Report table |
| M5 | Simple GNN if resources permit | `sgc_features` (2-hop SGC + LR) | `python scripts/run_baselines.py` | macro-F1 0.685 | Partial — linear SGC, not a trained nonlinear GNN | MS Table 2 |
| M6 | Published fair graph model if feasible | — | — | FairGNN code pins torch 1.2.0 / DGL 0.4.3 (incompatible with the Python 3.13 CPU environment) | N/A — infeasible here; documented, no substitute presented as reproduction | MS §2 |
| M7 | Conditional Markov network (unary + pairwise) | `src/graphical_model/pairwise_model.py`, `factors.py` | `pytest tests/test_factors.py` | positivity via clipping 1e-6; evidence as indicator factors | Complete | Report §2 equation |
| M8 | Fixed forest projection | `src/graphical_model/forest.py` | `pytest tests/test_forest.py` | max-Jaccard spanning forest, deterministic, label-free; 1,999 of 5,351 edges (37.4%) | Complete | Report §2 |
| M9 | Markov blankets | `pairwise_model.markov_blanket`, `conditional_given_blanket` | `pytest tests/test_inference_agreement.py::test_markov_blanket_conditional_matches_full_conditional` | closed-form conditional matches full conditional | Complete | MS §6 |
| M10 | Proposed attenuation (HCA) with fairness safeguard and leakage-free fitting | `src/graphical_model/attenuation.py` | `pytest tests/test_leakage.py` | λ on 5 training folds; η on validation under a 0.02 macro-F1 floor | Complete (executed; effect not supported, see E3) | Report §2 |
| M11 | Global-scalar comparator with the same tuning budget | `attenuation.select_on_validation` | — | same 11-point grid and rule | Complete | Report table |

## I — Inference and graphical-model correctness

| ID | Requirement | Implementation | Verification command | Evidence | Status |
|---|---|---|---|---|---|
| I1 | Exact forest sum-product (log space) | `src/graphical_model/sum_product.py` | `pytest tests/test_sum_product.py` | matches enumeration incl. components, isolated nodes, evidence | Complete |
| I2 | Variable elimination | `variable_elimination.py` | `pytest tests/test_variable_elimination.py` | order-invariant; Z matches enumeration | Complete |
| I3 | Clique-tree inference | `junction_tree.py` | `pytest tests/test_variable_elimination.py::test_clique_tree_matches_ve_and_is_calibrated` | calibrated sepsets | Complete |
| I4 | Exact enumeration | `enumeration.py` | used as reference in tests | — | Complete |
| I5 | Four-way agreement on fixtures | `scripts/verify_inference.py` | `python scripts/verify_inference.py --config configs/model.yaml` | 200 fixtures ≤ 12 nodes, max abs error ≈ 1.6e-15 vs tolerance 1e-8 (`results/metrics/inference_verification_summary.json`) | Complete |
| I6 | Factor dimensions / potential indexing | `tests/test_factors.py` | `pytest tests/test_factors.py` | scope and state-order tests | Complete |
| I7 | Normalisation | tests | `pytest tests/test_sum_product.py::test_marginals_normalised_and_evidence_clamped` | — | Complete |
| I8 | Evidence handling, contradictory evidence | tests | `pytest tests/test_sum_product.py` | zero-mass evidence raises | Complete |
| I9 | Disconnected components / isolated nodes | tests | `pytest tests/test_sum_product.py::test_isolated_node_equals_unary` | — | Complete |
| I10 | Numerical stability | tests | `pytest tests/test_sum_product.py::test_numerical_stability_extreme_potentials` | — | Complete |
| I11 | Joint conditional independence check | tests | `pytest tests/test_inference_agreement.py::test_joint_conditional_independence_given_separator` | {Y0,Y1} ⟂ {Y3,Y4,Y5} \| Y2 | Complete |
| I12 | Loopy BP on full graph (exploratory reference) | `loopy_bp.py` | `pytest tests/test_inference_agreement.py::test_loopy_bp_is_exact_on_trees` | converged in all 15 real runs (37–44 iterations) | Complete |

## F — Fairness, predictive and probabilistic metrics

| ID | Requirement | Implementation | Verification | Status |
|---|---|---|---|---|
| F1 | Demographic parity gap | `src/fairness/metrics.py` | `test_hand_computed_gaps` | Complete |
| F2 | Equality-of-opportunity gap (primary) | same | `test_hand_computed_gaps` | Complete |
| F3 | Equalised-odds gap = max(TPR range, FPR range) | same | `test_eodds_is_max_of_tpr_and_fpr_ranges` | Complete |
| F4 | Selection-rate ratio, undefined when denominator 0 | same | `test_selection_ratio_undefined_when_nobody_selected` | Complete |
| F5 | Group TPR/FPR/selection rates with denominators | same; `results/tables/group_rates_per_seed.csv` | `test_group_without_positives_gives_nan_tpr_and_flags` | Complete |
| F6 | Worst-group accuracy | same | in `*_metrics.csv` | Complete |
| F7 | Intersectional (gender × region) | same | `test_intersectional_groups_counted` | Complete |
| F8 | Small-group flag (< 30) | same | `test_small_group_flag_threshold`; smallest real cell 33 | Complete |
| F9 | AUROC, macro-F1, balanced accuracy | same | `test_probabilistic_metrics` and real outputs | Complete |
| F10 | Log loss, Brier, ECE | same | `test_probabilistic_metrics` | Complete |
| F11 | Group calibration | `reliability_table`; fig15 | `results/tables/table_group_reliability.csv` | Complete |
| F12 | Primary criterion and utility loss declared before testing | `configs/fairness.yaml`, `docs/protocol.md` | dated in `docs/proposal.md` | Complete |

## E — Mandatory experiments

| ID | Requirement | Command | Evidence (key finding) | Status | Report/slide |
|---|---|---|---|---|---|
| E1 | Reproduce unmitigated Node2Vec: utility, DP, EOpp, intersectional gaps, attribute predictability, hyper-parameters and seeds | `python scripts/run_experiments.py --config configs/experiments.yaml` | `table_methods_default.csv`, `table_embedding_diagnostics.csv`: macro-F1 0.667, ΔEOpp 0.101 [0.057, 0.157]; gender AUROC 0.511, region 0.991 | Complete | Report table, Slide 5 |
| E2 | Uniform smoothing on Node2Vec and FairWalk: changes in AUROC, macro-F1, balanced acc., worst-group acc., DP, EOpp, EOdds, Brier, log loss, group calibration | same + `python scripts/evaluate.py --config configs/evaluation.yaml` | `exploratory_paired_effects.csv` (H1-ctx, H2-ctx), `table_before_after_smoothing.csv`, fig07, fig15: no ΔEOpp change (H1 0.002 [−0.038, 0.022]); Node2Vec macro-F1 +0.011 [0.003, 0.022] | Complete | Report §4 |
| E3 | Attenuation vs embedding-only vs decision-only vs uniform: curves, matched utility, paired effects, intervals, parameter sensitivity | same | fig09, `table_methods_matched_utility.csv`, `table_attenuation_sensitivity.csv`, H3/H4: no difference; η unstable across seeds | Complete (negative result) | Report §4 |
| E4 | Synthetic graphs: vary demographic homophily, label homophily, degree imbalance, group sizes, label prevalence, training-label availability; multiple realisations | `python scripts/run_synthetic.py --config configs/synthetic.yaml` then `--extra` | `table_synthetic_homophily_degree.csv`, `table_synthetic_other_factors.csv`, `table_synthetic_embedding_check.csv`, fig11–12; 290 realisations (10 per cell). SIMULATED: smoothing raised ΔDP by 0.066 (SD 0.038) at demographic homophily 0.9 vs −0.009 at 0.5; 0.014 without a base-rate gap; HCA − global ΔEOpp −0.006 to 0.020 across cells | Complete (simulated; ΔEOpp effects noisy, SD ≈ mean) | Report §4, MS §8, Slide 9 |
| E5 | Missing/corrupted edges: dropout, corruption, rewiring; predictive, fairness, calibration degradation, inference stability, forest loss | `python scripts/run_robustness.py --config configs/robustness.yaml` | 35 runs (7 conditions × 5 seeds), all `ok`; `table_robustness.csv`, `table_robustness_paired_effects.csv`, `table_robustness_inference_stability.csv`, fig13. Utility and Brier degrade with perturbation; FairWalk smoothing effect on ΔEOpp −0.016 to 0.019, no consistent sign; loopy BP converged in all 70 runs; 50% dropout leaves ≈ 290 isolated forest nodes | Complete (seed SDs only, no block bootstrap for robustness) | MS §9 |

## A — Ablations

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| A1 | Remove pairwise factors ⇒ unary predictions | `*_nopair` equals `*_lr` to ≈ 5e-14 (E5 in `exploratory_paired_effects.csv`) | Complete |
| A2 | Global scalar instead of attenuation | H4: 0.000 [−0.037, 0.023], Holm p = 1.0 | Complete |
| A3 | Node2Vec instead of FairWalk | E3/E4 effects: HCA−uniform 0.004 [−0.013, 0.020]; HCA−global 0.001 [−0.018, 0.027] | Complete |

## RB — Robustness tests

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| RB1 | Degree-preserving rewiring | 10%, 30% double-edge swaps × 5 seeds; smoothing ΔEOpp effect −0.007 / 0.001 (FairWalk), macro-F1 gain 0.002 / 0.010 | Complete |
| RB2 | Edge dropout | 10/30/50% × 5 seeds; ΔEOpp effect 0.002 / 0.013 / −0.016, macro-F1 gain 0.012–0.014 | Complete |
| RB3 | Different forest constructions | BFS, low-degree, random vs max-Jaccard × 5 seeds; ΔEOpp effect within ±0.016; utility gain disappears (−0.006 to −0.002), so the Jaccard forest is what makes smoothing useful (fig14) | Complete |
| RB4 | Different embedding dimensions | d = 32, 128 (64 = main) × 5 seeds; ΔEOpp effect 0.016 / −0.019, macro-F1 gain 0.012 / 0.011 | Complete |
| RB5 | Reduced training-label availability | 50/25/10% of training labels × 5 seeds; ΔEOpp effect −0.019 / 0.009 / −0.027; fitted λ rises (up to 1.5) with fewer labels | Complete |

## S — Statistics

| ID | Requirement | Implementation | Evidence | Status |
|---|---|---|---|---|
| S1 | ≥ 5 seeds | `configs/experiments.yaml` | seeds 11, 22, 33, 44, 55 for main and robustness | Complete |
| S2 | Paired comparisons on identical units | `src/statistics/bootstrap.py` | identical bootstrap draws across methods, seed-averaged differences | Complete |
| S3 | ≥ 1,000 bootstrap replicates over the independent unit | Louvain block bootstrap | 1,000 replicates; only 15 blocks, 26.4% inter-block edges — coarse, documented | Partial — implemented as required; the unit is approximately independent only |
| S4 | Holm correction for confirmatory family | `holm()` | `confirmatory_tests.csv`; `test_holm_matches_hand_computation` | Complete |
| S5 | Effect sizes | `cohens_dz` | `dz_over_seeds` column | Complete |
| S6 | Small-group handling | metrics flags | no cell < 30 in the real test sets | Complete |
| S7 | Training vs evaluation vs scenario uncertainty separated | evaluate.py, synthetic SDs | `main_summary_training_uncertainty.csv`; synthetic reported as realisation SD | Complete |

## T — Tests and reproducibility

| ID | Requirement | Command | Evidence | Status |
|---|---|---|---|---|
| T1 | Data tests | `pytest tests/test_data.py` | schema, target, attributes, splits, integrity | Complete |
| T2 | Embedding tests | `pytest tests/test_embeddings.py` | dimensions, no missing, seed reproducibility, node mapping | Complete |
| T3 | Fairness tests | `pytest tests/test_fairness_metrics.py` | — | Complete |
| T4 | Graphical-model and inference tests | `pytest tests/test_factors.py tests/test_forest.py tests/test_sum_product.py tests/test_variable_elimination.py tests/test_inference_agreement.py` | — | Complete |
| T5 | Leakage tests | `pytest tests/test_leakage.py` | — | Complete |
| T6 | Statistical tests | `pytest tests/test_statistics.py` | — | Complete |
| T7 | Full suite | `python -m pytest tests` | 86 passed (2026-11-05) | Complete |
| T8 | Smoke test on simulated data | `python scripts/smoke_test.py` | "smoke test passed" | Complete |
| T9 | Headline result re-run from documented command (determinism) | `python scripts/run_experiments.py --config configs/determinism_check.yaml --seeds 11 --n-jobs 1` | seed-11 metrics (100 rows × 38 columns) and all saved predictions identical to the main run (max abs difference 0); `results/metrics/determinism_check/` | Complete (same machine and library versions) |
| T10 | Manifest with command, config, seed, data hash, commit, status, runtime, warnings | `results/manifest.csv` | written by every script via `src/utils.record` | Complete |
| T11 | Clean-environment install test | — | not done: dependencies were installed once in this environment | Missing — students should test a fresh clone |

## DOC — Documentation and deliverables

| ID | Requirement | Location | Status |
|---|---|---|---|
| DOC1 | README with 18 items | `README.md` | Partial — team ID, repository URL and the per-student contribution split are still placeholders |
| DOC2 | Proposal | `docs/proposal.md` | Complete |
| DOC3 | Related-work review | `docs/related_work.md` | Partial (see R1) |
| DOC4 | Data card | `docs/data_card.md` | Complete |
| DOC5 | Model card | `docs/model_card.md` | Complete |
| DOC6 | Assumptions register + 10 ethics items | `docs/assumptions.md` | Complete |
| DOC7 | Protocol | `docs/protocol.md` | Complete |
| DOC8 | Mathematical formulation, methodology | `docs/mathematical_formulation.md`, `docs/methodology.md` | Complete |
| DOC9 | Experiment log | `docs/experiment_log.md` | Complete |
| DOC10 | Contributions table | `docs/contributions.md` | Partial — structure only; students must fill names and individual statements |
| DOC11 | 16 figures | `results/figures/` (PNG + PDF, captions in `CAPTIONS.md`) | Complete |
| DOC12 | Results tables | `results/tables/` | Complete |
| DOC13 | 2–3 page report, compiled and inspected | `report/AFGML_Report.pdf` (3 pages incl. references; no overfull boxes) | Partial — compiled and inspected; student placeholders remain |
| DOC14 | Supporting manuscript 8–10 pages | `manuscript/Full_Research_Manuscript.pdf` (10 pages; main text ends on p. 9, then references and appendix) | Partial — compiled and inspected; student placeholders remain |
| DOC15 | Beamer slides ≈ 10–12 + backup | `slides/AFGML_Presentation.pdf` (11 main + 3 backup frames, from the course template) | Partial — compiled and inspected; student placeholders remain |
| DOC16 | Demo README + saved trace | `demo/` | Complete |
| DOC17 | Individual contribution statements | `docs/contributions.md` | Missing — must be written by each student |
| DOC18 | All placeholders replaced | report, slides, README | Partial — names and registration numbers filled; team ID, repository URL, submission commit, date remain |
| DOC19 | Submission tag / evaluator access | — | Missing — no remote repository exists yet |
