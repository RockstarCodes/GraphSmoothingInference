# Does Graph Smoothing Reintroduce Bias into Fair Node Embeddings?

**ICT 5143 Algorithms Fairness and Graphical Models Laboratory — Capstone Problem 4.**
An exact-inference audit of pairwise Markov smoothing on top of Node2Vec and FairWalk embeddings, on the Pokec-z benchmark and on simulated graphs.

**This is an offline research audit. Nothing here is a recruitment, employment or individual decision-making system.**

## 1. Identity
| | |
|---|---|
| Problem | 4 — Does Graph Smoothing Reintroduce Bias into Fair Node Embeddings? |
| Team | [Team ID] |
| Students | [Student 1 name, roll number]; [Student 2 name, roll number] |
| Repository | [repository URL] — submission commit `[commit]` |
| Dataset | Pokec-z (FairGNN release), fixed 2,000-node core subgraph |
| Y = 1 | the benchmark's positive working-field class: raw `I_am_working_in_field` code ≥ 1 (codes are undocumented, so Y has no occupational meaning) |
| D = 1 | the pipeline predicts Y = 1 |
| Protected attribute | recorded `gender` (0/1, primary); recorded `region` (0/1, secondary) |

## 2. Research question
When does probabilistic graph smoothing undo the fairness gains of debiased node embeddings, and can constrained edge potentials reduce this effect? Pre-declared hypothesis: with demographic homophily and a base-rate gap, uniform smoothing widens the gender equal-opportunity gap (ΔEOpp); falsified if the paired block-bootstrap effects (H1, H2) are not positive after Holm correction.

## 3. Contribution (bounded)
1. A controlled, exact-inference decomposition of decision-level fairness change between the embedding stage and a downstream conditional Markov network (CMN), on Pokec-z and on simulated graphs with known homophily, degree imbalance, label homophily, group sizes, prevalence and label availability.
2. Homophily-constrained attenuation (HCA), λ_ij = λ(1 − η·1[A_i = A_j]), tested against a global-scalar attenuation with the same tuning budget (H4). **It did not beat the global scalar.**
3. A tested implementation of factor algebra, forest sum-product, variable elimination, clique tree, enumeration and a graph-block bootstrap protocol.

Closest work: FairWalk (reproduced as a documented re-implementation), Correct & Smooth, FairDrop, FairPSL — see `docs/related_work.md` (12 papers).

## 4. Dataset
FairGNN's Pokec-z files (67,796 users, 617,958 undirected edges), downloaded by script and verified by SHA-256; not redistributed (no licence stated). Core: BFS ball of 2,000 labelled nodes (5,351 edges). P(Y=1) is 0.40 for gender 0 and 0.56 for gender 1. Edge homophily: gender 0.449 (0.500 under random mixing — no gender homophily), region 0.963 (0.514), label 0.586 (0.501). Details: `data/README.md`, `docs/data_card.md`.

## 5. Mathematical model
For node i: label Y_i, embedding H_i (Node2Vec or FairWalk), unary p_i from L2 logistic regression; training labels are evidence; validation/test labels are latent. On a label-free maximum-Jaccard spanning forest E_F (1,999 of 5,351 edges):

P(Y | H, E_F) = (1/Z) ∏_i ψ_i(Y_i) ∏_{(i,j)∈E_F} exp(λ_ij · 1[Y_i = Y_j]),  ψ_i(1) = p_i, ψ_i(0) = 1 − p_i.

Query: q_i = P(Y_i = 1 | y_train) by exact log-space sum-product; decision D_i = 1[q_i ≥ τ], τ maximising validation macro-F1. λ is fitted on five training folds with cross-fitted unaries; η (HCA) and a global scale c (ablation) are chosen on validation by: minimise ΔEOpp subject to macro-F1 ≥ uniform − 0.02. Full derivation: `docs/mathematical_formulation.md`.

## 6. Fairness definitions
For groups g: selection rate s_g, TPR t_g, FPR f_g. ΔDP = max s − min s; **ΔEOpp = max t − min t (primary, gender)**; ΔEOdds = max(Δt, Δf); R_sel = min s / max s (undefined if max s = 0). Also worst-group accuracy, gender × region intersectional gaps, per-group calibration, ECE, Brier, log loss. Groups with < 30 relevant observations are flagged exploratory (none occurred in the real test sets). Rationale for EOpp over DP: `docs/proposal.md`.

## 7. Headline findings (5 seeds; 95% intervals from 1,000 graph-block bootstrap replicates)
* **Negative real-data result.** Uniform smoothing changed the FairWalk pipeline's gender ΔEOpp by 0.002 [−0.038, 0.022] (H1) and Node2Vec's by 0.001 [−0.028, 0.039] (H2); HCA − uniform 0.004 [−0.020, 0.026] (H3); HCA − global scalar 0.000 [−0.037, 0.023] (H4). Holm-adjusted p = 1.0 for all four.
* Smoothing slightly improved utility and calibration (Node2Vec macro-F1 +0.011 [0.003, 0.022], Brier −0.004 [−0.008, −0.001]; exploratory).
* Gender was not recoverable from either embedding (AUROC 0.511 Node2Vec, 0.501 FairWalk); region was (0.991, 0.992). FairWalk on region did not reduce region predictability (0.986).
* Failure case: validation-selected η was unstable across seeds (0.8, 0, 0.3, 0.5, 1.0); fitted λ was 0.25 in every seed.
* Decision-only group thresholds reduced ΔEOpp to 0.013 but raised ΔDP to 0.098 — the two criteria conflict here.
* SIMULATED graphs (fair-by-construction unary, 10 realisations per cell): uniform smoothing raised ΔDP by 0.066 (SD 0.038) at demographic homophily 0.9 but −0.009 (SD 0.026) at 0.5, and only 0.014 at 0.9 when the groups had equal base rates; the ΔEOpp change followed the same direction (0.039 vs −0.002) with SDs as large as the means. Degree imbalance did not amplify it. HCA − global scalar ranged −0.006 to 0.020 across all homophily × degree cells: HCA never beat the scalar.
* Robustness (15 conditions × 5 seeds, all runs ok): the FairWalk smoothing effect on ΔEOpp ranged −0.027 to 0.019 with no consistent sign; its macro-F1 gain survived dropout, corruption and rewiring but vanished with BFS, low-degree and random forests.

Full tables: `results/tables/`; figures: `results/figures/` (captions in `CAPTIONS.md`).

## 8. Environment
CPython 3.13 on Linux x86_64; CPU only.
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt   # exact versions; requirements.txt has ranges
```
`environment.yml` is provided for conda users (not tested here). LaTeX (pdflatex with lmodern, beamer, tikz) is needed only to rebuild the PDFs.

## 9. Dataset acquisition
```bash
python scripts/acquire_data.py --config configs/data.yaml
python scripts/prepare_data.py --config configs/data.yaml
python scripts/audit_dataset.py --config configs/data.yaml
```
No credentials are needed. The download is ≈ 50 MB.

## 10. Smoke test (no dataset needed, about 1 minute)
```bash
python scripts/smoke_test.py
python scripts/verify_inference.py --config configs/model.yaml
python -m pytest tests
```
The smoke test runs the whole pipeline on a SIMULATED 300-node graph. It shows the code runs end to end; it substantiates no research claim.

## 11. Full reproduction
```bash
python scripts/run_experiments.py --config configs/experiments.yaml   # Exp. 1–3, ablations; ~7 min on 4 cores
python scripts/run_baselines.py   --config configs/experiments.yaml   # baseline table from the saved runs
python scripts/run_synthetic.py   --config configs/synthetic.yaml     # Exp. 4 (simulated)
python scripts/run_synthetic.py   --config configs/synthetic.yaml --extra   # group-size and prevalence designs
python scripts/run_robustness.py  --config configs/robustness.yaml    # Exp. 5 + robustness, 75 runs
python scripts/evaluate.py        --config configs/evaluation.yaml    # bootstrap, Holm, paired effects
python scripts/make_results.py    --config configs/evaluation.yaml    # figures 01-16, tables
python scripts/make_macros.py                                         # numbers used by report/slides
cd report && pdflatex AFGML_Report.tex && pdflatex AFGML_Report.tex && cd ..
cd slides && pdflatex AFGML_Presentation.tex && pdflatex AFGML_Presentation.tex && cd ..
cd manuscript && pdflatex Full_Research_Manuscript.tex && pdflatex Full_Research_Manuscript.tex && cd ..
```
Baselines are fitted inside `run_experiments.py` (same split and embeddings as the smoothed models, so comparisons are paired); `run_baselines.py` extracts them. One headline result alone: `python scripts/run_experiments.py --config configs/experiments.yaml --seeds 11 --n-jobs 1`.

## 12. Hardware
Executed on a 4-core x86_64 CPU container with 16 GB RAM, no GPU. Resident memory observed per worker during the robustness runs was about 0.25 GB (from `top`; not systematically profiled).

## 13. Expected runtime (measured here, from `results/manifest.csv`)
| step | runtime |
|---|---|
| smoke test (simulated) | 23–27 s |
| `verify_inference.py` | 2 s |
| `run_experiments.py`, 5 seeds | 204–222 s per seed; 7 min wall-clock on 4 workers |
| `run_synthetic.py` main grid, 230 realisations | 39 min on 1 worker (shared CPU) + 1 min embedding check |
| `run_synthetic.py --extra`, 60 realisations | 2.7 min on 4 workers |
| `run_robustness.py`, 75 runs | 90–104 s per run; 40 min wall-clock on 3 workers |
| `evaluate.py` | ≈ 25 s (1,000 bootstrap replicates) |
| `make_results.py` | about 15 s |


## 14. Report and presentation
* Evaluation report (2–3 pages): `report/AFGML_Report.pdf`
* Supporting manuscript: `manuscript/Full_Research_Manuscript.pdf`
* Slides: `slides/AFGML_Presentation.pdf`
* Demonstration: `demo/README.md`, saved trace `demo/saved_trace.txt`

## 15. Requirements matrix and evidence
`docs/requirements_matrix.md` maps every requirement to code, command, artifact and status. Also: `docs/proposal.md`, `docs/protocol.md`, `docs/related_work.md`, `docs/research_gap.md`, `docs/data_card.md`, `docs/model_card.md`, `docs/assumptions.md`, `docs/experiment_log.md`, `results/manifest.csv`.

## 16. Repository layout
```
configs/   frozen configurations          scripts/  entry points (acquire … make_macros)
src/       data, embeddings, graphical_model, baselines, evaluation, fairness, statistics, synthetic, experiments
tests/     pytest suite                   notebooks/ exploratory_analysis.ipynb (reads saved outputs only)
results/   metrics/, tables/, figures/, logs/, manifest.csv
docs/      proposal, related work, cards, assumptions, protocol, matrix, log, contributions
report/  manuscript/  slides/  demo/  data/ (raw and processed are git-ignored)
```

## 17. Individual contributions
See `docs/contributions.md`. Each student must replace the proposed allocation with what they actually did and write a short individual statement (one technical decision, one difficulty, one cross-check of the other student).

## 18. External attribution and assistance disclosure
* Data: Pokec (Takac & Zabovsky, 2012, via SNAP) as sampled and released by FairGNN (Dai & Wang, WSDM 2021).
* Methods re-implemented from papers: Node2Vec (Grover & Leskovec, 2016), FairWalk (Rahman et al., 2019). Libraries: NumPy, SciPy, pandas, scikit-learn, NetworkX (Louvain), gensim (Word2Vec), matplotlib.
* AI assistance: substantial use of an AI coding assistant (Claude, Anthropic) for code, experiments and drafting, disclosed in `docs/contributions.md`; disclose according to course policy.
* Licence: MIT for this project's code (`LICENSE`); data terms are those of the original sources.
