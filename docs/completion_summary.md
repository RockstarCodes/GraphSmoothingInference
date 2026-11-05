# Completion summary — AFGML Problem 4

Date: 2026-11-05. Everything below was executed in a Linux container (4 CPU cores, Python 3.13.18). Numbers come from `results/`; nothing was typed in by hand.

## The uncomfortable part first
1. **The real-data answer is negative.** None of the four pre-registered hypotheses holds: every paired effect on gender ΔEOpp is within ±0.004, CIs about ±0.03, Holm p = 1.0.
2. **The core subgraph does not have the property the question is about.** Gender edge homophily is 0.449 against 0.500 under random mixing, so there is no gender homophily for smoothing to amplify, and gender is not recoverable from either embedding (AUROC ≈ 0.50). FairWalk had nothing to remove. The real-data test of the hypothesis is weak by construction; the mechanism is only shown in simulation.
3. **The proposed method (HCA) did not beat the trivial alternative** (a global scalar on λ) on real data, in simulation, or under any robustness condition. Its validation-selected η jumps between 0 and 1 across seeds.
4. **Statistical resolution is limited:** only 15 Louvain blocks exist as resampling units and 26% of edges cross blocks.
5. **Not reproduced:** FairGNN (code pins torch 1.2.0 / DGL 0.4.3). FairWalk is a re-implementation from the paper, not the authors' code. The "simple GNN" is linear SGC.

## 1. Location
`/home/claude/AFGML_Problem4` (container, ephemeral) and a copy at `/mnt/project-files/AFGML_Problem4/` without raw/processed data and embedding files. Local git repository only; nothing pushed.

## 2. Repository tree
See README §16. Top level: `configs/ scripts/ src/ tests/ notebooks/ results/ docs/ report/ manuscript/ slides/ demo/ data/`.

## 3. Research question and contribution
When does probabilistic graph smoothing undo the fairness gains of debiased embeddings, and can constrained edge potentials reduce this effect? Contribution: an exact-inference decomposition of fairness change between the embedding and a downstream CMN (real + simulated), and HCA λ_ij = λ(1 − η·1[A_i = A_j]) tested against a global scalar. Outcome: the decomposition and the simulated conditions are the contribution; HCA is a refuted proposal.

## 4. Literature
12 papers reviewed, 9 fully verified against landing pages, 3 with one field to confirm (node2vec venue/DOI, FMP AAAI-2024 listing, FairPSL DOI). `docs/related_work.md`.

## 5. Dataset
FairGNN Pokec-z, SHA-256 verified; 67,796 × 279, no missing values, 617,958 undirected edges. Target reproduced from FairGNN code (code ≥ 1 → Y=1; codes undocumented). Core: 2,000 nodes, 5,351 edges, P(Y=1) 0.40 (gender 0) vs 0.56 (gender 1). FairGNN commit hash not recorded (GitHub API blocked).

## 6–7. Algorithms and model
Node2Vec, FairWalk, feature LR, SGC, group thresholds; pairwise CMN with L2-LR unaries, indicator evidence, attractive potentials on a max-Jaccard spanning forest (1,999 of 5,351 edges); exact log-space sum-product, variable elimination, Shafer–Shenoy clique tree, enumeration, loopy BP (exploratory); λ by 5-fold training cross-fitting; η / global scale by constrained validation selection.

## 8. Inference validation
200 fixtures (100 forests, 100 loopy, ≤ 12 nodes): max absolute marginal error 1.6e-15 vs tolerance 1e-8 across sum-product, VE, clique tree and enumeration; log Z error 2.2e-15. Unit tests cover scopes, indexing, evidence, normalisation, isolated nodes, components, zero mass, joint conditional independence and the Markov-blanket conditional.

## 9–10. Experiments
| | executed | status |
|---|---|---|
| Exp. 1 Node2Vec reproduction | 5 seeds | done |
| Exp. 2 uniform smoothing, both embeddings | 5 seeds | done |
| Exp. 3 HCA vs embedding-only / decision-only / uniform | 5 seeds, curves, matched utility | done (negative) |
| Exp. 4 synthetic | 290 realisations, 7 designs | done (simulated) |
| Exp. 5 dropout / corruption / rewiring | 35 runs | done |
| Ablations A–C | in main run | done |
| Robustness 1–5 | 75 runs total | done |
| Failed or blocked | FairGNN reproduction (environment), Pokec-n (optional, not run), clean-environment install test (not done) | — |

## 11–12. Results with uncertainty (test, mean of 5 seeds, 95% block-bootstrap CI)
| method | macro-F1 | ΔEOpp gender | ΔDP gender | Brier |
|---|---|---|---|---|
| Node2Vec + LR | 0.667 | 0.101 [0.057, 0.157] | 0.034 | 0.214 |
| FairWalk + LR | 0.657 | 0.087 [0.057, 0.157] | 0.032 | 0.215 |
| FairWalk + uniform CMN | 0.669 | 0.089 [0.045, 0.157] | 0.032 | 0.211 |
| FairWalk + global scalar | 0.660 | 0.092 [0.053, 0.160] | 0.035 | 0.214 |
| FairWalk + HCA | 0.660 | 0.092 [0.041, 0.161] | 0.038 | 0.212 |
| Node2Vec + group thresholds | 0.667 | 0.013 [0.020, 0.148]* | 0.098 | 0.214 |

\* percentile interval of an absolute gap; pushed away from zero, so the estimate can lie below it.

Confirmatory (ΔEOpp gender, Holm over 4): H1 0.002 [−0.038, 0.022]; H2 0.001 [−0.028, 0.039]; H3 0.004 [−0.020, 0.026]; H4 0.000 [−0.037, 0.023]; all p_Holm = 1.0. Exploratory: smoothing improves Node2Vec macro-F1 by 0.011 [0.003, 0.022] and Brier by −0.004 [−0.008, −0.001].

## 13. Main finding
Smoothing reintroduces group disparities when neighbourhoods carry group base rates (simulation: ΔDP +0.066 at homophily 0.9 vs −0.009 at 0.5; +0.014 without a base-rate gap). On this Pokec-z subgraph they do not for recorded gender, and exact smoothing leaves ΔEOpp unchanged while slightly improving utility and calibration. Group-aware attenuation adds nothing over weaker smoothing.

## 14. Main limitations
One BFS subgraph; no gender homophily; 15 bootstrap blocks; unstable η; forest keeps 37% of edges; FairWalk re-implemented; FairGNN not reproduced; linear SGC only; robustness uncertainty is seed SD only.

## 15. Tests
`python -m pytest tests`: 86 passed. `scripts/smoke_test.py`: passed. Determinism: re-running seed 11 reproduced every metric and prediction exactly (max abs difference 0).

## 16. Reproduce the headline
```bash
python scripts/acquire_data.py --config configs/data.yaml
python scripts/prepare_data.py --config configs/data.yaml
python scripts/run_experiments.py --config configs/experiments.yaml
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/make_results.py --config configs/evaluation.yaml && python scripts/make_macros.py
```

## 17. Compilation
Report: 3 pages, compiled twice with pdflatex, no errors or overfull boxes, inspected. Manuscript: 10 pages (main text to p. 9), compiled, inspected, two overfull boxes under 18 pt. Slides: 14 frames (11 + 3 backup) from the course template, compiled, inspected.

## 18. Remaining tasks requiring student input
1. Evaluation date on the title slide (names, registration numbers and repository URL are filled in; there is no team ID).
2. Create the remote repository; then the repository URL and submission commit/tag.
3. Individual statements and the real contribution split in `docs/contributions.md`.
4. Confirm three flagged references; record the FairGNN commit hash.
5. Test installation from a clean clone on your own machine.
6. Decide whether the course accepts the negative result framing as is (it is the honest one).
7. Rehearse: both students must be able to explain the factorization, sum-product, the EOpp choice, and why HCA failed.

## Requirement-by-requirement status
`docs/requirements_matrix.md` lists every requirement. Summary by state:
* **Implemented, executed and validated:** data pipeline, all models and baselines except published fair GNN, all four exact inference algorithms, all metrics, Experiments 1–5, Ablations A–C, robustness 1–5, bootstrap + Holm, synthetic generator, tests, figures, tables, manifest.
* **Executed with a documented limitation (Partial):** literature (3 flags), closest-method reproduction (re-implementation), simple GNN (linear), block bootstrap (15 blocks), provenance (no upstream commit), report/manuscript/slides (student placeholders).
* **Not executed:** clean-environment install test; Pokec-n (optional).
* **Blocked by environment:** FairGNN reproduction (torch 1.2 / DGL 0.4.3); GitHub commit lookup.
* **Missing, needs students:** identities, individual statements, repository URL and submission tag.
