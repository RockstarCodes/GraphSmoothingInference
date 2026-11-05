# Methodology (pipeline)

1. **Acquire and audit** Pokec-z (`scripts/acquire_data.py`, `scripts/audit_dataset.py`).
2. **Fix the core subgraph** once (`scripts/prepare_data.py`): label-value-blind BFS sample, 2,000 nodes.
3. **Per seed (11, 22, 33, 44, 55)** (`scripts/run_experiments.py` → `src/experiments/pipeline.py`):
   1. Stratified (Y × gender) 50/25/25 transductive label split.
   2. Unsupervised embeddings: Node2Vec, FairWalk(gender), FairWalk(region, exploratory) — `src/embeddings/random_walks.py`.
   3. Unary classifiers: L2 logistic regression on embeddings; feature-only LR; SGC (2-hop) — `src/baselines/classifiers.py`. C chosen on validation log loss.
   4. Forest projection (max-Jaccard spanning forest) — `src/graphical_model/forest.py`.
   5. λ fitted on 5 training folds with cross-fitted unaries — `src/graphical_model/attenuation.py`.
   6. Exact sum-product with training evidence for: no pairwise (ablation A), uniform λ, global scalar c·λ (ablation B), HCA λ(1−η·1[A_i=A_j]) (proposed), HCA on region (exploratory); loopy BP on the full graph (exploratory reference).
   7. Thresholds on validation; decision-only mitigation by per-gender thresholds; matched-utility operating points.
   8. Metrics on test nodes (`src/fairness/metrics.py`), curves over η and c.
4. **Statistics** (`scripts/evaluate.py`): graph-block bootstrap, confirmatory Holm family, exploratory paired effects, group denominators.
5. **Synthetic study** (`scripts/run_synthetic.py`): fair-by-construction unaries on generated graphs with controlled homophily, degree imbalance, label homophily, label availability, base-rate gap; plus Node2Vec/FairWalk on synthetic graphs.
6. **Robustness / Experiment 5** (`scripts/run_robustness.py`): edge dropout, corruption, degree-preserving rewiring, forest constructions, embedding dimension, training-label availability.
7. **Figures, tables, LaTeX macros** (`scripts/make_results.py`, `scripts/make_macros.py`).

Mathematical details: [`mathematical_formulation.md`](mathematical_formulation.md). Protocol and selection rules: [`protocol.md`](protocol.md).
