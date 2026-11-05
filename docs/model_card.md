# Model card — smoothed fair-embedding node classifiers (Problem 4)

**Status: offline research audit.** These models exist to measure how probabilistic graph smoothing changes group fairness of node classification on a benchmark. They are not, and must not be used as, a recruitment, employment, credit, moderation or any other individual decision-making system.

## Models covered
| key | pipeline |
|---|---|
| `feat_lr` | L2 logistic regression on the 275 profile features |
| `sgc` | Simplified graph convolution (2-hop feature propagation) + logistic regression — the "simple GNN" baseline |
| `n2v_lr`, `fw_lr` | Node2Vec / FairWalk (gender) embedding (d=64) + logistic regression |
| `n2v_thr` | `n2v_lr` with group-specific thresholds chosen on validation (decision-only mitigation) |
| `*_nopair` | conditional Markov network with all pairwise factors removed (Ablation A) |
| `*_uniform` | CMN with one shared edge strength λ on the max-Jaccard spanning forest, exact sum-product |
| `*_global` | uniform λ scaled by a single scalar c chosen on validation (Ablation B) |
| `*_hca` | homophily-constrained attenuation: λ_ij = λ(1 − η·1[A_i = A_j]), η chosen on validation under a utility floor |
| `*_lbp_uniform` | loopy BP on the full (non-forest) core graph — exploratory only |
| `fwr_*` | FairWalk on **region** with region-based attenuation — exploratory track added after the gender audit (see `docs/proposal.md`, amendment) |

Mathematical definitions: `docs/mathematical_formulation.md`. Hyper-parameters: `configs/*.yaml`.

## Intended use
Reproducing and extending the research question "does graph smoothing reintroduce bias into fair node embeddings?" on the Pokec-z core subgraph and on synthetic graphs. Out of scope: any prediction about real people, deployment, or claims about Pokec users beyond the 2,000-node core.

## Data
Pokec-z as released by FairGNN; 2,000-node BFS core, 50/25/25 transductive split stratified by Y × gender (FairGNN proportions), 5 seeds. See `docs/data_card.md`. The target is the benchmark's binarised `I_am_working_in_field` (raw code ≥ 1 ⇒ Y=1); the meaning of the codes is undocumented, so **Y has no defensible real-world occupational interpretation**.

## Protected attributes
Recorded `gender` (0/1, primary) and `region` (0/1, secondary). The models use the recorded attribute only inside FairWalk walks, HCA edge weights and group-threshold baselines. Nothing is inferred about any person's gender.

## Evaluation
Primary fairness criterion: gender equality-of-opportunity gap ΔEOpp = |TPR(g=1) − TPR(g=0)| on test nodes. Also ΔDP, ΔEOdds, selection-rate ratio, group TPR/FPR/selection rates, worst-group accuracy, gender×region intersectional gaps, macro-F1, balanced accuracy, AUROC, log loss, Brier score, ECE and per-group reliability. Uncertainty: 1,000 graph-block bootstrap replicates (Louvain blocks) with identical draws across methods, averaged over 5 training seeds; Holm correction over the four confirmatory comparisons. Full numbers: `results/tables/`.

## Headline results (5 seeds, test set, block-bootstrap 95% CIs)
* Embedding audit: neither Node2Vec nor FairWalk embeddings encode recorded gender in a recoverable way (gender AUROC ≈ 0.51 and ≈ 0.50); both encode region almost perfectly (≈ 0.99). FairWalk on region does not reduce region predictability (≈ 0.99), because 96% of edges are within-region and FairWalk only acts in mixed neighbourhoods.
* None of the four confirmatory hypotheses (uniform smoothing changes ΔEOpp for FairWalk or Node2Vec; HCA vs uniform; HCA vs global scalar) is supported: every paired effect is within ±0.004 with CIs spanning roughly ±0.03, Holm-adjusted p = 1.0.
* Smoothing gives a small, consistent utility gain for Node2Vec (macro-F1 +0.011, CI excludes 0).
* SIMULATED graphs: smoothing widens ΔDP only with demographic homophily and a base-rate gap together (0.066 at homophily 0.9 vs −0.009 at 0.5; 0.014 with equal base rates); HCA never beat a global scalar.
* Robustness (15 conditions × 5 seeds): no consistent sign of the smoothing effect on ΔEOpp; the utility gain depends on the max-Jaccard forest.
* Group-threshold adjustment lowers ΔEOpp (≈ 0.013 point estimate) but raises ΔDP (≈ 0.10): the two criteria conflict on this data.

## Known failure modes and limitations
* **Gender premise not met on this core.** Gender edge homophily is 0.449 (random mixing 0.500), so there is no gender homophily for smoothing to amplify; the real-data test of the core hypothesis is therefore weak. The synthetic experiments are where the mechanism is probed.
* **Attenuation parameters are unstable**: the selected η and global scale vary widely across seeds (see `results/tables/table_selected_parameters.csv`), consistent with a flat validation objective.
* **Bootstrap intervals for absolute gaps** are percentile intervals of |gap|; when the gap is near zero the bootstrap distribution is pushed away from zero and the point estimate can fall *below* the interval (e.g. `n2v_thr` ΔEOpp). Paired *differences* of gaps (used for H1–H4) do not have this artefact to the same degree but remain conservative near zero.
* Forest projection keeps 37.4% of core edges; information carried by the other 62.6% is only checked through the exploratory loopy-BP comparison.
* 2,000 nodes and about 500 test nodes per seed; the smallest gender×region test cell has 33 positives in one seed (`results/tables/table_group_rates_pooled.csv`), so intersectional estimates are noisy.
* Only 15 Louvain blocks are available as resampling units and 26% of edges cross blocks, so the block bootstrap is coarse and the independence assumption is approximate.
* FairWalk is our re-implementation from the paper's description, not the authors' code. No published fair GNN (FairGNN, NIFTY) was reproduced; see `docs/requirements_matrix.md`.
* Calibration differs by gender: the models under-predict for gender=1 and over-predict for gender=0 (reliability tables), consistent with the base-rate gap 0.558 vs 0.402.

## Ethical considerations
See `docs/assumptions.md` (numbered ethical items 1–10). In short: historical biases in the source network; recorded gender is not gender identity; region has a different normative status; subgraph fairness does not generalise; graph structure leaks region almost perfectly; attenuation itself changes whose neighbours count; DP and EOpp conflict here; small intersectional cells; synthetic results are not causal evidence about Pokec. No identifiable records are released.

## Provenance
Code: this repository (commit recorded in `results/manifest.csv`). Data hashes: `data/raw/PROVENANCE.json`. Authors: [Student 1 name, roll number], [Student 2 name, roll number] — to be filled in by the students.
