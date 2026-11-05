# Assumptions register

Associational claims only. Nothing in this project identifies a causal effect of gender, region or graph structure on the label in the real population; the synthetic study establishes mechanisms *within its own simulator*.

| # | Assumption | Justification | Checkable? / check | Sensitivity test | Consequence if violated |
|---|---|---|---|---|---|
| A1 | The FairGNN files are the benchmark release used in the literature | Downloaded from the authors' repository; checksums recorded | Partly: hashes recorded; upstream commit not readable from the build host | — | Results would describe a different release |
| A2 | Y = 1[raw code ≥ 1] is the benchmark target | Reproduced from `train_fairGNN.py` (`labels[labels>1]=1`) | Yes: `tests/test_data.py` | — | Different target, results not comparable to FairGNN |
| A3 | Recorded gender 1 corresponds to SNAP's "1 - man" | SNAP readme; no recoding documented by FairGNN | No (record-level link to SNAP not possible offline) | Results are reported by code, not by name | Group names would be swapped; gaps unchanged |
| A4 | The BFS core subgraph is a reasonable, fixed audit population | Connected, ~2,000 nodes, label-value blind | Yes: selection summary vs labelled population | Not re-sampled (spec requires one fixed subgraph) | Findings may not transfer to other subgraphs (likely, since homophily varies) |
| A5 | Transductive use of topology/attributes is not label leakage | Labels of val/test never used | Yes: `tests/test_leakage.py` | — | Optimistic estimates |
| A6 | Unary probabilities from LR are usable as node potentials | Product-of-experts view; λ re-fitted on held-out training folds absorbs over/under-confidence | Partly: log loss/Brier/ECE reported | λ grid includes 0 | Double counting of graph information (embeddings already encode neighbourhoods) could make smoothing redundant |
| A7 | A single forest is an adequate tractable projection | Exact inference requirement | Partly: forest keeps 37% of edges; LBP on full graph compared | Four forest constructions; loopy BP reference | Smoothing effect understated (fewer edges) or distorted (biased edge selection) |
| A8 | Attractive, homogeneous potentials (λ_ij ≥ 0, one λ) | Label homophily 0.586 > 0.501 random | Yes: fitted λ on training folds | λ grid, global scalar ablation | Heterophilous label structure would need repulsive potentials |
| A9 | Positivity of the distribution | Clipping eps = 1e-6, λ finite | Yes: tests | eps documented | Hammersley–Clifford equivalence fails; zero-mass evidence errors are raised, not hidden |
| A10 | Louvain blocks are approximately independent sampling units | Communities have few inter-block edges | Yes: inter-block edge share reported in `bootstrap_design.json` | — | Intervals too narrow |
| A11 | Seeds capture training variability | 5 refits of split + embeddings | — | — | Not population variability; do not read as sampling error |
| A12 | Synthetic generator isolates homophily/degree mechanisms | Parameters known by construction; fair-by-construction unary | Yes: realised homophily recorded | Several factors varied one at a time | Synthetic effects are not real-world facts |
| A13 | 0.02 macro-F1 is an acceptable utility loss | Pre-specified judgement, not derived from data | — | Matched-utility operating point reported | Different trade-off preference changes η selection |
| A14 | Validation set (≈500 nodes) is large enough to select η / thresholds | Standard practice | Partly: selected η varies across seeds (see results) | Curves over full η grid reported | Selection noise; unstable mitigation |

## Ethical assumptions and limits (required list)
1. The working-field label has no defensible employment or recruitment use; it is an undocumented integer code of a free-text profile field. This project is an offline representation audit only.
2. The 2012 Pokec data reflect historical, platform-specific social and demographic patterns.
3. Recorded gender is a binary profile field and does not represent gender identity.
4. Region has a different normative status from gender (it can be a legitimate context variable and is also strongly homophilous here); its gaps are reported as context, not as a protected-class verdict.
5. Fairness measured on one 2,000-node subgraph does not establish fairness of the whole network.
6. Graph structure can expose sensitive information: region is predictable from embeddings with AUROC ≈ 0.99 in the core graph (5-seed mean, Node2Vec and FairWalk alike); gender is not (≈ 0.50–0.51). Publishing embeddings could leak region.
7. Edge attenuation by group (HCA) is itself differential treatment: it changes how much a person's neighbours influence their prediction depending on group composition. The global scalar is the attribute-free alternative and is reported side by side.
8. DP and EOpp can conflict (group thresholds lower ΔEOpp but raise ΔDP here).
9. Intersectional cells are 409–620 nodes in the core and roughly 100–160 nodes per test split; positive/negative denominators per cell are reported; any cell under 30 is flagged.
10. Synthetic experiments do not establish real-world causal mechanisms.
