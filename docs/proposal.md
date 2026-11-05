# Research proposal — ICT 5143 AFGML Capstone, Problem 4

**Provisional title:** Does Graph Smoothing Reintroduce Bias into Fair Node Embeddings?
**Team:** `[Student 1 name, roll number]`, `[Student 2 name, roll number]` (to be supplied by the students)
**Domain:** offline audit of demographic disparities in social-network node classification. Not a recruitment, employment or individual decision system.

## Decision variables
* `Y = 1`: the benchmark's positive working-field class — raw `I_am_working_in_field` code ≥ 1 in the FairGNN Pokec-z release (`Y = 0` ⇔ raw code 0; code −1 = unlabelled, excluded). The real-world meaning of the integer codes is undocumented.
* `D = 1`: the pipeline predicts the positive benchmark class for a node. It confers no benefit or harm by itself; it is an audit decision.

## Research question
When does probabilistic graph smoothing undo the fairness gains of debiased embeddings, and can constrained edge potentials reduce this effect?

Secondary: How do demographic homophily and degree imbalance affect the result? Does lower protected-attribute predictability imply fairer decisions? What is lost by using a tractable forest?

## Hypothesis (tested, not assumed)
Under demographic homophily and a base-rate difference between groups, uniform attractive pairwise potentials in a conditional Markov network move posterior marginals toward neighbours' training labels, which carry group base rates, and thereby widen the group gap in true-positive rates (ΔEOpp) relative to the unary (embedding-only) predictor. Attenuating same-group edges reduces this widening at comparable validation utility.

## Falsifiable contribution statement
> On a fixed graph, split, forest and unary model, homophily-constrained attenuation of same-group pairwise potentials (one parameter η, chosen on validation) reduces the gender ΔEOpp of the smoothed FairWalk pipeline relative to (i) uniform smoothing and (ii) a global-scalar attenuation with the same tuning budget, at validation macro-F1 within 0.02 of uniform smoothing; the claim is refuted if the Holm-adjusted paired block-bootstrap tests H3/H4 are not significant or their intervals exclude reductions, and it is reported as context-limited if the reduction appears only in synthetic graphs with demographic homophily.

The second clause (comparison with the global scalar) is what separates the contribution from "just smooth less". The controlled synthetic study (Experiment 4) is the mechanistic contribution and stands even if HCA fails on Pokec-z.

## Primary criterion and utility (pre-specified)
* Primary fairness criterion: **equality of opportunity** for gender, ΔEOpp = max_g TPR_g − min_g TPR_g. Rationale: the benchmark label's base rates differ by recorded gender (core subgraph: 0.40 vs 0.56), and the label is a recorded profile field rather than an outcome with a documented measurement bias; DP would require equal prediction rates despite real differences in the recorded label, whereas EOpp asks whether the pipeline's errors on true positives are distributed unequally — exactly what neighbourhood smoothing can distort. DP is reported as the required secondary diagnostic. They are not interchangeable.
* Primary utility: macro-F1 at the validation-selected threshold. Acceptable utility loss: 0.02 absolute validation macro-F1.

## Dated amendments (honest record)
* **2026-11-05, before any test metric was computed:** protocol, seeds, grids, selection rules, primary metrics and the confirmatory family H1–H4 written to `configs/` and `docs/protocol.md`.
* **2026-11-05, after a development run of seed 11 that printed validation metrics and a test-set sensitive-attribute diagnostic:** the core subgraph shows *no* gender homophily (edge homophily 0.449 vs 0.500 under random mixing) and chance-level gender predictability from embeddings, but strong region homophily (0.963). We therefore **added an exploratory region track** (FairWalk on region, region-attenuation). This is a retrospective addition and is labelled exploratory everywhere; the confirmatory family was not changed.
* Pokec-n replication: not run (optional in the specification).
