# Contributions

Student names, roll numbers, commit links and individual statements must be written by the students themselves. The split below is a *proposed* allocation that balances the rubric; replace "Student 1/2" and edit freely. Commit count is not a proxy for contribution.

| Work item | Lead student | Supporting / reviewer student | Artifact / commit | Verification (what the reviewer checked) |
|---|---|---|---|---|
| Literature review, comparison table, gap analysis | Student 1 | Student 2 | `docs/related_work.md` | Reviewer re-opened every cited landing page and confirmed title/authors/venue/DOI |
| Dataset acquisition, schema audit, target reproduction | Student 2 | Student 1 | `src/data/`, `scripts/acquire_data.py`, `scripts/audit_dataset.py`, `docs/data_card.md` | Re-derived the label crosswalk from FairGNN `train_fairGNN.py`; reran checksums |
| Node2Vec / FairWalk embeddings and diagnostics | Student 1 | Student 2 | `src/embeddings/random_walks.py`, `tests/test_embeddings.py` | Checked FairWalk group-uniform step on the star fixture; reproducibility test |
| Conditional Markov network, factors, forest projection | Student 2 | Student 1 | `src/graphical_model/{factors,pairwise_model,forest}.py` | Hand-checked potential indexing and evidence factors |
| Exact inference and validation (sum-product, VE, clique tree, enumeration) | Student 2 | Student 1 | `src/graphical_model/`, `scripts/verify_inference.py`, `tests/test_inference_agreement.py` | Independently traced one message-passing example on paper |
| Fairness and probabilistic metrics | Student 1 | Student 2 | `src/fairness/metrics.py`, `tests/test_fairness_metrics.py` | Hand-computed the toy example in the test |
| Attenuation method (HCA), λ fitting, selection rules | Student 1 | Student 2 | `src/graphical_model/attenuation.py`, `docs/mathematical_formulation.md` | Checked no validation/test label enters λ (`tests/test_leakage.py`) |
| Statistical analysis (block bootstrap, Holm) | Student 2 | Student 1 | `src/statistics/bootstrap.py`, `scripts/evaluate.py` | Reproduced Holm adjustment by hand for H1–H4 |
| Synthetic generator and Experiment 4 | Student 1 | Student 2 | `src/synthetic/`, `scripts/run_synthetic.py` | Confirmed realised homophily matches targets |
| Robustness and Experiment 5 | Student 2 | Student 1 | `src/experiments/perturb.py`, `scripts/run_robustness.py` | Checked degree sequence preserved under rewiring |
| Report, manuscript, slides, demo | Both | Both | `report/`, `manuscript/`, `slides/`, `demo/` | Each student presents the other's half in rehearsal |

## Individual statements (to be written by each student)
* **Student 1:** one technical decision you made and why; one difficulty or failure; one cross-check you performed on Student 2's work.
* **Student 2:** same three items.

## Assistance disclosure
This repository was produced with substantial assistance from an AI coding assistant (Claude, Anthropic), which wrote code, ran experiments and drafted documents under the students' direction. Disclose this according to the course's policy, and make sure both students can explain and defend every part.
