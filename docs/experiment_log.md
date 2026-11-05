# Experiment log

Chronological record of what was run, what failed and what was changed. Times are UTC on 2026-11-05; runtimes come from `results/manifest.csv` and `results/logs/`. Hardware: Linux x86_64 container, 4 CPU cores, 16 GB RAM, no GPU; CPython 3.13.18; packages in `requirements-lock.txt`.

## 1. Specification audit and literature (Stage 1–2)
* Inputs inspected: `AFGML_Capstone_Problems.pdf`, `AFGML_Report_Template.tex`, `AFGML_Presentation_Template.tex`, `AFGML_Repository_Guide.md`. Report-length discrepancy resolved as documented in `docs/requirements_matrix.md`.
* Literature: 12 papers checked against their landing pages; 3 entries keep a "to confirm" flag (`docs/related_work.md`).
* Blocker: `github.com`, `api.github.com` and the SNAP site could not be fetched from the build environment; `raw.githubusercontent.com` could. Consequence: the FairGNN commit hash is not recorded (students must add it).

## 2. Data (Stage 3)
* `acquire_data.py` downloaded `region_job.csv` and `region_job_relationship.txt` from the FairGNN repository; SHA-256 recorded in `data/raw/PROVENANCE.json` and `configs/data.yaml`.
* `prepare_data.py`: labelled largest component (7,659 nodes) → BFS ball of 2,000 nodes from root 513797 (sampling seed 20261105). Saved to `data/processed/pokec_z_core.npz` (git-ignored).
* `audit_dataset.py`: schema, target cross-walk, group tables, homophily, forest projections (`results/metrics/dataset_audit.json`).
* Finding that changed the analysis plan: gender edge homophily 0.449 vs 0.500 under random mixing (no gender homophily), region 0.963 vs 0.514.

## 3. Development run and protocol freeze
* Protocol, seeds, grids, selection rules and the confirmatory family H1–H4 were written to `configs/` and `docs/protocol.md` before any test metric was computed.
* A development run of seed 11 printed validation metrics and a test-set sensitive-attribute diagnostic. After it, an **exploratory region track** (`fwr_*`) was added; the confirmatory family was not changed. Dated in `docs/proposal.md`.

## 4. Engineering issues found and fixed
| issue | effect | fix |
|---|---|---|
| gensim uses Python's salted `hash` for vector initialisation | embeddings not reproducible across processes | `hashfxn=stable_hash` (md5-based), `workers=1`; reproducibility test in `tests/test_embeddings.py` |
| leftover legacy loop in `pipeline.py` after a refactor | duplicate method evaluation | removed; full suite re-run |
| `evaluate.py` KeyError `dEOdds_r` | evaluation crashed | metric added to the evaluated list |
| matplotlib `vert=` deprecation in box plots | warning flood | `orientation="horizontal"` |
| NumPy divide-by-zero warnings in log-space messages with clipped potentials | warnings only, values correct | local `np.errstate` in `sum_product.py` and `pairwise_model.py` |
| one sentence in `docs/assumptions.md` (A13) claimed a data-derived justification that did not exist | unsupported claim | rewritten as "pre-specified judgement" |
| report/manuscript drafts contained claims written before results existed (region base-rate gap, "forest kept little label structure") | factual errors | checked against tables and corrected; see §8 |

## 5. Inference verification
`python scripts/verify_inference.py --config configs/model.yaml` — 200 random fixtures (≤ 12 nodes, half loopy, with evidence): enumeration, forest sum-product, variable elimination (min-fill and random order) and Shafer–Shenoy clique tree agree; max absolute marginal difference ≈ 1.6e-15 (tolerance 1e-8). Runtime 1.8 s.

## 6. Main experiment (Experiments 1–3, Ablations A–C)
`python scripts/run_experiments.py --config configs/experiments.yaml` — seeds 11, 22, 33, 44, 55, all status `ok`, 204–222 s per seed, 423 s wall-clock with 4 workers.
`python scripts/evaluate.py --config configs/evaluation.yaml` — 1,000 Louvain-block bootstrap replicates (15 blocks, inter-block edge share 0.264), 24.9 s for the bootstrap.
Observations:
* Fitted λ = 0.25 in every seed (lowest positive grid value).
* Selected FairWalk η per seed: 0.8, 0, 0.3, 0.5, 1.0; global scale: 0, 1, 0, 0, 0.9 — unstable.
* Loopy BP converged in all 15 runs (37–44 iterations).
* All four confirmatory effects within ±0.004, Holm p = 1.0. Negative result recorded as such.

## 7. Synthetic study (Experiment 4, SIMULATED)
* `run_synthetic.py --config configs/synthetic.yaml --n-jobs 1`: 230 realisations (homophily × degree grid 5 × 3, label homophily, training-label fraction, no-base-rate-gap control; 10 realisations per cell) in 2,320 s, plus the Node2Vec/FairWalk embedding check (3 realisations × 2 homophily levels, 58 s). Run concurrently with the robustness sweep, so wall-clock times are inflated.
* **Added after the main grid:** group-size (share 0.2/0.35/0.5) and label-prevalence (0.2/0.4, 0.4/0.6, 0.6/0.8) designs at homophily 0.8, because the specification lists both and the first grid did not vary them. `--extra`, 60 realisations, 162 s on 4 workers. These rows are merged into `synthetic_realisations.csv`; their own manifest row is `...[designs=group_share,prevalence]`. The `--extra` run first overwrote the main grid's manifest row (same artifact key); the row was restored by hand and the script changed to use a distinct key.
* Bug found and fixed in `make_macros.py`: training-fraction macro names collapsed to one name (digits stripped), so only the last value survived; replaced by word names.
* Outcome: ΔDP change rises with demographic homophily (−0.009 at 0.5 → 0.066 at 0.9, degree ratio 1), is small without a base-rate gap (0.014), is not amplified by degree imbalance, and HCA never beat the global scalar (HCA − global ΔEOpp −0.006 to 0.020).

## 8. Robustness (Experiment 5 and the five robustness tests)
* `run_robustness.py --config configs/robustness.yaml --n-jobs 3`: 15 conditions × 5 seeds = 75 runs, all status `ok`, 90–104 s per run, 2,413 s wall-clock. No failed configurations.
* Loopy BP is run only in the edge-perturbation conditions (it does not depend on the forest, dimensions or label budget in the way being tested); it converged in all 70 such runs (35 per embedding).
* Outcome: FairWalk smoothing effect on ΔEOpp −0.027 to 0.019 across conditions (no consistent sign); utility gain survives edge perturbations, disappears with non-Jaccard forests; λ fitted as 0 in some perturbed seeds and up to 1.5 with 10% of training labels.
* Added `table_robustness_inference_stability.csv` (forest edges kept, components, isolated nodes, λ, η spread, loopy-BP convergence) because Experiment 5 asks for inference stability and forest loss explicitly.

## 9. Figures, text checks and corrections
* `make_results.py` regenerated all 16 figures; fig14's legend covered bars and was moved below the axes; its caption wrongly implied loopy BP was run for every forest condition and was corrected.
* Every number in the report, manuscript and slides is a macro from `make_macros.py`. Statements written before the results existed were checked against the tables. Corrected: the region base-rate statement (actual 0.50 vs 0.45), "the forest kept little label structure" (forest label homophily is 0.629, higher than the full core's 0.586; the issue is mean forest degree 2.0), "simulations show it requires demographic homophily" (it requires homophily *and* a base-rate gap), a robustness claim that the smoothing effect stayed within ±0.03 (Node2Vec reaches 0.037), and "the global scalar captured most of the benefit" (it matched HCA in every cell).
* Known statistical artefact recorded in the papers and model card: percentile intervals of absolute gaps are pushed away from zero, so `n2v_thr`'s ΔEOpp point estimate (0.013) lies below its interval [0.020, 0.148].

## 10. Final checks
* `python -m pytest tests`: 86 passed.
* Notebook `notebooks/exploratory_analysis.ipynb` executed with `jupyter nbconvert --execute` (reads saved outputs only; nbformat/nbconvert/ipykernel were installed for this and are not in `requirements.txt`).
