# Data card — Pokec-z core subgraph

## Source and version
* **Release:** FairGNN repository by E. Dai and S. Wang, `dataset/pokec/region_job.csv` and `region_job_relationship.txt`, fetched from `https://raw.githubusercontent.com/EnyanDai/FairGNN/main/dataset/pokec/` on 2026-11-05 (branch `main`; the commit hash could not be read because github.com/api.github.com were blocked from the build environment — record it manually from the repository page).
* **SHA-256:** `region_job.csv` 62f299e6…bda3cd (38,608,131 bytes); `region_job_relationship.txt` a3494594…acf8 (11,351,174 bytes). Full hashes in `configs/data.yaml` and `data/raw/PROVENANCE.json`.
* **Upstream:** sampled by FairGNN from SNAP soc-Pokec (Takac & Zabovsky, 2012), crawled 25–27 May 2012.
* **Licence:** neither the SNAP page nor the FairGNN README states a data licence. We therefore do **not** redistribute the files; `scripts/acquire_data.py` downloads them.

## Released schema (audited, `results/metrics/dataset_audit.json`)
67,796 nodes × 279 columns (277 int64, 2 float64), no missing values, no duplicate `user_id`. 882,765 edge rows → 617,958 unique undirected edges (264,807 reverse/duplicate rows, 0 self-loops), 0 isolated nodes. Columns used: `user_id`, `gender` (0/1), `region` (0/1), `I_am_working_in_field` (−1, 0–4); the remaining 275 columns (e.g. `public`, `completion_percentage`, `AGE`, language and hobby indicators) are the feature matrix X.

## Label construction (reproduced from FairGNN code)
`load_pokec` treats `label ≥ 0` as labelled; `train_fairGNN.py` applies `labels[labels>1]=1`.

| raw code | count (full release) | Y |
|---|---|---|
| −1 | 57,534 | unlabelled, excluded |
| 0 | 4,764 | 0 |
| 1 | 1,964 | 1 |
| 2 | 1,266 | 1 |
| 3 | 1,287 | 1 |
| 4 | 981 | 1 |

The SNAP readme describes `I_am_working_in_field` only as a free-text user field; FairGNN does not document what codes 0–4 mean. **Y=1 therefore means only "raw code ≥ 1"**; any occupational interpretation would be invented.

## Protected attributes
* **gender:** recorded 0/1. The SNAP readme says "gender: bool, 1 - man" and does not describe 0; FairGNN documents no recoding, so we *believe* 1 = recorded "man" [Likely, not verified at record level]. We report groups as `gender=0`/`gender=1`. Recorded gender is a profile field, not gender identity; no other categories exist in the release; nothing is inferred.
* **region:** recorded 0/1; FairGNN's own sensitive attribute. Region names are undocumented. Region is a contextual dimension with different normative status from gender.

## Core subgraph (estimand population)
Procedure in `docs/protocol.md`. 2,000 nodes, 5,351 edges, one component, degree mean 5.35 / median 4 / max 38.

| group | n | Y=1 | Y=0 | P(Y=1) | mean degree |
|---|---|---|---|---|---|
| gender 0, region 0 | 550 | 229 | 321 | 0.416 | 4.47 |
| gender 0, region 1 | 421 | 161 | 260 | 0.382 | 6.57 |
| gender 1, region 0 | 620 | 360 | 260 | 0.581 | 4.40 |
| gender 1, region 1 | 409 | 214 | 195 | 0.523 | 6.73 |
| gender 0 | 971 | 390 | 581 | 0.402 | 5.38 |
| gender 1 | 1,029 | 574 | 455 | 0.558 | 5.32 |

Selection effects vs all labelled Pokec-z nodes: P(Y=1) 0.482 vs 0.536; gender=1 share 0.515 vs 0.502; region=1 share 0.415 vs 0.355. BFS sampling over-represents nodes close to the root and higher-degree nodes.

**Graph structure that matters for the question:** edge homophily is 0.449 for gender (0.500 expected under random mixing — mild heterophily), 0.963 for region (0.514 expected), 0.586 for the label (0.501 expected). Degree is balanced by gender but not by region (6.6 vs 4.4).

## Sampling unit and dependence
Nodes are linked; the evaluation unit for uncertainty is the Louvain graph block (see protocol). Results describe this subgraph, not Pokec, Slovakia, or social networks generally.

## Exclusions
Unlabelled nodes (code −1) and labelled nodes outside the BFS ball are excluded. No group was removed.

## Access, privacy
Raw files stay in `data/raw/` (git-ignored). No individual rows, ids or neighbourhoods are published in results; tables contain counts and rates only.
