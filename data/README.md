# Data

**Source.** Pokec-z as released in the FairGNN repository (E. Dai and S. Wang): `dataset/pokec/region_job.csv` (nodes) and `dataset/pokec/region_job_relationship.txt` (edges), downloaded from `https://raw.githubusercontent.com/EnyanDai/FairGNN/main/dataset/pokec/`. Upstream: SNAP soc-Pokec (L. Takac, M. Zabovsky, 2012; crawled 25–27 May 2012).

**Access date.** 2026-11-05 (UTC). Repository branch `main`; the commit hash could not be read from the build host (github.com and its API were blocked) — record it from the repository page when you clone.

**Licence / redistribution.** Neither the FairGNN README nor the SNAP page states a data licence. The files are therefore *not* committed (`.gitignore`); obtain them yourself:

```bash
python scripts/acquire_data.py --config configs/data.yaml
```

The script downloads to `data/raw/<file>.part`, resumes interrupted downloads with HTTP Range requests (4 attempts), verifies SHA-256 against `configs/data.yaml`, never silently replaces an existing file (a mismatching file stops the run; `--force` replaces it), and appends provenance to `data/raw/PROVENANCE.json`.

| file | bytes | SHA-256 |
|---|---|---|
| region_job.csv | 38,608,131 | 62f299e688cc07c29fa36f0d4c0f1c15137bd0fdced5931eec15e724e4bda3cd |
| region_job_relationship.txt | 11,351,174 | a34945943c13a8790439c2230bfb00e6573bb82fe781ccc30a295276b825acf8 |

**Preprocessing.** `python scripts/prepare_data.py --config configs/data.yaml` builds `data/processed/pokec_z_core.npz`: labelled nodes only (raw working-field code ≥ 0), largest connected component of the labelled induced graph, BFS ball of 2,000 nodes from a root drawn with seed 20261105. Target: Y = 1[code ≥ 1] (FairGNN). Features: all released columns except `user_id`, `gender`, `region`, `I_am_working_in_field` (275 columns). No imputation (no missing values exist). See `docs/data_card.md` for counts, exclusions and caveats.

**Without the data.** `python scripts/smoke_test.py` and `python scripts/run_synthetic.py` run entirely on simulated graphs.

**Privacy.** Do not commit `data/raw` or `data/processed`; results contain only aggregate counts and rates.
