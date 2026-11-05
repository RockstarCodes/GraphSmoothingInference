# Demonstration

## Live (about 1 minute, no dataset needed)
```bash
python scripts/smoke_test.py          # SIMULATED 300-node graph through the whole pipeline
python scripts/verify_inference.py    # 200 exact-inference agreement fixtures (~10 s)
```
Expected: `smoke test passed` and `all_within_tolerance: True`. Outputs: `results/smoke/`, `results/metrics/inference_verification*.{csv,json}`. Smoke numbers are not research results.

## One real result from a frozen config (requires the dataset; ~4 min per seed on 1 CPU core)
```bash
python scripts/acquire_data.py --config configs/data.yaml
python scripts/prepare_data.py --config configs/data.yaml
python scripts/run_experiments.py --config configs/experiments.yaml --seeds 11 --n-jobs 1
```
Expected artifacts: `results/metrics/main/seed11_{metrics.csv,info.json,predictions.npz,curves.csv}`. The seed-11 rows must match the committed ones (deterministic seeds; tested on the same library versions).

## Talking points for a live query
Pick one test node and its Markov blanket in the forest: `results/metrics/inference_verification_summary.json → markov_blanket_demo` tabulates P(Y_i=1 | k positive neighbours) for the highest-degree forest node with λ = 0.5 and p_i = 0.5 — each positive neighbour adds λ to the log-odds.

## Offline fallback (saved trace)
`demo/saved_trace.txt` contains the console output of the commands above as executed for this submission.
