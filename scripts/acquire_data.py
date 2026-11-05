"""Acquire Pokec-z from the official FairGNN repository with checksum verification.

Usage: python scripts/acquire_data.py --config configs/data.yaml [--force]
"""
import argparse
import time

import _bootstrap  # noqa: F401
from src.data.acquisition import acquire
from src.utils import ROOT, load_config, record, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--force", action="store_true", help="replace an existing file whose checksum differs")
    args = ap.parse_args()
    log = setup_logging("acquire_data")
    cfg = load_config(args.config)
    d = cfg["dataset"]
    t = time.time()
    recs = acquire(d["files"], d["base_url"], ROOT / d["raw_dir"], ROOT / d["provenance_path"], force=args.force)
    for name, r in recs.items():
        log.info("%s: %s sha256=%s verified=%s", name, r["status"], r["sha256"], r["checksum_verified"])
    record(d["provenance_path"], "python scripts/acquire_data.py --config " + args.config, cfg["_config_path"],
           runtime_s=time.time() - t, data_ver="raw")


if __name__ == "__main__":
    main()
