"""Shared helpers: config loading, logging, manifest recording."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path.relative_to(ROOT))
    return cfg


def setup_logging(name: str) -> logging.Logger:
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)r}')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"{name}.log") for h in root.handlers):
        fh = logging.FileHandler(ROOT / f"results/logs/{name}.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)
    for noisy in ("gensim", "smart_open"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def data_version() -> str:
    p = ROOT / "data/processed/pokec_z_core.npz"
    return file_hash(p) if p.exists() else "none"


MANIFEST_FIELDS = ["artifact", "command", "config", "seed", "data_version", "git_commit", "status",
                   "runtime_s", "warnings", "timestamp"]


def record(artifact: str | Path, command: str, config: str, seed: Any = "", status: str = "ok",
           runtime_s: Optional[float] = None, warnings: str = "", data_ver: Optional[str] = None) -> None:
    """Append (or replace) one artifact row in results/manifest.csv."""
    mpath = ROOT / "results/manifest.csv"
    rows = []
    if mpath.exists():
        with open(mpath) as fh:
            rows = [r for r in csv.DictReader(fh) if r["artifact"] != str(artifact)]
    rows.append({"artifact": str(artifact), "command": command, "config": config, "seed": seed,
                 "data_version": data_ver if data_ver is not None else data_version(),
                 "git_commit": git_commit(), "status": status,
                 "runtime_s": "" if runtime_s is None else f"{runtime_s:.1f}", "warnings": warnings,
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
