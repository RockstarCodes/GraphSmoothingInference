"""Load the raw FairGNN Pokec-z files into a node table and an undirected edge array."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RawPokec:
    nodes: pd.DataFrame          # one row per user; columns as released
    edges_raw: np.ndarray        # (m, 2) user_id pairs as released (possibly both directions)


def load_raw(raw_dir: Path, node_file: str, edge_file: str) -> RawPokec:
    raw_dir = Path(raw_dir)
    for f in (node_file, edge_file):
        if not (raw_dir / f).exists():
            raise FileNotFoundError(f"{raw_dir / f} not found. Run: python scripts/acquire_data.py --config configs/data.yaml")
    nodes = pd.read_csv(raw_dir / node_file)
    edges = np.loadtxt(raw_dir / edge_file, dtype=np.int64)
    return RawPokec(nodes=nodes, edges_raw=edges.reshape(-1, 2))
