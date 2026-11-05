"""Prepared core dataset: the fixed subgraph with arrays used by every experiment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class Prepared:
    user_ids: np.ndarray
    edges: np.ndarray            # (m,2) local indices
    y: np.ndarray                # binary benchmark target
    gender: np.ndarray
    region: np.ndarray
    features: np.ndarray         # remaining released columns (excl. id, gender, region, target)
    feature_names: List[str]
    meta: Dict

    @property
    def n(self) -> int:
        return len(self.y)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, user_ids=self.user_ids, edges=self.edges, y=self.y, gender=self.gender,
                            region=self.region, features=self.features,
                            feature_names=np.array(self.feature_names), meta=json.dumps(self.meta))

    @staticmethod
    def load(path: Path) -> "Prepared":
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/prepare_data.py --config configs/data.yaml")
        z = np.load(path, allow_pickle=False)
        return Prepared(z["user_ids"], z["edges"], z["y"], z["gender"], z["region"], z["features"],
                        [str(s) for s in z["feature_names"]], json.loads(str(z["meta"])))
