"""Protected attributes exactly as recorded in the release. Nothing is inferred.

gender : released as 0/1 (float column). The SNAP soc-Pokec readme documents
         "gender: bool, 1 - man" and does not describe 0. FairGNN does not
         document any recoding. We keep the codes and label them "gender=0"
         and "gender=1"; see docs/data_card.md for the caveat.
region : released as 0/1. FairGNN uses it as *its* sensitive attribute; the
         region names behind the codes are not documented in the release.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def recorded_attribute(df: pd.DataFrame, column: str) -> np.ndarray:
    v = df[column]
    if v.isna().any():
        raise ValueError(f"{column} has missing values; refusing to impute a protected attribute")
    vals = np.asarray(v)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"{column} has codes outside {{0,1}}: {sorted(set(vals.tolist()))}")
    return vals.astype(int)


def intersectional_groups(gender: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Group label strings like 'g0_r1' (gender code, region code)."""
    return np.array([f"g{g}_r{r}" for g, r in zip(gender, region)])


def group_table(groups: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"group": groups, "y": y})
    t = d.groupby("group")["y"].agg(n="size", positives="sum")
    t["negatives"] = t["n"] - t["positives"]
    t["positive_rate"] = t["positives"] / t["n"]
    return t.reset_index()
