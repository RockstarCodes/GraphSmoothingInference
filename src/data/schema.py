"""Schema audit of the released Pokec-z files. Reports what is there; assumes no encodings."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .loading import RawPokec

REQUIRED_COLUMNS = ("user_id", "gender", "region", "I_am_working_in_field")


def validate_schema(raw: RawPokec) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.nodes.columns]
    if missing:
        raise ValueError(f"released node table lacks required columns {missing}")
    if raw.nodes["user_id"].duplicated().any():
        raise ValueError("duplicate user_id values in node table")
    ids = set(raw.nodes["user_id"].tolist())
    unknown = ~np.isin(raw.edges_raw, list(ids)).all(axis=1)
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} edges reference user_ids absent from the node table")


def audit(raw: RawPokec) -> Dict[str, object]:
    df, e = raw.nodes, raw.edges_raw
    und = np.sort(e, axis=1)
    loops = int((und[:, 0] == und[:, 1]).sum())
    uniq = np.unique(und[und[:, 0] != und[:, 1]], axis=0)
    deg = pd.Series(uniq.ravel()).value_counts().reindex(df["user_id"], fill_value=0)
    out: Dict[str, object] = {
        "n_nodes": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_feature_columns_excl_id_gender_region_target": int(df.shape[1] - 4),
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_user_ids": int(df["user_id"].duplicated().sum()),
        "edge_rows_raw": int(len(e)),
        "self_loops": loops,
        "undirected_unique_edges": int(len(uniq)),
        "duplicate_or_reverse_edge_rows": int(len(e) - loops - len(uniq)),
        "isolated_nodes": int((deg == 0).sum()),
        "degree_mean": float(deg.mean()), "degree_median": float(deg.median()), "degree_max": int(deg.max()),
    }
    for c in ("gender", "region", "I_am_working_in_field", "public"):
        if c in df:
            out[f"values_{c}"] = {str(k): int(v) for k, v in df[c].value_counts(dropna=False).sort_index().items()}
    return out
