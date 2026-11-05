"""Reproduce the FairGNN working-field label transformation.

Source: EnyanDai/FairGNN, src/utils.py::load_pokec and src/train_fairGNN.py.
  * predict_attr = "I_am_working_in_field"
  * load_pokec keeps nodes with label >= 0 as labelled (label_idx = where(labels >= 0));
    value -1 therefore means "no label" and those nodes are never in train/val/test.
  * train_fairGNN.py then applies ``labels[labels > 1] = 1``.
Hence: raw 0 -> Y = 0; raw 1, 2, 3, 4 -> Y = 1; raw -1 -> unlabelled.
The semantic meaning of the integer codes is not documented by FairGNN or by
the SNAP soc-Pokec readme (which describes the field as free text); see
docs/data_card.md. Y=1 is therefore "the benchmark's positive working-field
class (raw code >= 1)", nothing more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "I_am_working_in_field"
UNLABELLED_CODE = -1


def transform_target(raw_values: pd.Series | np.ndarray) -> np.ndarray:
    """Return Y in {0, 1} for labelled nodes and -1 for unlabelled nodes."""
    v = np.asarray(raw_values).astype(int)
    if np.any(v < UNLABELLED_CODE):
        raise ValueError("unexpected working-field code below -1")
    y = np.where(v < 0, -1, np.where(v > 1, 1, v))
    return y.astype(int)


def target_crosswalk(raw_values) -> pd.DataFrame:
    v = pd.Series(np.asarray(raw_values).astype(int), name="raw_code")
    y = transform_target(v)
    return (pd.DataFrame({"raw_code": v, "Y": y}).value_counts().rename("count").reset_index()
            .sort_values("raw_code").reset_index(drop=True))
