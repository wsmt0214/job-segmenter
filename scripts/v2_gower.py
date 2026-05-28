"""v2.1 null-aware Gower 거리 — 범주형 4축, 불명확=미기재"""
from __future__ import annotations

import numpy as np
import pandas as pd
from schema_v2 import UNCLEAR_VALUE

# 한쪽만 미기재 → 약한 거리·낮은 가중
PARTIAL_PENALTY = 0.35
PARTIAL_WEIGHT = 0.5


def _is_missing(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == UNCLEAR_VALUE


def axis_contribution(a, b) -> tuple[float, float]:
    """축별 (거리, 가중) — 둘 다 미기재면 기여 0"""
    a_miss = _is_missing(a)
    b_miss = _is_missing(b)
    if a_miss and b_miss:
        return 0.0, 0.0
    if a_miss or b_miss:
        return PARTIAL_PENALTY, PARTIAL_WEIGHT
    if str(a) == str(b):
        return 0.0, 1.0
    return 1.0, 1.0


def gower_distance_matrix(
    df: pd.DataFrame,
    cols: list[str],
    dim_weights: list[float] | None = None,
) -> np.ndarray:
    """
    행 간 Gower 거리 행렬 (n×n)
    dim_weights: 축별 가중 (기본 균등)
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, 0))
    if n == 1:
        return np.zeros((1, 1))

    if dim_weights is None:
        dim_weights = [1.0] * len(cols)

    dist_sum = np.zeros((n, n), dtype=np.float64)
    weight_sum = np.zeros((n, n), dtype=np.float64)

    for col, w in zip(cols, dim_weights):
        vals = df[col].values
        miss = np.array([_is_missing(v) for v in vals])
        str_vals = np.array([str(v) if not _is_missing(v) else "" for v in vals])

        match = str_vals[:, None] == str_vals[None, :]
        mi = miss[:, None]
        mj = miss[None, :]
        both_miss = mi & mj
        one_miss = mi ^ mj
        both_ok = ~mi & ~mj

        d = np.zeros((n, n), dtype=np.float64)
        ww = np.zeros((n, n), dtype=np.float64)

        d[both_ok & match] = 0.0
        d[both_ok & ~match] = 1.0
        ww[both_ok] = 1.0

        d[one_miss] = PARTIAL_PENALTY
        ww[one_miss] = PARTIAL_WEIGHT

        dist_sum += w * d
        weight_sum += w * ww

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight_sum > 0, dist_sum / weight_sum, 0.0)
    np.fill_diagonal(out, 0.0)
    return out
