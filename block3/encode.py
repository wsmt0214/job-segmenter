"""스키마 values 순서 고정 원-핫 인코딩"""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_attribute_frame(df: pd.DataFrame, attr_defs: list[dict]) -> np.ndarray:
    """속성 컬럼만 있는 DataFrame → 원-핫 행렬 (recruitId 불필요)"""
    if df.empty:
        return np.zeros((0, 0), dtype=np.float64)
    blocks: list[np.ndarray] = []
    for ad in attr_defs:
        name = str(ad["name"])
        allowed = [str(v) for v in (ad.get("values") or [])]
        if not allowed:
            raise ValueError(f"속성 '{name}' 에 values 가 없음")
        idx_map = {v: i for i, v in enumerate(allowed)}
        col = df[name].astype(str)
        mat = np.zeros((len(df), len(allowed)), dtype=np.float64)
        for i, val in enumerate(col):
            j = idx_map.get(val)
            if j is None:
                j = idx_map.get("불명확", len(allowed) - 1)
            mat[i, j] = 1.0
        blocks.append(mat)
    return np.hstack(blocks)


def encode_dataframe(df: pd.DataFrame, attr_defs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """행 순서 유지, 반환 X (n_samples, n_features), ids int64"""
    if df.empty:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    attr_cols = [str(ad["name"]) for ad in attr_defs]
    ids = df["recruitId"].to_numpy(dtype=np.int64)
    X = encode_attribute_frame(df[attr_cols], attr_defs)
    return X, ids


def encode_attributes_dict(
    attributes: dict, attr_defs: list[dict]
) -> np.ndarray:
    """단일 행 벡터 (1, n_features)"""
    row = {}
    for ad in attr_defs:
        name = str(ad["name"])
        allowed = [str(v) for v in (ad.get("values") or [])]
        v = attributes.get(name)
        if v is None:
            v = "불명확"
        else:
            v = str(v).strip()
        if v not in allowed:
            v = "불명확" if "불명확" in allowed else allowed[-1]
        row[name] = v
    df = pd.DataFrame([row])
    X = encode_attribute_frame(df, attr_defs)
    return X
