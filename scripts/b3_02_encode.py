#!/usr/bin/env python3
"""Step 2 — 타입별 원-핫 행렬 및 recruitId 배열 저장"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block3.constants import BLOCK3_TYPES
from block3.encode import encode_dataframe
from block3.io_phase4 import build_typed_dataframes, ensure_block3_dirs, load_schema
from block3.schema_slice import block3_attr_defs

import config


def main() -> None:
    ensure_block3_dirs()
    schema = load_schema()
    dfs = build_typed_dataframes(schema)
    out_dir = Path(config.BLOCK3_ENCODED_DIR)

    for rt in BLOCK3_TYPES:
        df = dfs[rt]
        fname = "model" if rt == "model" else rt
        if df.empty:
            print(f"[{rt}] 건수 0 — 빈 배열 저장")
            np.save(out_dir / f"{fname}_X.npy", np.zeros((0, 0), dtype=np.float64))
            np.save(out_dir / f"{fname}_ids.npy", np.zeros((0,), dtype=np.int64))
            continue
        defs = block3_attr_defs(schema, rt)
        X, ids = encode_dataframe(df, defs)
        np.save(out_dir / f"{fname}_X.npy", X)
        np.save(out_dir / f"{fname}_ids.npy", ids)
        print(f"[{rt}] X.shape={X.shape}, ids={len(ids)} → {out_dir}/{fname}_*.npy")


if __name__ == "__main__":
    main()
