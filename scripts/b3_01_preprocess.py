#!/usr/bin/env python3
"""Step 1 — 로딩·타입 분리·통계 (산출 디렉터리 생성)"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import build_typed_dataframes, ensure_block3_dirs, load_schema


def main() -> None:
    ensure_block3_dirs()
    schema = load_schema()
    dfs = build_typed_dataframes(schema)

    print("=== Block 3 Step 1 — 타입별 요약 ===\n")
    for rt in BLOCK3_TYPES:
        df = dfs[rt]
        print(f"[{rt}] 건수: {len(df)}")
        if df.empty:
            print("  (데이터 없음)\n")
            continue
        cols = [c for c in df.columns if c != "recruitId"]
        for col in cols:
            vc = df[col].value_counts(dropna=False)
            top = vc.head(8)
            parts = ", ".join(f"{str(k)}:{int(v)}" for k, v in top.items())
            print(f"  {col}: {parts}")
        print()


if __name__ == "__main__":
    main()
