#!/usr/bin/env python3
"""Block 4 Step 1 — 타입별 원문 학습 CSV 구축 (MySQL + clustering 레이블)"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import ensure_block3_dirs

import config
from block4.recruit_text_repository import fetch_title_content_by_ids
from block4.text_dataset import (
    assemble_frame,
    load_ids_and_labels,
    print_type_stats,
    save_type_csv,
)


def main() -> None:
    ensure_block3_dirs()
    out_dir = Path(config.BLOCK4_TEXT_DIR)

    id_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_ids: list[int] = []
    for rt in BLOCK3_TYPES:
        ids, labels = load_ids_and_labels(rt)
        id_arrays[rt] = (ids, labels)
        all_ids.extend(int(x) for x in ids.tolist())

    unique_sorted = sorted(set(all_ids))
    texts_by_id = fetch_title_content_by_ids(config.DB_CONFIG, unique_sorted)

    print("=== Block 4 Step 1 — 텍스트 학습 데이터셋 ===\n")
    print(f"타입별 행 합계 recruitId: {len(all_ids)}건")
    print(f"DB 조회 고유 recruitId: {len(unique_sorted)}건")

    for rt in BLOCK3_TYPES:
        ids, labels = id_arrays[rt]
        df, missing = assemble_frame(ids, labels, texts_by_id)
        path = save_type_csv(rt, df, out_dir)
        print_type_stats(rt, df, missing)
        print(f"  저장: {path}")


if __name__ == "__main__":
    main()
