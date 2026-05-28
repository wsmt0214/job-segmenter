#!/usr/bin/env python3
"""급종별 segment_id → 특징(profiles) + 글 목록(recruitId·text) JSON 3종 출력"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import ensure_block3_dirs
from block4.segment_text_catalog import build_catalog_for_type, write_catalog_json
from block4.text_dataset import npy_stem


def main() -> None:
    ensure_block3_dirs()
    ap = argparse.ArgumentParser(description="분야별 세그먼트 카탈로그 JSON 생성")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(config.DATA_DIR) / "segment_catalog",
        help="출력 디렉터리 (기본: data/segment_catalog)",
    )
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()

    text_dir = Path(config.BLOCK4_TEXT_DIR)
    cl_dir = Path(config.BLOCK3_CLUSTERING_DIR)

    for rt in BLOCK3_TYPES:
        stem = npy_stem(rt)
        csv_p = text_dir / f"{stem}_dataset.csv"
        prof_p = cl_dir / f"{stem}_profiles.json"
        if not csv_p.is_file():
            print(f"[skip] 데이터셋 없음: {csv_p}")
            continue
        if not prof_p.is_file():
            print(f"[skip] 프로필 없음: {prof_p}")
            continue

        catalog = build_catalog_for_type(rt, csv_p, prof_p)
        out_path = out_dir / f"{stem}_segments.json"
        write_catalog_json(catalog, out_path)
        n_posts = sum(c["post_count"] for c in catalog["clusters"])
        print(f"[{rt}] clusters={len(catalog['clusters'])} posts={n_posts} → {out_path}")


if __name__ == "__main__":
    main()
