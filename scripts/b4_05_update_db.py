#!/usr/bin/env python3
"""Block 4 Step 5 — phase4·clustering 기준 recruit.segment_id 일괄 반영"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from block4.recruit_segment_db import sync_segments_to_db


def main() -> None:
    p = argparse.ArgumentParser(description="recruit 테이블 segment_id / segment_version 배치 반영")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 UPDATE 없이 계획·샘플만 리포트에 기록",
    )
    p.add_argument(
        "--ensure-columns",
        action="store_true",
        help="segment_id·segment_version 컬럼 없으면 ALTER 로 추가",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="executemany 배치 크기 (기본 1000)",
    )
    args = p.parse_args()

    out_path = Path(config.BLOCK3_EVAL_DIR) / "db_update_report.json"
    report = sync_segments_to_db(
        dry_run=args.dry_run,
        ensure_columns=args.ensure_columns,
        batch_size=max(1, args.batch_size),
        eval_report_path=out_path,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
