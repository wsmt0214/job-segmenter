#!/usr/bin/env python3
"""Block 4 Step 2 — 타입별 TF-IDF + 로지스틱 회귀 베이스라인"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block3.constants import BLOCK3_TYPES

import config
from block4.baseline_text import train_baseline_from_csv, write_baseline_report
from block4.text_dataset import npy_stem


def main() -> None:
    text_dir = Path(config.BLOCK4_TEXT_DIR)
    models_dir = Path(config.BLOCK3_MODELS_DIR)
    eval_dir = Path(config.BLOCK3_EVAL_DIR)

    report: dict = {}
    for rt in BLOCK3_TYPES:
        stem = npy_stem(rt)
        csv_path = text_dir / f"{stem}_dataset.csv"
        model_path = models_dir / f"baseline_{stem}.pkl"
        cm_path = eval_dir / f"baseline_{stem}_confusion_matrix.png"
        report[rt] = train_baseline_from_csv(
            recruit_type=rt,
            dataset_csv=csv_path,
            model_out=model_path,
            confusion_png=cm_path,
        )

    out_json = write_baseline_report(eval_dir, report)
    print(f"\nbaseline_report 저장: {out_json}")


if __name__ == "__main__":
    main()
