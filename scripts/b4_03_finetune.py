#!/usr/bin/env python3
"""Block 4 Step 3 — KLUE-RoBERTa 파인튜닝·베이스라인 대비 채택 및 리포트"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block3.constants import BLOCK3_TYPES

import config
from block4.roberta_finetune import finetune_roberta_for_type, write_json
from block4.text_dataset import npy_stem


def main() -> None:
    eval_dir = Path(config.BLOCK3_EVAL_DIR)
    baseline_path = eval_dir / "baseline_report.json"
    if not baseline_path.is_file():
        sys.exit(
            "baseline_report.json 없음 — 먼저 python scripts/b4_02_baseline.py 실행"
        )

    with baseline_path.open(encoding="utf-8") as f:
        baseline_report = json.load(f)

    text_dir = Path(config.BLOCK4_TEXT_DIR)
    models_dir = Path(config.BLOCK3_MODELS_DIR)

    roberta_report: dict = {}
    comparison: dict = {}

    for rt in BLOCK3_TYPES:
        stem = npy_stem(rt)
        csv_path = text_dir / f"{stem}_dataset.csv"
        br = baseline_report.get(rt, {})
        baseline_macro_f1 = None
        if not br.get("skipped"):
            mf = br.get("macro_f1")
            baseline_macro_f1 = float(mf) if mf is not None else None

        roberta_report[rt] = finetune_roberta_for_type(
            recruit_type=rt,
            dataset_csv=csv_path,
            models_dir=models_dir,
            baseline_macro_f1=baseline_macro_f1,
        )

        rr = roberta_report[rt]
        comparison[rt] = {
            "baseline": {
                "accuracy": br.get("accuracy"),
                "macro_f1": br.get("macro_f1"),
                "weighted_f1": br.get("weighted_f1"),
                "skipped": bool(br.get("skipped")),
            },
            "roberta": {
                "accuracy": rr.get("accuracy"),
                "macro_f1": rr.get("macro_f1"),
                "weighted_f1": rr.get("weighted_f1"),
                "skipped": bool(rr.get("skipped")),
            },
            "adopted": rr.get("adopted"),
            "macro_f1_gain": rr.get("macro_f1_gain"),
        }

    write_json(eval_dir / "roberta_report.json", roberta_report)
    write_json(eval_dir / "model_comparison.json", comparison)

    print(f"\nroberta_report 저장: {eval_dir / 'roberta_report.json'}")
    print(f"model_comparison 저장: {eval_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
