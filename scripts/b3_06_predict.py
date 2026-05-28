#!/usr/bin/env python3
"""Step 6 — predict_cluster 스모크 및 사용 예시"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import iter_phase4_ok, load_schema
from block3.predict import predict_cluster
from block3.schema_slice import block3_attr_defs


def main() -> None:
    schema = load_schema()

    example_model = {
        "보상_유형": "무페이",
        "활용_목적": "포트폴리오",
        "촬영_환경": "실내·스튜디오",
        "작업_지속성": "일회성",
        "긴급도": "불명확",
        "경력_조건": "신입환영",
        "신체_조건": "조건명시",
    }
    r = predict_cluster("model", example_model, schema=schema)
    print("=== 지시문 예시 (model) ===")
    print(json.dumps(r, ensure_ascii=False, indent=2))

    first_by_type: dict = {}
    for obj in iter_phase4_ok():
        rt = str(obj.get("recruitType", ""))
        if rt in BLOCK3_TYPES and rt not in first_by_type:
            attrs = dict(obj.get("attributes") or {})
            if rt == "beauty":
                attrs = {k: v for k, v in attrs.items() if k != "기록_촬영"}
            first_by_type[rt] = attrs
        if len(first_by_type) == len(BLOCK3_TYPES):
            break

    print("\n=== phase4 첫 샘플 타입별 스모크 ===")
    for rt, attrs in first_by_type.items():
        defs = block3_attr_defs(schema, rt)
        allowed_names = {str(d["name"]) for d in defs}
        attrs_f = {k: v for k, v in attrs.items() if k in allowed_names}
        try:
            out = predict_cluster(rt, attrs_f, schema=schema)
            print(f"[{rt}] cluster_id={out['cluster_id']} conf={out['confidence']:.4f}")
        except Exception as e:
            print(f"[{rt}] 오류: {e}")


if __name__ == "__main__":
    main()
