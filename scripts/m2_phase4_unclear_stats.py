"""phase4_results.jsonl + schema_definition.json 기준 속성별 불명확 비율 집계 (B2-05)"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from schema_attrs import attr_names_for_type

SCHEMA_PATH = Path(config.DATA_DIR) / "schema_definition.json"
PHASE4_PATH = Path(config.DATA_DIR) / "phase4_results.jsonl"
UNCLEAR = "불명확"


def run() -> None:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)

    # attr -> (unclear, total), by_type[attr][type] -> (unclear, total)
    tot_u: dict[str, int] = defaultdict(int)
    tot_n: dict[str, int] = defaultdict(int)
    by_u: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    n_rows = n_ok = 0
    with PHASE4_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            o = json.loads(line)
            if not o.get("ok"):
                continue
            n_ok += 1
            rt = str(o.get("recruitType", ""))
            attrs = o.get("attributes") or {}
            if not isinstance(attrs, dict):
                continue
            expected = attr_names_for_type(schema, rt)
            for name in expected:
                v = attrs.get(name)
                tot_n[name] += 1
                by_n[name][rt] += 1
                if v == UNCLEAR:
                    tot_u[name] += 1
                    by_u[name][rt] += 1

    print(f"읽은 줄: {n_rows}, ok=true: {n_ok}")
    print(f"\n=== 속성별 불명확 비율 (전체, ok 행만) ===\n")
    for name in sorted(tot_n.keys(), key=lambda x: (-(tot_u[x] / tot_n[x] if tot_n[x] else 0), x)):
        u, n = tot_u[name], tot_n[name]
        pct = 100.0 * u / n if n else 0.0
        flag = " ⚠ 30%+" if pct >= 30 else ""
        print(f"  {name}: {pct:.1f}%  ({u:,} / {n:,}){flag}")

    print("\n=== 타입별 (해당 속성이 정의된 행만 분모) ===\n")
    for name in sorted(tot_n.keys()):
        parts = []
        for rt in sorted(by_n[name].keys()):
            u = by_u[name][rt]
            n = by_n[name][rt]
            p = 100.0 * u / n if n else 0.0
            parts.append(f"{rt} {p:.1f}%")
        print(f"  {name}:  {' / '.join(parts)}")

    print("\n완료 기준(가이드): 속성별 불명확 **30% 미만** 권장 → ⚠ 항목은 스키마·프롬프트 재검토")


if __name__ == "__main__":
    if not SCHEMA_PATH.is_file():
        raise SystemExit(f"없음: {SCHEMA_PATH}")
    if not PHASE4_PATH.is_file():
        raise SystemExit(f"없음: {PHASE4_PATH}")
    run()
