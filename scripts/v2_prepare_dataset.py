"""Phase 1 결과 + 카테고리 + payment 그룹을 통합한 v2.0 입력 데이터셋 생성"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymysql

import config

PHASE1_PATH = Path(config.DATA_DIR) / "phase1_results.jsonl"
CATEGORY_MAP_PATH = Path(config.V2_DATA_DIR) / "category_map.json"
OUT_PATH = Path(config.V2_DATA_DIR) / "phase1_with_category.jsonl"


def fetch_category_map() -> dict[int, dict]:
    """DB에서 recruitId별 카테고리·payment 정보 조회"""
    conn = pymysql.connect(**config.DB_CONFIG)
    cur = conn.cursor()
    cur.execute(config.CATEGORY_JOIN_SQL)
    rows = cur.fetchall()
    conn.close()

    category_map: dict[int, dict] = {}
    for recruit_id, payment, categories_str in rows:
        cats = categories_str.split("|||") if categories_str else []
        category_map[int(recruit_id)] = {
            "payment": int(payment) if payment is not None else 0,
            "payment_group": config.get_payment_group(payment),
            "categories": cats,
        }

    no_cat = sum(1 for v in category_map.values() if not v["categories"])
    group_dist = Counter(v["payment_group"] for v in category_map.values())
    print(f"카테고리 조회: {len(category_map)}건")
    print(f"카테고리 없는 구인글: {no_cat}건")
    print(f"payment 그룹 분포: {dict(group_dist)}")
    return category_map


def merge_phase1_with_category(category_map: dict[int, dict]) -> list[dict]:
    """Phase 1 구인글 키워드에 카테고리·payment 그룹 병합"""
    merged: list[dict] = []
    skip_count = 0

    with PHASE1_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            if r.get("llm_result", {}).get("글_유형") != "구인글":
                continue

            rid = int(r["recruitId"])
            cat_info = category_map.get(rid)
            if cat_info is None:
                skip_count += 1
                continue

            keywords = r.get("llm_result", {}).get("keywords", [])
            categories = cat_info["categories"]
            category_signals = [f"카테고리:{cat}" for cat in categories]

            merged.append(
                {
                    "recruitId": rid,
                    "payment_group": cat_info["payment_group"],
                    "payment": cat_info["payment"],
                    "categories": categories,
                    "keywords": keywords,
                    "all_signals": keywords + category_signals,
                    "llm_domain": r.get("llm_result", {}).get("도메인", ""),
                    "ok": True,
                }
            )

    pg_dist = Counter(r["payment_group"] for r in merged)
    cat_freq = Counter(chain.from_iterable(r["categories"] for r in merged))
    print(f"병합 완료: {len(merged)}건 / category_map 미매칭 제외: {skip_count}건")
    print(f"payment 그룹별 건수: {dict(pg_dist)}")
    print("카테고리 빈도 상위 10:")
    for cat, cnt in cat_freq.most_common(10):
        print(f"  {cat}: {cnt}건")
    return merged


def run() -> None:
    Path(config.V2_DATA_DIR).mkdir(parents=True, exist_ok=True)

    print("=== 카테고리 데이터 조회 ===")
    category_map = fetch_category_map()

    with CATEGORY_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)
    print(f"저장: {CATEGORY_MAP_PATH}")

    print("\n=== Phase 1 + 카테고리 병합 ===")
    merged = merge_phase1_with_category(category_map)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"저장: {OUT_PATH} ({len(merged)}건)")
    print("\nTask 1 완료 — v2_phase2_discovery.py 실행 가능")


if __name__ == "__main__":
    run()
