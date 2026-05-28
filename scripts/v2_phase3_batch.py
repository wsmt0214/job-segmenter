"""v2.0 Phase 3 — 전체 배치 추출 (Task 3-B)"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import config
from schema_v2 import UNCLEAR_VALUE, load_schema, phase3_attr_names
from v2_phase3_core import extract_attributes

INPUT_PATH = Path(config.V2_DATA_DIR) / "phase1_with_category.jsonl"
RAW_CSV = Path(config.DATA_DIR) / "raw_recruits.csv"
OUT_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"


def load_posts() -> list[dict]:
    posts: list[dict] = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))
    return posts


def load_raw_map() -> dict[int, pd.Series]:
    if not RAW_CSV.is_file():
        raise SystemExit(f"없음: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, dtype={"recruitId": int})
    return {int(r["recruitId"]): r for _, r in df.iterrows()}


def load_done_ids(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["recruitId"]))
    return done


def run(sleep_sec: float = 0.3) -> None:
    schema = load_schema()
    attr_names = phase3_attr_names(schema)
    posts = load_posts()
    raw_map = load_raw_map()
    print(f"전체 처리 대상: {len(posts)}건")

    done_ids = load_done_ids(OUT_PATH)
    if done_ids:
        print(f"이미 처리됨: {len(done_ids)}건, 이어서 처리")

    remaining = [p for p in posts if int(p["recruitId"]) not in done_ids]
    print(f"남은 처리 대상: {len(remaining)}건")

    if not remaining:
        print("처리할 건 없음")
        return

    fail_count = 0
    unclear: dict[str, int] = {name: 0 for name in attr_names}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a", encoding="utf-8") as out_f:
        for post in tqdm(remaining, desc="Phase3 batch"):
            rid = int(post["recruitId"])
            raw = raw_map.get(rid)
            if raw is None:
                continue

            title = str(raw["title"]) if pd.notna(raw["title"]) else ""
            body = str(raw["content"]) if pd.notna(raw["content"]) else ""
            categories = post.get("categories") or []

            attrs = extract_attributes(schema, categories, title, body, retries=2)
            ok = attrs is not None
            if not ok:
                fail_count += 1
                attrs = {name: UNCLEAR_VALUE for name in attr_names}

            for name in attr_names:
                if attrs.get(name) == UNCLEAR_VALUE:
                    unclear[name] += 1

            out_f.write(json.dumps({
                "recruitId": rid,
                "payment_group": post.get("payment_group"),
                "payment": post.get("payment"),
                "categories": categories,
                "attributes": attrs,
                "ok": ok,
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            time.sleep(sleep_sec)

    batch_n = len(remaining)
    print(f"\n=== Phase 3 전체 배치 (이번 실행분) ===")
    print(f"처리: {batch_n}건 / JSON 파싱 실패: {fail_count}건")
    print("\n속성별 불명확 비율 (이번 실행분):")
    for name, cnt in unclear.items():
        pct = cnt / batch_n * 100 if batch_n else 0.0
        flag = " ⚠️" if pct > 40 else ""
        print(f"  {name}: {pct:.1f}%{flag}")
    print(f"\n결과 파일: {OUT_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="v2 Phase3 전체 배치")
    p.add_argument("--sleep", type=float, default=0.3, help="건당 대기(초)")
    args = p.parse_args()
    run(sleep_sec=args.sleep)


if __name__ == "__main__":
    main()
