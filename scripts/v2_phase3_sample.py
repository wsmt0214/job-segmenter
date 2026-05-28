"""v2.0 Phase 3 — 테스트셋 100건 검증 (Task 3-A)"""
from __future__ import annotations

import argparse
import json
import random
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
OUT_PATH = Path(config.V2_DATA_DIR) / "phase3_sample.jsonl"

# 차원별 불명확 비율 임계값 (초과 시 경고)
UNCLEAR_THRESHOLDS: dict[str, float] = {
    "촬영 장소": 0.70,
    "촬영 목적": 0.50,
    "촬영 주제": 0.85,
    "시술 종류": 0.72,
    "경력 조건": 0.50,
    "작업 지속성": 0.50,
}

PERM_EVIDENCE_WORDS = [
    "펌", "빈티지펌", "히피펌", "레이어드펌", "perm", "파마",
    "볼륨매직", "셋팅펌", "다운펌", "젤리펌", "매직",
]


def load_posts() -> list[dict]:
    posts: list[dict] = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))
    return posts


def load_posts_by_id() -> dict[int, dict]:
    return {int(p["recruitId"]): p for p in load_posts()}


def load_raw_map() -> dict[int, pd.Series]:
    if not RAW_CSV.is_file():
        raise SystemExit(f"없음: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, dtype={"recruitId": int})
    return {int(r["recruitId"]): r for _, r in df.iterrows()}


def load_sample_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(json.loads(line)["recruitId"]))
    return ids


def load_sample_file(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unclear_stats(rows: list[dict], attr_names: list[str]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {name: 0.0 for name in attr_names}
    out: dict[str, float] = {}
    for name in attr_names:
        cnt = sum(1 for r in rows if r.get("attributes", {}).get(name) == UNCLEAR_VALUE)
        out[name] = cnt / total * 100
    return out


def perm_hallucination_stats(
    rows: list[dict],
    raw_map: dict[int, pd.Series],
) -> tuple[int, int]:
    """펌 추출 건수, 원문(제목+본문)에 펌 근거 없는 건수"""
    perm_count = 0
    no_evidence = 0
    for row in rows:
        if row.get("attributes", {}).get("시술 종류") != "펌":
            continue
        perm_count += 1
        rid = int(row["recruitId"])
        raw = raw_map.get(rid)
        if raw is None:
            no_evidence += 1
            continue
        title = str(raw["title"]) if pd.notna(raw["title"]) else ""
        body = str(raw["content"]) if pd.notna(raw["content"]) else ""
        full_text = title + " " + body
        if not any(w in full_text for w in PERM_EVIDENCE_WORDS):
            no_evidence += 1
    return perm_count, no_evidence


def print_quality_report(
    results: list[dict],
    attr_names: list[str],
    fail_count: int,
    out_path: Path,
) -> None:
    total = len(results)
    success_rate = (total - fail_count) / total * 100 if total else 0.0
    unclear = unclear_stats(results, attr_names)

    print("\n=== 테스트셋 품질 리포트 ===")
    print(f"파싱 성공률: {success_rate:.1f}% (기준: 95%)")
    print("\n속성별 불명확 비율 (차원별 임계값):")
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        pct = unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        flag = "⚠️  재검토 필요" if pct >= threshold_pct else "✓"
        print(f"  {name}: {pct:.1f}% {flag} (기준: {threshold_pct:.0f}%)")

    print(f"\n결과 저장: {out_path}")


def print_v7_topic_comparison_report(
    v3_rows: list[dict],
    v6_rows: list[dict],
    v7_rows: list[dict],
    attr_names: list[str],
    raw_map: dict[int, pd.Series],
    fail_count: int,
) -> None:
    v3_unclear = unclear_stats(v3_rows, attr_names)
    v6_unclear = unclear_stats(v6_rows, attr_names)
    v7_unclear = unclear_stats(v7_rows, attr_names)
    v6_perm, v6_no_ev = perm_hallucination_stats(v6_rows, raw_map)
    v7_perm, v7_no_ev = perm_hallucination_stats(v7_rows, raw_map)
    total = len(v7_rows)
    success_rate = (total - fail_count) / total * 100 if total else 0.0

    print("\n=== Phase 3 v6 + 촬영주제 v3 (v7) 비교 ===")
    print(f"{'차원':<12} {'v3':>5} {'v6':>5} {'v7(이번)':>8} {'기준':>6} {'상태':>6}")
    print("-" * 58)
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        c = v3_unclear.get(name, 0.0)
        d = v6_unclear.get(name, 0.0)
        e = v7_unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        status = "✓" if e < threshold_pct else "⚠️"
        print(f"{name:<12} {c:>4.0f}% {d:>4.0f}% {e:>7.0f}% {threshold_pct:>5.0f}% {status:>6}")

    print()
    print(f"펌 추출 건수:       v6 {v6_perm}건 → v7 {v7_perm}건")
    print(f"펌 환각(근거 없음): v6 {v6_no_ev}건 → v7 {v7_no_ev}건")
    print(f"JSON 파싱 성공률: {success_rate:.1f}%")


def print_v6_hybrid_comparison_report(
    v4_rows: list[dict],
    v5_rows: list[dict],
    v6_rows: list[dict],
    attr_names: list[str],
    raw_map: dict[int, pd.Series],
    fail_count: int,
) -> None:
    v4_unclear = unclear_stats(v4_rows, attr_names)
    v5_unclear = unclear_stats(v5_rows, attr_names)
    v6_unclear = unclear_stats(v6_rows, attr_names)
    v4_perm, v4_no_ev = perm_hallucination_stats(v4_rows, raw_map)
    v5_perm, v5_no_ev = perm_hallucination_stats(v5_rows, raw_map)
    v6_perm, v6_no_ev = perm_hallucination_stats(v6_rows, raw_map)
    total = len(v6_rows)
    success_rate = (total - fail_count) / total * 100 if total else 0.0

    print("\n=== Phase 3 수정 v4/v5/v6 하이브리드 비교 ===")
    print(f"{'차원':<12} {'v4':>5} {'v5':>5} {'v6(이번)':>8} {'기준':>6} {'상태':>6}")
    print("-" * 58)
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        d = v4_unclear.get(name, 0.0)
        e = v5_unclear.get(name, 0.0)
        f = v6_unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        status = "✓" if f < threshold_pct else "⚠️"
        print(f"{name:<12} {d:>4.0f}% {e:>4.0f}% {f:>7.0f}% {threshold_pct:>5.0f}% {status:>6}")

    print()
    print(f"펌 추출 건수:       v4 {v4_perm}건 → v5 {v5_perm}건 → v6 {v6_perm}건")
    print(f"펌 환각(근거 없음): v4 {v4_no_ev}건  → v5 {v5_no_ev}건 → v6 {v6_no_ev}건")
    print(f"JSON 파싱 성공률: {success_rate:.1f}%")


def print_v5_hybrid_comparison_report(
    v3_rows: list[dict],
    v4_rows: list[dict],
    v5_rows: list[dict],
    attr_names: list[str],
    raw_map: dict[int, pd.Series],
    fail_count: int,
) -> None:
    v3_unclear = unclear_stats(v3_rows, attr_names)
    v4_unclear = unclear_stats(v4_rows, attr_names)
    v5_unclear = unclear_stats(v5_rows, attr_names)
    v3_perm, v3_no_ev = perm_hallucination_stats(v3_rows, raw_map)
    v4_perm, v4_no_ev = perm_hallucination_stats(v4_rows, raw_map)
    v5_perm, v5_no_ev = perm_hallucination_stats(v5_rows, raw_map)
    total = len(v5_rows)
    success_rate = (total - fail_count) / total * 100 if total else 0.0

    print("\n=== Phase 3 수정 v3/v4 하이브리드 (v5) 비교 ===")
    print(f"{'차원':<12} {'v3':>5} {'v4':>5} {'v5(이번)':>8} {'기준':>6} {'상태':>6}")
    print("-" * 58)
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        c = v3_unclear.get(name, 0.0)
        d = v4_unclear.get(name, 0.0)
        e = v5_unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        status = "✓" if e < threshold_pct else "⚠️"
        print(f"{name:<12} {c:>4.0f}% {d:>4.0f}% {e:>7.0f}% {threshold_pct:>5.0f}% {status:>6}")

    print()
    print(f"펌 추출 건수:       v3 {v3_perm}건 → v4 {v4_perm}건 → v5 {v5_perm}건")
    print(f"펌 환각(근거 없음): v3 {v3_no_ev}건  → v4 {v4_no_ev}건 → v5 {v5_no_ev}건")
    print(f"JSON 파싱 성공률: {success_rate:.1f}%")


def print_quad_comparison_report(
    v1_rows: list[dict],
    v2_rows: list[dict],
    v3_rows: list[dict],
    v4_rows: list[dict],
    attr_names: list[str],
    raw_map: dict[int, pd.Series],
    fail_count: int,
) -> None:
    v1_unclear = unclear_stats(v1_rows, attr_names)
    v2_unclear = unclear_stats(v2_rows, attr_names)
    v3_unclear = unclear_stats(v3_rows, attr_names)
    v4_unclear = unclear_stats(v4_rows, attr_names)
    v3_perm, v3_no_ev = perm_hallucination_stats(v3_rows, raw_map)
    v4_perm, v4_no_ev = perm_hallucination_stats(v4_rows, raw_map)
    total = len(v4_rows)
    success_rate = (total - fail_count) / total * 100 if total else 0.0

    print("\n=== Phase 3 수정 v3 → v4 비교 (샘플 100건) ===")
    print(f"{'차원':<12} {'v1':>5} {'v2':>5} {'v3':>5} {'v4(이번)':>8} {'기준':>6} {'상태':>6}")
    print("-" * 62)
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        a = v1_unclear.get(name, 0.0)
        b = v2_unclear.get(name, 0.0)
        c = v3_unclear.get(name, 0.0)
        d = v4_unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        status = "✓" if d < threshold_pct else "⚠️"
        print(f"{name:<12} {a:>4.0f}% {b:>4.0f}% {c:>4.0f}% {d:>7.0f}% {threshold_pct:>5.0f}% {status:>6}")

    print()
    print(f"펌 추출 건수:       v3 {v3_perm}건 → v4 {v4_perm}건")
    print(f"펌 환각(근거 없음): v3 {v3_no_ev}건  → v4 {v4_no_ev}건")
    print(f"JSON 파싱 성공률: {success_rate:.1f}%")


def print_triple_comparison_report(
    v1_rows: list[dict],
    v2_rows: list[dict],
    v3_rows: list[dict],
    attr_names: list[str],
    raw_map: dict[int, pd.Series],
    fail_count: int,
) -> None:
    v1_unclear = unclear_stats(v1_rows, attr_names)
    v2_unclear = unclear_stats(v2_rows, attr_names)
    v3_unclear = unclear_stats(v3_rows, attr_names)
    v1_perm, v1_no_ev = perm_hallucination_stats(v1_rows, raw_map)
    v2_perm, v2_no_ev = perm_hallucination_stats(v2_rows, raw_map)
    v3_perm, v3_no_ev = perm_hallucination_stats(v3_rows, raw_map)
    total = len(v3_rows)
    success_rate = (total - fail_count) / total * 100 if total else 0.0

    print("\n=== Phase 3 수정 v2 → v3 비교 (샘플 100건) ===")
    print(f"{'차원':<12} {'v1(수정 전)':>12} {'v2':>6} {'v3(이번)':>8} {'기준':>6} {'상태':>6}")
    print("-" * 58)
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        a = v1_unclear.get(name, 0.0)
        b = v2_unclear.get(name, 0.0)
        c = v3_unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        status = "✓" if c < threshold_pct else "⚠️"
        print(f"{name:<12} {a:>11.0f}% {b:>5.0f}% {c:>7.0f}% {threshold_pct:>5.0f}% {status:>6}")

    print()
    print(f"펌 추출 건수:          v1 {v1_perm}건 → v2 {v2_perm}건 → v3 {v3_perm}건")
    print(f"펌 환각(근거 없음):    v1 {v1_no_ev}건 → v2 {v2_no_ev}건  → v3 {v3_no_ev}건")
    print(f"JSON 파싱 성공률: {success_rate:.1f}%")


def print_comparison_report(
    before_rows: list[dict],
    after_rows: list[dict],
    attr_names: list[str],
    raw_map: dict[int, pd.Series],
) -> None:
    before_unclear = unclear_stats(before_rows, attr_names)
    after_unclear = unclear_stats(after_rows, attr_names)
    before_perm, before_no_ev = perm_hallucination_stats(before_rows, raw_map)
    after_perm, after_no_ev = perm_hallucination_stats(after_rows, raw_map)

    print("\n=== Phase 3 프롬프트 수정 전/후 비교 (샘플 100건) ===")
    print(f"{'차원':<12} {'수정 전 불명확':>12} {'수정 후 불명확':>12} {'기준':>8} {'상태':>6}")
    print("-" * 56)
    for name in attr_names:
        if name not in UNCLEAR_THRESHOLDS:
            continue
        b = before_unclear.get(name, 0.0)
        a = after_unclear.get(name, 0.0)
        threshold_pct = UNCLEAR_THRESHOLDS[name] * 100
        status = "✓" if a < threshold_pct else "⚠️"
        print(f"{name:<12} {b:>11.0f}% {a:>11.0f}% {threshold_pct:>7.0f}% {status:>6}")

    print()
    print(f"시술 종류 펌 추출 건수: 수정 전 {before_perm}건 → 수정 후 {after_perm}건")
    print(f"펌 중 원문(제목+본문)에 근거 없는 건수: 수정 전 {before_no_ev}건 → 수정 후 {after_no_ev}건")


def run(
    n: int = 100,
    seed: int = 42,
    sleep_sec: float = 0.3,
    out_path: Path | None = None,
    ids_from: Path | None = None,
    compare_with: Path | None = None,
    compare_v1: Path | None = None,
    compare_v2: Path | None = None,
    compare_v3: Path | None = None,
    compare_v4: Path | None = None,
    compare_v5: Path | None = None,
    compare_v6: Path | None = None,
) -> list[dict]:
    schema = load_schema()
    attr_names = phase3_attr_names(schema)
    raw_map = load_raw_map()
    output = out_path or OUT_PATH

    if ids_from and ids_from.is_file():
        sample_ids = load_sample_ids(ids_from)
        posts_by_id = load_posts_by_id()
        sample = [posts_by_id[rid] for rid in sample_ids if rid in posts_by_id]
        print(f"테스트셋: {len(sample)}건 (기존 {ids_from.name} recruitId 재사용)")
    else:
        posts = load_posts()
        rng = random.Random(seed)
        sample = rng.sample(posts, min(n, len(posts)))
        print(f"테스트셋: {len(sample)}건 샘플링 (seed={seed})")

    results: list[dict] = []
    fail_count = 0

    for post in tqdm(sample, desc="Phase3 sample"):
        rid = int(post["recruitId"])
        raw = raw_map.get(rid)
        if raw is None:
            raise RuntimeError(f"raw_recruits.csv 에 recruitId={rid} 없음")

        title = str(raw["title"]) if pd.notna(raw["title"]) else ""
        body = str(raw["content"]) if pd.notna(raw["content"]) else ""
        categories = post.get("categories") or []

        attrs = extract_attributes(schema, categories, title, body, retries=2)
        ok = attrs is not None
        if not ok:
            fail_count += 1
            attrs = {name: UNCLEAR_VALUE for name in attr_names}

        results.append({
            "recruitId": rid,
            "payment_group": post.get("payment_group"),
            "categories": categories,
            "title_preview": title[:80],
            "attributes": attrs,
            "ok": ok,
        })
        time.sleep(sleep_sec)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_quality_report(results, attr_names, fail_count, output)

    if compare_v3 and compare_v3.is_file() and compare_v6 and compare_v6.is_file():
        v3_rows = load_sample_file(compare_v3)
        v6_rows = load_sample_file(compare_v6)
        print_v7_topic_comparison_report(
            v3_rows, v6_rows, results, attr_names, raw_map, fail_count
        )
    elif compare_v4 and compare_v4.is_file() and compare_v5 and compare_v5.is_file():
        v4_rows = load_sample_file(compare_v4)
        v5_rows = load_sample_file(compare_v5)
        print_v6_hybrid_comparison_report(
            v4_rows, v5_rows, results, attr_names, raw_map, fail_count
        )
    elif compare_v3 and compare_v3.is_file() and compare_v4 and compare_v4.is_file():
        v3_rows = load_sample_file(compare_v3)
        v4_rows = load_sample_file(compare_v4)
        print_v5_hybrid_comparison_report(
            v3_rows, v4_rows, results, attr_names, raw_map, fail_count
        )
    elif (
        compare_v1 and compare_v1.is_file()
        and compare_v2 and compare_v2.is_file()
        and compare_v3 and compare_v3.is_file()
    ):
        v1_rows = load_sample_file(compare_v1)
        v2_rows = load_sample_file(compare_v2)
        v3_rows = load_sample_file(compare_v3)
        print_quad_comparison_report(
            v1_rows, v2_rows, v3_rows, results, attr_names, raw_map, fail_count
        )
    elif compare_v1 and compare_v1.is_file() and compare_v2 and compare_v2.is_file():
        v1_rows = load_sample_file(compare_v1)
        v2_rows = load_sample_file(compare_v2)
        print_triple_comparison_report(
            v1_rows, v2_rows, results, attr_names, raw_map, fail_count
        )
    elif compare_with and compare_with.is_file():
        before_rows = load_sample_file(compare_with)
        print_comparison_report(before_rows, results, attr_names, raw_map)

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="v2 Phase3 샘플 100건 검증")
    p.add_argument("-n", type=int, default=100, help="샘플 건수")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    p.add_argument("--sleep", type=float, default=0.3, help="건당 대기(초)")
    p.add_argument("--out", type=str, default=None, help="출력 jsonl 경로")
    p.add_argument(
        "--ids-from",
        type=str,
        default=None,
        help="기존 sample jsonl의 recruitId 목록 재사용",
    )
    p.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="수정 전 sample jsonl — 전/후 2-way 비교 리포트",
    )
    p.add_argument(
        "--compare-v1",
        type=str,
        default=None,
        help="v1 sample jsonl — 3-way 비교 리포트 (v1/v2/v3)",
    )
    p.add_argument(
        "--compare-v2",
        type=str,
        default=None,
        help="v2 sample jsonl — 3-way 비교 리포트 (v1/v2/v3)",
    )
    p.add_argument(
        "--compare-v3",
        type=str,
        default=None,
        help="v3 sample jsonl — v5 하이브리드 또는 4-way 비교",
    )
    p.add_argument(
        "--compare-v4",
        type=str,
        default=None,
        help="v4 sample jsonl — v5 하이브리드 비교 (v3/v4/v5)",
    )
    p.add_argument(
        "--compare-v5",
        type=str,
        default=None,
        help="v5 sample jsonl — v6 하이브리드 비교 (v4/v5/v6)",
    )
    p.add_argument(
        "--compare-v6",
        type=str,
        default=None,
        help="v6 sample jsonl — v7 촬영주제 v3 비교 (v3/v6/v7)",
    )
    args = p.parse_args()

    out = Path(args.out) if args.out else None
    ids_from = Path(args.ids_from) if args.ids_from else None
    compare_with = Path(args.compare_with) if args.compare_with else None
    compare_v1 = Path(args.compare_v1) if args.compare_v1 else None
    compare_v2 = Path(args.compare_v2) if args.compare_v2 else None
    compare_v3 = Path(args.compare_v3) if args.compare_v3 else None
    compare_v4 = Path(args.compare_v4) if args.compare_v4 else None
    compare_v5 = Path(args.compare_v5) if args.compare_v5 else None
    compare_v6 = Path(args.compare_v6) if args.compare_v6 else None

    run(
        n=args.n,
        seed=args.seed,
        sleep_sec=args.sleep,
        out_path=out,
        ids_from=ids_from,
        compare_with=compare_with,
        compare_v1=compare_v1,
        compare_v2=compare_v2,
        compare_v3=compare_v3,
        compare_v4=compare_v4,
        compare_v5=compare_v5,
        compare_v6=compare_v6,
    )


if __name__ == "__main__":
    main()
