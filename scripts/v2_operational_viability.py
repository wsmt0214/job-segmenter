"""운영 가능성 등급(A/B/C/D) · stop_check — docs/v2.1_클러스터링_확정.md"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from schema_v2 import UNCLEAR_VALUE
from v2_clustering_v21 import load_phase3_frame
from v2_segment_ops import COL_PLACE, COL_PURPOSE, COL_TOPIC
from v2_tune_metrics import (
    CATCH_ALL_PCT,
    MICRO_N,
    MODEL_N2_CELL,
    SMALL_CELL_SLUGS,
    cell_slug,
    pipeline_metrics,
    segment_rows,
)

DOM_TH = 0.70
GRADE_A_MIN_N = 50
GRADE_A_MAX_PCT = 25.0

STOP = {
    "model_n2_max_pct": 30.0,
    "model_n2_a_min": 10,
    "micro_lt30_max": 3,
    "catch_all_cells_max": 0,
    "n_poor_max": 120,
    "ab_grade_pct_min": 75.0,
}


@dataclass
class SegGrade:
    recruitType: str
    payment_group: str
    merged_segment_id: int
    segment_key: str
    n: int
    pct_cell: float
    grade: str
    flags: list[str]


def _axis_top(seg: pd.DataFrame, col: str) -> tuple[str, float]:
    if col not in seg.columns or seg.empty:
        return UNCLEAR_VALUE, 0.0
    vc = seg[col].value_counts()
    return str(vc.index[0]), float(vc.iloc[0] / len(seg))


def grade_segment(
    rt: str,
    pg: str,
    mid: int,
    seg: pd.DataFrame,
    *,
    cell_n: int,
) -> SegGrade:
    n = len(seg)
    pct = 100 * n / cell_n if cell_n else 100.0
    sk = str(seg["segment_key"].iloc[0]) if n else "-"
    flags: list[str] = []
    slug = cell_slug(rt, pg)

    if mid == -1 or sk == "정보 부족형":
        return SegGrade(rt, pg, mid, sk, n, pct, "D", ["info_poor"])

    if slug in SMALL_CELL_SLUGS:
        return SegGrade(rt, pg, mid, sk, n, pct, "D", ["small_cell"])

    if n < MICRO_N:
        flags.append("micro")
    if pct > CATCH_ALL_PCT:
        flags.append("catch_all")

    purpose, p_pct = _axis_top(seg, COL_PURPOSE)
    topic, t_pct = _axis_top(seg, COL_TOPIC)

    if "catch_all" in flags or "micro" in flags:
        return SegGrade(rt, pg, mid, sk, n, pct, "C", flags)

    if (
        n >= GRADE_A_MIN_N
        and pct <= GRADE_A_MAX_PCT
        and (p_pct >= DOM_TH or t_pct >= DOM_TH)
    ):
        return SegGrade(rt, pg, mid, sk, n, pct, "A", flags)

    return SegGrade(rt, pg, mid, sk, n, pct, "B", flags)


def grade_all(df: pd.DataFrame, phase_df: pd.DataFrame) -> list[SegGrade]:
    cols = [COL_PURPOSE, COL_PLACE, COL_TOPIC]
    merged = df.merge(phase_df[["recruitId"] + cols], on="recruitId", how="left")
    out: list[SegGrade] = []
    for (rt, pg), cell in merged.groupby(["recruitType", "payment_group"]):
        cell_n = len(cell[cell["merged_segment_id"] != -1])
        for mid in sorted(cell["merged_segment_id"].unique()):
            mid = int(mid)
            seg = cell[cell["merged_segment_id"] == mid]
            out.append(grade_segment(rt, pg, mid, seg, cell_n=cell_n))
    return out


def stop_check(df: pd.DataFrame, phase_df: pd.DataFrame) -> tuple[bool, dict]:
    m = pipeline_metrics(df, phase_df)
    grades = grade_all(df, phase_df)
    op = [g for g in grades if g.merged_segment_id != -1]
    rt, pg = MODEL_N2_CELL
    m2 = [g for g in op if g.recruitType == rt and g.payment_group == pg]
    m2_a = sum(1 for g in m2 if g.grade == "A")
    ab = sum(1 for g in op if g.grade in ("A", "B"))
    ab_pct = 100 * ab / len(op) if op else 0.0

    checks = {
        "model_n2_max_pct": m["model_n2_max_pct"] <= STOP["model_n2_max_pct"],
        "model_n2_a_count": m2_a >= STOP["model_n2_a_min"],
        "micro_lt30": m["micro_lt30"] <= STOP["micro_lt30_max"],
        "catch_all_cells": m["catch_all_cells"] <= STOP["catch_all_cells_max"],
        "n_poor": m["n_poor"] <= STOP["n_poor_max"],
        "ab_grade_pct": ab_pct >= STOP["ab_grade_pct_min"],
    }
    return all(checks.values()), {
        **m,
        "model_n2_a_count": m2_a,
        "model_n2_n_segs": len(m2),
        "ab_grade_pct": round(ab_pct, 1),
        "checks": checks,
    }


def print_report(df: pd.DataFrame, phase_df: pd.DataFrame) -> None:
    m = pipeline_metrics(df, phase_df)
    ok, detail = stop_check(df, phase_df)
    grades = grade_all(df, phase_df)
    op = [g for g in grades if g.merged_segment_id != -1]

    print("=== 운영 가능성 (KPI) ===")
    print(f"  segs={m['n_segments']} micro<{MICRO_N}={m['micro_lt30']} poor={m['n_poor']}")
    print(
        f"  model×n2 max={m['model_n2_max']} ({m['model_n2_max_pct']}%) "
        f"segs={m['model_n2_n_segs']}"
    )
    print(f"  catch-all cells (max%>{CATCH_ALL_PCT}, 소형셀 제외)={m['catch_all_cells']}")
    print(f"  A+B 등급={detail['ab_grade_pct']}%  model×n2 A={detail['model_n2_a_count']}")
    print(f"\n  stop_check={'PASS' if ok else 'FAIL'}")
    for k, v in detail["checks"].items():
        print(f"    {k}: {'OK' if v else 'NG'}")

    from collections import Counter

    c = Counter(g.grade for g in op)
    print(f"\n등급 분포: A={c['A']} B={c['B']} C={c['C']} D={sum(c[x] for x in c if x=='D')}")

    print("\n=== C/D 세그 ===")
    for g in sorted(op, key=lambda x: (-x.n, x.grade)):
        if g.grade in ("C", "D"):
            fl = ",".join(g.flags) if g.flags else "-"
            print(f"  [{g.grade}] {g.recruitType}×{g.payment_group} seg{g.merged_segment_id} "
                  f"n={g.n} ({g.pct_cell}%) {g.segment_key[:40]} ({fl})")


def main() -> None:
    parser = argparse.ArgumentParser(description="운영 가능성 등급·stop_check")
    parser.add_argument(
        "--assign",
        type=Path,
        default=Path(config.V2_DATA_DIR) / "cluster_assignments_v21_tune.csv",
    )
    parser.add_argument("--check", action="store_true", help="stop_check만 (exit 0=pass)")
    args = parser.parse_args()

    df = pd.read_csv(args.assign, dtype={"recruitId": int})
    phase_df, _, _ = load_phase3_frame()
    print_report(df, phase_df)
    if args.check:
        ok, _ = stop_check(df, phase_df)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
