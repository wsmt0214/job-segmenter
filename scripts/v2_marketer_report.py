"""v2.0 Task 4 — 마케터 검토 리포트 생성"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from schema_v2 import (
    UNCLEAR_VALUE,
    CLUSTERING_CORE_DIMS,
    clustering_3dim_feature_cols,
    clustering_feature_cols,
    filter_tag_names,
    load_schema,
)

ASSIGNMENTS_6DIM = Path(config.V2_DATA_DIR) / "cluster_assignments_v2.csv"
REPORT_6DIM = Path(config.V2_DATA_DIR) / "marketer_review_v2.txt"
ASSIGNMENTS_3DIM = Path(config.V2_DATA_DIR) / "cluster_assignments_v2_3dim.csv"
REPORT_3DIM = Path(config.V2_DATA_DIR) / "marketer_review_v2_3dim.txt"
ASSIGNMENTS_NOUNK = Path(config.V2_DATA_DIR) / "cluster_assignments_v2_nounk.csv"
REPORT_NOUNK = Path(config.V2_DATA_DIR) / "marketer_review_v2_nounk.txt"

LABEL_PRIORITY_6 = (
    "촬영 목적",
    "촬영 주제",
    "촬영 장소",
    "시술 종류",
    "경력 조건",
    "작업 지속성",
)

LABEL_PRIORITY_3 = CLUSTERING_CORE_DIMS


def _top_value(seg: pd.DataFrame, col: str, min_ratio: float = 0.25) -> str | None:
    if col not in seg.columns or seg.empty:
        return None
    vc = seg[col].value_counts()
    for val, cnt in vc.items():
        if str(val) != UNCLEAR_VALUE and cnt / len(seg) >= min_ratio:
            return str(val)
    return None


def suggest_cluster_label(
    seg: pd.DataFrame,
    clustering_cols: list[str],
    label_priority: tuple[str, ...],
) -> str:
    parts: list[str] = []
    for col in label_priority:
        if col not in clustering_cols:
            continue
        val = _top_value(seg, col)
        if val and val not in parts:
            parts.append(val)
        if len(parts) >= 3:
            break
    return " · ".join(parts) if parts else "(속성 혼재·불명확 다수)"


def load_labeled_frame(
    assignments_path: Path,
    feat_cols: list[str],
) -> pd.DataFrame:
    import v2_clustering as vc

    if not assignments_path.is_file():
        raise FileNotFoundError(f"없음: {assignments_path}")

    assign = pd.read_csv(assignments_path, dtype={"recruitId": int})
    type_map = vc.load_type_map()
    k_max = vc.CELL_K_MAX_3DIM if feat_cols == list(CLUSTERING_CORE_DIMS) else vc.CELL_K_MAX
    segments = vc.load_records_by_segment(type_map, k_max)

    frames: list[pd.DataFrame] = []
    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            records = segments.get((rt, pg), [])
            if not records:
                continue
            feat_df = vc.build_feature_frame(records, feat_cols, rt, pg)
            if feat_df.empty:
                continue
            sub = assign[
                (assign["recruitType"] == rt) & (assign["payment_group"] == pg)
            ][["recruitId", "cluster_id"]]
            merged = feat_df.merge(sub, on="recruitId", how="inner")
            frames.append(merged)

    if not frames:
        raise ValueError("리포트용 데이터 없음")

    return pd.concat(frames, ignore_index=True)


def write_marketer_report(
    all_df: pd.DataFrame,
    report_cols: list[str],
    path: Path = REPORT_6DIM,
    *,
    k_max: dict | None = None,
    vector_label: str | None = None,
    drop_unclear: bool = False,
) -> None:
    import v2_clustering as vc

    if k_max is None:
        k_max = (
            vc.CELL_K_MAX_3DIM
            if report_cols == list(CLUSTERING_CORE_DIMS)
            else vc.CELL_K_MAX
        )
    if vector_label is None:
        vector_label = (
            "dimensions 3 (촬영 장소·목적·시술 종류)"
            if report_cols == list(CLUSTERING_CORE_DIMS)
            else "dimensions 6"
        )
    label_priority = (
        LABEL_PRIORITY_3
        if report_cols == list(CLUSTERING_CORE_DIMS)
        else LABEL_PRIORITY_6
    )

    excluded_tags = filter_tag_names(load_schema())
    total_segments = all_df.groupby(["recruitType", "payment_group"])["cluster_id"].nunique().sum()

    with path.open("w", encoding="utf-8") as f:
        f.write("v2.0 클러스터링 마케터 검토 리포트\n")
        f.write("=" * 60 + "\n\n")
        f.write("분리 기준: recruitType × payment_group (9셀)\n")
        f.write("K 정책: CELL_K_MAX 범위 + Δsilhouette elbow K\n")
        f.write(f"전체 세그먼트: {total_segments}개 / 9,941건\n")
        f.write(f"클러스터링 입력 ({vector_label}): {', '.join(report_cols)}\n")
        if excluded_tags:
            f.write(f"미포함 — filter_tags: {', '.join(excluded_tags)}\n")
        if report_cols == list(CLUSTERING_CORE_DIMS):
            f.write(
                "미포함 — dimensions: 촬영 주제, 경력 조건, 작업 지속성 (변별력 낮음)\n"
            )
        if drop_unclear:
            f.write(
                "인코딩 — one-hot '불명확' 미포함: 불명확 값은 해당 차원 0벡터\n"
            )
        f.write("\n")

        f.write("검토 질문:\n")
        f.write("  Q1. 군집 후보명이 내용을 잘 표현하는가?\n")
        f.write("  Q2. 실제 운영에서 의미 있는 구분인가?\n")
        f.write("  Q3. 더 나눠야 하는가? 다른 군집과 합쳐야 하는가?\n\n")

        f.write("=" * 60 + "\n")
        f.write("셀별 요약\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'셀':<22} {'건수':>6} {'K상한':>5} {'선택K':>5} {'군집수':>5} {'최소':>5} {'중앙값':>6}\n")
        f.write("-" * 62 + "\n")

        for rt in config.RECRUIT_TYPES:
            for pg in config.PAYMENT_GROUPS:
                g_df = all_df[
                    (all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)
                ]
                if g_df.empty:
                    continue
                k_cap = k_max[(rt, pg)]
                k_chosen = g_df["cluster_id"].nunique()
                sizes = g_df["cluster_id"].value_counts()
                f.write(
                    f"{vc.cell_label(rt, pg):<22} {len(g_df):>6,} {k_cap:>5} {k_chosen:>5} "
                    f"{k_chosen:>5} {int(sizes.min()):>5} "
                    f"{sizes.median():>6.0f}\n"
                )

        f.write("\n")

        for rt in config.RECRUIT_TYPES:
            for pg in config.PAYMENT_GROUPS:
                g_df = all_df[
                    (all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)
                ]
                if g_df.empty:
                    continue

                k_cap = k_max[(rt, pg)]
                k_chosen = g_df["cluster_id"].nunique()

                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"[{vc.cell_label(rt, pg)}] {len(g_df):,}건\n"
                    f"  K 상한={k_cap} · 선택 K={k_chosen} · 군집 {k_chosen}개\n"
                )
                f.write(f"{'=' * 60}\n")

                for cid in sorted(g_df["cluster_id"].unique()):
                    seg = g_df[g_df["cluster_id"] == cid]
                    label = suggest_cluster_label(seg, report_cols, label_priority)
                    f.write(
                        f"\n  군집 {cid} — {label}\n"
                        f"  ({len(seg)}건, {len(seg) / len(g_df) * 100:.1f}%)\n"
                    )
                    for col in report_cols:
                        if col not in seg.columns:
                            continue
                        top = seg[col].value_counts().head(3)
                        f.write(
                            f"    {col}: "
                            + ", ".join(f"{v}({c})" for v, c in top.items())
                            + "\n"
                        )

    print(f"마케터 검토 리포트: {path} ({total_segments}세그먼트)")


def main() -> None:
    p = argparse.ArgumentParser(description="마케터 검토 리포트 재생성")
    p.add_argument(
        "--dims",
        type=int,
        choices=(3, 6),
        default=6,
        help="6dim(기본) 또는 3dim",
    )
    p.add_argument(
        "--nounk",
        action="store_true",
        help="3dim nounk 산출물 재생성",
    )
    args = p.parse_args()

    schema = load_schema()
    if args.nounk:
        feat_cols = clustering_3dim_feature_cols(schema)
        assignments = ASSIGNMENTS_NOUNK
        report_path = REPORT_NOUNK
        print("=== marketer_review_v2_nounk.txt 재생성 ===\n")
        df = load_labeled_frame(assignments, feat_cols)
        write_marketer_report(
            df, feat_cols, report_path,
            k_max=None,
            vector_label="dimensions 3, one-hot 불명확 제외",
            drop_unclear=True,
        )
    elif args.dims == 3:
        feat_cols = clustering_3dim_feature_cols(schema)
        assignments = ASSIGNMENTS_3DIM
        report_path = REPORT_3DIM
        print("=== marketer_review_v2_3dim.txt 재생성 ===\n")
        df = load_labeled_frame(assignments, feat_cols)
        write_marketer_report(df, feat_cols, report_path)
    else:
        feat_cols = clustering_feature_cols(schema)
        assignments = ASSIGNMENTS_6DIM
        report_path = REPORT_6DIM
        print("=== marketer_review_v2.txt 재생성 ===\n")
        df = load_labeled_frame(assignments, feat_cols)
        write_marketer_report(df, feat_cols, report_path)

    print(f"  입력: {assignments} ({len(df)}건)")


if __name__ == "__main__":
    main()
