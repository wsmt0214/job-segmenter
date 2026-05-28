"""v2 HDBSCAN 클러스터링 — K-means(v2_clustering.py)와 별도 비교용 파이프라인"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import numpy as np
import pandas as pd
from schema_v2 import (
    AUX_CLUSTER_TAGS,
    UNCLEAR_VALUE,
    clustering_4dim_feature_cols,
    filter_tag_names,
    load_schema,
)
from sklearn.cluster import HDBSCAN

import v2_clustering as vc

OUT_CSV = Path(config.V2_DATA_DIR) / "cluster_assignments_v2_hdbscan.csv"
REPORT_PATH = Path(config.V2_DATA_DIR) / "marketer_review_v2_hdbscan.txt"
KMEANS_NOUNK_CSV = Path(config.V2_DATA_DIR) / "cluster_assignments_v2_nounk.csv"

SegmentKey = tuple[str, str]

# 셀별 HDBSCAN min_cluster_size
CELL_MIN_CLUSTER_SIZE: dict[str, int] = {
    "model_n2": 50,
    "model_n3": 20,
    "model_pay": 30,
    "beauty_n2": 25,
    "beauty_n3": 15,
    "beauty_pay": 10,
    "photo_n2": 20,
    "photo_pay": 10,
}

LABEL_PRIORITY = ("촬영 목적", "촬영 주제", "촬영 장소", "시술 종류")


@dataclass
class HdbscanCellResult:
    recruit_type: str
    payment_group: str
    n: int
    min_cluster_size: int
    n_clusters: int
    n_noise: int
    noise_pct: float
    min_cluster: int | None
    median_cluster: float | None
    max_cluster: int | None


def _min_cluster_size(rt: str, pg: str) -> int:
    return CELL_MIN_CLUSTER_SIZE[vc.segment_slug(rt, pg)]


def build_cell_frame(
    records: list[dict],
    cluster_cols: list[str],
    aux_cols: list[str],
    recruit_type: str,
    payment_group: str,
) -> pd.DataFrame:
    """클러스터링·리포트용 프레임 — 핵심 4 + 보조 2"""
    rows: list[dict] = []
    for r in records:
        if not r.get("ok", True):
            continue
        attrs = r.get("attributes") or {}
        row: dict = {"recruitId": int(r["recruitId"])}
        for name in cluster_cols + aux_cols:
            row[name] = attrs.get(name, UNCLEAR_VALUE)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["recruitType"] = recruit_type
    df["payment_group"] = payment_group
    return df


def remap_cluster_ids(labels: np.ndarray) -> np.ndarray:
    """HDBSCAN raw label → 0..k-1, 노이즈(-1) 유지"""
    out = labels.astype(int).copy()
    non_noise = sorted({int(x) for x in labels if int(x) != -1})
    mapping = {old: new for new, old in enumerate(non_noise)}
    for i, lab in enumerate(out):
        if lab != -1:
            out[i] = mapping[int(lab)]
    return out


def cluster_cell_hdbscan(
    df: pd.DataFrame,
    feat_cols: list[str],
    recruit_type: str,
    payment_group: str,
) -> tuple[pd.DataFrame, HdbscanCellResult]:
    n = len(df)
    slug = vc.segment_slug(recruit_type, payment_group)

    # photo×n3: HDBSCAN 생략, 단일 군집
    if slug == "photo_n3":
        print(f"  {n}건 — HDBSCAN 생략, cluster_id=0 (단일 군집)")
        out = df.copy()
        out["cluster_id"] = 0
        return out, HdbscanCellResult(
            recruit_type, payment_group, n, 1, 1, 0, 0.0, n, float(n), n
        )

    mcs = _min_cluster_size(recruit_type, payment_group)

    X, _ = vc.prepare_features(df, feat_cols, drop_unclear=True)
    print(f"  원-핫 특성: {X.shape[1]}개 (불명확 제외)")

    if X.shape[1] == 0 or n < mcs:
        print(f"  경고: 인코딩 특성 없음 또는 n<{mcs} — 전체 cluster_id=-1 (정보 부족형)")
        out = df.copy()
        out["cluster_id"] = -1
        return out, HdbscanCellResult(
            recruit_type, payment_group, n, mcs, 0, n, 100.0, None, None, None
        )

    min_samples = max(1, mcs // 2)
    clusterer = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=1,
        copy=True,
    )
    raw_labels = clusterer.fit_predict(X)
    labels = remap_cluster_ids(raw_labels)

    n_noise = int((labels == -1).sum())
    n_clusters = len({int(x) for x in labels if int(x) != -1})

    if n_clusters == 0:
        print(f"  경고: 유효 군집 0개 — 전체 cluster_id=-1 (정보 부족형)")
        labels = np.full(n, -1, dtype=int)
        n_noise = n
        n_clusters = 0

    out = df.copy()
    out["cluster_id"] = labels

    cluster_sizes = out.loc[out["cluster_id"] != -1, "cluster_id"].value_counts()
    if cluster_sizes.empty:
        min_c = med_c = max_c = None
    else:
        min_c = int(cluster_sizes.min())
        med_c = float(cluster_sizes.median())
        max_c = int(cluster_sizes.max())

    noise_pct = n_noise / n * 100 if n else 0.0
    print(
        f"  HDBSCAN min_cluster_size={mcs}, min_samples={min_samples} "
        f"→ 군집 {n_clusters}개, 노이즈 {n_noise}건 ({noise_pct:.1f}%)"
    )

    return out, HdbscanCellResult(
        recruit_type,
        payment_group,
        n,
        mcs,
        n_clusters,
        n_noise,
        noise_pct,
        min_c,
        med_c,
        max_c,
    )


def _top_value(seg: pd.DataFrame, col: str, min_ratio: float = 0.25) -> str | None:
    if col not in seg.columns or seg.empty:
        return None
    for val, cnt in seg[col].value_counts().items():
        if str(val) != UNCLEAR_VALUE and cnt / len(seg) >= min_ratio:
            return str(val)
    return None


def suggest_cluster_label(seg: pd.DataFrame, cluster_cols: list[str]) -> str:
    if not seg.empty and int(seg["cluster_id"].iloc[0]) == -1:
        return "정보 부족형"
    parts: list[str] = []
    for col in LABEL_PRIORITY:
        if col not in cluster_cols:
            continue
        val = _top_value(seg, col)
        if val and val not in parts:
            parts.append(val)
        if len(parts) >= 3:
            break
    return " · ".join(parts) if parts else "(속성 혼재·불명확 다수)"


def write_hdbscan_report(
    all_df: pd.DataFrame,
    cluster_cols: list[str],
    aux_cols: list[str],
    results: list[HdbscanCellResult],
    path: Path = REPORT_PATH,
) -> None:
    excluded_tags = filter_tag_names(load_schema())
    n_segments = sum(r.n_clusters for r in results) + sum(1 for r in results if r.n_noise > 0)

    with path.open("w", encoding="utf-8") as f:
        f.write("v2 HDBSCAN 클러스터링 마케터 검토 리포트\n")
        f.write("=" * 60 + "\n\n")
        f.write("분리 기준: recruitType × payment_group (9셀)\n")
        f.write("알고리즘: HDBSCAN (metric=euclidean, cluster_selection_method=eom)\n")
        f.write(f"운영 세그먼트(군집+노이즈): {n_segments}개 / 9,941건\n")
        f.write(
            f"클러스터링 입력 (4dim, one-hot 불명확 제외): {', '.join(cluster_cols)}\n"
        )
        f.write(f"보조 통계만: {', '.join(aux_cols)}\n")
        if excluded_tags:
            f.write(f"미포함 — filter_tags: {', '.join(excluded_tags)}\n")
        f.write("인코딩 — one-hot '불명확' 미포함: 불명확 값은 해당 차원 0벡터\n")
        f.write("cluster_id=-1 — 정보 부족형 (HDBSCAN 노이즈)\n\n")

        f.write("검토 질문:\n")
        f.write("  Q1. 군집 후보명이 내용을 잘 표현하는가?\n")
        f.write("  Q2. 실제 운영에서 의미 있는 구분인가?\n")
        f.write("  Q3. 노이즈(정보 부족형) 비율이 허용 가능한가?\n\n")

        f.write("=" * 60 + "\n")
        f.write("셀별 요약\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"{'셀':<22} {'건수':>6} {'군집수':>5} {'노이즈':>6} {'노이즈%':>7} "
            f"{'최소':>5} {'중앙값':>6}\n"
        )
        f.write("-" * 68 + "\n")

        for r in results:
            min_s = "—" if r.min_cluster is None else str(r.min_cluster)
            med_s = "—" if r.median_cluster is None else f"{r.median_cluster:.0f}"
            f.write(
                f"{vc.cell_label(r.recruit_type, r.payment_group):<22} "
                f"{r.n:>6,} {r.n_clusters:>5} {r.n_noise:>6} {r.noise_pct:>6.1f}% "
                f"{min_s:>5} {med_s:>6}\n"
            )

        f.write("\n")

        for rt in config.RECRUIT_TYPES:
            for pg in config.PAYMENT_GROUPS:
                g_df = all_df[
                    (all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)
                ]
                if g_df.empty:
                    continue

                r = next(
                    x for x in results
                    if x.recruit_type == rt and x.payment_group == pg
                )
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"[{vc.cell_label(rt, pg)}] {len(g_df):,}건\n"
                    f"  min_cluster_size={r.min_cluster_size} · "
                    f"군집 {r.n_clusters}개 · "
                    f"노이즈 {r.n_noise}건 ({r.noise_pct:.1f}%)\n"
                )
                f.write(f"{'=' * 60}\n")

                # 노이즈 먼저
                noise_df = g_df[g_df["cluster_id"] == -1]
                if not noise_df.empty:
                    f.write(
                        f"\n  군집 -1 — 정보 부족형\n"
                        f"  ({len(noise_df)}건, {len(noise_df) / len(g_df) * 100:.1f}%)\n"
                    )
                    for col in cluster_cols + list(aux_cols):
                        top = noise_df[col].value_counts().head(3)
                        f.write(
                            f"    {col}: "
                            + ", ".join(f"{v}({c})" for v, c in top.items())
                            + "\n"
                        )

                for cid in sorted(g_df["cluster_id"].unique()):
                    if cid == -1:
                        continue
                    seg = g_df[g_df["cluster_id"] == cid]
                    label = suggest_cluster_label(seg, cluster_cols)
                    f.write(
                        f"\n  군집 {cid} — {label}\n"
                        f"  ({len(seg)}건, {len(seg) / len(g_df) * 100:.1f}%)\n"
                    )
                    for col in cluster_cols:
                        top = seg[col].value_counts().head(3)
                        f.write(
                            f"    {col}: "
                            + ", ".join(f"{v}({c})" for v, c in top.items())
                            + "\n"
                        )
                    aux_lines: list[str] = []
                    for col in aux_cols:
                        top = seg[col].value_counts().head(2)
                        aux_lines.append(
                            f"{col}: " + ", ".join(f"{v}({c})" for v, c in top.items())
                        )
                    if aux_lines:
                        f.write("    [보조] " + " | ".join(aux_lines) + "\n")

    print(f"마케터 검토 리포트: {path}")


def print_cell_summaries(results: list[HdbscanCellResult]) -> None:
    print("\n=== HDBSCAN 셀별 요약 ===\n")
    for r in results:
        min_s = "—" if r.min_cluster is None else str(r.min_cluster)
        print(
            f"[{vc.cell_short_label(r.recruit_type, r.payment_group)}] "
            f"군집 수: {r.n_clusters} / "
            f"노이즈: {r.n_noise}건 ({r.noise_pct:.1f}%) / "
            f"최소군집: {min_s}건"
        )


def print_kmeans_comparison(results: list[HdbscanCellResult]) -> None:
    if not KMEANS_NOUNK_CSV.is_file():
        print(f"\n경고: K-means 비교 파일 없음 — {KMEANS_NOUNK_CSV}")
        return

    km = pd.read_csv(KMEANS_NOUNK_CSV, dtype={"recruitId": int})
    print("\n=== K-means vs HDBSCAN 비교 (K-means: nounk 3dim) ===\n")
    header = (
        f"{'셀':<22} {'K-means 군집수':>14} {'HDBSCAN 군집수':>15} "
        f"{'HDBSCAN 노이즈%':>16}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        sub = km[
            (km["recruitType"] == r.recruit_type)
            & (km["payment_group"] == r.payment_group)
        ]
        km_clusters = sub["cluster_id"].nunique() if not sub.empty else 0
        print(
            f"{vc.cell_label(r.recruit_type, r.payment_group):<22} "
            f"{km_clusters:>14} {r.n_clusters:>15} {r.noise_pct:>15.1f}%"
        )


def run() -> list[HdbscanCellResult]:
    np.random.seed(42)

    schema = load_schema()
    cluster_cols = clustering_4dim_feature_cols(schema)
    aux_cols = list(AUX_CLUSTER_TAGS)

    print("=== v2 HDBSCAN 클러스터링 ===\n")
    print(f"클러스터링 벡터 (4dim, 불명확 제외): {cluster_cols}")
    print(f"보조 통계: {aux_cols}\n")

    type_map = vc.load_type_map()
    # 건수 출력용 — min_cluster_size 표시
    print("=== recruitType × payment_group 건수 ===")
    segments: dict[SegmentKey, list[dict]] = {k: [] for k in vc.all_segment_keys()}
    with vc.PHASE3_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = int(r["recruitId"])
            rt = type_map.get(rid, "")
            pg = r.get("payment_group")
            if rt in config.RECRUIT_TYPES and pg in config.PAYMENT_GROUPS:
                segments[(rt, pg)].append(r)

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            n = len(segments[(rt, pg)])
            if n:
                slug = vc.segment_slug(rt, pg)
                mcs = CELL_MIN_CLUSTER_SIZE.get(slug, "—")
                print(
                    f"  {vc.cell_label(rt, pg)} ({slug}): {n}건, "
                    f"min_cluster_size={mcs}"
                )

    all_frames: list[pd.DataFrame] = []
    cell_results: list[HdbscanCellResult] = []

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            records = segments[(rt, pg)]
            if not records:
                continue

            print(f"\n[{vc.cell_label(rt, pg)}] {len(records)}건")
            df = build_cell_frame(records, cluster_cols, aux_cols, rt, pg)
            if df.empty:
                print("  ok=true usable 0건 — 건너뜀")
                continue

            df, result = cluster_cell_hdbscan(df, cluster_cols, rt, pg)
            all_frames.append(df)
            cell_results.append(result)

    if not all_frames:
        raise SystemExit("클러스터링 결과 없음")

    all_df = pd.concat(all_frames, ignore_index=True)
    all_df[["recruitId", "recruitType", "payment_group", "cluster_id"]].to_csv(
        OUT_CSV, index=False
    )
    print(f"\n군집 배정 저장: {OUT_CSV} ({len(all_df)}건)")

    write_hdbscan_report(all_df, cluster_cols, aux_cols, cell_results)
    print_cell_summaries(cell_results)
    print_kmeans_comparison(cell_results)
    return cell_results


if __name__ == "__main__":
    run()
