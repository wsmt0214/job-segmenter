"""v2.1 클러스터링 — null-aware Gower + 정보 충실도 선분리 + 운영 세그먼트 병합"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import numpy as np
import pandas as pd
from schema_v2 import (
    AUX_CLUSTER_TAGS,
    CLUSTERING_4DIM,
    CLUSTERING_6DIM,
    GOWER_DIM_WEIGHTS,
    P3_SNAP_TIER_DEFAULT,
    UNCLEAR_VALUE,
    clustering_6dim_feature_cols,
    filter_tag_names,
    load_schema,
)
from sklearn.cluster import AgglomerativeClustering

from v2_phase3_core import apply_p3_corrections_phase3_frame, load_recruit_texts
import v2_clustering as vc
from v2_gower import gower_distance_matrix
from v2_segment_ops import (
    COL_CAREER,
    COL_CONTINUITY,
    COL_PLACE,
    COL_PURPOSE,
    COL_TOPIC,
    COL_TREATMENT,
    CLUSTER_AXIS_COLS,
    MERGE_COLS,
    ClusterInfo,
    MergedSegment,
    build_place_missing_name,
    apply_display_segment_names,
    build_display_segment_name,
    build_segment_name,
    dominant_value,
    info_density_row,
    merge_clusters_by_key,
)

PHASE3_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"
OUT_CSV = Path(config.V2_DATA_DIR) / "cluster_assignments_v21_tune.csv"
REPORT_PATH = Path(config.V2_DATA_DIR) / "marketer_review_v21_tune.txt"

# Fix C-1: 4축 전부 불명확(score=0)만 정보 부족형
INFO_POOR_THRESHOLD = 1  # score < 1

# Fix F-1: 대형 세그먼트 장소 post-split
SPLIT_MIN_SEGMENT = 1000
SPLIT_MIN_UNCLEAR_RATIO = 0.25
SPLIT_B_MIN_SIZE = 30
SPLIT_MIN_PORTFOLIO_RATIO = 0.25
PURPOSE_PORTFOLIO_VALUE = "포트폴리오"
TOPIC_SNAP_VALUE = "스냅"

# Fix C-2 / F-2A: 흡수 대상 dominant
AMBIGUOUS_DOM = ("혼재", UNCLEAR_VALUE)

# 정보 충실도 — 4축만 (경력·지속성 기본값 제외)
INFO_DENSITY_COLS = list(CLUSTERING_4DIM)

# Ward 1차 군집 수 (셀별) — 병합 전 raw 클러스터
CELL_RAW_K: dict[str, int] = {
    "model_n2": 16,
    "model_n3": 5,
    "model_pay": 7,
    "beauty_n2": 5,
    "beauty_n3": 3,
    "beauty_pay": 3,
    "photo_n2": 5,
    "photo_pay": 2,
}


@dataclass(frozen=True)
class ClusterRunConfig:
    """클러스터링 실행 파라미터 — 실험·프로덕션 공용"""
    gower_weights: tuple[float, ...] = GOWER_DIM_WEIGHTS
    include_treatment_in_merge: bool = True
    dominant_threshold: float = 0.70
    skip_coarse_merge: bool = True  # coarse 2차 병합 — experiments (v2.1_coarse_병합_실험.md)
    raw_k_scale: float = 1.0  # K0 기본 — docs/v2.1_클러스터링_확정.md
    cell_raw_k_scale: dict[str, float] = field(default_factory=dict)
    min_segment_size: int = 10
    absorb_small: bool = True
    # experiments 후처리 — 튜닝 베이스 기본 OFF (v2.1_Fix_C2_F1_F2_후처리_실험.md)
    apply_c2_absorb: bool = False
    apply_f1_split: bool = False
    apply_f1p_purpose_split: bool = False
    apply_f2b_rename: bool = False
    # Fix F-1 장소 post-split (apply_f1_split=True 일 때)
    split_min_segment: int = SPLIT_MIN_SEGMENT
    split_min_unclear_ratio: float = SPLIT_MIN_UNCLEAR_RATIO
    split_b_min_size: int = SPLIT_B_MIN_SIZE
    # Fix F-1-P 목적 post-split — 포트폴리오 / 비포트폴리오
    split_min_portfolio_ratio: float = SPLIT_MIN_PORTFOLIO_RATIO
    split_f1p_require_snap_topic: bool = True

    def merge_include_treatment(self, rt: str) -> bool:
        return self.include_treatment_in_merge and rt != "photo"


DEFAULT_CLUSTER_CONFIG = ClusterRunConfig()

# 운영 가능성 튜닝 확정 — docs/v2.1_클러스터링_확정.md
TUNE_CLUSTER_CONFIG = ClusterRunConfig(
    raw_k_scale=1.0,
    cell_raw_k_scale={
        "model_n2": 1.5,
        "model_n3": 2.0,
        "model_pay": 1.5,
        "beauty_n2": 2.5,
        "beauty_n3": 2.0,
        "photo_n2": 2.0,
    },
    min_segment_size=25,
)

# coarse만 켠 experiments 재현
COARSE_CLUSTER_CONFIG = ClusterRunConfig(skip_coarse_merge=False)


@dataclass
class CellResult:
    recruit_type: str
    payment_group: str
    n: int
    n_info_poor: int
    raw_k: int
    n_raw_clusters: int
    n_after_merge1: int
    n_final_segments: int


def _full_merge_key(ci: ClusterInfo) -> tuple[str, ...]:
    return ci.merge_key


def _coarse_merge_key(ci: ClusterInfo) -> tuple[str, ...]:
    return ci.coarse_key


def load_phase3_frame(
    *,
    apply_p3_hair: bool = True,
    apply_p3_snap: bool = True,
    apply_p3_purpose: bool = True,
    apply_p3_place: bool = True,
    p3_snap_tier: str = P3_SNAP_TIER_DEFAULT,
) -> pd.DataFrame:
    schema = load_schema()
    cluster_cols = clustering_6dim_feature_cols(schema)
    aux_cols = list(AUX_CLUSTER_TAGS)
    all_cols = cluster_cols + aux_cols

    rows: list[dict] = []
    categories: dict[int, list[str]] = {}
    with PHASE3_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok", True):
                continue
            rid = int(r["recruitId"])
            categories[rid] = r.get("categories") or []
            attrs = r.get("attributes") or {}
            row = {
                "recruitId": rid,
                "payment_group": r.get("payment_group"),
            }
            for c in all_cols:
                row[c] = attrs.get(c, UNCLEAR_VALUE)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and (
        apply_p3_hair or apply_p3_snap or apply_p3_purpose or apply_p3_place
    ):
        texts = load_recruit_texts(set(df["recruitId"].astype(int)))
        df = apply_p3_corrections_phase3_frame(
            df,
            categories,
            texts,
            apply_hair=apply_p3_hair,
            apply_snap=apply_p3_snap,
            apply_purpose=apply_p3_purpose,
            apply_place=apply_p3_place,
            snap_tier=p3_snap_tier,
        )
    return df, cluster_cols, aux_cols


def target_raw_k(
    rt: str,
    pg: str,
    n_clusterable: int,
    *,
    scale: float = 1.0,
    cell_scales: dict[str, float] | None = None,
) -> int:
    slug = vc.segment_slug(rt, pg)
    if slug not in CELL_RAW_K:
        return 1
    eff_scale = scale
    if cell_scales and slug in cell_scales:
        eff_scale = cell_scales[slug]
    k = max(1, int(round(CELL_RAW_K[slug] * eff_scale)))
    return max(1, min(k, n_clusterable))


def ward_cluster(dist: np.ndarray, k: int) -> np.ndarray:
    if len(dist) <= 1:
        return np.zeros(len(dist), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(dist)


def density_label_v21(n: int) -> str:
    """Fix C-1 기준 라벨 — score=0만 정보 부족형"""
    if n < INFO_POOR_THRESHOLD:
        return "정보 부족형"
    if n <= 3:
        return "부분 정보형"
    return "충분 정보형"


def is_info_poor_score(score: int) -> bool:
    return score < INFO_POOR_THRESHOLD


def renumber_operational_segment_ids(cell_df: pd.DataFrame) -> pd.DataFrame:
    """운영 세그먼트 id를 0부터 연속 재부여 (-1 유지)"""
    out = cell_df.copy()
    op_mask = out["merged_segment_id"] != -1
    if not op_mask.any():
        return out
    old_ids = sorted(out.loc[op_mask, "merged_segment_id"].unique())
    id_map = {old: new for new, old in enumerate(old_ids)}
    out.loc[op_mask, "merged_segment_id"] = out.loc[op_mask, "merged_segment_id"].map(id_map)
    return out


def absorb_ambiguous_purpose_place_segments(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fix C-2 / F-2A — 목적·장소 dom 모두 혼재 또는 불명확인 운영 세그먼트 → -1 흡수
    """
    out = cell_df.copy()
    absorb_ids: list[int] = []

    for mid in out.loc[out["merged_segment_id"] != -1, "merged_segment_id"].unique():
        seg = out[out["merged_segment_id"] == mid]
        purpose_dom = dominant_value(seg[COL_PURPOSE])
        place_dom = dominant_value(seg[COL_PLACE])
        if purpose_dom in AMBIGUOUS_DOM and place_dom in AMBIGUOUS_DOM:
            absorb_ids.append(int(mid))

    if not absorb_ids:
        return out, 0

    mask = out["merged_segment_id"].isin(absorb_ids)
    absorbed = int(mask.sum())
    print(f"[Fix C-2/F-2A] 목적·장소 혼재/불명확 흡수: {absorb_ids} → -1 ({absorbed}건)")
    out.loc[mask, "merged_segment_id"] = -1
    out.loc[mask, "segment_key"] = "정보 부족형"
    return renumber_operational_segment_ids(out), absorbed


def post_split_large_by_place(
    cell_df: pd.DataFrame,
    rt: str,
    pg: str,
    *,
    config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fix F-1 — 대형 세그먼트를 장소 명시 / 장소 불명확으로 post-split
    """
    out = cell_df.copy()
    split_logs: list[dict] = []
    next_id = int(out["merged_segment_id"].max()) + 1
    min_seg = config.split_min_segment
    min_ratio = config.split_min_unclear_ratio
    min_b = config.split_b_min_size

    for mid in sorted(out.loc[out["merged_segment_id"] != -1, "merged_segment_id"].unique()):
        seg_mask = out["merged_segment_id"] == mid
        seg = out[seg_mask]
        n = len(seg)
        if n < min_seg:
            continue

        unclear_n = int((seg[COL_PLACE] == UNCLEAR_VALUE).sum())
        if unclear_n / n < min_ratio:
            continue
        if unclear_n < min_b:
            continue

        place_dom = dominant_value(seg[COL_PLACE])
        if place_dom == "혼재":
            continue

        purpose_dom = dominant_value(seg[COL_PURPOSE])
        topic_dom = dominant_value(seg[COL_TOPIC])
        original_name = str(seg["segment_key"].iloc[0])

        split_a_mask = seg_mask & (out[COL_PLACE] != UNCLEAR_VALUE)
        split_b_mask = seg_mask & (out[COL_PLACE] == UNCLEAR_VALUE)
        n_a = int(split_a_mask.sum())
        n_b = int(split_b_mask.sum())
        if n_b < min_b:
            continue

        name_a = build_display_segment_name(out[split_a_mask], rt)
        name_b = build_display_segment_name(
            out[split_b_mask], rt, place_missing=True
        )

        out.loc[split_a_mask, "segment_key"] = name_a
        out.loc[split_b_mask, "merged_segment_id"] = next_id
        out.loc[split_b_mask, "segment_key"] = name_b

        split_logs.append({
            "cell": vc.cell_short_label(rt, pg),
            "original_name": original_name,
            "total": n,
            "split_a_name": name_a,
            "split_a_n": n_a,
            "split_b_name": name_b,
            "split_b_n": n_b,
        })
        print(
            f"[Fix F-1] {vc.cell_short_label(rt, pg)} '{original_name}' "
            f"→ A:{name_a}({n_a}) / B:{name_b}({n_b})"
        )
        next_id += 1

    return renumber_operational_segment_ids(out), split_logs


def post_split_large_by_purpose(
    cell_df: pd.DataFrame,
    rt: str,
    pg: str,
    *,
    config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fix F-1-P — 대형 세그를 촬영 목적 포트폴리오 / 그 외로 post-split

    조건: 크기≥split_min_segment · 포트폴리오 비율≥split_min_portfolio_ratio
    · A/B 각각≥split_b_min_size · (옵션) 주제 스냅 지배
    """
    out = cell_df.copy()
    split_logs: list[dict] = []
    next_id = int(out["merged_segment_id"].max()) + 1
    min_seg = config.split_min_segment
    min_b = config.split_b_min_size
    min_port_ratio = config.split_min_portfolio_ratio

    for mid in sorted(out.loc[out["merged_segment_id"] != -1, "merged_segment_id"].unique()):
        seg_mask = out["merged_segment_id"] == mid
        seg = out[seg_mask]
        n = len(seg)
        if n < min_seg:
            continue

        if config.split_f1p_require_snap_topic:
            topic_dom = dominant_value(seg[COL_TOPIC])
            if topic_dom != TOPIC_SNAP_VALUE:
                continue

        port_n = int((seg[COL_PURPOSE] == PURPOSE_PORTFOLIO_VALUE).sum())
        if port_n / n < min_port_ratio or port_n < min_b:
            continue
        non_port_n = n - port_n
        if non_port_n < min_b:
            continue

        purpose_dom = dominant_value(seg[COL_PURPOSE])
        if purpose_dom == PURPOSE_PORTFOLIO_VALUE:
            continue

        original_name = str(seg["segment_key"].iloc[0])
        split_a_mask = seg_mask & (out[COL_PURPOSE] == PURPOSE_PORTFOLIO_VALUE)
        split_b_mask = seg_mask & (out[COL_PURPOSE] != PURPOSE_PORTFOLIO_VALUE)
        name_a = build_display_segment_name(out[split_a_mask], rt)
        name_b = build_display_segment_name(out[split_b_mask], rt)

        out.loc[split_a_mask, "segment_key"] = name_a
        out.loc[split_b_mask, "merged_segment_id"] = next_id
        out.loc[split_b_mask, "segment_key"] = name_b

        split_logs.append({
            "cell": vc.cell_short_label(rt, pg),
            "original_name": original_name,
            "total": n,
            "split_a_name": name_a,
            "split_a_n": port_n,
            "split_b_name": name_b,
            "split_b_n": non_port_n,
        })
        print(
            f"[Fix F-1-P] {vc.cell_short_label(rt, pg)} '{original_name}' "
            f"→ A:{name_a}({port_n}) / B:{name_b}({non_port_n})"
        )
        next_id += 1

    return renumber_operational_segment_ids(out), split_logs


def regenerate_segment_names(cell_df: pd.DataFrame, recruit_type: str) -> pd.DataFrame:
    """표시명 재생성 — merge_key dominant와 동일 (1순위 규칙)"""
    return apply_display_segment_names(cell_df, recruit_type)


def finalize_cell_postprocess(
    cell_df: pd.DataFrame,
    rt: str,
    pg: str,
    *,
    config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
) -> tuple[pd.DataFrame, list[dict]]:
    """experiments 후처리 — C-2 / F-1 / F-2B (플래그로 ON/OFF)"""
    out = cell_df.copy()
    split_logs: list[dict] = []
    if config.apply_c2_absorb:
        out, _ = absorb_ambiguous_purpose_place_segments(out)
    if config.apply_f1_split:
        out, place_logs = post_split_large_by_place(out, rt, pg, config=config)
        split_logs.extend(place_logs)
    if config.apply_f1p_purpose_split:
        out, purpose_logs = post_split_large_by_purpose(out, rt, pg, config=config)
        split_logs.extend(purpose_logs)
    if config.apply_f2b_rename:
        out = regenerate_segment_names(out, rt)
    return out, split_logs


def absorb_hybrid_hybrid_segments(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """"""
    return absorb_ambiguous_purpose_place_segments(cell_df)


def rebuild_merged_meta(cell_df: pd.DataFrame) -> dict[int, MergedSegment]:
    """Fix C-2 이후 리포트용 merged_segment 메타 재구성"""
    merged_by_id: dict[int, MergedSegment] = {}
    op = cell_df[cell_df["merged_segment_id"] != -1]
    if op.empty:
        return merged_by_id

    for mid, grp in op.groupby("merged_segment_id"):
        mid = int(mid)
        sk = str(grp["segment_key"].iloc[0])
        if "raw_cluster_id" in grp.columns:
            sources = [
                (int(cid), int(n))
                for cid, n in grp.groupby("raw_cluster_id").size().items()
                if int(cid) != -1
            ]
        else:
            sources = [(mid, len(grp))]
        merged_by_id[mid] = MergedSegment((sk,), sk, sources, len(grp))
    return merged_by_id


def second_pass_coarse_merge(
    cell_df: pd.DataFrame,
    rt: str,
    *,
    config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
) -> tuple[pd.DataFrame, dict[tuple, MergedSegment], dict[int, MergedSegment], int]:
    """병합 1차 결과 → 주제 혼재/불명확 축 2차 병합"""
    work = cell_df[cell_df["merged_segment_id"] != -1].copy()
    if work.empty:
        return cell_df, {}, {}, 0

    work = work.drop(columns=["raw_cluster_id"], errors="ignore")
    work = work.rename(columns={"merged_segment_id": "stage1_id"})
    out, segments, merged_by_id, n_before = merge_clusters_by_key(
        work,
        "stage1_id",
        _coarse_merge_key,
        include_treatment=config.merge_include_treatment(rt),
        dominant_threshold=config.dominant_threshold,
        absorb_small=config.absorb_small,
    )
    poor = cell_df[cell_df["merged_segment_id"] == -1].copy()
    if not poor.empty:
        out = pd.concat([out, poor], ignore_index=True)
    return out, segments, merged_by_id, n_before


def cluster_cell_v21(
    cell_df: pd.DataFrame,
    cluster_cols: list[str],
    rt: str,
    pg: str,
    *,
    config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
) -> tuple[pd.DataFrame, CellResult, dict[int, MergedSegment], list[dict]]:
    slug = vc.segment_slug(rt, pg)
    n = len(cell_df)

    cell_df = cell_df.copy()
    cell_df["info_density"] = cell_df.apply(
        lambda r: info_density_row(r, INFO_DENSITY_COLS), axis=1
    )
    cell_df["density_label"] = cell_df["info_density"].map(density_label_v21)

    # photo×n3: HDBSCAN/v2.0과 동일 — 단일 세그먼트
    if slug == "photo_n3":
        out = cell_df.copy()
        out["raw_cluster_id"] = np.where(
            cell_df["info_density"].map(is_info_poor_score), -1, 0
        )
        out["merged_segment_id"] = out["raw_cluster_id"]
        n_poor = int((out["merged_segment_id"] == -1).sum())
        out["segment_key"] = "정보 부족형"
        merged_by_id: dict[int, MergedSegment] = {}
        if n_poor < n:
            g = out[out["merged_segment_id"] == 0]
            sk = build_display_segment_name(g, rt)
            out.loc[out["merged_segment_id"] == 0, "segment_key"] = sk
            merged_by_id[0] = MergedSegment((sk,), sk, [(0, n - n_poor)], n - n_poor)
        out, split_logs = finalize_cell_postprocess(out, rt, pg, config=config)
        n_poor = int((out["merged_segment_id"] == -1).sum())
        merged_by_id = rebuild_merged_meta(out)
        n_final = int(
            out.loc[out["merged_segment_id"] != -1, "merged_segment_id"].nunique()
        )
        return (
            out,
            CellResult(rt, pg, n, n_poor, 1, 1, 1, n_final),
            merged_by_id,
            split_logs,
        )

    poor_mask = cell_df["info_density"].map(is_info_poor_score)
    n_poor = int(poor_mask.sum())
    clusterable = cell_df[~poor_mask].copy()

    out_parts: list[pd.DataFrame] = []
    if n_poor:
        poor_df = cell_df[poor_mask].copy()
        poor_df["raw_cluster_id"] = -1
        poor_df["merged_segment_id"] = -1
        poor_df["segment_key"] = "정보 부족형"
        out_parts.append(poor_df)

    merged_by_id: dict[int, MergedSegment] = {}
    n_raw = n_merge1 = n_final = 0
    raw_k = 0

    if len(clusterable) >= 2:
        raw_k = target_raw_k(
            rt,
            pg,
            len(clusterable),
            scale=config.raw_k_scale,
            cell_scales=config.cell_raw_k_scale or None,
        )
        dist = gower_distance_matrix(
            clusterable,
            list(CLUSTERING_6DIM),
            dim_weights=list(config.gower_weights),
        )
        labels = ward_cluster(dist, raw_k)
        clusterable["raw_cluster_id"] = labels

        # 1차 병합 — 목적·장소·주제 full key
        merged1, _, _, n_raw = merge_clusters_by_key(
            clusterable,
            "raw_cluster_id",
            _full_merge_key,
            include_treatment=config.merge_include_treatment(rt),
            dominant_threshold=config.dominant_threshold,
            absorb_small=config.absorb_small,
            min_segment_size=config.min_segment_size,
        )
        n_merge1 = int(
            merged1.loc[merged1["merged_segment_id"] != -1, "merged_segment_id"].nunique()
        )

        if config.skip_coarse_merge:
            merged2 = merged1
            merged_by_id = rebuild_merged_meta(merged1)
        else:
            merged2, _, merged_by_id, _ = second_pass_coarse_merge(
                merged1, rt, config=config
            )
        # 2차 병합 후에도 원본 Ward 라벨 보존
        if "raw_cluster_id" not in merged2.columns and "raw_cluster_id" in merged1.columns:
            id_map = merged1.set_index("recruitId")["raw_cluster_id"]
            merged2["raw_cluster_id"] = merged2["recruitId"].map(id_map)
        n_final = int(
            merged2.loc[merged2["merged_segment_id"] != -1, "merged_segment_id"].nunique()
        )
        out_parts.append(merged2)
    elif len(clusterable) == 1:
        clusterable = clusterable.copy()
        clusterable["raw_cluster_id"] = 0
        clusterable["merged_segment_id"] = 0
        row = clusterable.iloc[0]
        clusterable["segment_key"] = (
            f"{row[COL_PURPOSE]}·{row[COL_PLACE]}·{row[COL_TOPIC]}"
            if row[COL_PURPOSE] != UNCLEAR_VALUE
            else "단일·부분정보"
        )
        raw_k = n_raw = n_merge1 = n_final = 1
        merged_by_id = {
            0: MergedSegment(
                (clusterable["segment_key"].iloc[0],),
                clusterable["segment_key"].iloc[0],
                [(0, 1)],
                1,
            )
        }
        out_parts.append(clusterable)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else cell_df.copy()

    out, split_logs = finalize_cell_postprocess(out, rt, pg, config=config)
    n_poor = int((out["merged_segment_id"] == -1).sum())
    n_final = int(
        out.loc[out["merged_segment_id"] != -1, "merged_segment_id"].nunique()
    )
    merged_by_id = rebuild_merged_meta(out)

    result = CellResult(
        rt, pg, n, n_poor, raw_k, n_raw, n_merge1, n_final
    )
    return out, result, merged_by_id, split_logs


def _tag_lines(series: pd.Series, top_n: int = 6) -> str:
    total = len(series)
    if total == 0:
        return "(없음)"
    return ", ".join(
        f"{v}({c}건, {c/total*100:.0f}%)"
        for v, c in series.value_counts().head(top_n).items()
    )


def write_v21_report(
    all_df: pd.DataFrame,
    cell_results: list[CellResult],
    cell_merged: dict[tuple[str, str], dict[int, MergedSegment]],
    aux_cols: list[str],
    report_path: Path = REPORT_PATH,
    *,
    cluster_config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
) -> None:
    excluded = filter_tag_names(load_schema())
    n_seg = sum(r.n_final_segments for r in cell_results) + sum(
        1 for r in cell_results if r.n_info_poor > 0
    )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("v2.1 튜닝 베이스 — 운영 세그먼트 마케터 검토 리포트\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            "파이프라인: P3-헤어 → P3-스냅 → P3-목적 → P3-장소 → C-1 → "
            "Gower(6축) → raw K → 1차 dominant 병합"
        )
        if not cluster_config.skip_coarse_merge:
            f.write(" → coarse 병합")
        f.write("\n")
        f.write(
            "experiments: coarse·C-2/F-1/F-2B 기본 OFF — docs/v2.1_클러스터링_확정.md §7\n"
        )
        f.write(f"운영 세그먼트: {n_seg}개 / 9,941건\n")
        f.write(f"클러스터링 6축: {', '.join(CLUSTERING_6DIM)}\n")
        f.write(
            f"Fix C-1 정보충실도(4축만): {', '.join(CLUSTERING_4DIM)}\n"
        )
        if aux_cols:
            f.write(f"보조 태그: {', '.join(aux_cols)}\n")
        f.write(
            "정보 부족형: 4축 전부 불명확(score=0)만 (Fix C-1, segment_id=-1)\n"
        )
        if excluded:
            f.write(f"미포함 filter_tags: {', '.join(excluded)}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("셀별 요약\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"{'셀':<22} {'건수':>6} {'부족':>5} {'rawK':>5} "
            f"{'1차병합':>6} {'최종':>5}\n"
        )
        f.write("-" * 55 + "\n")
        for r in cell_results:
            f.write(
                f"{vc.cell_label(r.recruit_type, r.payment_group):<22} "
                f"{r.n:>6,} {r.n_info_poor:>5} {r.raw_k:>5} "
                f"{r.n_after_merge1:>6} {r.n_final_segments:>5}\n"
            )
        f.write("\n")

        for rt in config.RECRUIT_TYPES:
            for pg in config.PAYMENT_GROUPS:
                g = all_df[(all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)]
                if g.empty:
                    continue
                cr = next(
                    x for x in cell_results
                    if x.recruit_type == rt and x.payment_group == pg
                )
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"[{vc.cell_label(rt, pg)}] {len(g):,}건 · "
                    f"정보 부족 {cr.n_info_poor} · 최종 세그먼트 {cr.n_final_segments}\n"
                )
                f.write(f"{'=' * 60}\n")

                poor = g[g["merged_segment_id"] == -1]
                if not poor.empty:
                    f.write(
                        f"\n  세그먼트 -1 — 정보 부족형\n"
                        f"  ({len(poor)}건, {len(poor)/len(g)*100:.1f}%)\n"
                    )
                    f.write(f"    [시술 태그]: {_tag_lines(poor[COL_TREATMENT])}\n")
                    for c in aux_cols:
                        f.write(f"    [보조] {c}: {_tag_lines(poor[c], 3)}\n")

                meta_map = cell_merged.get((rt, pg), {})
                for mid in sorted(g["merged_segment_id"].unique()):
                    if mid == -1:
                        continue
                    seg = g[g["merged_segment_id"] == mid]
                    sk = seg["segment_key"].iloc[0]
                    f.write(
                        f"\n  세그먼트 {mid} — {sk}\n"
                        f"  ({len(seg)}건, {len(seg)/len(g)*100:.1f}%)\n"
                    )
                    if int(mid) in meta_map:
                        src = ", ".join(
                            f"raw{cid}({n}건)"
                            for cid, n in sorted(meta_map[int(mid)].source_clusters)
                        )
                        f.write(f"    [병합 raw 군집]: {src}\n")
                    f.write(f"    [시술 태그]: {_tag_lines(seg[COL_TREATMENT])}\n")
                    if "혼재" in sk or UNCLEAR_VALUE in sk:
                        f.write(f"    [주제 태그]: {_tag_lines(seg[COL_TOPIC])}\n")
                    for c in CLUSTER_AXIS_COLS:
                        top = seg[c].value_counts().head(3)
                        f.write(
                            f"    {c}: "
                            + ", ".join(f"{v}({c2})" for v, c2 in top.items())
                            + "\n"
                        )
                    if aux_cols:
                        aux = " | ".join(
                            f"{c}: {_tag_lines(seg[c], 2)}" for c in aux_cols
                        )
                        f.write(f"    [보조] {aux}\n")

    print(f"마케터 리포트: {report_path}")


def _is_hybrid_hybrid_key(segment_key: str) -> bool:
    """목적·장소 모두 혼재 dominant — '혼재·혼재' 패턴"""
    parts = str(segment_key).split("·")
    if len(parts) < 2:
        return False
    return parts[0] == "혼재" and parts[1] == "혼재"


def pipeline_metrics(all_df: pd.DataFrame, results: list[CellResult]) -> dict:
    """비교용 요약 지표"""
    non_noise = all_df[all_df["merged_segment_id"] != -1]
    n_segments = int(non_noise.groupby(["recruitType", "payment_group"])["merged_segment_id"].nunique().sum())
    n_poor = int((all_df["merged_segment_id"] == -1).sum())

    m2 = all_df[(all_df["recruitType"] == "model") & (all_df["payment_group"] == "n2")]
    m2_op = m2[m2["merged_segment_id"] != -1]
    m2_max = int(m2_op.groupby("merged_segment_id").size().max()) if not m2_op.empty else 0

    m2_max_pct = m2_max / 4908 * 100 if m2_max else 0.0

    hybrid_keys = {
        sk
        for sk in non_noise["segment_key"].unique()
        if _is_hybrid_hybrid_key(sk)
    }
    n_hybrid = len(hybrid_keys)

    n_unclear_dom = int(
        sum(
            1
            for sk in non_noise["segment_key"].unique()
            if UNCLEAR_VALUE in str(sk)
        )
    )

    return {
        "n_segments": n_segments,
        "n_poor": n_poor,
        "model_n2_max": m2_max,
        "model_n2_max_pct": m2_max_pct,
        "n_hybrid_hybrid": n_hybrid,
        "n_unclear_dom": n_unclear_dom,
    }


def cell_poor_ratio(all_df: pd.DataFrame, rt: str, pg: str) -> float:
    g = all_df[(all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)]
    if g.empty:
        return 0.0
    return float((g["merged_segment_id"] == -1).sum()) / len(g) * 100


def print_summary(results: list[CellResult]) -> None:
    total_poor = sum(r.n_info_poor for r in results)
    total_final = sum(r.n_final_segments for r in results)
    n_with_poor = sum(1 for r in results if r.n_info_poor > 0)

    print("\n=== v2.1 셀별 요약 ===\n")
    for r in results:
        print(
            f"[{vc.cell_short_label(r.recruit_type, r.payment_group)}] "
            f"{r.n:,}건 · 정보 부족 {r.n_info_poor} · "
            f"raw K={r.raw_k} → 1차 {r.n_after_merge1} → 최종 {r.n_final_segments} 세그먼트"
        )
    print(
        f"\n전체: 운영 세그먼트 {total_final} + "
        f"정보 부족 셀 {n_with_poor}개 (총 {total_poor:,}건 부족형)"
    )


def run(
    *,
    cluster_config: ClusterRunConfig = DEFAULT_CLUSTER_CONFIG,
    out_csv: Path = OUT_CSV,
    report_path: Path = REPORT_PATH,
    verbose: bool = True,
    phase_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[CellResult]]:
    np.random.seed(42)
    if phase_df is None:
        phase_df, cluster_cols, aux_cols = load_phase3_frame()
    else:
        _, cluster_cols, aux_cols = load_phase3_frame()

    type_map = vc.load_type_map()
    phase_df["recruitType"] = phase_df["recruitId"].map(type_map)

    if verbose:
        print("=== v2.1 튜닝 베이스 클러스터링 ===\n")
        print("P3-헤어: 헤어+시술불명확 → 헤어 시술 미언급")
        print("P3-스냅: 카테고리·본문 스냅 신호 → 촬영 주제=스냅 (프로필 guard)")
        print(
            f"P3-스냅 Tier: {P3_SNAP_TIER_DEFAULT} "
            "(Tier2=인물스냅+본문 스냅 키워드 있을 때만)"
        )
        print("P3-목적: 불명확 → 비포애프터·프로필·뷰티·패션·포트폴리오")
        print("P3-장소: 불명확 → 스튜디오·야외·홈스냅 (충돌 skip)")
        print(f"Fix C-1: 정보충실도 4축, score<{INFO_POOR_THRESHOLD}")
        print(f"Fix B: model×n2 raw K={CELL_RAW_K['model_n2']}")
        print("파이프라인: Gower(6축) → raw K → 1차 dominant 병합")
        exp = []
        if not cluster_config.skip_coarse_merge:
            exp.append("coarse")
        if cluster_config.apply_c2_absorb:
            exp.append("C-2/F-2A")
        if cluster_config.apply_f1_split:
            exp.append("F-1")
        if cluster_config.apply_f2b_rename:
            exp.append("F-2B")
        if exp:
            print(f"experiments ON: {', '.join(exp)}")
        else:
            print("experiments: OFF (coarse, C-2, F-1, F-2B)")
        print(
            f"Gower 가중 ({'/'.join(CLUSTERING_6DIM)}): "
            f"{', '.join(str(w) for w in cluster_config.gower_weights)}"
        )
        print(
            f"병합: dominant {cluster_config.dominant_threshold:.0%} · "
            f"6축(photo 시술 제외) · coarse={'OFF' if cluster_config.skip_coarse_merge else 'ON'}\n"
        )

    all_frames: list[pd.DataFrame] = []
    results: list[CellResult] = []
    cell_merged: dict[tuple[str, str], dict[int, MergedSegment]] = {}

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            sub = phase_df[
                (phase_df["recruitType"] == rt) & (phase_df["payment_group"] == pg)
            ]
            if sub.empty:
                continue
            if verbose:
                print(f"[{vc.cell_label(rt, pg)}] {len(sub)}건 클러스터링...")
            out, res, merged_by_id, _ = cluster_cell_v21(
                sub, cluster_cols, rt, pg, config=cluster_config
            )
            results.append(res)
            cell_merged[(rt, pg)] = merged_by_id
            all_frames.append(out)

    all_df = pd.concat(all_frames, ignore_index=True)
    out_cols = [
        "recruitId",
        "recruitType",
        "payment_group",
        "info_density",
        "density_label",
        "raw_cluster_id",
        "merged_segment_id",
        "segment_key",
    ]
    result_df = all_df[out_cols].copy()
    result_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"\n배정 저장: {out_csv} ({len(all_df)}건)")

    if verbose:
        write_v21_report(
            all_df,
            results,
            cell_merged,
            aux_cols,
            report_path,
            cluster_config=cluster_config,
        )
        print_summary(results)
        if cluster_config == TUNE_CLUSTER_CONFIG:
            from v2_operational_viability import stop_check

            ok, _ = stop_check(result_df, phase_df)
            print(f"\nstop_check={'PASS' if ok else 'FAIL'}")
    return result_df, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v2.1 클러스터링")
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="experiments — coarse 2차 병합 ON",
    )
    args = parser.parse_args()
    if args.coarse:
        run(cluster_config=COARSE_CLUSTER_CONFIG)
    else:
        run(cluster_config=TUNE_CLUSTER_CONFIG)
