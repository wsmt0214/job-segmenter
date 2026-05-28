"""HDBSCAN 군집 후처리 병합 — 시술 종류 차이만으로 분리된 군집 통합"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from schema_v2 import (
    AUX_CLUSTER_TAGS,
    UNCLEAR_VALUE,
    clustering_4dim_feature_cols,
    filter_tag_names,
    load_schema,
)

import v2_clustering as vc

HDBSCAN_CSV = Path(config.V2_DATA_DIR) / "cluster_assignments_v2_hdbscan.csv"
PHASE3_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"
OUT_CSV = Path(config.V2_DATA_DIR) / "cluster_assignments_v2_merged.csv"
REPORT_PATH = Path(config.V2_DATA_DIR) / "marketer_review_v2_merged.txt"

COL_PURPOSE = "촬영 목적"
COL_PLACE = "촬영 장소"
COL_TOPIC = "촬영 주제"
COL_TREATMENT = "시술 종류"
MERGE_COLS = (COL_PURPOSE, COL_PLACE, COL_TOPIC)

DOMINANT_THRESHOLD = 0.60
MIN_SEGMENT_SIZE = 10


@dataclass
class ClusterInfo:
    hdbscan_id: int
    n: int
    purpose_dom: str
    place_dom: str
    topic_dom: str

    @property
    def merge_key(self) -> tuple[str, str, str]:
        return (self.purpose_dom, self.place_dom, self.topic_dom)

    @property
    def segment_key(self) -> str:
        return f"{self.purpose_dom}·{self.place_dom}·{self.topic_dom}"


@dataclass
class MergedSegment:
    merge_key: tuple[str, str, str]
    segment_key: str
    source_clusters: list[tuple[int, int]] = field(default_factory=list)  # (hdbscan_id, n)
    n: int = 0


def dominant_value(series: pd.Series, threshold: float = DOMINANT_THRESHOLD) -> str:
    """비율 threshold 이상이면 최빈값, 미만이면 혼재"""
    if series.empty:
        return "혼재"
    vc = series.value_counts()
    top_val = str(vc.index[0])
    if vc.iloc[0] / len(series) >= threshold:
        return top_val
    return "혼재"


def load_attributes() -> pd.DataFrame:
    rows: list[dict] = []
    with PHASE3_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok", True):
                continue
            attrs = r.get("attributes") or {}
            row = {"recruitId": int(r["recruitId"])}
            for col in clustering_4dim_feature_cols(load_schema()) + list(AUX_CLUSTER_TAGS):
                row[col] = attrs.get(col, UNCLEAR_VALUE)
            rows.append(row)
    return pd.DataFrame(rows)


def cluster_info_for_cell(cell_df: pd.DataFrame) -> dict[int, ClusterInfo]:
    """셀 내 HDBSCAN 군집별 dominant·병합 키"""
    info: dict[int, ClusterInfo] = {}
    for cid, grp in cell_df.groupby("hdbscan_cluster_id"):
        cid = int(cid)
        if cid == -1:
            continue
        info[cid] = ClusterInfo(
            hdbscan_id=cid,
            n=len(grp),
            purpose_dom=dominant_value(grp[COL_PURPOSE]),
            place_dom=dominant_value(grp[COL_PLACE]),
            topic_dom=dominant_value(grp[COL_TOPIC]),
        )
    return info


def merge_key_similarity(a: tuple[str, str, str], b: tuple[str, str, str]) -> int:
    return sum(1 for x, y in zip(a, b) if x == y)


def absorb_small_segments(
    segments: dict[tuple[str, str, str], MergedSegment],
) -> dict[tuple[str, str, str], MergedSegment]:
    """병합 후 MIN_SEGMENT_SIZE 미만 세그먼트를 유사 키·최대 건수 세그먼트로 흡수"""
    changed = True
    while changed:
        changed = False
        small_keys = [k for k, s in segments.items() if s.n < MIN_SEGMENT_SIZE]
        if not small_keys:
            break

        for key in sorted(small_keys, key=lambda k: segments[k].n):
            if key not in segments or segments[key].n >= MIN_SEGMENT_SIZE:
                continue
            if len(segments) <= 1:
                break

            seg = segments[key]
            candidates = [(k, v) for k, v in segments.items() if k != key]
            if not candidates:
                break

            # 유사 병합 키 우선, 동률이면 건수 큰 쪽
            target_key, target_seg = max(
                candidates,
                key=lambda kv: (merge_key_similarity(key, kv[0]), kv[1].n),
            )
            target_seg.source_clusters.extend(seg.source_clusters)
            target_seg.n += seg.n
            del segments[key]
            changed = True

    return segments


def merge_cell(
    cell_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], MergedSegment], dict[int, MergedSegment], int]:
    """셀 단위 병합 — noise 유지, merged_segment_id 부여"""
    cluster_info = cluster_info_for_cell(cell_df)

    segments: dict[tuple[str, str, str], MergedSegment] = {}
    for cid, ci in cluster_info.items():
        if ci.merge_key not in segments:
            segments[ci.merge_key] = MergedSegment(
                merge_key=ci.merge_key,
                segment_key=ci.segment_key,
            )
        segments[ci.merge_key].source_clusters.append((cid, ci.n))
        segments[ci.merge_key].n += ci.n

    n_clusters_before = len(cluster_info)
    segments = absorb_small_segments(segments)

    sorted_keys = sorted(segments.keys(), key=lambda k: segments[k].n, reverse=True)
    merged_by_id: dict[int, MergedSegment] = {
        i: segments[k] for i, k in enumerate(sorted_keys)
    }

    hdbscan_to_merged: dict[int, int] = {}
    for mid, seg in merged_by_id.items():
        for cid, _ in seg.source_clusters:
            hdbscan_to_merged[cid] = mid

    id_to_segment_key = {mid: seg.segment_key for mid, seg in merged_by_id.items()}

    out = cell_df.copy()
    out["merged_segment_id"] = out["hdbscan_cluster_id"].map(
        lambda x: -1 if int(x) == -1 else hdbscan_to_merged[int(x)]
    )
    out["segment_key"] = out["merged_segment_id"].map(
        lambda x: "정보 부족형" if int(x) == -1 else id_to_segment_key[int(x)]
    )

    return out, segments, merged_by_id, n_clusters_before


def _tag_lines(series: pd.Series, top_n: int = 6) -> str:
    total = len(series)
    if total == 0:
        return "(없음)"
    parts: list[str] = []
    for val, cnt in series.value_counts().head(top_n).items():
        parts.append(f"{val}({cnt}건, {cnt / total * 100:.0f}%)")
    return ", ".join(parts)


def write_merged_report(
    all_df: pd.DataFrame,
    cell_merged_by_id: dict[tuple[str, str], dict[int, MergedSegment]],
    cell_stats: list[dict],
) -> None:
    excluded_tags = filter_tag_names(load_schema())
    n_merged = sum(s["n_after"] for s in cell_stats) + sum(
        1 for s in cell_stats if s["n_noise"] > 0
    )

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("v2 HDBSCAN 병합 운영 세그먼트 마케터 검토 리포트\n")
        f.write("=" * 60 + "\n\n")
        f.write("분리: recruitType × payment_group → 병합 세그먼트\n")
        f.write(
            f"병합 키: 촬영 목적·장소·주제 dominant (≥{DOMINANT_THRESHOLD:.0%}), "
            f"시술 종류는 보조 태그\n"
        )
        f.write(f"운영 세그먼트: {n_merged}개 / 9,941건\n")
        f.write("cluster_id=-1 / merged_segment_id=-1: 정보 부족형\n\n")

        if excluded_tags:
            f.write(f"미포함 — filter_tags: {', '.join(excluded_tags)}\n\n")

        f.write("=" * 60 + "\n")
        f.write("셀별 요약\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"{'셀':<22} {'병합전':>6} {'병합후':>6} {'노이즈':>6} {'최소':>5} {'중앙값':>6}\n"
        )
        f.write("-" * 58 + "\n")
        for st in cell_stats:
            min_s = "—" if st["min_seg"] is None else str(st["min_seg"])
            med_s = "—" if st["med_seg"] is None else f"{st['med_seg']:.0f}"
            f.write(
                f"{st['label']:<22} {st['n_before']:>6} {st['n_after']:>6} "
                f"{st['n_noise']:>6} {min_s:>5} {med_s:>6}\n"
            )
        f.write("\n")

        for rt in config.RECRUIT_TYPES:
            for pg in config.PAYMENT_GROUPS:
                g_df = all_df[
                    (all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)
                ]
                if g_df.empty:
                    continue

                st = next(
                    s for s in cell_stats
                    if s["rt"] == rt and s["pg"] == pg
                )
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"[{vc.cell_label(rt, pg)}] {len(g_df):,}건\n"
                    f"  병합 전 {st['n_before']}개 군집 · "
                    f"병합 후 {st['n_after']}개 세그먼트 · "
                    f"노이즈 {st['n_noise']}건\n"
                )
                f.write(f"{'=' * 60}\n")

                noise = g_df[g_df["merged_segment_id"] == -1]
                if not noise.empty:
                    f.write(
                        f"\n  세그먼트 -1 — 정보 부족형\n"
                        f"  ({len(noise)}건, {len(noise) / len(g_df) * 100:.1f}%)\n"
                    )
                    f.write(f"    [시술 태그]: {_tag_lines(noise[COL_TREATMENT])}\n")
                    for col in AUX_CLUSTER_TAGS:
                        f.write(f"    [보조] {col}: {_tag_lines(noise[col], top_n=3)}\n")

                for mid in sorted(g_df["merged_segment_id"].unique()):
                    if mid == -1:
                        continue
                    seg_df = g_df[g_df["merged_segment_id"] == mid]
                    sk = seg_df["segment_key"].iloc[0]
                    meta = cell_merged_by_id[(rt, pg)][int(mid)]
                    src_parts = ", ".join(
                        f"군집{cid}({n}건)" for cid, n in sorted(meta.source_clusters)
                    )
                    f.write(
                        f"\n  세그먼트 {mid} — {sk}\n"
                        f"  ({len(seg_df)}건, {len(seg_df) / len(g_df) * 100:.1f}%)\n"
                    )
                    f.write(f"    [병합된 원본 군집]: {src_parts}\n")
                    f.write(f"    [시술 태그]: {_tag_lines(seg_df[COL_TREATMENT])}\n")

                    topic_dom = meta.merge_key[2]
                    if topic_dom == "혼재":
                        f.write(f"    [주제 태그]: {_tag_lines(seg_df[COL_TOPIC])}\n")

                    for col in MERGE_COLS:
                        top = seg_df[col].value_counts().head(3)
                        f.write(
                            f"    {col}: "
                            + ", ".join(f"{v}({c})" for v, c in top.items())
                            + "\n"
                        )
                    aux_parts = []
                    for col in AUX_CLUSTER_TAGS:
                        aux_parts.append(f"{col}: {_tag_lines(seg_df[col], top_n=2)}")
                    f.write("    [보조] " + " | ".join(aux_parts) + "\n")

    print(f"마케터 검토 리포트: {REPORT_PATH}")


def print_merge_summary(
    cell_stats: list[dict],
    cell_segments: dict[tuple[str, str], dict[tuple[str, str, str], MergedSegment]],
    total_noise: int,
) -> None:
    total_before = sum(s["n_before"] for s in cell_stats)
    total_after = sum(s["n_after"] for s in cell_stats)

    print("\n=== 병합 전후 셀별 상세 ===\n")
    for st in cell_stats:
        rt, pg = st["rt"], st["pg"]
        print(f"[{vc.cell_short_label(rt, pg)}]")
        print(
            f"  병합 전: {st['n_before']}개 군집 + 노이즈 {st['n_noise']}건"
        )
        print(
            f"  병합 후: {st['n_after']}개 세그먼트 + 노이즈 {st['n_noise']}건"
        )
        print("  병합 상세:")
        segs = cell_segments[(rt, pg)]
        for seg in sorted(segs.values(), key=lambda s: s.n, reverse=True):
            parts = "+".join(f"군집{cid}({n})" for cid, n in sorted(seg.source_clusters))
            print(f"    {seg.segment_key} ← {parts} = {seg.n:,}건")
        print()

    print("=== 전체 요약 ===")
    print(f"  병합 전 총 군집: {total_before}개 (노이즈 제외)")
    print(f"  병합 후 총 세그먼트: {total_after}개")
    print(f"  노이즈(정보 부족형): {total_noise:,}건 유지")


def run() -> pd.DataFrame:
    if not HDBSCAN_CSV.is_file():
        raise FileNotFoundError(f"없음: {HDBSCAN_CSV}")

    assign = pd.read_csv(HDBSCAN_CSV, dtype={"recruitId": int})
    assign = assign.rename(columns={"cluster_id": "hdbscan_cluster_id"})

    attrs = load_attributes()
    df = assign.merge(attrs, on="recruitId", how="inner")

    print("=== HDBSCAN 병합 후처리 ===\n")
    print(f"입력: {HDBSCAN_CSV} ({len(df)}건)\n")

    all_frames: list[pd.DataFrame] = []
    cell_segments: dict[tuple[str, str], dict[tuple[str, str, str], MergedSegment]] = {}
    cell_merged_by_id: dict[tuple[str, str], dict[int, MergedSegment]] = {}
    cell_stats: list[dict] = []

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            cell_df = df[(df["recruitType"] == rt) & (df["payment_group"] == pg)].copy()
            if cell_df.empty:
                continue

            merged_df, segments, merged_by_id, n_before = merge_cell(cell_df)
            n_noise = int((merged_df["hdbscan_cluster_id"] == -1).sum())
            n_after = len(segments)

            non_noise_sizes = merged_df.loc[
                merged_df["merged_segment_id"] != -1, "merged_segment_id"
            ].value_counts()
            min_seg = int(non_noise_sizes.min()) if not non_noise_sizes.empty else None
            med_seg = float(non_noise_sizes.median()) if not non_noise_sizes.empty else None

            cell_segments[(rt, pg)] = segments
            cell_merged_by_id[(rt, pg)] = merged_by_id
            cell_stats.append({
                "rt": rt,
                "pg": pg,
                "label": vc.cell_label(rt, pg),
                "n_before": n_before,
                "n_after": n_after,
                "n_noise": n_noise,
                "min_seg": min_seg,
                "med_seg": med_seg,
            })
            all_frames.append(merged_df)

    all_df = pd.concat(all_frames, ignore_index=True)
    out_cols = [
        "recruitId",
        "recruitType",
        "payment_group",
        "hdbscan_cluster_id",
        "merged_segment_id",
        "segment_key",
    ]
    all_df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"병합 배정 저장: {OUT_CSV} ({len(all_df)}건)")

    total_noise = int((all_df["merged_segment_id"] == -1).sum())
    write_merged_report(all_df, cell_merged_by_id, cell_stats)
    print_merge_summary(cell_stats, cell_segments, total_noise)
    return all_df


if __name__ == "__main__":
    run()
