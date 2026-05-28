"""v2.1 TUNE 세그먼트별 구인글 JSON (급종별 · 셀 → 세그 → 글 ~30건)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from v2_cluster_segment_examples import (
    ASSIGN_CSV,
    CELL_LABELS,
    RT_LABELS,
    load_frame,
    load_recruit_meta,
    merge_key_display,
    segment_name,
)
from v2_operational_viability import grade_segment

OUT_JSON_DIR = Path(config.DATA_DIR) / "segment_catalog"
DEFAULT_POSTS_PER_SEGMENT = 30


def post_record(rid: int, rd: dict) -> dict:
    title = str(rd.get("title") or "").strip()
    content = str(rd.get("content") or "").strip()
    if title and content:
        text = f"{title}\n{content}"
    else:
        text = title or content
    return {
        "recruitId": rid,
        "text": text,
        "title": title,
        "content": content,
        "categories": list(rd.get("categories") or []),
    }


def sample_posts(
    seg: pd.DataFrame,
    recruit_data: dict[int, dict],
    cap: int,
) -> list[dict]:
    ordered = seg.sort_values("recruitId").head(cap)
    return [
        post_record(int(row.recruitId), recruit_data.get(int(row.recruitId), {}))
        for _, row in ordered.iterrows()
    ]


def build_json_for_type(
    df: pd.DataFrame,
    rt: str,
    recruit_data: dict[int, dict],
    *,
    posts_cap: int,
) -> dict:
    rt_df = df[df.recruitType == rt]
    payment_groups: dict = {}

    for pg in config.PAYMENT_GROUPS:
        cell_df = rt_df[rt_df.payment_group == pg]
        if cell_df.empty:
            continue
        cell_label = CELL_LABELS.get((rt, pg), f"{rt}×{pg}")
        segments: dict = {}
        seg_ids = sorted(
            cell_df["merged_segment_id"].unique(),
            key=lambda x: (x == -1, x),
        )
        cell_op = cell_df[cell_df.merged_segment_id != -1]
        cell_n = len(cell_op)

        for seg_id in seg_ids:
            seg_id = int(seg_id)
            seg = cell_df[cell_df.merged_segment_id == seg_id]
            n = len(seg)
            cap = min(posts_cap, n)
            g = grade_segment(rt, pg, seg_id, seg, cell_n=cell_n)
            segments[str(seg_id)] = {
                "merged_segment_id": seg_id,
                "segment_key": segment_name(seg, seg_id),
                "grade": g.grade,
                "merge_key": merge_key_display(seg, rt) if seg_id != -1 else None,
                "post_count": n,
                "posts_shown": cap,
                "posts": sample_posts(seg, recruit_data, cap),
            }

        payment_groups[pg] = {
            "cell_key": f"{pg}_{rt}",
            "label": cell_label,
            "cell_count": len(cell_df),
            "segments": segments,
        }

    return {
        "recruit_type": rt,
        "recruit_type_label": RT_LABELS[rt],
        "cluster_version": "v2.1-tune",
        "assignments_csv": str(ASSIGN_CSV.resolve()),
        "posts_per_segment_cap": posts_cap,
        "payment_groups": payment_groups,
    }


def write_json(catalog: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="v2.1 세그먼트 뷰 JSON 생성")
    ap.add_argument(
        "--posts-per-segment",
        type=int,
        default=DEFAULT_POSTS_PER_SEGMENT,
        help=f"세그당 글 수 상한 (기본 {DEFAULT_POSTS_PER_SEGMENT})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_JSON_DIR,
        help="JSON 출력 디렉터리",
    )
    args = ap.parse_args()
    cap = max(1, int(args.posts_per_segment))

    df = load_frame()
    recruit_ids = set(df["recruitId"].astype(int))
    recruit_data = load_recruit_meta(recruit_ids)

    out_dir = args.out_dir.resolve()

    for rt in config.RECRUIT_TYPES:
        catalog = build_json_for_type(df, rt, recruit_data, posts_cap=cap)
        json_path = out_dir / f"{rt}_segments_view.json"
        write_json(catalog, json_path)

        n_posts = sum(
            len(seg.get("posts") or [])
            for cell in catalog["payment_groups"].values()
            for seg in cell["segments"].values()
        )
        n_seg = sum(
            len(cell["segments"])
            for cell in catalog["payment_groups"].values()
        )
        print(f"[{rt}] segments={n_seg} posts={n_posts} → {json_path}")


if __name__ == "__main__":
    main()
