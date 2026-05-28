"""P3-스냅 Tier A/B — full vs tier1_only vs tier2_conditional"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from schema_v2 import (
    CLUSTERING_4DIM,
    P3_SNAP_TIER1_ONLY,
    P3_SNAP_TIER2_BODY_SNAP,
    P3_SNAP_TIER2_CONDITIONAL,
    P3_SNAP_TIER_DEFAULT,
    P3_SNAP_TIER_FULL,
    UNCLEAR_VALUE,
    clustering_6dim_feature_cols,
    load_schema,
)
from v2_clustering import load_type_map
from v2_clustering_v21 import INFO_POOR_THRESHOLD, load_phase3_frame, run
from v2_phase3_core import load_recruit_texts, apply_p3_corrections_phase3_frame

OUT_MD = ROOT / "docs" / "43_v2.1_P3스냅_Tier_AB.md"
PHASE3_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"

TIER_LABELS = {
    P3_SNAP_TIER_FULL: "A0 full (Tier1+Tier2 무조건)",
    P3_SNAP_TIER1_ONLY: "A Tier1 only",
    P3_SNAP_TIER2_BODY_SNAP: "C Tier2 본문스냅 (프로덕션)",
    P3_SNAP_TIER2_CONDITIONAL: "B Tier2+포폴",
}


def load_raw_frame() -> tuple[pd.DataFrame, dict[int, list[str]]]:
    schema = load_schema()
    cluster_cols = clustering_6dim_feature_cols(schema)
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
            row = {"recruitId": rid, "payment_group": r.get("payment_group")}
            for c in cluster_cols:
                row[c] = attrs.get(c, UNCLEAR_VALUE)
            rows.append(row)
    df = pd.DataFrame(rows)
    type_map = load_type_map()
    df["recruitType"] = df["recruitId"].map(type_map)
    return df, categories


def frame_pre_cluster_stats(frame: pd.DataFrame) -> dict:
    n = len(frame)
    scores = []
    for _, row in frame.iterrows():
        scores.append(
            sum(
                1
                for c in CLUSTERING_4DIM
                if str(row.get(c, UNCLEAR_VALUE)) != UNCLEAR_VALUE
            )
        )
    poor = sum(1 for s in scores if s < INFO_POOR_THRESHOLD)
    snap_fill = int((frame["촬영 주제"] == "스냅").sum())
    topic_u = int((frame["촬영 주제"] == UNCLEAR_VALUE).sum())
    m = frame[(frame["recruitType"] == "model") & (frame["payment_group"] == "n2")]
    return {
        "info_poor_pre": poor,
        "info_poor_pre_pct": round(poor / n * 100, 1),
        "snap_total": snap_fill,
        "topic_unclear": topic_u,
        "topic_unclear_pct": round(topic_u / n * 100, 1),
        "mn2_snap": int((m["촬영 주제"] == "스냅").sum()),
        "mn2_topic_unclear": int((m["촬영 주제"] == UNCLEAR_VALUE).sum()),
    }


def cluster_kpi(frame: pd.DataFrame) -> dict:
    df, results = run(phase_df=frame, verbose=False, out_csv="/tmp/p3_snap_tier_ab.csv")
    g = df[(df["recruitType"] == "model") & (df["payment_group"] == "n2")]
    sizes = (
        g[g["merged_segment_id"] != -1]
        .groupby("merged_segment_id")
        .size()
        .sort_values(ascending=False)
    )
    max_id = int(sizes.index[0])
    max_n = int(sizes.iloc[0])
    max_key = str(g[g["merged_segment_id"] == max_id]["segment_key"].iloc[0])
    mn2_poor = int((g["merged_segment_id"] == -1).sum())
    mn2_segs = next(
        r.n_final_segments
        for r in results
        if r.recruit_type == "model" and r.payment_group == "n2"
    )
    return {
        "info_poor": int((df["merged_segment_id"] == -1).sum()),
        "n_operating": sum(r.n_final_segments for r in results)
        + sum(1 for r in results if r.n_info_poor > 0),
        "model_n2_max": max_n,
        "model_n2_max_pct": round(max_n / len(g) * 100, 1),
        "model_n2_max_key": max_key,
        "model_n2_poor": mn2_poor,
        "model_n2_segs": mn2_segs,
        "kpi_30": max_n / len(g) <= 0.30,
    }


def run_ab() -> list[dict]:
    raw, categories = load_raw_frame()
    texts = load_recruit_texts(set(raw["recruitId"].astype(int)))
    rows: list[dict] = []
    for tier in (P3_SNAP_TIER_FULL, P3_SNAP_TIER1_ONLY, P3_SNAP_TIER2_CONDITIONAL):
        frame = apply_p3_corrections_phase3_frame(
            raw.drop(columns=["recruitType"]),
            categories,
            texts,
            apply_hair=True,
            apply_snap=True,
            apply_purpose=True,
            apply_place=True,
            snap_tier=tier,
        )
        frame["recruitType"] = raw["recruitType"]
        frame["payment_group"] = raw["payment_group"]
        pre = frame_pre_cluster_stats(frame)
        post = cluster_kpi(frame)
        rows.append({"tier": tier, "label": TIER_LABELS[tier], **pre, **post})
    return rows


def print_table(rows: list[dict]) -> None:
    print("\n=== P3-스냅 Tier A/B (P3-헤어·목적·장소 ON) ===\n")
    hdr = (
        f"{'variant':<28} {'pre poor':>9} {'snap':>6} "
        f"{'post poor':>10} {'mn2 max':>8} {'max%':>6} {'30%':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ok = "OK" if r["kpi_30"] else "NG"
        print(
            f"{r['label']:<28} {r['info_poor_pre']:>6} ({r['info_poor_pre_pct']:>4}%) "
            f"{r['snap_total']:>6} {r['info_poor']:>6} ({r['info_poor']/9941*100:>4.1f}%) "
            f"{r['model_n2_max']:>8} {r['model_n2_max_pct']:>5.1f}% {ok:>4}"
        )
        print(f"  max seg: {r['model_n2_max_key']}")
        print(
            f"  mn2 snap={r['mn2_snap']} topic_불명확={r['mn2_topic_unclear']} "
            f"mn2 poor={r['model_n2_poor']} segs={r['model_n2_segs']}"
        )


def write_doc(rows: list[dict]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# v2.1 P3-스냅 Tier A/B 실험",
        "",
        f"**생성:** {ts} · `scripts/v2_p3_snap_tier_ab.py`",
        "",
        "전제: P3-헤어 · P3-목적 · P3-장소 **ON** · K1·GW1·Fix C-1 동일",
        "",
        "## Tier 정의",
        "",
        "| ID | `snap_tier` | Tier2 (인물스냅) |",
        "|---|---|---|",
        f"| A0 | `{P3_SNAP_TIER_FULL}` | 카테고리만으로 → 스냅 (**현행**) |",
        f"| A | `{P3_SNAP_TIER1_ONLY}` | **OFF** — Tier1(일상·여행·우정·본문 스냅)만 |",
        f"| B | `{P3_SNAP_TIER2_CONDITIONAL}` | 본문 **스냅 키워드 OR 포트폴리오** 맥락 있을 때만 |",
        "",
        "## KPI",
        "",
        "| variant | pre poor | P3→스냅 | post poor | model×n2 max | max% | 30% | max seg |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for r in rows:
        ok = "✓" if r["kpi_30"] else "✗"
        lines.append(
            f"| {r['label']} | {r['info_poor_pre']} ({r['info_poor_pre_pct']}%) | "
            f"{r['snap_total']} | {r['info_poor']} | {r['model_n2_max']} | "
            f"{r['model_n2_max_pct']}% | {ok} | `{r['model_n2_max_key']}` |"
        )
    lines.extend(
        [
            "",
            "## model×n2 상세",
            "",
        ]
    )
    for r in rows:
        lines.append(f"### {r['label']}")
        lines.append("")
        lines.append(
            f"- mn2 스냅: {r['mn2_snap']} · 주제 불명확: {r['mn2_topic_unclear']} · "
            f"정보 부족: {r['model_n2_poor']} · 세그: {r['model_n2_segs']}"
        )
        lines.append("")

    a0 = next(x for x in rows if x["tier"] == P3_SNAP_TIER_FULL)
    a = next(x for x in rows if x["tier"] == P3_SNAP_TIER1_ONLY)
    b = next(x for x in rows if x["tier"] == P3_SNAP_TIER2_CONDITIONAL)
    tier2_only = a0["snap_total"] - a["snap_total"]
    lines.extend(
        [
            "## 해석",
            "",
            f"- Tier2-only 스냅 (A0−A): **{tier2_only}건** — 인물스냅 무조건 매핑분",
            f"- A vs A0 max: {a0['model_n2_max_pct']}% → **{a['model_n2_max_pct']}%**",
            f"- B vs A0 max: {a0['model_n2_max_pct']}% → **{b['model_n2_max_pct']}%**",
            "",
            "## 재현",
            "",
            "```bash",
            "./venv/bin/python scripts/v2_p3_snap_tier_ab.py --write-doc",
            "```",
            "",
            "코드: `apply_snap_topic_correction(..., snap_tier=)` · "
            "`load_phase3_frame(p3_snap_tier=)`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n문서 저장: {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P3-스냅 Tier A/B")
    parser.add_argument("--write-doc", action="store_true", help="docs/43 MD 생성")
    args = parser.parse_args()
    rows = run_ab()
    print_table(rows)
    if args.write_doc:
        write_doc(rows)


if __name__ == "__main__":
    main()
