#!/usr/bin/env python3
"""model×n2 seg1 재분리 실험 — docs/v2.1_클러스터링_확정.md §7"""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v2_clustering_v21 import (
    DEFAULT_CLUSTER_CONFIG,
    ClusterRunConfig,
    load_phase3_frame,
    run,
)
from v2_segment_ops import COL_PLACE
from v2_tune_metrics import METRICS_TABLE_HEADER, format_metrics_row, pipeline_metrics

SCRATCH = ROOT / "data" / "v2" / "_scratch"
OUT_MD = ROOT / "docs" / "48_v2.1_model_n2_seg1_재분리.md"

VARIANTS: list[tuple[str, ClusterRunConfig]] = [
    ("K0 baseline", DEFAULT_CLUSTER_CONFIG),
    (
        "K0 + min20",
        replace(DEFAULT_CLUSTER_CONFIG, min_segment_size=20),
    ),
    (
        "model_n2 K×1.5",
        replace(
            DEFAULT_CLUSTER_CONFIG,
            cell_raw_k_scale={"model_n2": 1.5},
        ),
    ),
    (
        "model_n2 K×1.5 + min20",
        replace(
            DEFAULT_CLUSTER_CONFIG,
            cell_raw_k_scale={"model_n2": 1.5},
            min_segment_size=20,
        ),
    ),
    (
        "K0 + F-1 장소",
        replace(DEFAULT_CLUSTER_CONFIG, apply_f1_split=True),
    ),
]


def seg1_stats(df, phase_df) -> dict:
    m2 = df[(df.recruitType == "model") & (df.payment_group == "n2")]
    m2 = m2[m2["merged_segment_id"] != -1]
    merged = m2.merge(
        phase_df[["recruitId", COL_PLACE]], on="recruitId", how="left"
    )
    # largest non-snap portfolio-like seg
    sizes = merged.groupby(["merged_segment_id", "segment_key"]).size()
    if sizes.empty:
        return {"key": "-", "n": 0, "studio_pct": 0.0}
    top = sizes.sort_values(ascending=False)
    mid, sk = top.index[0]
    seg = merged[merged["merged_segment_id"] == mid]
    studio = (seg[COL_PLACE] == "스튜디오").mean() if len(seg) else 0
    return {"key": sk, "n": int(len(seg)), "studio_pct": round(studio * 100, 1)}


def main() -> None:
    phase_df, _, _ = load_phase3_frame()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, cfg in VARIANTS:
        slug = re.sub(r"[^\w\-]+", "_", label).strip("_")
        df, _ = run(
            cluster_config=cfg,
            out_csv=SCRATCH / f"seg1_{slug}.csv",
            report_path=SCRATCH / f"seg1_{slug}.txt",
            verbose=False,
            phase_df=phase_df,
        )
        m = pipeline_metrics(df, phase_df)
        s1 = seg1_stats(df, phase_df)
        rows.append({"label": label, **m, **s1})

    lines = [
        "# v2.1 model×n2 seg1 재분리 실험",
        "",
        "> 포트폴리오·경력무관 1,049건(K0) — 스튜디오 dom 분리 검증",
        "",
        METRICS_TABLE_HEADER,
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(format_metrics_row(r["label"], r))
    lines.extend(["", "## seg1 대형 블록 (스튜디오 dom)", ""])
    lines.append("| variant | seg key | n | studio% |")
    lines.append("|---|---|---:|---:|")
    for r in rows:
        lines.append(f"| {r['label']} | {r['key'][:35]} | {r['n']} | {r['studio_pct']}% |")

    best = max(
        rows,
        key=lambda r: (
            r["studio_pct"] >= 60,
            r["model_n2_max_pct"] <= 30,
            -r["micro_lt30"],
        ),
    )
    lines.extend([
        "",
        "## 결정",
        "",
        f"- **채택 후보:** `{best['label']}` — studio dom {best['studio_pct']}%, "
        f"m2 max {best['model_n2_max_pct']}%, micro {best['micro_lt30']}",
        "- model_n2만 K×1.5 → 스튜디오·포트폴리오 블록 분리 (K1 674+330 패턴 복원)",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    for r in rows:
        print(f"{r['label']}: m2max={r['model_n2_max_pct']}% studio={r['studio_pct']}% micro={r['micro_lt30']}")


if __name__ == "__main__":
    main()
