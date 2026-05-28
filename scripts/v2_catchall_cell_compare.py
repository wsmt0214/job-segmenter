#!/usr/bin/env python3
"""beauty×n2·photo×n2 catch-all 분할 실험 — docs/v2.1_클러스터링_확정.md §7"""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v2_clustering_v21 import DEFAULT_CLUSTER_CONFIG, ClusterRunConfig, load_phase3_frame, run
from v2_tune_metrics import METRICS_TABLE_HEADER, format_metrics_row, pipeline_metrics

SCRATCH = ROOT / "data" / "v2" / "_scratch"
OUT_MD = ROOT / "docs" / "49_v2.1_catchall_셀_분할.md"

BASE = replace(
    DEFAULT_CLUSTER_CONFIG,
    cell_raw_k_scale={"model_n2": 1.5},
    min_segment_size=20,
)

VARIANTS: list[tuple[str, ClusterRunConfig]] = [
    ("base (m2 K1.5 min20)", BASE),
    (
        "+ beauty_n2 K1.5",
        replace(BASE, cell_raw_k_scale={"model_n2": 1.5, "beauty_n2": 1.5}),
    ),
    (
        "+ photo_n2 K1.5",
        replace(BASE, cell_raw_k_scale={"model_n2": 1.5, "photo_n2": 1.5}),
    ),
    (
        "TUNE (m2+beauty+photo K1.5)",
        replace(
            BASE,
            cell_raw_k_scale={"model_n2": 1.5, "beauty_n2": 1.5, "photo_n2": 1.5},
        ),
    ),
]


def cell_max(df, rt, pg) -> tuple[int, float, str]:
    g = df[(df.recruitType == rt) & (df.payment_group == pg)]
    g = g[g["merged_segment_id"] != -1]
    if g.empty:
        return 0, 0.0, "-"
    sizes = g.groupby(["merged_segment_id", "segment_key"]).size()
    (mid, sk), n = max(sizes.items(), key=lambda x: x[1])
    return int(n), round(100 * n / len(g), 1), str(sk)


def main() -> None:
    phase_df, _, _ = load_phase3_frame()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, cfg in VARIANTS:
        slug = re.sub(r"[^\w\-]+", "_", label).strip("_")
        df, _ = run(
            cluster_config=cfg,
            out_csv=SCRATCH / f"catch_{slug}.csv",
            report_path=SCRATCH / f"catch_{slug}.txt",
            verbose=False,
            phase_df=phase_df,
        )
        m = pipeline_metrics(df, phase_df)
        bn, bp, bk = cell_max(df, "beauty", "n2")
        pn, pp, pk = cell_max(df, "photo", "n2")
        rows.append({**m, "label": label, "beauty_max_pct": bp, "photo_max_pct": pp, "beauty_key": bk, "photo_key": pk})

    lines = [
        "# v2.1 catch-all 셀 분할 실험",
        "",
        METRICS_TABLE_HEADER,
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(format_metrics_row(r["label"], r))
    lines.extend(["", "## 셀별 max 세그", "", "| variant | beauty×n2 max% | photo×n2 max% |", "|---|---:|---:|"])
    for r in rows:
        lines.append(f"| {r['label']} | {r['beauty_max_pct']}% | {r['photo_max_pct']}% |")

    best = min(rows, key=lambda r: (r["beauty_max_pct"] + r["photo_max_pct"], r["micro_lt30"]))
    lines.extend([
        "",
        "## 결정",
        "",
        f"- **채택:** `{best['label']}` — beauty max {best['beauty_max_pct']}%, photo max {best['photo_max_pct']}%",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    for r in rows:
        print(f"{r['label']}: catch={r['catch_all_cells']} beauty={r['beauty_max_pct']}% photo={r['photo_max_pct']}%")


if __name__ == "__main__":
    main()
