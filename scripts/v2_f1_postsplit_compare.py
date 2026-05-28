"""Fix F-1 / F-1-P post-split 비교 — 현행 P3 full 베이스"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v2_clustering_v21 import (
    DEFAULT_CLUSTER_CONFIG,
    ClusterRunConfig,
    load_phase3_frame,
    pipeline_metrics,
    run,
)

OUT_MD = ROOT / "docs" / "44_v2.1_F1_postsplit_비교.md"

VARIANTS: list[tuple[str, ClusterRunConfig]] = [
    ("baseline", DEFAULT_CLUSTER_CONFIG),
    (
        "F-1 장소",
        ClusterRunConfig(apply_f1_split=True),
    ),
    (
        "F-1-P 목적",
        ClusterRunConfig(apply_f1p_purpose_split=True),
    ),
    (
        "F-1 + F-1-P",
        ClusterRunConfig(apply_f1_split=True, apply_f1p_purpose_split=True),
    ),
]


def model_n2_detail(df) -> dict:
    g = df[(df["recruitType"] == "model") & (df["payment_group"] == "n2")]
    g = g[g["merged_segment_id"] != -1]
    if g.empty:
        return {"max": 0, "max_pct": 0.0, "max_key": "-", "top3": []}
    sizes = g.groupby(["merged_segment_id", "segment_key"]).size().sort_values(
        ascending=False
    )
    top3 = [
        (str(sk), int(n), round(n / len(g) * 100, 1))
        for (_, sk), n in sizes.head(3).items()
    ]
    (mid, sk), max_n = next(iter(sizes.items()))
    return {
        "max": int(max_n),
        "max_pct": round(max_n / len(g) * 100, 1),
        "max_key": str(sk),
        "top3": top3,
        "n_segs": int(g["merged_segment_id"].nunique()),
    }


def run_variant(label: str, config: ClusterRunConfig) -> dict:
    phase_df, _, _ = load_phase3_frame()
    slug = re.sub(r"[^\w\-]+", "_", label).strip("_")
    scratch = ROOT / "data" / "v2" / "_scratch" / f"f1_compare_{slug}.csv"
    df, results = run(
        phase_df=phase_df,
        verbose=False,
        cluster_config=config,
        out_csv=scratch,
        report_path=scratch.with_suffix(".txt"),
    )
    m = pipeline_metrics(df, results)
    mn2 = model_n2_detail(df)
    splits = []
    if config.apply_f1_split:
        splits.append("F-1")
    if config.apply_f1p_purpose_split:
        splits.append("F-1-P")
    return {
        "label": label,
        "splits": splits,
        **m,
        **{f"mn2_{k}": v for k, v in mn2.items() if k != "top3"},
        "mn2_top3": mn2["top3"],
        "kpi_30": m["model_n2_max_pct"] <= 30.0,
    }


def print_table(rows: list[dict]) -> None:
    print("\n=== F-1 post-split 비교 (P3 full · K1·GW1) ===\n")
    hdr = f"{'variant':<16} {'poor':>6} {'mn2 max':>8} {'max%':>6} {'30%':>4} {'segs':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ok = "OK" if r["kpi_30"] else "NG"
        print(
            f"{r['label']:<16} {r['n_poor']:>6} {r['mn2_max']:>8} "
            f"{r['mn2_max_pct']:>5.1f}% {ok:>4} {r['mn2_n_segs']:>5}"
        )
        print(f"  max: {r['mn2_max_key']}")
        for sk, n, pct in r["mn2_top3"]:
            print(f"    {n:4d} ({pct:4.1f}%) {sk}")


def write_doc(rows: list[dict]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# v2.1 Fix F-1 / F-1-P post-split 비교",
        "",
        f"**생성:** {ts} · `scripts/v2_f1_postsplit_compare.py`",
        "",
        "전제: P3-헤어·스냅(full Tier)·목적·장소 ON · Fix C-1 · K1·GW1 · **클러스터 후** split",
        "",
        "## 규칙",
        "",
        "| Fix | 조건 | split |",
        "|---|---|---|",
        "| **F-1** (장소) | n≥1000 · 장소 불명확≥25% · 장소_dom≠혼재 | 장소 명시 / 불명확 |",
        "| **F-1-P** (목적) | n≥1000 · **주제=스냅** · 포트폴리오≥25% · purpose_dom≠포트폴리오 | 포트폴리오 / 그 외 |",
        "",
        "## KPI",
        "",
        "| variant | 정보 부족 | model×n2 max | max% | 30% | mn2 세그 | max seg |",
        "|---|---:|---:|---:|:---:|---:|---|",
    ]
    for r in rows:
        ok = "✓" if r["kpi_30"] else "✗"
        lines.append(
            f"| {r['label']} | {r['n_poor']} | {r['mn2_max']} | "
            f"{r['mn2_max_pct']}% | {ok} | {r['mn2_n_segs']} | `{r['mn2_max_key']}` |"
        )
    lines.extend(["", "## model×n2 상위 3세그", ""])
    for r in rows:
        lines.append(f"### {r['label']}")
        lines.append("")
        for sk, n, pct in r["mn2_top3"]:
            lines.append(f"- {n:,} ({pct}%) — `{sk}`")
        lines.append("")

    base = rows[0]
    f1p = next(r for r in rows if r["label"] == "F-1-P 목적")
    lines.extend(
        [
            "## 해석",
            "",
            f"- **baseline** max {base['mn2_max_pct']}% → **F-1-P** max **{f1p['mn2_max_pct']}%**",
            f"- F-1(장소): seg0 장소=혼재 → **split 0건** 기대 (실행 로그 확인)",
            "- F-1-P: 스냅 bulk를 **포트폴리오 / 비포트폴리오** 로 쪼갬 — P3-스냅 유지",
            "",
            "## 재현",
            "",
            "```bash",
            "./venv/bin/python scripts/v2_f1_postsplit_compare.py --write-doc",
            "```",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n문서 저장: {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-doc", action="store_true")
    args = parser.parse_args()
    rows = [run_variant(label, cfg) for label, cfg in VARIANTS]
    print_table(rows)
    if args.write_doc:
        write_doc(rows)


if __name__ == "__main__":
    main()
