"""v2.1 클러스터링 4버전 비교 실험 — Gower 가중 × 병합키 조합"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
import pymysql
from schema_v2 import CLUSTERING_4DIM, UNCLEAR_VALUE
from v2_clustering_v21 import (
    COL_PLACE,
    COL_PURPOSE,
    COL_TOPIC,
    COL_TREATMENT,
    ClusterRunConfig,
    load_phase3_frame,
    pipeline_metrics,
    run,
)
import v2_clustering as vc

EXPERIMENTS_DIR = Path(config.V2_DATA_DIR) / "_experiments"
DOCS_DIR = ROOT / "docs"
EXPERIMENTS_DOCS_DIR = Path(config.V2_DATA_DIR) / "cluster_experiments"
MAX_EXAMPLES = 5

RT_LABELS = {"model": "모델", "beauty": "뷰티", "photo": "포토"}


def _cell_label(recruit_type: str, payment_group: str) -> str:
    rt = RT_LABELS.get(recruit_type, recruit_type)
    pg = config.PAYMENT_GROUPS.get(payment_group, payment_group)
    return f"{rt} × {pg}"


CELL_LABELS = {
    (rt, pg): _cell_label(rt, pg)
    for rt in ("model", "beauty", "photo")
    for pg in ("n2", "n3", "pay")
}
DIM_COLS = list(CLUSTERING_4DIM)


@dataclass(frozen=True)
class ExperimentVariant:
    slug: str
    label: str
    description: str
    cluster_config: ClusterRunConfig


VARIANTS: list[ExperimentVariant] = [
    ExperimentVariant(
        slug="V1_w1111_3axis",
        label="V1 — 균등 가중 + 3축 병합",
        description=(
            "Gower [1,1,1,1] — 4축 동일 가중. "
            "병합키 3축(목적·장소·주제), 시술은 병합에 미포함. "
            "주제·시술 변별이 약한 baseline"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 1.0, 1.0),
            include_treatment_in_merge=False,
        ),
    ),
    ExperimentVariant(
        slug="V2_w1122_3axis",
        label="V2 — 주제·시술 2배 + 3축 병합",
        description=(
            "Gower [1,1,2,2] — Ward 거리에서 주제·시술 2배. "
            "병합키는 3축만 — raw 군집은 시술 반영, 병합 단계에서 시술 무시"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 2.0, 2.0),
            include_treatment_in_merge=False,
        ),
    ),
    ExperimentVariant(
        slug="V3_w1122_4axis",
        label="V3 — 주제·시술 2배 + 4축 병합 (현재 후보)",
        description=(
            "Gower [1,1,2,2] + model/beauty 4축 병합(시술 dominant 포함). "
            "구 Gower 가중 실험 보관"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 2.0, 2.0),
            include_treatment_in_merge=True,
        ),
    ),
    ExperimentVariant(
        slug="V4_w1133_4axis",
        label="V4 — 주제·시술 3배 + 4축 병합",
        description=(
            "Gower [1,1,3,3] — 주제·시술 변별 강화. "
            "4축 병합으로 펌/컷/웨딩 등 시술별 세그 분리 극대화 시도"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 3.0, 3.0),
            include_treatment_in_merge=True,
        ),
    ),
]

# 밀도 강화 — V3(4축) 대비 coarse 생략·dominant↑·raw K↑
DENSITY_VARIANTS: list[ExperimentVariant] = [
    ExperimentVariant(
        slug="V5_dense_nocoarse",
        label="V5 — 4축 + coarse 병합 생략",
        description=(
            "V3와 동일 Gower·4축이나 2차 coarse 병합 생략. "
            "주제 혼재 세그를 목적·장소로 뭉개지 않아 셀 내 동질성↑"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 2.0, 2.0),
            include_treatment_in_merge=True,
            skip_coarse_merge=True,
        ),
    ),
    ExperimentVariant(
        slug="V6_dense_dom70",
        label="V6 — 4축 + coarse 생략 + dominant 70%",
        description=(
            "V5 + merge dominant 70%. "
            "60% 미만 축은 혼재 처리 → 병합키가 더 보수적"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 2.0, 2.0),
            include_treatment_in_merge=True,
            skip_coarse_merge=True,
            dominant_threshold=0.70,
        ),
    ),
    ExperimentVariant(
        slug="V7_dense_k24",
        label="V7 — 4축 + coarse 생략 + raw K×1.5",
        description=(
            "V5 + Ward raw K 1.5배(model×n2 16→24). "
            "초기 군집을 더 잘게 만든 뒤 4축 병합만 적용"
        ),
        cluster_config=ClusterRunConfig(
            gower_weights=(1.0, 1.0, 2.0, 2.0),
            include_treatment_in_merge=True,
            skip_coarse_merge=True,
            raw_k_scale=1.5,
        ),
    ),
]

# V7 베이스 — raw K×1.5 + 4축 + coarse OFF
V7_BASE_KW = dict(
    gower_weights=(1.0, 1.0, 2.0, 2.0),
    include_treatment_in_merge=True,
    skip_coarse_merge=True,
    raw_k_scale=1.5,
)


def _v7_config(**overrides) -> ClusterRunConfig:
    base = dict(V7_BASE_KW)
    base.update(overrides)
    return ClusterRunConfig(**base)


EXTENDED_DENSITY_VARIANTS: list[ExperimentVariant] = [
    ExperimentVariant(
        slug="V7_dense_k24",
        label="V7 — raw K×1.5 + 4축 + coarse OFF",
        description="밀도 강화 기준 — Ward K 1.5배, 4축 병합, coarse 생략",
        cluster_config=_v7_config(),
    ),
    ExperimentVariant(
        slug="V8_dense_k32",
        label="V8 — V7 + raw K×2",
        description="model×n2 Ward K 16→32. 세그 수·동질성 추가 개선 시도",
        cluster_config=_v7_config(raw_k_scale=2.0),
    ),
    ExperimentVariant(
        slug="V9_dense_w1144",
        label="V9 — V7 + Gower [1,1,4,4]",
        description="주제·시술 Gower 거리 4배 — 스냅/펌/컷 변별 강화",
        cluster_config=_v7_config(gower_weights=(1.0, 1.0, 4.0, 4.0)),
    ),
    ExperimentVariant(
        slug="V10_dense_f1strong",
        label="V10 — V7 + F-1 장소 split 강화",
        description=(
            "F-1 임계 완화: 최소 400건·장소불명확 18%·B군 20건. "
            "장소미기재·혼재 장소 대형 세그 추가 분리"
        ),
        cluster_config=_v7_config(
            split_min_segment=400,
            split_min_unclear_ratio=0.18,
            split_b_min_size=20,
        ),
    ),
]

# V11 — V10 + 기타헤어 후처리 + raw K×2 + F-1 강화
V11_VARIANT = ExperimentVariant(
    slug="V11_generic_hair_k2_f1",
    label="V11 — V10 + 기타헤어 + K×2 + F-1 강화",
    description=(
        "Phase3 후처리: 헤어+시술불명확 → 기타 헤어. "
        "클러스터: V10 F-1 베이스 + raw K×2 + F-1 추가 강화(300/15%/B≥15). "
        "Gower [1,1,2,2], 4축 병합, coarse OFF"
    ),
    cluster_config=_v7_config(
        raw_k_scale=2.0,
        split_min_segment=300,
        split_min_unclear_ratio=0.15,
        split_b_min_size=15,
    ),
)

BASELINE_FOR_DENSITY = ExperimentVariant(
    slug="V3_w1122_4axis",
    label="V3 — 기준 (coarse 병합 O)",
    description="비교 기준 — 현재 후보",
    cluster_config=ClusterRunConfig(
        gower_weights=(1.0, 1.0, 2.0, 2.0),
        include_treatment_in_merge=True,
    ),
)


def compute_purity(all_df: pd.DataFrame, phase_df: pd.DataFrame) -> dict[str, float]:
    """세그먼트 동질성 — 4축 top1 비율의 가중 평균 (건수 가중)"""
    attrs = phase_df[DIM_COLS + ["recruitId"]].drop_duplicates(subset="recruitId")
    merged = all_df.merge(attrs, on="recruitId", how="left")
    op = merged[merged["merged_segment_id"] != -1]
    if op.empty:
        return {"avg_purity": 0.0, "model_n2_purity": 0.0}

    def weighted_purity(sub: pd.DataFrame) -> float:
        total = 0.0
        weight = 0
        for _, seg in sub.groupby("merged_segment_id"):
            n = len(seg)
            if n == 0:
                continue
            axis_tops = []
            for dim in DIM_COLS:
                vc = seg[dim].value_counts()
                axis_tops.append(float(vc.iloc[0] / n))
            total += sum(axis_tops) / len(DIM_COLS) * n
            weight += n
        return total / weight if weight else 0.0

    m2 = op[(op["recruitType"] == "model") & (op["payment_group"] == "n2")]
    return {
        "avg_purity": weighted_purity(op),
        "model_n2_purity": weighted_purity(m2) if not m2.empty else 0.0,
    }


def load_recruit_texts(recruit_ids: list[int]) -> dict[int, dict[str, str]]:
    """DB title/content + phase1 카테고리"""
    recruit_data: dict[int, dict[str, str]] = {}
    phase1_cats: dict[int, list[str]] = {}
    with open(Path(config.V2_DATA_DIR) / "phase1_with_category.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            phase1_cats[int(r["recruitId"])] = r.get("categories") or []

    conn = pymysql.connect(**config.DB_CONFIG)
    cur = conn.cursor()
    batch_size = 2000
    for i in range(0, len(recruit_ids), batch_size):
        batch = recruit_ids[i : i + batch_size]
        ph = ",".join(["%s"] * len(batch))
        cur.execute(
            f"SELECT {config.JOB_POSTING_ID_COL}, {config.TITLE_COL}, {config.CONTENT_COL} "
            f"FROM {config.JOB_POSTINGS_TABLE} WHERE {config.JOB_POSTING_ID_COL} IN ({ph})",
            batch,
        )
        for rid, title, content in cur.fetchall():
            rid = int(rid)
            cats = phase1_cats.get(rid, [])
            recruit_data[rid] = {
                "title": (title or "").strip(),
                "content": (content or "").strip(),
                "categories": ", ".join(cats[:3]) if cats else "-",
            }
    conn.close()
    return recruit_data


def summarize(text: str, n: int = 100) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t[:n] + ("..." if len(t) > n else "")


def dim_distribution(seg_df: pd.DataFrame, dim: str, top_n: int = 4) -> str:
    total = len(seg_df)
    if total == 0:
        return "-"
    vc = seg_df[dim].value_counts().head(top_n)
    return ", ".join(f"{v} {c/total*100:.0f}%" for v, c in vc.items())


def build_review_md(
    variant: ExperimentVariant,
    all_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    recruit_data: dict[int, dict[str, str]],
) -> str:
    attrs = phase_df[["recruitId"] + DIM_COLS].drop_duplicates(subset="recruitId")
    merged = all_df.merge(attrs, on="recruitId", how="left")
    missing = [c for c in DIM_COLS if c not in merged.columns]
    if missing:
        raise KeyError(f"Phase3 속성 컬럼 누락: {missing}")
    metrics = pipeline_metrics(all_df, [])
    purity = compute_purity(all_df, phase_df)
    cfg = variant.cluster_config

    lines: list[str] = [
        f"# v2.1 클러스터링 검토 — {variant.slug}",
        "",
        f"> **{variant.label}**",
        "",
        variant.description,
        "",
        "## KPI 요약",
        "",
        "| 지표 | 값 |",
        "|---|---:|",
        f"| 운영 세그먼트 | {metrics['n_segments']} |",
        f"| 정보부족형 (-1) | {metrics['n_poor']} |",
        f"| model×n2 최대 세그 | {metrics['model_n2_max']} ({metrics['model_n2_max_pct']:.1f}%) |",
        f"| **4축 평균 동질성** | **{purity['avg_purity']*100:.1f}%** |",
        f"| model×n2 동질성 | {purity['model_n2_purity']*100:.1f}% |",
        f"| 혼재·혼재 이름 | {metrics['n_hybrid_hybrid']} |",
        f"| 불명확 포함 이름 | {metrics['n_unclear_dom']} |",
        "",
        f"Gower: `{cfg.gower_weights}` · 병합: `{'4축' if cfg.include_treatment_in_merge else '3축'}` · "
        f"dominant: `{cfg.dominant_threshold:.0%}` · coarse: `{'OFF' if cfg.skip_coarse_merge else 'ON'}` · "
        f"raw K×{cfg.raw_k_scale}",
        "",
        f"F-1 split: 최소 {cfg.split_min_segment}건 · 장소불명확 ≥{cfg.split_min_unclear_ratio:.0%} · "
        f"B군 ≥{cfg.split_b_min_size}건",
        "",
        "---",
        "",
    ]

    for rt in config.RECRUIT_TYPES:
        lines.append(f"## {RT_LABELS[rt]} ({rt})")
        lines.append("")
        for pg in config.PAYMENT_GROUPS:
            cell = merged[(merged.recruitType == rt) & (merged.payment_group == pg)]
            if cell.empty:
                continue
            label = CELL_LABELS[(rt, pg)]
            op = cell[cell.merged_segment_id != -1]
            n_poor = int((cell.merged_segment_id == -1).sum())
            lines.append(f"### {label} ({len(cell):,}건 · 정보 부족 {n_poor})")
            lines.append("")
            lines.append("| seg | 이름 | 건수 | 비율 | 목적 | 장소 | 주제 | 시술 |")
            lines.append("|---:|---|---:|---:|---|---|---|---|")

            for sid in sorted(cell.merged_segment_id.unique()):
                if sid == -1:
                    continue
                seg = cell[cell.merged_segment_id == sid]
                name = seg.segment_key.iloc[0]
                n = len(seg)
                pct = n / len(cell) * 100
                lines.append(
                    f"| {sid} | {name} | {n} | {pct:.1f}% | "
                    f"{dim_distribution(seg, COL_PURPOSE, 2)} | "
                    f"{dim_distribution(seg, COL_PLACE, 2)} | "
                    f"{dim_distribution(seg, COL_TOPIC, 2)} | "
                    f"{dim_distribution(seg, COL_TREATMENT, 2)} |"
                )
            lines.append("")

            for sid in sorted(cell.merged_segment_id.unique()):
                seg = cell[cell.merged_segment_id == sid]
                name = seg.segment_key.iloc[0]
                n = len(seg)
                lines.append(f"#### 세그먼트 {sid}: {name} ({n}건)")
                lines.append("")
                if sid == -1:
                    lines.append("_정보 부족형 — 예시 생략 가능_")
                    lines.append("")
                    continue

                lines.append(
                    f"- **목적** {dim_distribution(seg, COL_PURPOSE)}  "
                    f"- **장소** {dim_distribution(seg, COL_PLACE)}  "
                    f"- **주제** {dim_distribution(seg, COL_TOPIC)}  "
                    f"- **시술** {dim_distribution(seg, COL_TREATMENT)}"
                )
                lines.append("")
                lines.append("| # | recruitId | 카테고리 | 제목 | 4축 | 내용 요약 |")
                lines.append("|---:|---:|---|---|---|---|")

                for i, (_, row) in enumerate(
                    seg.sort_values("recruitId").head(MAX_EXAMPLES).iterrows(), 1
                ):
                    rid = int(row.recruitId)
                    rd = recruit_data.get(rid, {})
                    axes = " · ".join(
                        str(row.get(c, UNCLEAR_VALUE))[:8] for c in DIM_COLS
                    )
                    lines.append(
                        f"| {i} | {rid} | {rd.get('categories', '-')} | "
                        f"{rd.get('title', '')} | {axes} | {summarize(rd.get('content', ''))} |"
                    )
                lines.append("")

    return "\n".join(lines)


def build_summary_md(
    variant_results: list[tuple[ExperimentVariant, pd.DataFrame]],
    *,
    title: str = "v2.1 클러스터링 비교 요약",
    summary_filename: str = "v2.1_클러스터링_비교_요약.md",
) -> str:
    variants = [v for v, _ in variant_results]
    lines = [
        f"# {title}",
        "",
        "> 검토용 — 버전 선택 후 RF 재학습·프로덕션 반영",
        "",
        "## 버전 정의",
        "",
        "| 버전 | Gower | 병합 | coarse | raw K× | F-1 min | 설명 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for v in variants:
        c = v.cluster_config
        merge = "4축" if c.include_treatment_in_merge else "3축"
        coarse = "OFF" if c.skip_coarse_merge else "ON"
        f1 = f"{c.split_min_segment}/{c.split_min_unclear_ratio:.0%}"
        lines.append(
            f"| [{v.slug}](./v2.1_클러스터링_검토_{v.slug}.md) | "
            f"{c.gower_weights} | {merge} | {coarse} | {c.raw_k_scale} | {f1} | "
            f"{v.description[:40]}... |"
        )

    slug_list = [v.slug for v in variants]
    lines.extend([
        "",
        "## KPI 비교",
        "",
        "| 지표 | " + " | ".join(slug_list) + " |",
        "|---|" + "|".join(["---:"] * len(variants)) + "|",
    ])

    rows = [
        ("운영 세그먼트", "n_segments", "d"),
        ("정보부족형", "n_poor", "d"),
        ("4축 평균 동질성", "avg_purity", "pct"),
        ("model×n2 동질성", "model_n2_purity", "pct"),
        ("model×n2 최대 세그", "model_n2_max", "d"),
        ("model×n2 최대 %", "model_n2_max_pct", "pct2"),
    ]
    metrics_by_slug = {}
    purity_by_slug = {}
    phase_df, _, _ = load_phase3_frame()
    for v, df in variant_results:
        metrics_by_slug[v.slug] = pipeline_metrics(df, [])
        purity_by_slug[v.slug] = compute_purity(df, phase_df)

    for label, key, fmt in rows:
        vals = []
        for v in variants:
            if key in ("avg_purity", "model_n2_purity"):
                m = purity_by_slug[v.slug][key] * 100
            else:
                m = metrics_by_slug[v.slug][key]
            if fmt == "pct":
                vals.append(f"{m:.1f}%")
            elif fmt == "pct2":
                vals.append(f"{m:.1f}%")
            else:
                vals.append(str(int(m)))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.extend(["", f"## {_cell_label('model', 'n2')} 세그먼트 목록", ""])
    for v, df in variant_results:
        sub = df[
            (df.recruitType == "model")
            & (df.payment_group == "n2")
            & (df.merged_segment_id != -1)
        ]
        lines.append(f"### {v.slug}")
        lines.append("")
        for sid, g in sub.groupby("merged_segment_id"):
            lines.append(f"- **{sid}** {g.segment_key.iloc[0]} ({len(g)}건)")
        lines.append("")

    lines.extend([
        "",
        "## 검토 체크리스트",
        "",
        "- [ ] 세그먼트 이름이 실제 글 특성(펌/컷/웨딩/스냅 등)과 맞는가",
        "- [ ] 같은 이름인데 내용이 크게 다른 세그가 없는가",
        "- [ ] model×n2 최대 세그 비율 30% 미만인가",
        "- [ ] 혼재·불명확 이름 세그가 없는가",
        "- [ ] 세그 수가 마케터 운영에 적절한가 (30~45개)",
        "",
        "## 산출물 경로",
        "",
        "| 버전 | CSV | MD |",
        "|---|---|---|",
    ])
    for v in variants:
        csv_path = EXPERIMENTS_DIR / f"cluster_assignments_{v.slug}.csv"
        md_path = EXPERIMENTS_DOCS_DIR / f"v2.1_클러스터링_검토_{v.slug}.md"
        lines.append(f"| {v.slug} | `{csv_path.relative_to(ROOT)}` | `{md_path.relative_to(ROOT)}` |")

    return "\n".join(lines)


def run_variant_batch(
    variants: list[ExperimentVariant],
    phase_df: pd.DataFrame,
    recruit_data: dict[int, dict[str, str]],
    *,
    cluster_phase_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> list[tuple[ExperimentVariant, pd.DataFrame]]:
    """cluster_phase_df — 클러스터링 입력용 (None이면 phase_df와 동일)"""
    cluster_input = cluster_phase_df if cluster_phase_df is not None else phase_df
    results: list[tuple[ExperimentVariant, pd.DataFrame]] = []
    for i, variant in enumerate(variants, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(variants)}] {variant.slug}")
        print(f"{'='*60}")
        out_csv = EXPERIMENTS_DIR / f"cluster_assignments_{variant.slug}.csv"
        all_df, _ = run(
            cluster_config=variant.cluster_config,
            out_csv=out_csv,
            report_path=EXPERIMENTS_DIR / f"marketer_review_{variant.slug}.txt",
            verbose=verbose,
            phase_df=cluster_input,
        )
        results.append((variant, all_df))
        md_path = EXPERIMENTS_DOCS_DIR / f"v2.1_클러스터링_검토_{variant.slug}.md"
        md_path.write_text(
            build_review_md(variant, all_df, phase_df, recruit_data),
            encoding="utf-8",
        )
        print(f"검토 MD: {md_path}")
    return results


def run_experiments(variants: list[ExperimentVariant] | None = None) -> None:
    variants = variants or VARIANTS
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    phase_df, _, _ = load_phase3_frame()
    recruit_ids = phase_df["recruitId"].astype(int).tolist()
    print("구인글 텍스트 로드...")
    recruit_data = load_recruit_texts(recruit_ids)

    variant_results = run_variant_batch(variants, phase_df, recruit_data)

    summary_path = EXPERIMENTS_DOCS_DIR / "v2.1_클러스터링_비교_요약.md"
    summary_path.write_text(build_summary_md(variant_results), encoding="utf-8")
    print(f"\n비교 요약: {summary_path}")


def run_density_experiments() -> None:
    """V3 기준 + V5~V7 밀도 강화 비교"""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    phase_df, _, _ = load_phase3_frame()
    recruit_data = load_recruit_texts(phase_df["recruitId"].astype(int).tolist())

    # V3는 이미 있으면 CSV 로드, 없으면 실행
    variants = [BASELINE_FOR_DENSITY] + DENSITY_VARIANTS
    results: list[tuple[ExperimentVariant, pd.DataFrame]] = []
    for v in variants:
        csv_path = EXPERIMENTS_DIR / f"cluster_assignments_{v.slug}.csv"
        if v.slug == "V3_w1122_4axis" and csv_path.is_file():
            print(f"기존 CSV 사용: {csv_path}")
            results.append((v, pd.read_csv(csv_path, dtype={"recruitId": int})))
            md_path = EXPERIMENTS_DOCS_DIR / f"v2.1_클러스터링_검토_{v.slug}.md"
            if not md_path.is_file():
                md_path.write_text(
                    build_review_md(v, results[-1][1], phase_df, recruit_data),
                    encoding="utf-8",
                )
            continue
        batch = run_variant_batch([v], phase_df, recruit_data)
        results.extend(batch)

    summary_path = EXPERIMENTS_DOCS_DIR / "v2.1_클러스터링_밀도_비교_요약.md"
    summary_path.write_text(
        build_summary_md(
            results,
            title="v2.1 클러스터링 밀도 강화 비교 (V3 vs V5~V7)",
            summary_filename="v2.1_클러스터링_밀도_비교_요약.md",
        ),
        encoding="utf-8",
    )
    print(f"\n밀도 비교 요약: {summary_path}")
    print("밀도 실험 완료 — 검토 후 채택 버전을 알려주세요.")


def _load_or_run_variant(
    v: ExperimentVariant,
    phase_df: pd.DataFrame,
    recruit_data: dict[int, dict[str, str]],
    *,
    force: bool = False,
) -> tuple[ExperimentVariant, pd.DataFrame]:
    csv_path = EXPERIMENTS_DIR / f"cluster_assignments_{v.slug}.csv"
    md_path = EXPERIMENTS_DOCS_DIR / f"v2.1_클러스터링_검토_{v.slug}.md"
    if csv_path.is_file() and not force:
        print(f"기존 CSV 사용: {csv_path}")
        df = pd.read_csv(csv_path, dtype={"recruitId": int})
        if not md_path.is_file():
            md_path.write_text(
                build_review_md(v, df, phase_df, recruit_data), encoding="utf-8"
            )
        return v, df
    return run_variant_batch([v], phase_df, recruit_data)[0]


def run_extended_density_experiments(*, force: bool = False) -> None:
    """V7~V10 밀도 확장 실험 (K×2, w1144, F-1 강화)"""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    phase_df, _, _ = load_phase3_frame()
    recruit_data = load_recruit_texts(phase_df["recruitId"].astype(int).tolist())

    results: list[tuple[ExperimentVariant, pd.DataFrame]] = []
    for v in EXTENDED_DENSITY_VARIANTS:
        if v.slug == "V7_dense_k24" and not force:
            results.append(_load_or_run_variant(v, phase_df, recruit_data, force=False))
        else:
            batch = run_variant_batch([v], phase_df, recruit_data)
            results.extend(batch)

    summary_path = EXPERIMENTS_DOCS_DIR / "v2.1_클러스터링_밀도확장_비교_요약.md"
    summary_path.write_text(
        build_summary_md(
            results,
            title="v2.1 클러스터링 밀도 확장 비교 (V7~V10)",
            summary_filename="v2.1_클러스터링_밀도확장_비교_요약.md",
        ),
        encoding="utf-8",
    )
    print(f"\n밀도 확장 비교: {summary_path}")


def build_v11_comparison_md(
    v11_df: pd.DataFrame,
    phase_generic: pd.DataFrame,
    stats_applied: int,
) -> str:
    """V10 / V10+기타헤어 / V11 KPI 비교"""
    rows: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

    v10_csv = EXPERIMENTS_DIR / "cluster_assignments_V10_dense_f1strong.csv"
    gh_csv = EXPERIMENTS_DIR / "cluster_assignments_V10_generic_hair.csv"
    phase_baseline, _, _ = load_phase3_frame()

    if v10_csv.is_file():
        rows.append(("V10 baseline", pd.read_csv(v10_csv), phase_baseline))
    if gh_csv.is_file():
        rows.append(("V10 + 기타헤어", pd.read_csv(gh_csv), phase_generic))
    rows.append(("**V11 (채택 후보)**", v11_df, phase_generic))

    lines = [
        "# v2.1 V11 비교 요약 — 기타헤어 + 밀도 강화",
        "",
        f"> **V11** = V10 + `기타 헤어` 후처리({stats_applied}건) + raw K×2 + F-1(300/15%/B≥15)",
        "",
        "검토 MD: [v2.1_클러스터링_검토_V11_generic_hair_k2_f1.md]"
        "(v2.1_클러스터링_검토_V11_generic_hair_k2_f1.md) · "
        "기타헤어 예시: [v2.1_기타헤어_검토_예시.md](v2.1_기타헤어_검토_예시.md)",
        "",
        "## KPI 비교",
        "",
        "| 버전 | 운영 세그 | m2 동질성 | 전체 동질성 | m2 최대 | m2 최대 % |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for label, df, pdf in rows:
        m = pipeline_metrics(df, [])
        p = compute_purity(df, pdf)
        m2 = df[(df.recruitType == "model") & (df.payment_group == "n2") & (df.merged_segment_id != -1)]
        m2_max = int(m2.groupby("merged_segment_id").size().max()) if not m2.empty else 0
        lines.append(
            f"| {label} | {m['n_segments']} | {p['model_n2_purity']*100:.1f}% | "
            f"{p['avg_purity']*100:.1f}% | {m2_max} | {m2_max/4908*100:.1f}% |"
        )

    lines.extend([
        "",
        f"## V11 {_cell_label('model', 'n2')} 상위 세그",
        "",
        "| seg | 이름 | 건수 |",
        "|---:|---|---:|",
    ])
    m2 = v11_df[
        (v11_df.recruitType == "model")
        & (v11_df.payment_group == "n2")
        & (v11_df.merged_segment_id != -1)
    ]
    for sid, g in m2.groupby("merged_segment_id"):
        key = g.segment_key.iloc[0]
        lines.append(f"| {sid} | {key} | {len(g)} |")

    lines.extend([
        "",
        "## 고정 기준 (변경 없음)",
        "",
        "- Phase3 v7 프롬프트 + `guard_perm_hallucination`",
        "- `apply_generic_hair_treatment` 후처리 (헤어+시술불명확+키워드 없음)",
        "- Gower `[1,1,2,2]`, 4축 병합, coarse OFF",
        "",
        "## V11에서만 변경",
        "",
        "- `raw_k_scale = 2.0` (V8 레버)",
        "- F-1: 최소 300건 · 장소불명확 ≥15% · B군 ≥15건",
        "",
    ])
    return "\n".join(lines)


def run_v11_experiment(*, force: bool = False) -> None:
    """V11 실험 — 기타헤어 후처리 + K×2 + F-1 강화"""
    from v2_generic_hair_simulation import apply_rule_to_frame, load_meta

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    phase_df, _, _ = load_phase3_frame()
    categories, texts = load_meta()
    modified_df, stats, _ = apply_rule_to_frame(phase_df, categories, texts)
    print(f"기타 헤어 적용: {stats.applied}건 (제외 {stats.skipped_text}건)")

    recruit_data = load_recruit_texts(phase_df["recruitId"].astype(int).tolist())
    csv_path = EXPERIMENTS_DIR / f"cluster_assignments_{V11_VARIANT.slug}.csv"
    md_path = EXPERIMENTS_DOCS_DIR / f"v2.1_클러스터링_검토_{V11_VARIANT.slug}.md"
    summary_path = EXPERIMENTS_DOCS_DIR / "v2.1_V11_비교_요약.md"

    if csv_path.is_file() and not force:
        print(f"기존 CSV 사용: {csv_path}")
        v11_df = pd.read_csv(csv_path, dtype={"recruitId": int})
    else:
        _, v11_df = run_variant_batch(
            [V11_VARIANT],
            modified_df,
            recruit_data,
            cluster_phase_df=modified_df,
        )[0]

    if not md_path.is_file() or force:
        md_path.write_text(
            build_review_md(V11_VARIANT, v11_df, modified_df, recruit_data),
            encoding="utf-8",
        )
        print(f"검토 MD: {md_path}")

    summary_path.write_text(
        build_v11_comparison_md(v11_df, modified_df, stats.applied),
        encoding="utf-8",
    )
    print(f"비교 요약: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v2.1 클러스터링 실험")
    parser.add_argument(
        "--density",
        action="store_true",
        help="밀도 강화 V5~V7 (+ V3 기준)만 실행",
    )
    parser.add_argument(
        "--density-ext",
        action="store_true",
        help="밀도 확장 V7~V10 (K×2, w1144, F-1 강화) 실행",
    )
    parser.add_argument(
        "--v11",
        action="store_true",
        help="V11 — V10 + 기타헤어 + K×2 + F-1 강화",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 CSV 있어도 재실행",
    )
    args = parser.parse_args()
    if args.v11:
        run_v11_experiment(force=args.force)
    elif args.density_ext:
        run_extended_density_experiments(force=args.force)
    elif args.density:
        run_density_experiments()
    else:
        run_experiments()
        print("\n4버전 실험 완료 — 검토 후 채택 버전을 알려주세요.")
