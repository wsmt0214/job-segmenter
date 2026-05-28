"""v2.1 TUNE 세그먼트별 구인글 예시 MD/HTML — 브라우저에서 열어 볼 수 있음"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from schema_v2 import CLUSTERING_6DIM, UNCLEAR_VALUE
from v2_clustering_experiment import CELL_LABELS, RT_LABELS
from v2_clustering_v21 import load_phase3_frame
from v2_operational_viability import (
    GRADE_A_MAX_PCT,
    GRADE_A_MIN_N,
    DOM_TH,
    grade_segment,
)
from v2_segment_ops import (
    COL_CAREER,
    COL_CONTINUITY,
    COL_PLACE,
    COL_PURPOSE,
    COL_TOPIC,
    COL_TREATMENT,
    DOMINANT_THRESHOLD,
    ClusterInfo,
    compute_dominant,
)
import v2_clustering as vc

ASSIGN_CSV = Path(config.V2_DATA_DIR) / "cluster_assignments_v21_tune.csv"
OUT_MD = Path(config.V2_DATA_DIR) / "cluster_segment_examples.md"
OUT_HTML = OUT_MD.with_suffix(".html")
DIM_COLS = list(CLUSTERING_6DIM)
EXAMPLES_PER_SEGMENT = 3
DIST_TOP_N = 8

AXIS_SHORT = {
    COL_PLACE: "장소",
    COL_PURPOSE: "목적",
    COL_TOPIC: "주제",
    COL_TREATMENT: "시술",
    COL_CAREER: "경력",
    COL_CONTINUITY: "지속성",
}


def segment_anchor(cell_label: str, seg_id: int) -> str:
    slug = re.sub(r"[^\w\-]", "-", cell_label.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"{slug}-seg-{seg_id}"


def sanitize_fence(text: str) -> str:
    return (text or "").replace("```", "'''")


def load_recruit_meta(recruit_ids: set[int]) -> dict[int, dict]:
    categories: dict[int, list[str]] = {}
    phase1_path = Path(config.V2_DATA_DIR) / "phase1_with_category.jsonl"
    with phase1_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = int(r["recruitId"])
            if rid in recruit_ids:
                categories[rid] = r.get("categories") or []

    raw = pd.read_csv(
        Path(config.DATA_DIR) / "raw_recruits.csv",
        usecols=["recruitId", "title", "content"],
    )
    raw = raw[raw["recruitId"].isin(recruit_ids)]
    return {
        int(row.recruitId): {
            "title": str(row.title or "").strip(),
            "content": str(row.content or "").strip(),
            "categories": categories.get(int(row.recruitId), []),
        }
        for _, row in raw.iterrows()
    }


def load_frame() -> pd.DataFrame:
    assign = pd.read_csv(ASSIGN_CSV, dtype={"recruitId": int})
    phase_df, _, _ = load_phase3_frame()
    return assign.merge(
        phase_df.drop(columns=["payment_group"], errors="ignore"),
        on="recruitId",
        how="left",
    )


def axis_columns_for_rt(rt: str) -> list[str]:
    if rt == "photo":
        return [c for c in DIM_COLS if c != COL_TREATMENT]
    return list(DIM_COLS)


def axes_line(row: pd.Series, rt: str) -> str:
    return " · ".join(
        f"{AXIS_SHORT[c]}={row.get(c, UNCLEAR_VALUE)}" for c in axis_columns_for_rt(rt)
    )


def segment_name(seg: pd.DataFrame, seg_id: int) -> str:
    if seg_id == -1:
        return "정보 부족형"
    return str(seg["segment_key"].iloc[0])


def merge_key_display(seg: pd.DataFrame, rt: str) -> str:
    ci = ClusterInfo(
        cluster_id=0,
        n=len(seg),
        purpose_dom=compute_dominant(seg, COL_PURPOSE),
        place_dom=compute_dominant(seg, COL_PLACE),
        topic_dom=compute_dominant(seg, COL_TOPIC),
        treatment_dom=compute_dominant(seg, COL_TREATMENT),
        career_dom=compute_dominant(seg, COL_CAREER),
        continuity_dom=compute_dominant(seg, COL_CONTINUITY),
        include_treatment_in_key=(rt != "photo"),
    )
    inner = ", ".join(str(x) for x in ci.merge_key)
    return f"({inner})"


def axis_value_distribution(
    seg: pd.DataFrame,
    col: str,
    *,
    top_n: int = DIST_TOP_N,
) -> list[tuple[str, int, float]]:
    total = len(seg)
    if total == 0:
        return []
    counts = seg[col].fillna(UNCLEAR_VALUE).astype(str).value_counts()
    return [
        (str(val), int(cnt), 100.0 * int(cnt) / total)
        for val, cnt in counts.head(top_n).items()
    ]


def format_value_pct(val: str, pct: float, *, qualifies: bool = False) -> str:
    pct_s = f"{pct:.1f}%"
    if qualifies:
        return f"**{val}: {pct_s}**"
    return f"{val}: {pct_s}"


def distribution_section(seg: pd.DataFrame, rt: str) -> list[str]:
    total = len(seg)
    th_pct = int(DOMINANT_THRESHOLD * 100)
    lines = [
        f"**6축 value 분포** ({total:,}건 · 굵게=1등 **≥{th_pct}%** · merge_key 포함 축)",
        "",
    ]
    for col in axis_columns_for_rt(rt):
        short = AXIS_SHORT[col]
        rows = axis_value_distribution(seg, col)
        if not rows:
            lines.append(f"- **{short}** — (없음)")
            lines.append("")
            continue
        top_val = rows[0][0]
        parts: list[str] = []
        for val, _cnt, pct in rows:
            qualifies = (
                val == top_val
                and val != UNCLEAR_VALUE
                and pct >= DOMINANT_THRESHOLD * 100
            )
            parts.append(format_value_pct(val, pct, qualifies=qualifies))
        n_unique = int(seg[col].fillna(UNCLEAR_VALUE).astype(str).nunique())
        if len(rows) < n_unique:
            parts.append(f"… (외 {n_unique - len(rows)}값)")
        lines.append(f"- **{short}:** {', '.join(parts)}")
        lines.append("")
    return lines


def top_categories(
    seg: pd.DataFrame,
    recruit_data: dict[int, dict],
    top_n: int = 3,
) -> str:
    counter: dict[str, int] = {}
    for rid in seg["recruitId"].astype(int):
        for c in recruit_data.get(rid, {}).get("categories") or []:
            counter[c] = counter.get(c, 0) + 1
    if not counter:
        return "-"
    total = len(seg)
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    return ", ".join(
        f"{k}({v}, {100 * v / total:.1f}%)" for k, v in items
    )


def iter_segments(df: pd.DataFrame):
    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            cell_df = df[(df.recruitType == rt) & (df.payment_group == pg)]
            if cell_df.empty:
                continue
            cell_label = CELL_LABELS.get((rt, pg), vc.cell_short_label(rt, pg))
            cell_op = cell_df[cell_df.merged_segment_id != -1]
            cell_n = len(cell_op)
            seg_ids = sorted(
                cell_df["merged_segment_id"].unique(),
                key=lambda x: (x == -1, x),
            )
            for seg_id in seg_ids:
                seg_id = int(seg_id)
                seg = cell_df[cell_df.merged_segment_id == seg_id]
                yield rt, pg, cell_label, cell_n, seg_id, seg


def summary_section(df: pd.DataFrame) -> list[str]:
    op = df[df.merged_segment_id != -1]
    n_poor = int((df.merged_segment_id == -1).sum())
    n_seg = int(
        op.groupby(["recruitType", "payment_group"])["merged_segment_id"].nunique().sum()
    )
    m2 = df[
        (df.recruitType == "model")
        & (df.payment_group == "n2")
        & (df.merged_segment_id != -1)
    ]
    m2_max = int(m2.groupby("merged_segment_id").size().max()) if not m2.empty else 0
    m2_pct = 100 * m2_max / len(m2) if len(m2) else 0.0

    return [
        "## 1. 요약",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 전체 구인글 | {len(df):,}건 |",
        f"| 운영 세그먼트 | **{n_seg}개** (9셀 합) |",
        f"| 정보 부족형 (-1) | {n_poor:,}건 ({100 * n_poor / len(df):.1f}%) |",
        f"| model×n2 최대 세그 | {m2_max:,}건 ({m2_pct:.1f}%) |",
        f"| 세그당 예시 | {EXAMPLES_PER_SEGMENT}건 |",
        "",
    ]


def grade_rules_section() -> list[str]:
    th_pct = int(DOMINANT_THRESHOLD * 100)
    a_dom = int(DOM_TH * 100)
    return [
        "## 2. 등급·분포 기준",
        "",
        "**운영 등급** (`docs/v2.1_클러스터링_확정.md` §6 · `v2_operational_viability.py`):",
        "",
        f"- **A** — n≥{GRADE_A_MIN_N}, 셀 내 비율≤{GRADE_A_MAX_PCT:.0f}%, 목적 또는 주제 1등 **≥{a_dom}%**",
        "- **B** — catch-all·micro 아님, 명확한 segment_key",
        "- **C** — catch-all(셀 max%>40%) 또는 micro(n<30)",
        "- **D** — 소형셀 KPI 면제 또는 정보 부족형",
        "",
        f"**6축 value 분포** — 세그 내 각 축 value별 **건수 대비 %**. "
        f"1등이 **≥{th_pct}%**이면 merge_key·dominant 후보(굵게 표시).",
        "",
        "---",
        "",
    ]


def cell_summary_section(df: pd.DataFrame, recruit_data: dict[int, dict]) -> list[str]:
    op = df[df.merged_segment_id != -1]
    n_seg = int(
        op.groupby(["recruitType", "payment_group"])["merged_segment_id"].nunique().sum()
    )
    lines = [
        "## 3. 셀별 요약",
        "",
        "| 셀 | 건수 | 운영세그 | 정보부족 | A | B | C | D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grade_totals: Counter[str] = Counter()

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            cell_df = df[(df.recruitType == rt) & (df.payment_group == pg)]
            if cell_df.empty:
                continue
            cell_label = CELL_LABELS.get((rt, pg), vc.cell_short_label(rt, pg))
            cell_n = len(cell_df[cell_df.merged_segment_id != -1])
            n_poor = int((cell_df.merged_segment_id == -1).sum())
            n_op_seg = int(
                cell_df[cell_df.merged_segment_id != -1]["merged_segment_id"].nunique()
            )
            grades: Counter[str] = Counter()
            for _, _, _, cn, seg_id, seg in iter_segments(cell_df):
                if seg_id == -1:
                    grades["D"] += 1
                    grade_totals["D"] += 1
                    continue
                g = grade_segment(rt, pg, seg_id, seg, cell_n=cn).grade
                grades[g] += 1
                grade_totals[g] += 1

            lines.append(
                f"| {cell_label} | {len(cell_df):,} | {n_op_seg} | {n_poor} | "
                f"{grades.get('A', 0)} | {grades.get('B', 0)} | "
                f"{grades.get('C', 0)} | {grades.get('D', 0)} |"
            )

    n_rows = sum(grade_totals.values())
    n_poor_rows = sum(1 for _, _, _, _, sid, _ in iter_segments(df) if sid == -1)
    lines.extend(
        [
            "",
            f"**전체 등급 합계** (§4 표 {n_rows}행 · 운영 세그 {n_seg} + 정보 부족형 {n_poor_rows}): "
            f"A {grade_totals.get('A', 0)} · B {grade_totals.get('B', 0)} · "
            f"C {grade_totals.get('C', 0)} · D {grade_totals.get('D', 0)}",
            "",
            "---",
            "",
        ]
    )
    return lines


def segment_index_section(df: pd.DataFrame, recruit_data: dict[int, dict]) -> list[str]:
    lines = [
        "## 4. 전체 세그먼트 표",
        "",
        "§5 각 세그에 **6축 value 분포(%)** · 구인글 예시 포함.",
        "",
        "| 셀 | 세그 | 등급 | segment_key | merge_key | 건수 | 셀% | 상위 카테고리 |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]

    for rt, pg, cell_label, cell_n, seg_id, seg in iter_segments(df):
        name = segment_name(seg, seg_id)
        anchor = segment_anchor(cell_label, seg_id)
        mk_str = merge_key_display(seg, rt) if seg_id != -1 else "-"
        g = grade_segment(rt, pg, seg_id, seg, cell_n=cell_n)
        pct = f"{g.pct_cell:.1f}" if seg_id != -1 else "-"
        lines.append(
            f"| {cell_label} | [{seg_id}](#{anchor}) | {g.grade} | {name} | {mk_str} | "
            f"{len(seg):,} | {pct} | {top_categories(seg, recruit_data)} |"
        )

    lines.extend(["", "---", ""])
    return lines


def examples_section(df: pd.DataFrame, recruit_data: dict[int, dict]) -> list[str]:
    lines = [
        "## 5. 세그먼트별 6축 분포 및 구인글 예시",
        "",
        f"각 세그 **6축 value %** + **{EXAMPLES_PER_SEGMENT}건** 예시 (recruitId 오름차순).",
        "",
    ]

    current_rt = ""
    current_cell = ""

    for rt, pg, cell_label, cell_n, seg_id, seg in iter_segments(df):
        if rt != current_rt:
            current_rt = rt
            current_cell = ""
            lines.append(f"### {RT_LABELS[rt]} ({rt})")
            lines.append("")

        if cell_label != current_cell:
            current_cell = cell_label
            cell_df = df[(df.recruitType == rt) & (df.payment_group == pg)]
            lines.append(f"#### {cell_label} ({len(cell_df):,}건)")
            lines.append("")

        name = segment_name(seg, seg_id)
        anchor = segment_anchor(cell_label, seg_id)
        g = grade_segment(rt, pg, seg_id, seg, cell_n=cell_n)

        lines.append(f'<a id="{anchor}"></a>')
        lines.append("")
        lines.append(f"##### 세그 {seg_id}: {name}")
        lines.append("")
        lines.append(
            f"- **건수:** {len(seg):,}건"
            + (f" · **셀%:** {g.pct_cell:.1f}%" if seg_id != -1 else "")
            + f" · **등급:** {g.grade}"
        )
        if seg_id != -1:
            lines.append(f"- **merge_key:** `{merge_key_display(seg, rt)}`")
        lines.append("")
        lines.extend(distribution_section(seg, rt))

        sample = seg.sort_values("recruitId").head(EXAMPLES_PER_SEGMENT)
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            rid = int(row.recruitId)
            rd = recruit_data.get(rid, {})
            cats = rd.get("categories") or []
            cat_str = ", ".join(cats) if cats else "-"
            title = sanitize_fence(rd.get("title") or "")
            body = sanitize_fence(rd.get("content") or "")

            lines.append(f"###### 예시 {i} · recruitId `{rid}`")
            lines.append("")
            lines.append(f"- **카테고리:** {cat_str}")
            lines.append(f"- **6축:** {axes_line(row, rt)}")
            lines.append("")
            lines.append("**제목**")
            lines.append("")
            lines.append(title if title else "(제목 없음)")
            lines.append("")
            lines.append("**본문**")
            lines.append("")
            lines.append("```")
            lines.append(body if body else "(본문 없음)")
            lines.append("```")
            lines.append("")

        remaining = len(seg) - len(sample)
        if remaining > 0:
            lines.append(
                f"*외 {remaining:,}건 — recruitId 순 상위 {EXAMPLES_PER_SEGMENT}건만 표시*"
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    return lines


def build_markdown(df: pd.DataFrame, recruit_data: dict[int, dict]) -> str:
    rel_csv = ASSIGN_CSV.relative_to(ROOT).as_posix()
    lines: list[str] = [
        "# v2.1 클러스터(세그먼트)별 구인글 예시",
        "",
        f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 배정: `{rel_csv}`",
        "",
        "브라우저에서 보려면 **`cluster_segment_examples.html`** 을 엽니다.",
        "확정 파이프라인·KPI: [v2.1_클러스터링_확정.md](../docs/v2.1_클러스터링_확정.md).",
        "",
        *summary_section(df),
        *grade_rules_section(),
        *cell_summary_section(df, recruit_data),
        *segment_index_section(df, recruit_data),
        *examples_section(df, recruit_data),
        "## 재생성",
        "",
        "```bash",
        "./venv/bin/python scripts/v2_cluster_segment_examples.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_html(md_path: Path, html_path: Path) -> None:
    script_path = ROOT / "scripts/cluster_examples_md_to_html.py"
    spec = importlib.util.spec_from_file_location("cluster_examples_md_to_html", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cluster_examples_md_to_html 로드 실패")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.convert(
        md_path,
        html_path,
        page_title="v2.1 클러스터(세그먼트)별 구인글 예시",
        md_filename=md_path.name,
    )


def main() -> None:
    if not ASSIGN_CSV.exists():
        raise SystemExit(f"배정 CSV 없음: {ASSIGN_CSV} — 먼저 v2_clustering_v21.py 실행")

    df = load_frame()
    recruit_data = load_recruit_meta(set(df["recruitId"].astype(int)))
    md_text = build_markdown(df, recruit_data)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md_text, encoding="utf-8")
    build_html(OUT_MD, OUT_HTML)

    op = df[df.merged_segment_id != -1]
    n_seg = op.groupby(["recruitType", "payment_group"])["merged_segment_id"].nunique().sum()
    print(
        f"세그 {n_seg} (+정보부족) · 예시 {EXAMPLES_PER_SEGMENT}건/세그\n"
        f"  MD:   {OUT_MD}\n"
        f"  HTML: {OUT_HTML}"
    )


if __name__ == "__main__":
    main()
