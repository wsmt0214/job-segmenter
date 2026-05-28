"""기타 헤어 후처리 규칙 시뮬레이션 — V10 클러스터링 기준"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from schema_v2 import UNCLEAR_VALUE
from v2_clustering_experiment import (
    CELL_LABELS,
    EXTENDED_DENSITY_VARIANTS,
    RT_LABELS,
    compute_purity,
    dim_distribution,
    summarize,
)
from v2_clustering_v21 import load_phase3_frame, pipeline_metrics, run
from v2_phase3_core import (
    GENERIC_HAIR_VALUE,
    apply_generic_hair_treatment,
    has_explicit_treatment_text,
)
from v2_segment_ops import COL_PLACE, COL_PURPOSE, COL_TOPIC, COL_TREATMENT

V10 = next(v for v in EXTENDED_DENSITY_VARIANTS if v.slug == "V10_dense_f1strong")
BASELINE_CSV = Path(config.V2_DATA_DIR) / "_experiments" / "cluster_assignments_V10_dense_f1strong.csv"
OUT_CSV = Path(config.V2_DATA_DIR) / "_experiments" / "cluster_assignments_V10_generic_hair.csv"
OUT_MD = ROOT / "docs" / "_experiments" / "v2.1_기타헤어_V10시뮬_요약.md"
OUT_EXAMPLES_MD = ROOT / "docs" / "_experiments" / "v2.1_기타헤어_검토_예시.md"
DIM_COLS = [COL_PLACE, COL_PURPOSE, COL_TOPIC, COL_TREATMENT]
MAX_EXAMPLES = 5
MOVE_PAIR_EXAMPLES = 3


@dataclass
class RuleStats:
    hair_unclear_total: int
    applied: int
    skipped_text: int


def load_meta() -> tuple[dict[int, list[str]], dict[int, tuple[str, str]]]:
    """phase3 카테고리 + raw_recruits 제목·본문"""
    categories: dict[int, list[str]] = {}
    with open(Path(config.V2_DATA_DIR) / "phase3_results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            categories[int(r["recruitId"])] = r.get("categories") or []

    raw = pd.read_csv(Path(config.DATA_DIR) / "raw_recruits.csv")
    texts = {
        int(row.recruitId): (str(row.title), str(row.content))
        for _, row in raw.iterrows()
    }
    return categories, texts


def build_recruit_data(
    categories: dict[int, list[str]],
    texts: dict[int, tuple[str, str]],
) -> dict[int, dict[str, str]]:
    """검토 MD용 구인글 메타"""
    data: dict[int, dict[str, str]] = {}
    for rid, (title, content) in texts.items():
        cats = categories.get(rid, [])
        data[rid] = {
            "title": title,
            "content": content,
            "categories": ", ".join(cats[:3]) if cats else "-",
        }
    return data


def axes_label(row: pd.Series) -> str:
    return " · ".join(str(row.get(c, UNCLEAR_VALUE)) for c in DIM_COLS)


def example_row_table(
    rows: list[tuple],
    *,
    show_move: bool = False,
) -> list[str]:
    """구인글 예시 테이블 행 생성"""
    if show_move:
        header = "| # | recruitId | V10 세그 | → 신규 세그 | 카테고리 | 제목 | 4축 | 내용 요약 |"
        sep = "|---:|---:|---|---|---|---|---|---|"
    else:
        header = "| # | recruitId | 카테고리 | 제목 | 4축 | 내용 요약 |"
        sep = "|---:|---:|---|---|---|---|"
    lines = [header, sep]
    for item in rows:
        if show_move:
            i, rid, b_key, g_key, rd, axes = item
            lines.append(
                f"| {i} | {rid} | `{b_key}` | `{g_key}` | {rd.get('categories', '-')} | "
                f"{rd.get('title', '')} | {axes} | {summarize(rd.get('content', ''))} |"
            )
        else:
            i, rid, rd, axes = item
            lines.append(
                f"| {i} | {rid} | {rd.get('categories', '-')} | "
                f"{rd.get('title', '')} | {axes} | {summarize(rd.get('content', ''))} |"
            )
    return lines


def collect_move_examples(
    baseline_df: pd.DataFrame,
    generic_df: pd.DataFrame,
    applied_ids: set[int],
    phase_generic: pd.DataFrame,
    recruit_data: dict[int, dict[str, str]],
    src_key: str,
    dst_key: str,
    limit: int = MOVE_PAIR_EXAMPLES,
) -> list[tuple]:
    """특정 이동 쌍의 구인글 샘플"""
    base = baseline_df.set_index("recruitId")
    gen = generic_df.set_index("recruitId")
    attrs = phase_generic.set_index("recruitId")
    samples: list[tuple] = []

    for rid in sorted(applied_ids):
        if rid not in base.index or rid not in gen.index:
            continue
        if str(base.loc[rid]["segment_key"]) != src_key:
            continue
        if str(gen.loc[rid]["segment_key"]) != dst_key:
            continue
        rd = recruit_data.get(rid, {})
        axes = axes_label(attrs.loc[rid])
        samples.append((len(samples) + 1, rid, src_key, dst_key, rd, axes))
        if len(samples) >= limit:
            break
    return samples


def build_examples_md(
    stats: RuleStats,
    cmp: dict,
    baseline_df: pd.DataFrame,
    generic_df: pd.DataFrame,
    phase_baseline: pd.DataFrame,
    phase_generic: pd.DataFrame,
    applied_ids: set[int],
    hair_unclear_ids: set[int],
    skipped_ids: set[int],
    recruit_data: dict[int, dict[str, str]],
    metrics_gen: dict,
    purity_gen: dict,
) -> str:
    """V10 검토 MD 형식 — 기타 헤어 세그·이동·제외 구인글 예시"""
    attrs = phase_generic[["recruitId"] + DIM_COLS].drop_duplicates(subset="recruitId")
    merged = generic_df.merge(attrs, on="recruitId", how="left")

    lines: list[str] = [
        "# v2.1 기타 헤어 검토 — 구인글 예시",
        "",
        "> **V10 + `기타 헤어` 후처리 시뮬레이션** — 확정 규칙·수치는 "
        f"[37_v2.1_기타헤어_후처리_확정.md](../37_v2.1_기타헤어_후처리_확정.md) 참조",
        "",
        f"기타 헤어 적용 **{stats.applied}건** · 세그 이동 **{cmp['moved']}건** "
        f"({100 * cmp['moved'] / stats.applied:.1f}%) · "
        f"model×n2 동질성 **{purity_gen['model_n2_purity'] * 100:.1f}%**",
        "",
        "---",
        "",
        "## 1. `기타 헤어` 세그먼트별 구인글 예시",
        "",
        "V10+기타헤어 재클러스터링 결과, 이름에 `기타 헤어`가 포함된 세그 (셀별)",
        "",
    ]

    for rt in config.RECRUIT_TYPES:
        lines.append(f"### {RT_LABELS[rt]} ({rt})")
        lines.append("")
        for pg in config.PAYMENT_GROUPS:
            cell = merged[(merged.recruitType == rt) & (merged.payment_group == pg)]
            if cell.empty:
                continue
            cell_hair = cell[
                (cell.merged_segment_id != -1)
                & (cell.segment_key.str.contains("기타 헤어", na=False))
            ]
            if cell_hair.empty:
                continue
            label = CELL_LABELS[(rt, pg)]
            n_poor = int((cell.merged_segment_id == -1).sum())
            lines.append(f"#### {label} ({len(cell):,}건 · 정보 부족 {n_poor})")
            lines.append("")
            lines.append("| seg | 이름 | 건수 | 비율 | 목적 | 장소 | 주제 | 시술 |")
            lines.append("|---:|---|---:|---:|---|---|---|---|")
            for sid in sorted(cell_hair.merged_segment_id.unique()):
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

            for sid in sorted(cell_hair.merged_segment_id.unique()):
                seg = cell[cell.merged_segment_id == sid]
                name = seg.segment_key.iloc[0]
                n = len(seg)
                lines.append(f"##### 세그먼트 {sid}: {name} ({n}건)")
                lines.append("")
                lines.append(
                    f"- **목적** {dim_distribution(seg, COL_PURPOSE)}  "
                    f"- **장소** {dim_distribution(seg, COL_PLACE)}  "
                    f"- **주제** {dim_distribution(seg, COL_TOPIC)}  "
                    f"- **시술** {dim_distribution(seg, COL_TREATMENT)}"
                )
                lines.append("")
                rows: list[tuple] = []
                for i, (_, row) in enumerate(
                    seg.sort_values("recruitId").head(MAX_EXAMPLES).iterrows(), 1
                ):
                    rid = int(row.recruitId)
                    rd = recruit_data.get(rid, {})
                    rows.append((i, rid, rd, axes_label(row)))
                lines.extend(example_row_table(rows))
                lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. V10 → 기타헤어 주요 이동 — 구인글 사례",
        "",
        "상위 이동 쌍별 샘플 (V10 세그 → 신규 세그)",
        "",
    ])

    for (src, dst), cnt in cmp["move_pairs"].most_common(8):
        lines.append(f"### `{src}` → `{dst}` ({cnt}건)")
        lines.append("")
        samples = collect_move_examples(
            baseline_df, generic_df, applied_ids, phase_generic,
            recruit_data, src, dst,
        )
        if samples:
            lines.extend(example_row_table(samples, show_move=True))
        else:
            lines.append("_샘플 없음_")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. 펌 세그 오분류 해소 사례",
        "",
        "V10 `포트폴리오·스튜디오·펌`에 있었으나 시술=불명확 → `기타 헤어`로 이동",
        "",
    ])
    perm_fix = collect_move_examples(
        baseline_df, generic_df, applied_ids, phase_generic, recruit_data,
        "포트폴리오·스튜디오·펌", "포트폴리오·스튜디오·기타 헤어",
        limit=MAX_EXAMPLES,
    )
    if perm_fix:
        lines.extend(example_row_table(perm_fix, show_move=True))
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. 규칙 미적용 — 시술 키워드 제외 250건",
        "",
        "본문에 `펌`·`컷`·`염색` 등이 있어 `기타 헤어` 미적용 — V10과 동일 세그 유지",
        "",
    ])
    skip_rows: list[tuple] = []
    base_idx = baseline_df.set_index("recruitId")
    attrs_skip = phase_baseline.set_index("recruitId")
    lines.append("| # | recruitId | V10 세그 | 감지 키워드 | 카테고리 | 제목 | 4축 | 내용 요약 |")
    lines.append("|---:|---:|---|---|---|---|---|---|")
    for i, rid in enumerate(sorted(skipped_ids)[:MAX_EXAMPLES], 1):
        rd = recruit_data.get(rid, {})
        title, body = texts_from_recruit(rd)
        kw = _treatment_keywords_found(title, body)
        axes = axes_label(attrs_skip.loc[rid]) if rid in attrs_skip.index else "-"
        seg = str(base_idx.loc[rid]["segment_key"]) if rid in base_idx.index else "-"
        lines.append(
            f"| {i} | {rid} | `{seg}` | {kw} | {rd.get('categories', '-')} | "
            f"{rd.get('title', '')} | {axes} | {summarize(rd.get('content', ''))} |"
        )
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. 세그 유지 — 기타 헤어 적용 후에도 V10과 동일 (121건 중 샘플)",
        "",
    ])
    same_rows: list[tuple] = []
    gen_idx = generic_df.set_index("recruitId")
    for rid in sorted(applied_ids):
        if rid not in base_idx.index or rid not in gen_idx.index:
            continue
        if int(base_idx.loc[rid]["merged_segment_id"]) == int(gen_idx.loc[rid]["merged_segment_id"]):
            rd = recruit_data.get(rid, {})
            attrs_row = phase_generic.set_index("recruitId").loc[rid]
            same_rows.append((len(same_rows) + 1, rid, rd, axes_label(attrs_row)))
            if len(same_rows) >= MAX_EXAMPLES:
                break
    lines.extend(example_row_table(same_rows))
    lines.append("")

    return "\n".join(lines)


def texts_from_recruit(rd: dict[str, str]) -> tuple[str, str]:
    return rd.get("title", ""), rd.get("content", "")


def _treatment_keywords_found(title: str, body: str) -> str:
    """제목·본문에서 감지된 시술 키워드 요약"""
    from v2_phase3_core import EXPLICIT_TREATMENT_MARKERS
    full = f"{title} {body}"
    found = [m for m in EXPLICIT_TREATMENT_MARKERS if m in full]
    # 긴 키워드 우선, 중복 제거
    found = sorted(set(found), key=len, reverse=True)[:4]
    return ", ".join(found) if found else "-"


def hair_unclear_id_set(
    phase_df: pd.DataFrame,
    categories: dict[int, list[str]],
) -> set[int]:
    ids: set[int] = set()
    for _, row in phase_df.iterrows():
        rid = int(row["recruitId"])
        if row[COL_TREATMENT] == UNCLEAR_VALUE and "헤어" in categories.get(rid, []):
            ids.add(rid)
    return ids


def apply_rule_to_frame(
    phase_df: pd.DataFrame,
    categories: dict[int, list[str]],
    texts: dict[int, tuple[str, str]],
) -> tuple[pd.DataFrame, RuleStats, set[int]]:
    """기타 헤어 후처리 적용"""
    out = phase_df.copy()
    applied_ids: set[int] = set()
    skipped_text = 0
    hair_unclear = 0

    for idx, row in out.iterrows():
        rid = int(row["recruitId"])
        cats = categories.get(rid, [])
        title, body = texts.get(rid, ("", ""))
        before = row[COL_TREATMENT]
        if before != UNCLEAR_VALUE or "헤어" not in cats:
            continue
        hair_unclear += 1
        if has_explicit_treatment_text(title, body):
            skipped_text += 1
            continue
        after = apply_generic_hair_treatment(
            before, cats, title=title, body=body, skip_when_treatment_text=False
        )
        if after == GENERIC_HAIR_VALUE:
            out.at[idx, COL_TREATMENT] = after
            applied_ids.add(rid)

    stats = RuleStats(
        hair_unclear_total=hair_unclear,
        applied=len(applied_ids),
        skipped_text=skipped_text,
    )
    return out, stats, applied_ids


def compare_assignments(
    baseline: pd.DataFrame,
    generic: pd.DataFrame,
    target_ids: set[int],
    phase_baseline: pd.DataFrame,
    phase_generic: pd.DataFrame,
) -> dict:
    """대상 건의 세그먼트 이동 집계"""
    base = baseline.set_index("recruitId")
    gen = generic.set_index("recruitId")

    same = moved = info_poor = 0
    move_pairs: Counter[tuple[str, str]] = Counter()
    dest_segments: Counter[str] = Counter()
    src_segments: Counter[str] = Counter()

    for rid in target_ids:
        if rid not in base.index or rid not in gen.index:
            continue
        b = base.loc[rid]
        g = gen.loc[rid]
        b_key = str(b["segment_key"])
        g_key = str(g["segment_key"])
        src_segments[b_key] += 1

        if int(g["merged_segment_id"]) == -1:
            info_poor += 1
            dest_segments[g_key] += 1
            if b_key != g_key:
                moved += 1
                move_pairs[(b_key, g_key)] += 1
            else:
                same += 1
            continue

        dest_segments[g_key] += 1
        if int(b["merged_segment_id"]) == int(g["merged_segment_id"]):
            same += 1
        else:
            moved += 1
            move_pairs[(b_key, g_key)] += 1

    return {
        "same": same,
        "moved": moved,
        "info_poor": info_poor,
        "move_pairs": move_pairs,
        "dest_segments": dest_segments,
        "src_segments": src_segments,
    }


def top_move_table(pairs: Counter, n: int = 15) -> str:
    lines = ["| V10 세그 | → 기타헤어 V10 세그 | 건수 |", "|---|---|---|"]
    for (src, dst), cnt in pairs.most_common(n):
        lines.append(f"| `{src}` | `{dst}` | {cnt} |")
    return "\n".join(lines)


def segment_dest_table(dest: Counter, generic_df: pd.DataFrame, n: int = 12) -> str:
    lines = ["| 세그먼트 | 배정 건수 |", "|---|---|"]
    for key, cnt in dest.most_common(n):
        seg_ids = generic_df[generic_df["segment_key"] == key]["merged_segment_id"].unique()
        sid = seg_ids[0] if len(seg_ids) else "-"
        lines.append(f"| `{key}` (id={sid}) | {cnt} |")
    return "\n".join(lines)


def build_md(
    stats: RuleStats,
    cmp: dict,
    baseline_df: pd.DataFrame,
    generic_df: pd.DataFrame,
    phase_baseline: pd.DataFrame,
    phase_generic: pd.DataFrame,
    metrics_base: dict,
    metrics_gen: dict,
    purity_base: dict,
    purity_gen: dict,
) -> str:
    target_n = stats.applied
    same_pct = 100 * cmp["same"] / target_n if target_n else 0
    moved_pct = 100 * cmp["moved"] / target_n if target_n else 0

    lines = [
        "# v2.1 기타 헤어 후처리 시뮬레이션 (V10 기준)",
        "",
        "## 1. 후처리 규칙 초안",
        "",
        "**적용 위치:** `guard_perm_hallucination` 이후, 클러스터링 직전 (LLM 재추출 없음)",
        "",
        "**조건 (모두 만족 시 `시술 종류` → `기타 헤어`):**",
        "",
        "1. Phase1/Phase3 카테고리에 `'헤어'` 포함",
        "2. Phase3 `시술 종류` == `불명확`",
        "3. 제목·본문에 구체 시술 키워드 **없음** (`펌`·`컷`·`염색`·`속눈썹` 등 — `EXPLICIT_TREATMENT_MARKERS`)",
        "",
        "**제외 (불명확 유지):**",
        "",
        "- 이미 `펌`·`컷`·`컬러` 등 명시 시술이 Phase3에 있는 건",
        "- 본문에 시술 키워드가 있는 건 (메뉴 나열 `펌/컷/염색 가능` 포함 — 보수적 제외)",
        "",
        "**구현:** `scripts/v2_phase3_core.py` — `apply_generic_hair_treatment()`",
        "",
        "---",
        "",
        "## 2. 적용 규모",
        "",
        f"| 항목 | 건수 |",
        f"|---|---:|",
        f"| 헤어 카테고리 + 시술 불명확 | {stats.hair_unclear_total} |",
        f"| 시술 키워드로 제외 | {stats.skipped_text} |",
        f"| **기타 헤어 적용** | **{stats.applied}** |",
        "",
        "---",
        "",
        "## 3. V10 클러스터링 영향 (전체)",
        "",
        f"| KPI | V10 baseline | V10 + 기타 헤어 |",
        f"|---|---:|---:|",
        f"| 운영 세그먼트 수 | {metrics_base['n_segments']} | {metrics_gen['n_segments']} |",
        f"| model×n2 동질성 | {purity_base['model_n2_purity']*100:.1f}% | {purity_gen['model_n2_purity']*100:.1f}% |",
        f"| 전체 동질성 | {purity_base['avg_purity']*100:.1f}% | {purity_gen['avg_purity']*100:.1f}% |",
        "",
        "---",
        "",
        "## 4. 기타 헤어 적용 970건 — 세그 재배치",
        "",
        f"| 결과 | 건수 | 비율 |",
        f"|---|---:|---:|",
        f"| **동일 세그 유지** | {cmp['same']} | {same_pct:.1f}% |",
        f"| **세그 이동** | {cmp['moved']} | {moved_pct:.1f}% |",
        f"| 이동 중 정보 부족형(-1) | {cmp['info_poor']} | — |",
        "",
        "### 4-1. V10 → 기타헤어 V10 주요 이동 (상위 15)",
        "",
        top_move_table(cmp["move_pairs"]),
        "",
        "### 4-2. 재배치 후 도착 세그 (상위 12)",
        "",
        segment_dest_table(cmp["dest_segments"], generic_df),
        "",
        "### 4-3. V10에서 출발했던 세그 (상위 10)",
        "",
        "| V10 세그 | 출발 건수 |",
        "|---|---:|",
    ]
    for key, cnt in cmp["src_segments"].most_common(10):
        lines.append(f"| `{key}` | {cnt} |")

    lines.extend([
        "",
        "### 4-4. 헤어+시술불명확 1220건 전체 관점",
        "",
        f"| 그룹 | 건수 | 세그 변화 |",
        f"|---|---:|---|",
        f"| 시술 키워드 제외 (규칙 미적용) | {stats.skipped_text} | V10과 동일 |",
        f"| 기타 헤어 적용 | {stats.applied} | 위 4-1~4-3 참조 |",
        f"| **합계** | **{stats.hair_unclear_total}** | 이동 {cmp['moved']}+0={cmp['moved']}건 ({100*cmp['moved']/stats.hair_unclear_total:.1f}%) |",
        "",
        "---",
        "",
        "## 5. 해석",
        "",
        f"- **이동률 {moved_pct:.1f}%** — `불명확`끼리는 시술 축 기여 0이었으나, `기타 헤어`는 "
        f"펌/컷 세그와 **거리 1.0**·동일 시술끼리 **거리 0**이라 Ward/병합키가 크게 바뀜",
        "- **주 도착지 `포트폴리오·스튜디오·기타 헤어` (738건)** — V10 `포트폴리오·스튜디오`·`장소미기재`·"
        "`뷰티·메이크업·스튜디오` 등에 흩어져 있던 일반 헤어 모델이 **한 덩어리로 수렴**",
        "- **펌 세그에서 빠져나온 건 76+51건** — 시술 불명확인데 `포트폴리오·스튜디오·펌` 등에 "
        "섞여 있던 케이스가 `기타 헤어`로 **오분류 해소**",
        "- **동질성 trade-off** — model×n2 92.5%→91.0%, 전체 76.5%→73.3%. "
        "세그 +2개(61→63), 일반 헤어 전용 세그 생기는 대가",
        "- **250건 제외** — 본문에 `펌/컷/염색` 등 키워드(메뉴 나열 포함). "
        "규칙 완화 시 추가 이동 가능하나 펌 환각·메뉴 오분류 리스크",
        "",
        "## 6. 산출물",
        "",
        f"- 배정 CSV: `{OUT_CSV.relative_to(ROOT)}`",
        f"- baseline: `{BASELINE_CSV.relative_to(ROOT)}`",
        f"- **구인글 예시 검토:** `{OUT_EXAMPLES_MD.relative_to(ROOT)}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    print("=== 기타 헤어 후처리 V10 시뮬레이션 ===\n")

    phase_df, _, _ = load_phase3_frame()
    categories, texts = load_meta()
    modified_df, stats, applied_ids = apply_rule_to_frame(phase_df, categories, texts)

    print(f"헤어+시술불명확: {stats.hair_unclear_total}")
    print(f"기타 헤어 적용: {stats.applied}")
    print(f"시술 키워드로 제외: {stats.skipped_text}")

    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"V10 baseline 없음: {BASELINE_CSV}")

    baseline_df = pd.read_csv(BASELINE_CSV)
    print(f"V10 baseline 로드: {len(baseline_df)}건")

    print("V10 + 기타 헤어 재클러스터링...")
    generic_df, _ = run(
        cluster_config=V10.cluster_config,
        out_csv=OUT_CSV,
        report_path=Path(config.V2_DATA_DIR) / "_experiments" / "marketer_review_V10_generic_hair.txt",
        verbose=False,
        phase_df=modified_df,
    )
    print(f"재클러스터링 완료: {len(generic_df)}건 → {OUT_CSV}")

    cmp = compare_assignments(
        baseline_df, generic_df, applied_ids, phase_df, modified_df
    )
    print(f"\n대상 {stats.applied}건 — 유지 {cmp['same']}, 이동 {cmp['moved']}")

    hair_unclear_ids = hair_unclear_id_set(phase_df, categories)
    skipped_ids = hair_unclear_ids - applied_ids
    recruit_data = build_recruit_data(categories, texts)

    metrics_base = pipeline_metrics(baseline_df, [])
    metrics_gen = pipeline_metrics(generic_df, [])
    purity_base = compute_purity(baseline_df, phase_df)
    purity_gen = compute_purity(generic_df, modified_df)

    md = build_md(
        stats, cmp, baseline_df, generic_df,
        phase_df, modified_df,
        metrics_base, metrics_gen, purity_base, purity_gen,
    )
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"수치 리포트: {OUT_MD}")

    examples_md = build_examples_md(
        stats, cmp, baseline_df, generic_df,
        phase_df, modified_df, applied_ids,
        hair_unclear_ids, skipped_ids, recruit_data,
        metrics_gen, purity_gen,
    )
    OUT_EXAMPLES_MD.write_text(examples_md, encoding="utf-8")
    print(f"예시 검토 MD: {OUT_EXAMPLES_MD}")


if __name__ == "__main__":
    main()
