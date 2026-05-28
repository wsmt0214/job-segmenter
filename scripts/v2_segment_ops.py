"""v2.1 세그먼트 공통 — 정보 충실도·dominant·병합"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from schema_v2 import UNCLEAR_VALUE

DOMINANT_THRESHOLD = 0.70
MIN_SEGMENT_SIZE = 10
INFO_POOR_MAX = 1  # 0~1개 명시 → 정보 부족형

COL_PURPOSE = "촬영 목적"
COL_PLACE = "촬영 장소"
COL_TOPIC = "촬영 주제"
COL_TREATMENT = "시술 종류"
COL_CAREER = "경력 조건"
COL_CONTINUITY = "작업 지속성"
MERGE_COLS = (COL_PURPOSE, COL_PLACE, COL_TOPIC)
# 리포트·검토용 6축 (촬영 4 + 경력·지속성)
CLUSTER_AXIS_COLS = (
    COL_PURPOSE,
    COL_PLACE,
    COL_TOPIC,
    COL_TREATMENT,
    COL_CAREER,
    COL_CONTINUITY,
)

HYBRID_LABEL = "혼재"


@dataclass
class ClusterInfo:
    cluster_id: int
    n: int
    purpose_dom: str
    place_dom: str
    topic_dom: str
    treatment_dom: str = UNCLEAR_VALUE
    career_dom: str = UNCLEAR_VALUE
    continuity_dom: str = UNCLEAR_VALUE
    include_treatment_in_key: bool = False

    def _append_explicit(self, parts: list[str], dom: str) -> None:
        if dom not in (HYBRID_LABEL, UNCLEAR_VALUE):
            parts.append(dom)

    @property
    def merge_key(self) -> tuple[str, ...]:
        """병합 키 — photo는 시술 제외, 경력·지속성은 recruitType 공통"""
        parts: list[str] = [
            self.purpose_dom,
            self.place_dom,
            self.topic_dom,
        ]
        if self.include_treatment_in_key:
            self._append_explicit(parts, self.treatment_dom)
        self._append_explicit(parts, self.career_dom)
        self._append_explicit(parts, self.continuity_dom)
        return tuple(parts)

    @property
    def segment_key(self) -> str:
        return segment_key_from_merge_key(self.merge_key)

    @property
    def coarse_key(self) -> tuple[str, ...]:
        """주제 혼재/불명확 시 축소 병합 — 시술·경력·지속성은 유지"""
        if self.topic_dom in (HYBRID_LABEL, UNCLEAR_VALUE):
            parts: list[str] = [self.purpose_dom, self.place_dom]
            if self.include_treatment_in_key:
                self._append_explicit(parts, self.treatment_dom)
            self._append_explicit(parts, self.career_dom)
            self._append_explicit(parts, self.continuity_dom)
            return tuple(parts)
        return self.merge_key


@dataclass
class MergedSegment:
    merge_key: tuple
    segment_key: str
    source_clusters: list[tuple[int, int]] = field(default_factory=list)
    n: int = 0


def is_explicit(val) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip() != UNCLEAR_VALUE


def info_density_row(row: pd.Series, density_cols: list[str]) -> int:
    """dimensions 6 중 명시된 개수"""
    return sum(1 for c in density_cols if is_explicit(row.get(c)))


def density_label(n: int) -> str:
    if n <= INFO_POOR_MAX:
        return "정보 부족형"
    if n <= 3:
        return "부분 정보형"
    return "충분 정보형"


def compute_dominant(
    cluster_df: pd.DataFrame,
    dim: str,
    threshold: float = DOMINANT_THRESHOLD,
) -> str:
    """
    세그 전체(불명확 포함) 1등 value 기준 dominant — 병합키·표시명 공통
    - 1등이 불명확 → '불명확'
    - 1등이 명시값이고 전체 대비 >= threshold → 해당 값
    - 1등이 명시값인데 threshold 미달 → '혼재'
    """
    total = len(cluster_df)
    if total == 0:
        return UNCLEAR_VALUE

    vc = cluster_df[dim].value_counts()
    top_val = str(vc.index[0])
    top_ratio = vc.iloc[0] / total

    if top_val == UNCLEAR_VALUE:
        return UNCLEAR_VALUE

    if top_ratio >= threshold:
        return top_val
    return HYBRID_LABEL


def build_segment_name(
    purpose_dom: str | None,
    place_dom: str | None,
    topic_dom: str | None = None,
) -> str:
    """
    Fix F-2B — dominant '불명확' 축은 이름에서 제외, '혼재'는 유지
    """
    parts: list[str] = []
    for dom in (purpose_dom, place_dom, topic_dom):
        if dom is None:
            continue
        s = str(dom).strip()
        if s in (UNCLEAR_VALUE, ""):
            continue
        parts.append(s)
    return "·".join(parts) if parts else "기타"


def build_place_missing_name(purpose_dom: str, topic_dom: str) -> str:
    """Fix F-1 split_B — 장소_dom 제외 + 장소미기재 접미"""
    base = build_segment_name(purpose_dom, None, topic_dom)
    if base == "기타":
        return "장소미기재"
    return f"{base}·장소미기재"


def build_merge_key_display_name(
    cluster_df: pd.DataFrame,
    recruit_type: str,
    *,
    place_missing: bool = False,
    dominant_threshold: float = DOMINANT_THRESHOLD,
) -> str:
    """
    표시명 — dominant 병합키와 동일
    전체 1등 + 70% · 목적·장소·주제·(시술) · 불명확·혼재 제외
    """
    purpose = compute_dominant(cluster_df, COL_PURPOSE, dominant_threshold)
    topic = compute_dominant(cluster_df, COL_TOPIC, dominant_threshold)

    if place_missing:
        return build_place_missing_name(purpose, topic)

    place = compute_dominant(cluster_df, COL_PLACE, dominant_threshold)
    include_treatment = recruit_type != "photo"
    treatment = compute_dominant(cluster_df, COL_TREATMENT, dominant_threshold)

    ci = ClusterInfo(
        cluster_id=0,
        n=len(cluster_df),
        purpose_dom=purpose,
        place_dom=place,
        topic_dom=topic,
        treatment_dom=treatment,
        career_dom=compute_dominant(cluster_df, COL_CAREER, dominant_threshold),
        continuity_dom=compute_dominant(
            cluster_df, COL_CONTINUITY, dominant_threshold
        ),
        include_treatment_in_key=include_treatment,
    )
    return ci.segment_key


def build_display_segment_name(
    cluster_df: pd.DataFrame,
    recruit_type: str,
    *,
    place_missing: bool = False,
    dominant_threshold: float = DOMINANT_THRESHOLD,
) -> str:
    """표시명 — merge_key dominant 규칙"""
    return build_merge_key_display_name(
        cluster_df,
        recruit_type,
        place_missing=place_missing,
        dominant_threshold=dominant_threshold,
    )


def apply_display_segment_names(
    cell_df: pd.DataFrame,
    recruit_type: str,
    *,
    dominant_threshold: float = DOMINANT_THRESHOLD,
) -> pd.DataFrame:
    """셀 단위 표시명 — merge_key dominant 규칙으로 재부여"""
    out = cell_df.copy()
    op = out[out["merged_segment_id"] != -1]
    if op.empty:
        return out

    for mid, grp in op.groupby("merged_segment_id"):
        mid = int(mid)
        current = str(grp["segment_key"].iloc[0])
        place_missing = current.endswith("·장소미기재")
        name = build_merge_key_display_name(
            grp,
            recruit_type,
            place_missing=place_missing,
            dominant_threshold=dominant_threshold,
        )
        out.loc[out["merged_segment_id"] == mid, "segment_key"] = name
    return out


def segment_key_from_merge_key(mk: tuple) -> str:
    """merge_key tuple → 표시용 segment_key (3~4축)"""
    parts: list[str] = []
    for dom in mk:
        s = str(dom).strip()
        if s in (UNCLEAR_VALUE, HYBRID_LABEL, ""):
            continue
        parts.append(s)
    return "·".join(parts) if parts else "기타"


def dominant_value(series: pd.Series, threshold: float = DOMINANT_THRESHOLD) -> str:
    """Series 입력 래퍼 — compute_dominant와 동일 로직"""
    if series.empty:
        return "혼재"
    col = series.name or "__dim__"
    return compute_dominant(series.to_frame(name=col), col, threshold)


def cluster_info_for_labels(
    cell_df: pd.DataFrame,
    label_col: str,
    *,
    include_treatment: bool = False,
    dominant_threshold: float = DOMINANT_THRESHOLD,
) -> dict[int, ClusterInfo]:
    info: dict[int, ClusterInfo] = {}
    for cid, grp in cell_df.groupby(label_col):
        cid = int(cid)
        if cid == -1:
            continue
        info[cid] = ClusterInfo(
            cluster_id=cid,
            n=len(grp),
            purpose_dom=dominant_value(grp[COL_PURPOSE], dominant_threshold),
            place_dom=dominant_value(grp[COL_PLACE], dominant_threshold),
            topic_dom=dominant_value(grp[COL_TOPIC], dominant_threshold),
            treatment_dom=dominant_value(grp[COL_TREATMENT], dominant_threshold),
            career_dom=dominant_value(grp[COL_CAREER], dominant_threshold),
            continuity_dom=dominant_value(grp[COL_CONTINUITY], dominant_threshold),
            include_treatment_in_key=include_treatment,
        )
    return info


def merge_key_similarity(a: tuple, b: tuple) -> int:
    return sum(1 for x, y in zip(a, b) if x == y)


def absorb_small_segments(
    segments: dict[tuple, MergedSegment],
    *,
    min_size: int = MIN_SEGMENT_SIZE,
) -> dict[tuple, MergedSegment]:
    changed = True
    while changed:
        changed = False
        small_keys = [k for k, s in segments.items() if s.n < min_size]
        if not small_keys:
            break
        for key in sorted(small_keys, key=lambda k: segments[k].n):
            if key not in segments or segments[key].n >= min_size:
                continue
            if len(segments) <= 1:
                break
            seg = segments[key]
            candidates = [(k, v) for k, v in segments.items() if k != key]
            target_key, target_seg = max(
                candidates,
                key=lambda kv: (merge_key_similarity(key, kv[0]), kv[1].n),
            )
            target_seg.source_clusters.extend(seg.source_clusters)
            target_seg.n += seg.n
            del segments[key]
            changed = True
    return segments


def merge_clusters_by_key(
    cell_df: pd.DataFrame,
    label_col: str,
    key_fn,
    *,
    include_treatment: bool = False,
    dominant_threshold: float = DOMINANT_THRESHOLD,
    absorb_small: bool = True,
    min_segment_size: int = MIN_SEGMENT_SIZE,
) -> tuple[pd.DataFrame, dict[tuple, MergedSegment], dict[int, MergedSegment], int]:
    """
    군집 라벨 → merge_key 병합 → merged_segment_id
    key_fn: ClusterInfo → merge key tuple
    """
    cluster_info = cluster_info_for_labels(
        cell_df,
        label_col,
        include_treatment=include_treatment,
        dominant_threshold=dominant_threshold,
    )
    segments: dict[tuple, MergedSegment] = {}
    label_to_key: dict[int, tuple] = {}

    for cid, ci in cluster_info.items():
        mk = key_fn(ci)
        label_to_key[cid] = mk
        if mk not in segments:
            label = segment_key_from_merge_key(mk)
            segments[mk] = MergedSegment(merge_key=mk, segment_key=label)
        segments[mk].source_clusters.append((cid, ci.n))
        segments[mk].n += ci.n

    n_before = len(cluster_info)
    if absorb_small:
        segments = absorb_small_segments(segments, min_size=min_segment_size)

    sorted_keys = sorted(segments.keys(), key=lambda k: segments[k].n, reverse=True)
    merged_by_id: dict[int, MergedSegment] = {
        i: segments[k] for i, k in enumerate(sorted_keys)
    }

    label_to_merged: dict[int, int] = {}
    for mid, seg in merged_by_id.items():
        for cid, _ in seg.source_clusters:
            label_to_merged[cid] = mid

    out = cell_df.copy()
    out["merged_segment_id"] = out[label_col].map(
        lambda x: -1 if int(x) == -1 else label_to_merged[int(x)]
    )
    id_to_key = {mid: seg.segment_key for mid, seg in merged_by_id.items()}
    out["segment_key"] = out["merged_segment_id"].map(
        lambda x: "정보 부족형" if int(x) == -1 else id_to_key[int(x)]
    )
    return out, segments, merged_by_id, n_before
