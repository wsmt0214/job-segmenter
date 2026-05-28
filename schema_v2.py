"""schema_definition_v2.json 해석 공통 (Phase3 추출·클러스터링·RF 분류기 공유)"""
from __future__ import annotations

import json
from pathlib import Path

import config

SCHEMA_PATH = Path(config.V2_DATA_DIR) / "schema_definition_v2.json"
UNCLEAR_VALUE = "불명확"

# Phase3 클러스터링 직전 결정적 보정 단계 (구 Fix G / Fix T)
P3_CORRECT_HAIR = "P3-헤어"
P3_CORRECT_SNAP = "P3-스냅"
P3_CORRECT_PURPOSE = "P3-목적"
P3_CORRECT_PLACE = "P3-장소"

# P3-스냅 Tier — 인물스냅(Tier2) 적용 범위
P3_SNAP_TIER_FULL = "full"  # Tier1 + Tier2 (인물스냅 무조건)
P3_SNAP_TIER1_ONLY = "tier1_only"  # Tier1만 — 일상·여행·우정·본문 스냅
P3_SNAP_TIER2_BODY_SNAP = "tier2_body_snap"  # Tier2 — 인물스냅 + 본문 스냅 키워드 있을 때만
P3_SNAP_TIER2_CONDITIONAL = "tier2_conditional"  # Tier2 — 인물스냅 + 본문 스냅 OR 포폴 맥락
P3_SNAP_TIER_DEFAULT = P3_SNAP_TIER2_BODY_SNAP
P3_SNAP_TIERS: tuple[str, ...] = (
    P3_SNAP_TIER_FULL,
    P3_SNAP_TIER1_ONLY,
    P3_SNAP_TIER2_BODY_SNAP,
    P3_SNAP_TIER2_CONDITIONAL,
)

# 클러스터링 핵심 3차원 — 변별력 낮은 dimensions 제외
CLUSTERING_CORE_DIMS: tuple[str, ...] = ("촬영 장소", "촬영 목적", "시술 종류")

# Fix C-1 정보충실도·4축 Gower 실험 호환 — 경력·지속성 제외
CLUSTERING_4DIM: tuple[str, ...] = (
    "촬영 장소",
    "촬영 목적",
    "촬영 주제",
    "시술 종류",
)

# v2.1 튜닝 베이스 클러스터링 6차원 (Phase3 dimensions 전체)
CLUSTERING_6DIM: tuple[str, ...] = (
    "촬영 장소",
    "촬영 목적",
    "촬영 주제",
    "시술 종류",
    "경력 조건",
    "작업 지속성",
)

# Gower 축별 가중 — CLUSTERING_6DIM 순서 (장소·목적·주제·시술·경력·지속성)
# GW1 확정 — docs/v2.1_클러스터링_확정.md · 경력·지속성은 merge_key만 (Gower 거리=0)
GOWER_DIM_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 3.0, 3.0, 0.0, 0.0)

AUX_CLUSTER_TAGS: tuple[str, ...] = ()


def load_schema(path: Path | str | None = None) -> dict:
    """v2 스키마 JSON 로드"""
    p = Path(path) if path else SCHEMA_PATH
    if not p.is_file():
        raise FileNotFoundError(f"schema_definition_v2.json 없음: {p}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _as_attr_list(items: list | None) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [a for a in items if isinstance(a, dict) and a.get("name")]


def clustering_attr_defs(schema: dict) -> list[dict]:
    """클러스터링 벡터용 dimensions 6개"""
    return _as_attr_list(schema.get("dimensions"))


def filter_tag_defs(schema: dict) -> list[dict]:
    """Phase3 추출·클러스터링 제외 filter_tags"""
    return _as_attr_list(schema.get("filter_tags"))


def db_feature_defs(schema: dict) -> list[dict]:
    """DB 직접 조회 feature — Phase3 LLM 추출 대상 아님"""
    return _as_attr_list(schema.get("db_features"))


def phase3_extract_attrs(schema: dict) -> list[dict]:
    """Phase3 LLM 추출 대상: dimensions + filter_tags (7개)"""
    return clustering_attr_defs(schema) + filter_tag_defs(schema)


def clustering_attr_names(schema: dict) -> list[str]:
    """클러스터링 원-핫 칼럼 순서용 이름 목록"""
    return [str(a["name"]) for a in clustering_attr_defs(schema)]


def filter_tag_names(schema: dict) -> list[str]:
    """클러스터링 벡터 미포함 — filter_tags 이름"""
    return [str(a["name"]) for a in filter_tag_defs(schema)]


def clustering_feature_cols(schema: dict) -> list[str]:
    """클러스터링 벡터 — Phase3 dimensions 6개 (db_features·filter_tags 제외)"""
    return clustering_attr_names(schema)


def clustering_3dim_feature_cols(schema: dict) -> list[str]:
    """클러스터링 핵심 3차원 — 촬영 장소·목적·시술 종류"""
    names = set(clustering_attr_names(schema))
    missing = [d for d in CLUSTERING_CORE_DIMS if d not in names]
    if missing:
        raise ValueError(f"스키마에 없는 핵심 차원: {missing}")
    return list(CLUSTERING_CORE_DIMS)


def clustering_4dim_feature_cols(schema: dict) -> list[str]:
    """클러스터링 4차원 — Fix C-1 정보충실도·구버전 실험용"""
    names = set(clustering_attr_names(schema))
    missing = [d for d in CLUSTERING_4DIM if d not in names]
    if missing:
        raise ValueError(f"스키마에 없는 4dim 차원: {missing}")
    return list(CLUSTERING_4DIM)


def clustering_6dim_feature_cols(schema: dict) -> list[str]:
    """클러스터링 6차원 — 촬영 4축 + 경력·지속성"""
    names = set(clustering_attr_names(schema))
    missing = [d for d in CLUSTERING_6DIM if d not in names]
    if missing:
        raise ValueError(f"스키마에 없는 6dim 차원: {missing}")
    return list(CLUSTERING_6DIM)


def clustering_report_cols(schema: dict) -> list[str]:
    """마케터 리포트 표시용 — clustering_feature_cols와 동일"""
    return clustering_attr_names(schema)


def phase3_attr_names(schema: dict) -> list[str]:
    """Phase3 출력 JSON 키 순서"""
    return [str(a["name"]) for a in phase3_extract_attrs(schema)]


def valid_values_map(attrs: list[dict]) -> dict[str, set[str]]:
    """속성별 허용값 집합"""
    out: dict[str, set[str]] = {}
    for a in attrs:
        name = str(a["name"])
        values = a.get("values") or []
        out[name] = {str(v) for v in values}
    return out
