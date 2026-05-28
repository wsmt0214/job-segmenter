"""schema_definition.json 해석 공통 (Phase4 추출·불명확 통계·클러스터링 공유)"""
from __future__ import annotations

# 타입에 속하지 않는 속성 칼럼은 원-핫에서 구별 가능한 더미 값
CLUSTERING_NA_VALUE = "__해당없음__"


def attrs_for_type(schema: dict, recruit_type: str) -> list[dict]:
    """급종별 허용 속성 정의 목록 (common + type_specific)"""
    common_ok = schema.get("common_attributes")
    if not isinstance(common_ok, list):
        raise ValueError("common_attributes 는 배열이어야 함")
    spec = schema.get("type_specific") or {}
    extra = spec.get(recruit_type)
    if extra is None:
        extra = []
    if not isinstance(extra, list):
        raise ValueError(f"type_specific[{recruit_type}] 는 배열이어야 함")
    return list(common_ok) + list(extra)


def attr_names_for_type(schema: dict, recruit_type: str) -> list[str]:
    """급종별 속성 이름만 순서 유지 추출"""
    return [str(a["name"]) for a in attrs_for_type(schema, recruit_type) if a.get("name")]


def clustering_attr_union(schema: dict) -> list[str]:
    """모든 급종에서 등장할 수 있는 속성 이름 합집합 (정렬, 원-핫 칼럼 순서 고정용)"""
    names: set[str] = set()
    for a in schema.get("common_attributes") or []:
        n = a.get("name")
        if n:
            names.add(str(n))
    spec = schema.get("type_specific") or {}
    for _rt, defs in spec.items():
        if not isinstance(defs, list):
            continue
        for a in defs:
            n = a.get("name")
            if n:
                names.add(str(n))
    return sorted(names)
