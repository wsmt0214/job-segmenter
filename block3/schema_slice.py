"""급종별 Block 3 입력 속성 정의 (beauty는 공통 6만 — 기록_촬영 제외)"""

from __future__ import annotations

from block3.constants import BLOCK3_TYPES


def block3_attr_defs(schema: dict, recruit_type: str) -> list[dict]:
    """스키마에서 허용 값 순서가 보존된 속성 정의 목록"""
    if recruit_type not in BLOCK3_TYPES:
        raise ValueError(f"지원 급종 아님: {recruit_type}")

    common = schema.get("common_attributes")
    if not isinstance(common, list):
        raise ValueError("common_attributes 는 배열이어야 함")

    if recruit_type == "beauty":
        return [dict(a) for a in common]

    spec = schema.get("type_specific") or {}
    extra = spec.get(recruit_type)
    if extra is None:
        extra = []
    if not isinstance(extra, list):
        raise ValueError(f"type_specific[{recruit_type}] 는 배열이어야 함")
    return [dict(a) for a in common] + [dict(a) for a in extra]
