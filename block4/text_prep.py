"""구인글 제목·본문 결합 및 길이 제한 (학습·추론 공통)"""

from __future__ import annotations

import re

import pandas as pd

# 지시문: BERT 토크나이저 기준 최대 512자, 제목 선두 유지
MAX_TEXT_CHARS = 512


def normalize_field(value: object) -> str:
    """None·NaN·비문자 → 빈 문자열, 앞뒤 공백 제거"""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return s


def collapse_whitespace(s: str) -> str:
    """연속 공백·개행을 단일 공백으로"""
    return re.sub(r"\s+", " ", s).strip()


def build_training_text(title: object, content: object, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    제목과 본문을 한 문자열로 붙인 뒤 max_chars로 절단
    제목을 앞에 두므로 통상 제목 전체가 포함됨
    """
    t = collapse_whitespace(normalize_field(title))
    c = collapse_whitespace(normalize_field(content))
    if not t and not c:
        return ""
    if not t:
        combined = c
    elif not c:
        combined = t
    else:
        combined = f"{t} {c}"
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars]
