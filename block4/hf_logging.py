"""transformers 모델 로드 시 LOAD REPORT(경고 표) 노이즈 억제"""

from __future__ import annotations

import logging


def suppress_transformers_load_report() -> None:
    """RoBERTa-base → 분류 헤드 로드 시 예상되는 mismatch 표가 warning 으로 찍히는 것 방지"""
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
