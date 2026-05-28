"""phase4_results.jsonl 로딩 및 타입별 표 구성"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
from block3.constants import BLOCK3_TYPES
from block3.schema_slice import block3_attr_defs


def schema_path() -> Path:
    return Path(config.DATA_DIR) / "schema_definition.json"


def phase4_path() -> Path:
    return Path(config.DATA_DIR) / "phase4_results.jsonl"


def load_schema() -> dict:
    p = schema_path()
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def iter_phase4_ok(path: Path | None = None) -> list[dict]:
    """ok=true 행만 순서대로"""
    p = path or phase4_path()
    out: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("ok"):
                continue
            out.append(obj)
    return out


def normalize_attr_row(attrs: dict, attr_defs: list[dict]) -> dict[str, str]:
    """키 누락·허용 목록 밖 값은 불명확 또는 스키마 폴백"""
    row: dict[str, str] = {}
    raw = attrs if isinstance(attrs, dict) else {}
    for ad in attr_defs:
        name = str(ad["name"])
        allowed_list = list(ad.get("values") or [])
        allowed_set = set(allowed_list)
        v = raw.get(name)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = "불명확"
        else:
            v = str(v).strip()
        if v not in allowed_set:
            v = "불명확" if "불명확" in allowed_set else allowed_list[-1]
        row[name] = v
    return row


def build_typed_dataframes(schema: dict | None = None) -> dict[str, pd.DataFrame]:
    """급종별 DataFrame (recruitId + 해당 타입 속성 컬럼만), recruitId 정렬"""
    schema = schema or load_schema()
    buckets: dict[str, list[dict]] = defaultdict(list)
    for obj in iter_phase4_ok():
        rt = str(obj.get("recruitType", ""))
        if rt not in BLOCK3_TYPES:
            continue
        defs = block3_attr_defs(schema, rt)
        attrs = normalize_attr_row(obj.get("attributes") or {}, defs)
        rec = {"recruitId": int(obj["recruitId"]), **attrs}
        buckets[rt].append(rec)

    result: dict[str, pd.DataFrame] = {}
    for rt in BLOCK3_TYPES:
        rows = buckets[rt]
        if not rows:
            result[rt] = pd.DataFrame()
            continue
        df = pd.DataFrame(rows)
        df = df.sort_values("recruitId").reset_index(drop=True)
        result[rt] = df
    return result


def ensure_block3_dirs() -> None:
    """산출 디렉터리 생성"""
    for d in (
        config.BLOCK3_ENCODED_DIR,
        config.BLOCK3_CLUSTERING_DIR,
        config.BLOCK3_EVAL_DIR,
        config.BLOCK3_MODELS_DIR,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)
