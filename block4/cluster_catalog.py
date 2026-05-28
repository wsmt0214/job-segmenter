"""클러스터 프로필 JSON에서 세그먼트 표시 이름 생성"""

from __future__ import annotations

import json
from pathlib import Path

from block3.schema_slice import block3_attr_defs

from block4.text_dataset import npy_stem


def format_segment_name(dominant: dict[str, str], attr_order: list[str]) -> str:
    """dominant_values를 스키마 속성 순으로 이어 붙임"""
    parts: list[str] = []
    for key in attr_order:
        if key in dominant:
            parts.append(str(dominant[key]))
    return " - ".join(parts) if parts else "미분류"


def load_cluster_catalog(
    recruit_type: str,
    clustering_dir: Path,
    schema: dict,
) -> dict[int, dict]:
    """cluster_id → {name, size}"""
    stem = npy_stem(recruit_type)
    path = clustering_dir / f"{stem}_profiles.json"
    attr_order = [str(a["name"]) for a in block3_attr_defs(schema, recruit_type)]

    with path.open(encoding="utf-8") as f:
        blob = json.load(f)

    out: dict[int, dict] = {}
    for c in blob.get("clusters") or []:
        cid = int(c["cluster_id"])
        dom = c.get("dominant_values") or {}
        out[cid] = {
            "name": format_segment_name(dom, attr_order),
            "size": int(c.get("size", 0)),
        }
    return out
