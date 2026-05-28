"""학습된 RF·프로필 로드 후 군집 예측"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

import config
from block3.constants import BLOCK3_TYPES, CLASSIFIER_PKL
from block3.encode import encode_attributes_dict
from block3.io_phase4 import load_schema
from block3.schema_slice import block3_attr_defs


def _profiles_path(clustering_dir: Path, recruit_type: str) -> Path:
    return clustering_dir / f"{recruit_type}_profiles.json"


def predict_cluster(
    recruit_type: str,
    attributes: dict,
    *,
    models_dir: str | Path | None = None,
    clustering_dir: str | Path | None = None,
    schema: dict | None = None,
) -> dict:
    """타입별 분류기로 군집 예측 — 반환에 cluster_id, cluster_size, confidence, dominant_values"""
    rt = str(recruit_type)
    if rt not in BLOCK3_TYPES:
        raise ValueError(f"지원 급종 아님: {rt}")

    models_dir = Path(models_dir or config.BLOCK3_MODELS_DIR)
    clustering_dir = Path(clustering_dir or config.BLOCK3_CLUSTERING_DIR)
    schema = schema if schema is not None else load_schema()

    attr_defs = block3_attr_defs(schema, rt)
    X = encode_attributes_dict(attributes, attr_defs)

    pkl_name = CLASSIFIER_PKL[rt]
    with (models_dir / pkl_name).open("rb") as f:
        clf = pickle.load(f)

    proba = clf.predict_proba(X)[0]
    cid = int(np.argmax(proba))
    confidence = float(np.max(proba))

    prof_path = _profiles_path(clustering_dir, rt)
    with prof_path.open(encoding="utf-8") as f:
        prof = json.load(f)

    clusters = prof.get("clusters") or []
    meta = next((c for c in clusters if int(c.get("cluster_id", -1)) == cid), None)
    if meta is None:
        raise RuntimeError(f"프로필에 cluster_id={cid} 없음 ({prof_path})")

    return {
        "cluster_id": cid,
        "cluster_size": int(meta.get("size", 0)),
        "confidence": confidence,
        "dominant_values": dict(meta.get("dominant_values") or {}),
    }
