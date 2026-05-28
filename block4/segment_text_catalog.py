"""급종별 cluster_id → 프로필 특징 + 해당 글(recruitId·원문) JSON 산출"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from block4.text_dataset import npy_stem


def _profiles_index_from_blob(blob: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """cluster_id → 클러스터 블록(dict)"""
    out: dict[int, dict[str, Any]] = {}
    for c in blob.get("clusters") or []:
        cid = int(c["cluster_id"])
        out[cid] = c
    return out


def build_catalog_for_type(
    recruit_type: str,
    dataset_csv: Path,
    profiles_json: Path,
) -> dict[str, Any]:
    """
    단일 급종 카탈로그 루트 dict.
    텍스트는 data/text/*_dataset.csv, 특징은 data/clustering/*_profiles.json 기준
    """
    stem = npy_stem(recruit_type)
    df = pd.read_csv(dataset_csv, dtype={"recruitId": "int64"})
    if "cluster_id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"필수 열 없음: {dataset_csv} (recruitId, cluster_id, text)")

    with profiles_json.open(encoding="utf-8") as f:
        profile_root = json.load(f)

    prof_by_cid = _profiles_index_from_blob(profile_root)

    clusters_out: list[dict[str, Any]] = []
    for cid in sorted(df["cluster_id"].unique()):
        cid_i = int(cid)
        sub = df.loc[df["cluster_id"] == cid].sort_values("recruitId")
        posts: list[dict[str, Any]] = []
        for _, row in sub.iterrows():
            tid = int(row["recruitId"])
            raw = row["text"]
            text_s = "" if pd.isna(raw) else str(raw)
            posts.append({"recruitId": tid, "text": text_s})

        meta = prof_by_cid.get(cid_i)
        if meta is None:
            # 프로필에 없는 레이블 (파이프 불일치 시) — 특징 없이 글만 유지
            segment_traits: dict[str, Any] = {
                "note": "profiles_json에 해당 cluster_id 없음",
            }
        else:
            segment_traits = {
                "size_in_clustering_profile": int(meta.get("size", 0)),
                "dominant_attribute_values": dict(meta.get("dominant_values") or {}),
                "attribute_value_ratios": dict(meta.get("attributes") or {}),
            }

        clusters_out.append(
            {
                "cluster_id": cid_i,
                "segment_traits": segment_traits,
                "post_count": len(posts),
                "posts": posts,
            }
        )

    return {
        "recruit_type": recruit_type,
        "file_stem": stem,
        "source_dataset_csv": str(dataset_csv.resolve()),
        "source_profiles_json": str(profiles_json.resolve()),
        "k": profile_root.get("k"),
        "silhouette_score": profile_root.get("silhouette_score"),
        "clusters": clusters_out,
    }


def write_catalog_json(catalog: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
