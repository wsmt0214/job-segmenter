#!/usr/bin/env python3
"""Step 4 — 군집별 속성 분포·dominant_values JSON"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import build_typed_dataframes, load_schema
from block3.schema_slice import block3_attr_defs


def _fname(rt: str) -> str:
    return "model" if rt == "model" else rt


def cluster_profiles_for_type(
    df: pd.DataFrame,
    labels: np.ndarray,
    attr_defs: list[dict],
    silhouette: float | None,
    k: int,
) -> dict:
    work = df.copy()
    work["_cluster"] = labels
    clusters_out = []
    for cid in sorted(work["_cluster"].unique()):
        sub = work[work["_cluster"] == cid]
        n = len(sub)
        attrs_json: dict = {}
        dominant: dict = {}
        for ad in attr_defs:
            name = str(ad["name"])
            allowed = [str(v) for v in (ad.get("values") or [])]
            vc = sub[name].value_counts()
            dist = {}
            for v in allowed:
                c = int(vc.get(v, 0))
                dist[v] = round(c / n, 6) if n else 0.0
            attrs_json[name] = dist
            dominant[name] = max(dist, key=dist.get)
        clusters_out.append(
            {
                "cluster_id": int(cid),
                "size": n,
                "attributes": attrs_json,
                "dominant_values": dominant,
            }
        )
    return {
        "type": None,
        "k": k,
        "silhouette_score": silhouette,
        "clusters": clusters_out,
    }


def main() -> None:
    schema = load_schema()
    dfs = build_typed_dataframes(schema)
    cl_dir = Path(config.BLOCK3_CLUSTERING_DIR)

    with (cl_dir / "k_scores.json").open(encoding="utf-8") as f:
        k_meta = json.load(f)

    for rt in BLOCK3_TYPES:
        fname = _fname(rt)
        df = dfs[rt]
        labels_path = cl_dir / f"{fname}_labels.npy"
        labels = np.load(labels_path)

        meta = k_meta.get(rt) or {}
        if meta.get("skipped"):
            print(f"[{rt}] Step 3 스킵됨 — 빈 프로필")
            blob = {
                "type": rt,
                "k": 0,
                "silhouette_score": None,
                "clusters": [],
            }
            out_path = cl_dir / f"{fname}_profiles.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            continue

        if df.empty:
            blob = {"type": rt, "k": 0, "silhouette_score": None, "clusters": []}
            out_path = cl_dir / f"{fname}_profiles.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            continue

        if len(df) != len(labels):
            raise SystemExit(
                f"[{rt}] DataFrame 행 수({len(df)})와 레이블 수({len(labels)}) 불일치"
            )

        defs = block3_attr_defs(schema, rt)
        best_k = int(meta.get("best_k", len(np.unique(labels))))
        sil = meta.get("best_silhouette")
        sil_f = float(sil) if isinstance(sil, (int, float)) else None

        blob = cluster_profiles_for_type(df, labels, defs, sil_f, best_k)
        blob["type"] = rt
        out_path = cl_dir / f"{fname}_profiles.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)
        print(f"[{rt}] 저장 {out_path}")


if __name__ == "__main__":
    main()
