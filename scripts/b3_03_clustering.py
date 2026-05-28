#!/usr/bin/env python3
"""Step 3 — 타입별 KMeans·실루엣·레이블 및 k_scores.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from block3.constants import BLOCK3_TYPES, K_RANGE_BY_TYPE


def _fname(rt: str) -> str:
    return "model" if rt == "model" else rt


def sweep_k(X: np.ndarray, k_list: list[int]) -> list[dict]:
    rows = []
    for k in k_list:
        if len(X) < k + 1:
            rows.append({"k": k, "silhouette": None, "note": "skip_n_lt_k"})
            continue
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=42,
            max_iter=500,
        )
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            rows.append({"k": k, "silhouette": -1.0, "note": "single_cluster"})
            continue
        ss = silhouette_score(
            X, labels, sample_size=min(8000, len(X)), random_state=42
        )
        rows.append({"k": k, "silhouette": round(float(ss), 6)})
    return rows


def main() -> None:
    enc = Path(config.BLOCK3_ENCODED_DIR)
    out_dir = Path(config.BLOCK3_CLUSTERING_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}

    for rt in BLOCK3_TYPES:
        fname = _fname(rt)
        X_path = enc / f"{fname}_X.npy"
        X = np.load(X_path)
        print(f"\n{'=' * 60}\n[{rt}] 샘플 {len(X)} / 특성 {X.shape[1] if len(X) else 0}")

        if len(X) < 3:
            print("  샘플 부족 — 스킵")
            summary[rt] = {"skipped": True, "reason": "too_few_samples"}
            np.save(out_dir / f"{fname}_labels.npy", np.zeros((len(X),), dtype=np.int64))
            continue

        k_candidates = [k for k in K_RANGE_BY_TYPE[rt] if k >= 2 and k <= len(X)]
        if not k_candidates:
            k_candidates = [min(2, len(X))]

        rows = sweep_k(X, k_candidates)
        valid_scores = [
            r
            for r in rows
            if isinstance(r.get("silhouette"), (int, float)) and r["silhouette"] >= -0.5
        ]

        if not valid_scores:
            best_k = k_candidates[0]
            best_sil = -1.0
        else:
            best = max(valid_scores, key=lambda r: r["silhouette"])
            best_k = int(best["k"])
            best_sil = float(best["silhouette"])

        all_low = all(
            r["silhouette"] < 0.15
            for r in valid_scores
            if isinstance(r.get("silhouette"), (int, float))
        )
        if valid_scores and all_low:
            print(
                "\n⚠ 경고: 해당 타입의 모든 K에 대해 실루엣이 0.15 미만 — 해석·재검토 권고\n"
            )

        print(f"\n{'K':>4} {'silhouette':>12}")
        print("-" * 20)
        for r in rows:
            sil = r.get("silhouette")
            sil_s = f"{sil:.4f}" if isinstance(sil, (int, float)) else str(sil)
            print(f"{r['k']:>4} {sil_s:>12}")

        print(f"\n→ 최적 K={best_k}, silhouette={best_sil:.4f}")

        km = KMeans(
            n_clusters=best_k,
            init="k-means++",
            n_init=20,
            random_state=42,
            max_iter=500,
        )
        labels = km.fit_predict(X)
        np.save(out_dir / f"{fname}_labels.npy", labels.astype(np.int64))

        summary[rt] = {
            "k_scores": rows,
            "best_k": best_k,
            "best_silhouette": round(best_sil, 6) if best_sil >= -0.5 else None,
            "all_below_0_15": bool(valid_scores and all_low),
        }

    with (out_dir / "k_scores.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nk_scores 저장: {out_dir / 'k_scores.json'}")


if __name__ == "__main__":
    main()
