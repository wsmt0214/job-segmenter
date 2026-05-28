"""Phase 5: 구조화 속성 원-핫 후 KMeans 세그먼트 및 마케터 검토용 리포트"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from schema_attrs import (
    CLUSTERING_NA_VALUE,
    attr_names_for_type,
    clustering_attr_union,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder


def load_data():
    """ok 행만 사용해 타입별 속성 격차는 CLUSTERING_NA_VALUE 로 패딩"""
    schema_path = f"{config.DATA_DIR}/schema_definition.json"
    phase4_path = f"{config.DATA_DIR}/phase4_results.jsonl"

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    union_cols = clustering_attr_union(schema)
    records = []
    with open(phase4_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok"):
                continue
            rt = str(r.get("recruitType", ""))
            attrs = r.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            expected = set(attr_names_for_type(schema, rt))
            rec = {
                "recruitId": int(r["recruitId"]),
                "recruitType": rt,
            }
            for c in union_cols:
                if c in expected:
                    rec[c] = attrs.get(c, "불명확")
                else:
                    rec[c] = CLUSTERING_NA_VALUE
            records.append(rec)

    if not records:
        raise SystemExit("클러스터링할 ok=true 레코드가 없음")

    df = pd.DataFrame(records)
    feat_cols = list(union_cols)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(df[feat_cols])
    print(f"데이터: {len(df)}건 / 원-핫 특성: {X.shape[1]}개")
    return df, X, feat_cols


def find_k(X, k_range=range(5, 25)):
    scores = []
    print(f"\nK 탐색 ({k_range.start}~{k_range.stop - 1}):")
    for k in k_range:
        best = -1.0
        for _ in range(10):
            km = KMeans(n_clusters=k, init="k-means++", n_init=1)
            labels = km.fit_predict(X)
            s = silhouette_score(X, labels, sample_size=min(5000, len(X)))
            best = max(best, s)
        scores.append((k, best))
        print(f"  K={k}: 실루엣={best:.4f}")

    ks, ss = zip(*scores)
    plt.figure(figsize=(10, 4))
    plt.plot(ks, ss, "b-o")
    plt.xlabel("K")
    plt.ylabel("실루엣 점수")
    plt.title("K별 실루엣 점수")
    plt.tight_layout()
    plt.savefig(f"{config.DATA_DIR}/silhouette_scores.png")
    plt.close()
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\n최적 K: {best_k} | 그래프: data/silhouette_scores.png")
    return best_k


def cluster_and_report(df, X, k, attr_names):
    print(f"\nK={k} 최종 클러스터링 (10회 반복)...")
    best_km, best_s = None, -1.0
    for i in range(10):
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=i)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        if s > best_s:
            best_s, best_km = s, km

    df = df.copy()
    df["segment_id"] = best_km.predict(X)
    df[["recruitId", "segment_id"]].to_csv(
        f"{config.DATA_DIR}/segment_assignments.csv", index=False
    )

    # 마케터 검토 리포트 (세그먼트별 속성·급종 상위 분포)
    report_path = f"{config.DATA_DIR}/marketer_review_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"세그먼트 수: {k} / 총 구인글: {len(df)}\n\n")
        for sid in sorted(df["segment_id"].unique()):
            seg = df[df["segment_id"] == sid]
            f.write(
                f"{'=' * 50}\n세그먼트 {sid} ({len(seg)}건, {len(seg) / len(df) * 100:.1f}%)\n"
            )
            for col in attr_names:
                top = seg[col].value_counts().head(3)
                f.write(
                    f"  {col}: "
                    + ", ".join(f"{v}({c})" for v, c in top.items())
                    + "\n"
                )
            rt_dist = seg["recruitType"].value_counts().head(5)
            f.write(
                "  급종(recruitType): "
                + ", ".join(f"{v}({c})" for v, c in rt_dist.items())
                + "\n\n"
            )

    print("세그먼트 배정 저장: data/segment_assignments.csv")
    print(f"마케터 검토 리포트: {report_path}")
    print("\n⚠️  marketer_review_report.txt 검토 후 세그먼트 의미 확인")
    print("   병합 필요 시 segment_assignments.csv 수동 수정 가능")


def run(k_fixed: int | None = None, k_min: int = 5, k_max: int = 24) -> None:
    df, X, attr_names = load_data()
    if k_fixed is not None:
        best_k = k_fixed
        print(f"\n고정 K 사용: {best_k} (실루엣 스윕 생략)")
    else:
        best_k = find_k(X, k_range=range(k_min, k_max + 1))
    cluster_and_report(df, X, best_k, attr_names)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 KMeans 세그먼트")
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="고정 클러스터 수 (지정 시 실루엣 K 탐색 생략)",
    )
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=24)
    args = ap.parse_args()
    run(k_fixed=args.k, k_min=args.k_min, k_max=args.k_max)


if __name__ == "__main__":
    main()
