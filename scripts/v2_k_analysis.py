"""v2.0 Task 4 — CH 지수 + 실루엣 한계 이득으로 의미 있는 K 탐색"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from schema_v2 import clustering_feature_cols, load_schema
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

import v2_clustering as vc
from v2_k_selection import N_SEEDS, recommend_k, silhouette_elbow_k

SegmentKey = tuple[str, str]
ANALYSIS_K_MAX: dict[SegmentKey, int] = {
    k: v for k, v in vc.CELL_K_MAX.items() if v > 1
}


@dataclass
class KAnalysisResult:
    recruit_type: str
    payment_group: str
    n: int
    k_cap: int
    ks: list[int]
    silhouettes: list[float]
    ch_scores: list[float]
    delta_sil: list[tuple[int, float]]  # (K, Δsil) for K >= 4
    ch_best_k: int
    sil_elbow_k: int
    match: str
    recommended_k: int


def evaluate_k(X: np.ndarray, k: int) -> tuple[float, float, np.ndarray]:
    """5 seed 중 최고 실루엣 run의 실루엣·CH·라벨 반환"""
    best_s, best_labels = -1.0, None
    for seed in range(N_SEEDS):
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed)
        labels = km.fit_predict(X)
        s = silhouette_score(
            X, labels, sample_size=min(3000, len(X)), random_state=42
        )
        if s > best_s:
            best_s, best_labels = s, labels
    ch = calinski_harabasz_score(X, best_labels)
    return best_s, ch, best_labels


def analyze_cell(
    X: np.ndarray,
    recruit_type: str,
    payment_group: str,
) -> KAnalysisResult:
    k_cap = ANALYSIS_K_MAX[(recruit_type, payment_group)]
    effective_max = min(k_cap, len(X) - 1)
    effective_min = min(vc.K_MIN, effective_max)
    ks = list(range(effective_min, effective_max + 1))

    silhouettes: list[float] = []
    ch_scores: list[float] = []

    print(f"  K={effective_min}~{effective_max} (상한={k_cap}):")
    for k in ks:
        sil, ch, _ = evaluate_k(X, k)
        silhouettes.append(sil)
        ch_scores.append(ch)
        delta = sil - silhouettes[-2] if len(silhouettes) > 1 else 0.0
        d_str = f", Δsil={delta:+.4f}" if len(silhouettes) > 1 else ""
        print(f"    K={k}: 실루엣={sil:.4f}, CH={ch:.1f}{d_str}")

    ch_best_k = ks[int(np.argmax(ch_scores))]
    sil_elbow_k = silhouette_elbow_k(ks, silhouettes)

    delta_sil = [
        (ks[i], silhouettes[i] - silhouettes[i - 1])
        for i in range(1, len(ks))
    ]

    recommended_k, match = recommend_k(ch_best_k, sil_elbow_k)

    return KAnalysisResult(
        recruit_type=recruit_type,
        payment_group=payment_group,
        n=len(X),
        k_cap=k_cap,
        ks=ks,
        silhouettes=silhouettes,
        ch_scores=ch_scores,
        delta_sil=delta_sil,
        ch_best_k=ch_best_k,
        sil_elbow_k=sil_elbow_k,
        match=match,
        recommended_k=recommended_k,
    )


def plot_cell_analysis(result: KAnalysisResult, out_dir: Path) -> Path:
    slug = vc.segment_slug(result.recruit_type, result.payment_group)
    rec_k = result.recommended_k
    ks = result.ks

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"{result.recruit_type} x {result.payment_group} "
        f"(n={result.n:,}, recommended K={rec_k})",
        fontsize=12,
    )

    # 위: 실루엣 + Δsilhouette (이중 y축)
    ax_top.plot(ks, result.silhouettes, "b-o", label="Silhouette", linewidth=2)
    ax_top.set_ylabel("Silhouette", color="b")
    ax_top.tick_params(axis="y", labelcolor="b")
    ax_top.axvline(rec_k, color="r", linestyle="--", linewidth=1.5, label=f"rec K={rec_k}")
    ax_top.grid(True, alpha=0.3)

    ax_delta = ax_top.twinx()
    delta_ks = [d[0] for d in result.delta_sil]
    delta_vals = [d[1] for d in result.delta_sil]
    if delta_ks:
        ax_delta.bar(
            delta_ks, delta_vals, alpha=0.35, color="orange", width=0.6, label="Δsilhouette"
        )
    ax_delta.set_ylabel("Δsilhouette", color="darkorange")
    ax_delta.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_delta.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # 아래: CH 지수
    ax_bot.plot(ks, result.ch_scores, "g-s", label="CH index", linewidth=2)
    ax_bot.axvline(rec_k, color="r", linestyle="--", linewidth=1.5, label=f"rec K={rec_k}")
    ax_bot.axvline(
        result.ch_best_k, color="g", linestyle=":", linewidth=1, alpha=0.7,
        label=f"CH max K={result.ch_best_k}",
    )
    ax_bot.axvline(
        result.sil_elbow_k, color="orange", linestyle=":", linewidth=1, alpha=0.7,
        label=f"Δsil elbow K={result.sil_elbow_k}",
    )
    ax_bot.set_xlabel("K")
    ax_bot.set_ylabel("Calinski-Harabasz")
    ax_bot.legend(loc="upper left")
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"k_analysis_{slug}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def print_summary(results: list[KAnalysisResult]) -> None:
    print("\n=== K 결정 분석 결과 ===\n")
    header = (
        f"{'셀':<16} {'CH 최대 K':>8} {'실루엣 elbow K':>14} "
        f"{'일치 여부':<28} {'추천 K':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{vc.cell_short_label(r.recruit_type, r.payment_group):<16} "
            f"{r.ch_best_k:>8} {r.sil_elbow_k:>14} "
            f"{r.match:<28} {r.recommended_k:>6}"
        )

    print("\n추천 K 기준:")
    print("  - 일치 / ±1 이내 → Δsil elbow K")
    print("  - 불일치 → Δsil elbow K 우선 (CH max=3 단조 문제 회피)")


def run() -> list[KAnalysisResult]:
    schema = load_schema()
    feat_cols = clustering_feature_cols(schema)
    out_dir = Path(config.V2_DATA_DIR)

    print("=== CH + Δsilhouette K 분석 ===\n")
    print(f"벡터 (dimensions 6): {feat_cols}")
    print(f"분석 셀: {len(ANALYSIS_K_MAX)}개 (photo×n3 제외)\n")

    type_map = vc.load_type_map()
    segments = vc.load_records_by_segment(type_map, vc.CELL_K_MAX)

    results: list[KAnalysisResult] = []

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            if (rt, pg) not in ANALYSIS_K_MAX:
                continue

            records = segments[(rt, pg)]
            print(f"\n[{vc.cell_short_label(rt, pg)}] {len(records)}건")
            feat_df = vc.build_feature_frame(records, feat_cols, rt, pg)
            if feat_df.empty or len(feat_df) < vc.K_MIN:
                print("  데이터 부족 — 건너뜀")
                continue

            X, _ = vc.prepare_features(feat_df, feat_cols)
            result = analyze_cell(X, rt, pg)
            plot_path = plot_cell_analysis(result, out_dir)
            print(f"  → 그래프: {plot_path}")
            print(
                f"  → CH max K={result.ch_best_k}, "
                f"Δsil elbow K={result.sil_elbow_k}, "
                f"추천 K={result.recommended_k} ({result.match})"
            )
            results.append(result)

    print_summary(results)
    return results


if __name__ == "__main__":
    run()
