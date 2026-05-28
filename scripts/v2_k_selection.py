"""v2.0 Task 4 — K 선택 공통 로직 (Δsilhouette elbow · CH 비교)"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

N_SEEDS = 5


def kneedle_elbow(x_vals: list[int], y_vals: list[float]) -> int:
    """엔드포인트 직선으로부터 최대 거리점 = elbow"""
    if len(x_vals) == 1:
        return x_vals[0]
    if len(x_vals) == 2:
        return x_vals[0]

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    x_n = (x - x.min()) / x_span if x_span > 0 else np.zeros_like(x)
    y_n = (y - y.min()) / y_span if y_span > 0 else np.zeros_like(y)

    dx, dy = x_n[-1] - x_n[0], y_n[-1] - y_n[0]
    norm = np.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm

    dists = [
        abs((x_n[i] - x_n[0]) * uy - (y_n[i] - y_n[0]) * ux)
        for i in range(len(x_n))
    ]
    return int(x_vals[int(np.argmax(dists))])


def silhouette_elbow_k(ks: list[int], silhouettes: list[float]) -> int:
    """Δsilhouette 시계열에서 elbow K 도출"""
    if len(ks) <= 1:
        return ks[0]

    delta_ks = [ks[i] for i in range(1, len(ks))]
    deltas = [silhouettes[i] - silhouettes[i - 1] for i in range(1, len(ks))]

    if len(delta_ks) == 1:
        return delta_ks[0]

    return kneedle_elbow(delta_ks, deltas)


def scan_silhouettes(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    n_seeds: int = N_SEEDS,
) -> tuple[list[int], list[float]]:
    """K 범위 실루엣 스캔 — seed별 최고값 중 max"""
    ks = list(range(k_min, k_max + 1))
    silhouettes: list[float] = []
    for k in ks:
        best_s = -1.0
        for seed in range(n_seeds):
            km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed)
            labels = km.fit_predict(X)
            s = silhouette_score(
                X, labels, sample_size=min(3000, len(X)), random_state=42
            )
            best_s = max(best_s, s)
        silhouettes.append(best_s)
    return ks, silhouettes


def recommend_k(ch_best_k: int, sil_elbow_k: int) -> tuple[int, str]:
    """
    추천 K 결정
    - 일치 / ±1 이내 → Δsil elbow K (CH max=3 고정 문제 회피)
    - 불일치 → Δsil elbow K 우선
    """
    diff = abs(ch_best_k - sil_elbow_k)
    if ch_best_k == sil_elbow_k:
        return sil_elbow_k, "일치 (high confidence)"
    if diff <= 1:
        return sil_elbow_k, f"±1 이내 (high confidence, Δ={diff})"
    return sil_elbow_k, f"불일치 (Δ={diff}, Δsil elbow 우선)"


def find_delta_elbow_k(ks: list[int], silhouettes: list[float]) -> int:
    """실루엣 스캔 결과에서 Δsil elbow K 반환"""
    return silhouette_elbow_k(ks, silhouettes)
