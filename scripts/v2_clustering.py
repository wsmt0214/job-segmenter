"""v2.0 Task 4 — recruitType × payment_group 교차 K-means 클러스터링"""
from __future__ import annotations

import argparse
import json
import pickle
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
import pandas as pd
from schema_v2 import (
    UNCLEAR_VALUE,
    clustering_3dim_feature_cols,
    clustering_feature_cols,
    load_schema,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

from v2_k_selection import find_delta_elbow_k, scan_silhouettes
from v2_marketer_report import write_marketer_report

PHASE3_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"
RAW_CSV = Path(config.DATA_DIR) / "raw_recruits.csv"

SegmentKey = tuple[str, str]
K_MIN = 3

# floor(셀 건수 / 30) — dimensions 6 기준
CELL_K_MAX: dict[SegmentKey, int] = {
    ("model", "n2"): 20,
    ("model", "n3"): 10,
    ("model", "pay"): 15,
    ("beauty", "n2"): 10,
    ("beauty", "n3"): 6,
    ("beauty", "pay"): 5,
    ("photo", "n2"): 10,
    ("photo", "n3"): 1,
    ("photo", "pay"): 3,
}


def derive_3dim_k_max(cap: int) -> int:
    """dimensions 3 — 기존 상한 절반, 최대 15, 소형 셀은 탐색 하한 보장"""
    if cap <= 1:
        return 1
    half = min(15, cap // 2)
    floor = 3 if cap >= 6 else 2
    return max(half, floor)


CELL_K_MAX_3DIM: dict[SegmentKey, int] = {
    key: derive_3dim_k_max(cap) for key, cap in CELL_K_MAX.items()
}


@dataclass(frozen=True)
class ClusterRunConfig:
    """클러스터링 실행 프로파일 — 6dim / 3dim / 3dim+nounk"""

    dims: int
    feat_cols: list[str]
    k_max: dict[SegmentKey, int]
    file_suffix: str  # "" | "_3dim" | "_nounk"
    drop_unclear: bool
    out_csv: Path
    report_path: Path
    vector_label: str


def get_run_config(dims: int = 6, *, nounk: bool = False) -> ClusterRunConfig:
    schema = load_schema()
    if nounk:
        return ClusterRunConfig(
            dims=3,
            feat_cols=clustering_3dim_feature_cols(schema),
            k_max=CELL_K_MAX_3DIM,
            file_suffix="_nounk",
            drop_unclear=True,
            out_csv=Path(config.V2_DATA_DIR) / "cluster_assignments_v2_nounk.csv",
            report_path=Path(config.V2_DATA_DIR) / "marketer_review_v2_nounk.txt",
            vector_label="dimensions 3, one-hot 불명확 제외",
        )
    if dims == 3:
        return ClusterRunConfig(
            dims=3,
            feat_cols=clustering_3dim_feature_cols(schema),
            k_max=CELL_K_MAX_3DIM,
            file_suffix="_3dim",
            drop_unclear=False,
            out_csv=Path(config.V2_DATA_DIR) / "cluster_assignments_v2_3dim.csv",
            report_path=Path(config.V2_DATA_DIR) / "marketer_review_v2_3dim.txt",
            vector_label="dimensions 3 (촬영 장소·목적·시술 종류)",
        )
    if dims == 6:
        return ClusterRunConfig(
            dims=6,
            feat_cols=clustering_feature_cols(schema),
            k_max=CELL_K_MAX,
            file_suffix="",
            drop_unclear=False,
            out_csv=Path(config.V2_DATA_DIR) / "cluster_assignments_v2.csv",
            report_path=Path(config.V2_DATA_DIR) / "marketer_review_v2.txt",
            vector_label="dimensions 6",
        )
    raise ValueError(f"지원하지 않는 dims: {dims}")


def load_type_map() -> dict[int, str]:
    """recruitId → recruitType (9셀 분리용)"""
    if not RAW_CSV.is_file():
        raise FileNotFoundError(f"없음: {RAW_CSV}")
    df = pd.read_csv(
        RAW_CSV,
        dtype={"recruitId": int},
        usecols=["recruitId", "recruitType"],
    )
    type_map: dict[int, str] = {}
    for _, r in df.iterrows():
        rid = int(r["recruitId"])
        rt = str(r["recruitType"]).strip().lower() if pd.notna(r["recruitType"]) else ""
        type_map[rid] = rt if rt in config.RECRUIT_TYPES else ""
    return type_map


@dataclass
class CellResult:
    recruit_type: str
    payment_group: str
    n: int
    k_cap: int
    chosen_k: int
    silhouette: float | None
    min_cluster: int
    median_cluster: float
    max_cluster: int


def segment_slug(recruit_type: str, payment_group: str) -> str:
    return f"{recruit_type}_{payment_group}"


def cell_short_label(rt: str, pg: str) -> str:
    return f"{rt} × {pg}"


def cell_label(rt: str, pg: str) -> str:
    return f"{config.RECRUIT_TYPE_LABELS[rt]} × {config.PAYMENT_GROUPS[pg]}"


def all_segment_keys() -> list[SegmentKey]:
    return [(rt, pg) for rt in config.RECRUIT_TYPES for pg in config.PAYMENT_GROUPS]


def load_records_by_segment(
    type_map: dict[int, str],
    k_max: dict[SegmentKey, int],
) -> dict[SegmentKey, list[dict]]:
    if not PHASE3_PATH.is_file():
        raise FileNotFoundError(f"없음: {PHASE3_PATH}")

    groups: dict[SegmentKey, list[dict]] = {key: [] for key in all_segment_keys()}
    skipped = 0
    with PHASE3_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = int(r["recruitId"])
            rt = type_map.get(rid, "")
            pg = r.get("payment_group")
            if rt in config.RECRUIT_TYPES and pg in config.PAYMENT_GROUPS:
                groups[(rt, pg)].append(r)
            else:
                skipped += 1

    print("=== recruitType × payment_group 건수 ===")
    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            n = len(groups[(rt, pg)])
            if n:
                print(f"  {cell_label(rt, pg)} ({rt}×{pg}): {n}건, K 상한={k_max[(rt, pg)]}")
    if skipped:
        print(f"  미매칭 건너뜀: {skipped}건")
    return groups


def build_feature_frame(
    records: list[dict],
    dim_names: list[str],
    recruit_type: str,
    payment_group: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for r in records:
        if not r.get("ok", True):
            continue
        rid = int(r["recruitId"])
        attrs = r.get("attributes") or {}
        row: dict = {"recruitId": rid}
        for name in dim_names:
            row[name] = attrs.get(name, UNCLEAR_VALUE)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["recruitType"] = recruit_type
    df["payment_group"] = payment_group
    return df


def prepare_features(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    drop_unclear: bool = False,
) -> tuple[np.ndarray, OneHotEncoder]:
    """
    원-핫 인코딩
    drop_unclear=True: '불명확'을 카테고리에서 제외 → 해당 차원 전부 0벡터
    전부 불명확인 차원은 컬럼 자체를 생략 (0-width 기여)
    """
    data = df[feat_cols].astype(str)
    active_cols: list[str] = []
    categories: list[list[str]] = []
    for col in feat_cols:
        vals = data[col]
        if drop_unclear:
            cats = sorted({v for v in vals.unique() if v != UNCLEAR_VALUE})
            if not cats:
                continue
        else:
            cats = sorted(vals.unique())
        active_cols.append(col)
        categories.append(cats)

    enc = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )
    if not active_cols:
        enc.categories_ = np.array([], dtype=object)
        enc.feature_names_in_ = np.array([], dtype=object)
        return np.zeros((len(df), 0)), enc

    enc.set_params(categories=categories)
    X = enc.fit_transform(data[active_cols])
    enc.active_cols_ = active_cols  # type: ignore[attr-defined]
    return X, enc


def find_best_k(
    X: np.ndarray,
    recruit_type: str,
    payment_group: str,
    slug: str,
    title: str,
    out_dir: Path,
    k_max: dict[SegmentKey, int],
    file_suffix: str,
    k_policy: str = "delta_elbow",
) -> int:
    """k_max 범위 K 탐색 — delta_elbow(기본) 또는 max_sil"""
    k_cap = k_max[(recruit_type, payment_group)]
    effective_max = min(k_cap, len(X) - 1)
    effective_min = min(K_MIN, effective_max)

    ks, silhouettes = scan_silhouettes(X, effective_min, effective_max)
    print(f"  K 탐색 ({effective_min}~{effective_max}, 상한={k_cap}, policy={k_policy}):")
    for k, s in zip(ks, silhouettes):
        idx = ks.index(k)
        delta = s - silhouettes[idx - 1] if idx > 0 else 0.0
        d_str = f", Δsil={delta:+.4f}" if idx > 0 else ""
        print(f"    K={k}: 실루엣={s:.4f}{d_str}")

    if k_policy == "max_sil":
        chosen_k = ks[int(np.argmax(silhouettes))]
        label = "실루엣 max"
    else:
        chosen_k = find_delta_elbow_k(ks, silhouettes)
        label = "Δsil elbow"

    plt.figure(figsize=(8, 4))
    plt.plot(ks, silhouettes, "b-o")
    plt.axvline(chosen_k, color="r", linestyle="--", label=f"{label} K={chosen_k}")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.title(f"{title} (cap={k_cap}, {label})")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / f"silhouette_v2_{slug}{file_suffix}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  → 그래프: {plot_path}")
    print(f"  → 선택 K: {chosen_k} ({label})")
    return chosen_k


def assign_single_cluster(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cluster_id"] = 0
    return out


def cluster_and_save(
    df: pd.DataFrame,
    X: np.ndarray,
    enc: OneHotEncoder,
    slug: str,
    k: int,
    model_dir: Path,
    file_suffix: str,
) -> tuple[pd.DataFrame, float]:
    print(f"  K={k} 최종 클러스터링 (10회 반복)...")
    best_km, best_s = None, -1.0
    for seed in range(10):
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed)
        labels = km.fit_predict(X)
        s = silhouette_score(
            X, labels, sample_size=min(3000, len(X)), random_state=42
        )
        if s > best_s:
            best_s, best_km = s, km

    out = df.copy()
    out["cluster_id"] = best_km.predict(X)

    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / f"kmeans_{slug}{file_suffix}.pkl").open("wb") as f:
        pickle.dump(best_km, f)
    with (model_dir / f"encoder_{slug}{file_suffix}.pkl").open("wb") as f:
        pickle.dump(enc, f)

    print(f"  실루엣 점수: {best_s:.4f}")
    return out, best_s


def remove_cell_artifacts(
    slug: str,
    model_dir: Path,
    out_dir: Path,
    file_suffix: str,
) -> None:
    for prefix in ("kmeans", "encoder"):
        p = model_dir / f"{prefix}_{slug}{file_suffix}.pkl"
        if p.is_file():
            p.unlink()
    sil = out_dir / f"silhouette_v2_{slug}{file_suffix}.png"
    if sil.is_file():
        sil.unlink()


def cluster_segment(
    df: pd.DataFrame,
    feat_cols: list[str],
    recruit_type: str,
    payment_group: str,
    out_dir: Path,
    model_dir: Path,
    k_max: dict[SegmentKey, int],
    file_suffix: str,
    drop_unclear: bool,
    k_fixed: int | None,
    k_policy: str = "delta_elbow",
) -> tuple[pd.DataFrame, CellResult]:
    n = len(df)
    seg_key = (recruit_type, payment_group)
    k_cap = k_max[seg_key]
    slug = segment_slug(recruit_type, payment_group)

    if k_cap <= 1:
        print(f"  {n}건 — K=1 (단일 군집, cluster_id=0)")
        remove_cell_artifacts(slug, model_dir, out_dir, file_suffix)
        out = assign_single_cluster(df)
        sizes = out["cluster_id"].value_counts()
        return out, CellResult(
            recruit_type, payment_group, n, k_cap, 1, None,
            int(sizes.min()), float(sizes.median()), int(sizes.max()),
        )

    if n < 3:
        print(f"  {n}건 — K-means 생략, cluster_id=0")
        out = assign_single_cluster(df)
        sizes = out["cluster_id"].value_counts()
        return out, CellResult(
            recruit_type, payment_group, n, k_cap, 1, None,
            int(sizes.min()), float(sizes.median()), int(sizes.max()),
        )

    X, enc = prepare_features(df, feat_cols, drop_unclear=drop_unclear)
    enc_note = " (불명확 제외)" if drop_unclear else ""
    print(f"  원-핫 특성: {X.shape[1]}개{enc_note}")

    if X.shape[1] == 0:
        print("  인코딩 가능 특성 0개 — K=1 (단일 군집, cluster_id=0)")
        remove_cell_artifacts(slug, model_dir, out_dir, file_suffix)
        out = assign_single_cluster(df)
        sizes = out["cluster_id"].value_counts()
        return out, CellResult(
            recruit_type, payment_group, n, k_cap, 1, None,
            int(sizes.min()), float(sizes.median()), int(sizes.max()),
        )

    if k_fixed is not None:
        chosen_k = min(k_fixed, k_cap, n - 1)
        print(f"  CLI 고정 K={chosen_k}")
    else:
        chosen_k = find_best_k(
            X, recruit_type, payment_group, slug,
            f"{recruit_type} x {payment_group}", out_dir,
            k_max, file_suffix, k_policy,
        )

    out, sil = cluster_and_save(df, X, enc, slug, chosen_k, model_dir, file_suffix)
    sizes = out["cluster_id"].value_counts()
    return out, CellResult(
        recruit_type, payment_group, n, k_cap, chosen_k, sil,
        int(sizes.min()), float(sizes.median()), int(sizes.max()),
    )


def print_results_summary(results: list[CellResult], all_df: pd.DataFrame) -> None:
    print("\n=== 재클러스터링 결과 (군집당 최소 30건 기준) ===\n")
    header = f"{'셀':<22} {'건수':>6} {'K상한':>5} {'선택K':>5} {'실루엣':>7} {'최소':>5} {'중앙값':>6} {'최대':>5}"
    print(header)
    print("-" * len(header))

    for r in results:
        sil = "—" if r.silhouette is None else f"{r.silhouette:.4f}"
        print(
            f"{cell_short_label(r.recruit_type, r.payment_group):<22} "
            f"{r.n:>6,} {r.k_cap:>5} {r.chosen_k:>5} {sil:>7} "
            f"{r.min_cluster:>5} {r.median_cluster:>6.0f} {r.max_cluster:>5}"
        )

    # 최소 1건 군집 — 과분할 잔존 여부 점검
    singleton_cells: list[str] = []
    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            sub = all_df[(all_df["recruitType"] == rt) & (all_df["payment_group"] == pg)]
            if sub.empty:
                continue
            if sub["cluster_id"].value_counts().min() == 1:
                singleton_cells.append(cell_short_label(rt, pg))

    print(f"\n최소 1건 군집 존재 셀: {len(singleton_cells)}")
    if singleton_cells:
        for name in singleton_cells:
            print(f"  - {name}")


def cleanup_stale_artifacts(model_dir: Path, out_dir: Path) -> None:
    for rt in config.RECRUIT_TYPES:
        for name in (f"kmeans_{rt}.pkl", f"encoder_{rt}.pkl", f"silhouette_v2_{rt}.png"):
            p = model_dir / name if name.endswith(".pkl") else out_dir / name
            if p.is_file():
                p.unlink()
    for pg in config.PAYMENT_GROUPS:
        for name in (f"kmeans_{pg}.pkl", f"encoder_{pg}.pkl", f"silhouette_v2_{pg}.png"):
            p = model_dir / name if name.endswith(".pkl") else out_dir / name
            if p.is_file():
                p.unlink()


def run(
    k_fixed: int | None = None,
    k_policy: str = "delta_elbow",
    dims: int = 6,
    *,
    nounk: bool = False,
) -> list[CellResult]:
    cfg = get_run_config(dims, nounk=nounk)

    mode = "nounk" if nounk else f"dims={cfg.dims}"
    print(f"=== v2.0 recruitType × payment_group 클러스터링 ({mode}, K policy={k_policy}) ===\n")
    print(f"클러스터링 벡터 ({cfg.vector_label}): {cfg.feat_cols}")
    if cfg.drop_unclear:
        print("  → one-hot: '불명확' 카테고리 미포함 (해당 차원 0벡터)\n")
    else:
        print()

    type_map = load_type_map()
    segments = load_records_by_segment(type_map, cfg.k_max)
    out_dir = Path(config.V2_DATA_DIR)
    model_dir = Path(config.V2_MODEL_DIR)

    cleanup_stale_artifacts(model_dir, out_dir)

    all_frames: list[pd.DataFrame] = []
    cell_results: list[CellResult] = []

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            records = segments[(rt, pg)]
            if not records:
                continue

            print(f"\n[{cell_label(rt, pg)}] {len(records)}건")
            df = build_feature_frame(records, cfg.feat_cols, rt, pg)
            if df.empty:
                print("  ok=true usable 0건 — 건너뜀")
                continue

            df, result = cluster_segment(
                df, cfg.feat_cols, rt, pg, out_dir, model_dir,
                cfg.k_max, cfg.file_suffix, cfg.drop_unclear, k_fixed, k_policy,
            )
            all_frames.append(df)
            cell_results.append(result)

    if not all_frames:
        raise SystemExit("클러스터링 결과 없음")

    all_df = pd.concat(all_frames, ignore_index=True)
    all_df[["recruitId", "recruitType", "payment_group", "cluster_id"]].to_csv(
        cfg.out_csv, index=False
    )
    print(f"\n군집 배정 저장: {cfg.out_csv} ({len(all_df)}건)")

    write_marketer_report(
        all_df,
        cfg.feat_cols,
        cfg.report_path,
        k_max=cfg.k_max,
        vector_label=cfg.vector_label,
        drop_unclear=cfg.drop_unclear,
    )
    print_results_summary(cell_results, all_df)
    return cell_results


def main() -> None:
    p = argparse.ArgumentParser(
        description="v2 recruitType × payment_group K-means"
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="모든 셀 고정 K",
    )
    p.add_argument(
        "--k-policy",
        choices=("delta_elbow", "max_sil"),
        default="delta_elbow",
        help="K 선택: delta_elbow(기본) 또는 max_sil",
    )
    p.add_argument(
        "--dims",
        type=int,
        choices=(3, 6),
        default=6,
        help="클러스터링 차원: 6(기본) 또는 3(핵심 차원)",
    )
    p.add_argument(
        "--nounk",
        action="store_true",
        help="3dim + one-hot에서 '불명확' 제외 → cluster_assignments_v2_nounk.csv",
    )
    args = p.parse_args()
    if args.nounk and args.dims == 6:
        dims = 3
    else:
        dims = args.dims
    if args.nounk and dims != 3:
        raise SystemExit("--nounk는 3dim 전용 (--dims 3)")
    run(k_fixed=args.k, k_policy=args.k_policy, dims=dims, nounk=args.nounk)


if __name__ == "__main__":
    main()
