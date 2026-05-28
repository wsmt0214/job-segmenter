"""v2 Task 5 — (recruitType × payment_group) 셀별 RF 분류기 학습"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import numpy as np
import pandas as pd
from schema_v2 import UNCLEAR_VALUE, load_schema, phase3_attr_names
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import v2_clustering as vc

PHASE3_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"
DEFAULT_ASSIGNMENTS = Path(config.V2_CLUSTER_ASSIGNMENTS_CSV)
EVAL_PATH = Path(config.V2_DATA_DIR) / "rf_eval_v21.json"
SEGMENT_CATALOG_PATH = Path(config.V2_DATA_DIR) / "segment_catalog_v21.json"

INFO_POOR_SEGMENT_ID = -1
CLUSTER_VERSION = config.CLUSTER_VERSION
MIN_TRAIN_SAMPLES = 5


def group_key(pg: str, rt: str) -> str:
    return f"{pg}_{rt}"


def load_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recruitId": int})
    required = {"recruitId", "recruitType", "payment_group", "merged_segment_id", "segment_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"배정 CSV 필수 컬럼 누락: {missing} ({path})")
    return df


def load_phase3_index(path: Path = PHASE3_PATH) -> dict[int, dict]:
    """recruitId → attributes (LLM JSONL 원본, P3 보정 없음)"""
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok", True):
                continue
            attrs = r.get("attributes") or {}
            out[int(r["recruitId"])] = attrs
    return out


def load_phase3_training_index() -> dict[int, dict]:
    """
    RF 학습용 속성 — 클러스터링·추론(extract_attributes)과 동일.

    6축: load_phase3_frame() P3-헤어/스냅/목적/장소 보정
    긴급도: JSONL 원값 (P3 보정 대상 아님)
    """
    from v2_clustering_v21 import load_phase3_frame

    phase_df, cluster_cols, _ = load_phase3_frame()
    raw = load_phase3_index()
    schema = load_schema()
    attr_names = phase3_attr_names(schema)
    extra = [n for n in attr_names if n not in cluster_cols]

    out: dict[int, dict] = {}
    for _, row in phase_df.iterrows():
        rid = int(row["recruitId"])
        rec = {c: str(row.get(c, UNCLEAR_VALUE)) for c in cluster_cols}
        raw_a = raw.get(rid, {})
        for name in extra:
            rec[name] = str(raw_a.get(name, UNCLEAR_VALUE))
        out[rid] = rec
    return out


def build_cell_training_frame(
    pg: str,
    rt: str,
    assignments: pd.DataFrame,
    phase3: dict[int, dict],
    attr_names: list[str],
) -> pd.DataFrame:
    """운영 세그먼트(≠-1)만 학습 대상"""
    cell = assignments[
        (assignments["recruitType"] == rt) & (assignments["payment_group"] == pg)
    ]
    train = cell[cell["merged_segment_id"] != INFO_POOR_SEGMENT_ID].copy()
    rows: list[dict] = []
    for _, row in train.iterrows():
        rid = int(row["recruitId"])
        attrs = phase3.get(rid)
        if not attrs:
            continue
        rec = {name: attrs.get(name, UNCLEAR_VALUE) for name in attr_names}
        rec["segment_id"] = int(row["merged_segment_id"])
        rec["segment_key"] = str(row["segment_key"])
        rec["recruitId"] = rid
        rows.append(rec)
    return pd.DataFrame(rows)


def encode_features(df: pd.DataFrame, attr_names: list[str]) -> tuple[np.ndarray, OneHotEncoder]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(df[attr_names].astype(str))
    return X, enc


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RandomForestClassifier, dict]:
    """셀별 RF 학습 + hold-out 평가"""
    n_classes = len(np.unique(y))
    strat = y if n_classes > 1 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    f1_macro = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
    probas = clf.predict_proba(X_te).max(axis=1)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_classes": int(n_classes),
        "confidence_ge_0.9": round(float((probas >= 0.9).mean()), 4),
        "confidence_ge_0.7": round(float((probas >= 0.7).mean()), 4),
        "confidence_lt_0.5": round(float((probas < 0.5).mean()), 4),
    }
    if len(y_te) > 0 and n_classes > 1:
        metrics["classification_report"] = classification_report(
            y_te, y_pred, zero_division=0, output_dict=True
        )
    return clf, metrics


def build_segment_catalog(
    df: pd.DataFrame,
    pg: str,
    rt: str,
) -> dict:
    """셀별 segment_id → segment_key·건수"""
    segments: dict[str, dict] = {}
    for sid, grp in df.groupby("segment_id"):
        sid = int(sid)
        segments[str(sid)] = {
            "segment_key": str(grp["segment_key"].iloc[0]),
            "n": int(len(grp)),
        }
    return {
        "recruit_type": rt,
        "payment_group": pg,
        "label": vc.cell_label(rt, pg),
        "cluster_version": CLUSTER_VERSION,
        "info_poor_segment_id": INFO_POOR_SEGMENT_ID,
        "n_train": int(len(df)),
        "n_segments": int(df["segment_id"].nunique()),
        "segments": segments,
    }


def train_cell(
    pg: str,
    rt: str,
    assignments: pd.DataFrame,
    phase3: dict[int, dict],
    attr_names: list[str],
    model_dir: Path,
) -> tuple[dict | None, dict | None]:
    key = group_key(pg, rt)
    label = vc.cell_label(rt, pg)
    print(f"\n[{label}] RF 분류기 학습 ({key})")

    df = build_cell_training_frame(pg, rt, assignments, phase3, attr_names)
    n_poor = int(
        (
            (assignments["recruitType"] == rt)
            & (assignments["payment_group"] == pg)
            & (assignments["merged_segment_id"] == INFO_POOR_SEGMENT_ID)
        ).sum()
    )
    print(f"  학습 대상: {len(df)}건 (정보부족 {n_poor}건 제외)")

    if len(df) < MIN_TRAIN_SAMPLES:
        print(f"  샘플 부족 — 건너뜀")
        return None, {"skipped": True, "reason": "too_few_samples", "label": label}

    n_classes = int(df["segment_id"].nunique())
    if n_classes < 2:
        print(f"  운영 세그먼트 {n_classes}개 — RF 불필요, 건너뜀")
        seg_catalog = build_segment_catalog(df, pg, rt) if not df.empty else {}
        return None, {
            "skipped": True,
            "reason": "single_segment",
            "label": label,
            "n_train": int(len(df)),
            "segment_catalog": seg_catalog,
        }

    X, enc = encode_features(df, attr_names)
    y = df["segment_id"].values
    clf, metrics = train_rf(X, y)

    clf_path = model_dir / f"rf_{key}.pkl"
    enc_path = model_dir / f"enc_{key}.pkl"
    meta_path = model_dir / f"rf_meta_{key}.json"

    with clf_path.open("wb") as f:
        pickle.dump(clf, f)
    with enc_path.open("wb") as f:
        pickle.dump({"encoder": enc, "attr_names": attr_names}, f)

    seg_catalog = build_segment_catalog(df, pg, rt)
    meta = {
        **seg_catalog,
        "attr_names": attr_names,
        "metrics": metrics,
        "clf_path": str(clf_path),
        "enc_path": str(enc_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"  정확도 {metrics['accuracy']:.4f} · "
        f"macro F1 {metrics['macro_f1']:.4f} · "
        f"클래스 {metrics['n_classes']}개"
    )
    print(
        f"  신뢰도 — ≥0.9: {metrics['confidence_ge_0.9']*100:.1f}% · "
        f"≥0.7: {metrics['confidence_ge_0.7']*100:.1f}% · "
        f"<0.5: {metrics['confidence_lt_0.5']*100:.1f}%"
    )
    print(f"  저장: {clf_path.name}, {enc_path.name}")

    eval_entry = {
        "label": label,
        "recruit_type": rt,
        "payment_group": pg,
        "skipped": False,
        **metrics,
        "clf_path": str(clf_path),
        "enc_path": str(enc_path),
        "meta_path": str(meta_path),
    }
    return seg_catalog, eval_entry


def run(assignments_path: Path = DEFAULT_ASSIGNMENTS) -> dict:
    model_dir = Path(config.V2_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema()
    attr_names = phase3_attr_names(schema)
    assignments = load_assignments(assignments_path)
    phase3 = load_phase3_training_index()

    print("=== v2 Task 5 — RF 분류기 학습 ===")
    print(f"배정 CSV: {assignments_path}")
    print(f"속성: P3 보정 6축 + JSONL 긴급도 (클러스터링·추론과 동일)")
    print(f"입력 속성 ({len(attr_names)}): {', '.join(attr_names)}")

    eval_results: dict = {
        "cluster_version": CLUSTER_VERSION,
        "assignments_csv": str(assignments_path),
        "phase3_path": str(PHASE3_PATH),
        "attr_names": attr_names,
        "cells": {},
    }
    segment_catalog: dict = {}

    for rt in config.RECRUIT_TYPES:
        for pg in config.PAYMENT_GROUPS:
            key = group_key(pg, rt)
            seg_cat, eval_entry = train_cell(
                pg, rt, assignments, phase3, attr_names, model_dir
            )
            if seg_cat:
                segment_catalog[key] = seg_cat
            if eval_entry:
                eval_results["cells"][key] = eval_entry

    with SEGMENT_CATALOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cluster_version": CLUSTER_VERSION,
                "assignments_csv": str(assignments_path),
                "info_poor_segment_id": INFO_POOR_SEGMENT_ID,
                "cells": segment_catalog,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with EVAL_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print("\n=== RF 학습 요약 ===")
    trained = 0
    skipped = 0
    for key, res in eval_results["cells"].items():
        if res.get("skipped"):
            skipped += 1
            print(f"  [{res.get('label', key)}] SKIP — {res.get('reason')}")
        else:
            trained += 1
            print(
                f"  [{res['label']}] acc={res['accuracy']:.4f} "
                f"F1={res['macro_f1']:.4f} ({res['n_classes']} classes)"
            )
    print(f"\n학습 완료: {trained}셀 · 건너뜀: {skipped}셀")
    print(f"평가 저장: {EVAL_PATH}")
    print(f"세그먼트 카탈로그: {SEGMENT_CATALOG_PATH}")
    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Task 5 RF 분류기 학습")
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS,
        help=f"클러스터 배정 CSV (기본: {DEFAULT_ASSIGNMENTS.name})",
    )
    args = parser.parse_args()
    run(args.assignments)


if __name__ == "__main__":
    main()
