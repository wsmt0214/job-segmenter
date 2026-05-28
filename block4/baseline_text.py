"""타입별 TF-IDF + 로지스틱 회귀 베이스라인 학습·평가"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_tfidf_logistic_pipeline() -> Pipeline:
    """지시문 고정 하이퍼파라미터"""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    analyzer="char_wb",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline_from_csv(
    *,
    recruit_type: str,
    dataset_csv: Path,
    model_out: Path,
    confusion_png: Path,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """
    단일 급종 CSV(recruitId, text, cluster_id) 학습
    반환: 메트릭 dict 또는 skipped 사유
    """
    df = pd.read_csv(dataset_csv, dtype={"recruitId": np.int64})
    if df.empty:
        return {
            "skipped": True,
            "reason": "empty_csv",
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    X = df["text"].fillna("").astype(str)
    y = df["cluster_id"].to_numpy()

    if len(df) < 5:
        return {
            "skipped": True,
            "reason": "too_few_samples",
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    uniq = np.unique(y)
    strat = y if len(uniq) > 1 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    pipe = build_tfidf_logistic_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title(f"{recruit_type} baseline (TF-IDF+LR)")
    plt.ylabel("true")
    plt.xlabel("pred")
    plt.tight_layout()
    confusion_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_png)
    plt.close()

    print(
        f"[{recruit_type}] accuracy={acc:.4f} macro_f1={macro_f1:.4f} "
        f"weighted_f1={weighted_f1:.4f} → {model_out}"
    )
    return {
        "skipped": False,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
    }


def write_baseline_report(eval_dir: Path, report: dict) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / "baseline_report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path
