#!/usr/bin/env python3
"""Step 5 — 타입별 RandomForest 학습·평가·모델 저장"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from block3.constants import BLOCK3_TYPES, CLASSIFIER_PKL


def _fname(rt: str) -> str:
    return "model" if rt == "model" else rt


def train_one(rt: str) -> dict:
    enc = Path(config.BLOCK3_ENCODED_DIR)
    cl = Path(config.BLOCK3_CLUSTERING_DIR)
    fname = _fname(rt)

    X = np.load(enc / f"{fname}_X.npy")
    y = np.load(cl / f"{fname}_labels.npy")

    if len(X) < 5:
        return {
            "skipped": True,
            "reason": "too_few_samples",
            "accuracy": None,
            "f1_weighted": None,
        }

    strat = y if len(np.unique(y)) > 1 else None
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
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    f1w = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    models_dir = Path(config.BLOCK3_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = models_dir / CLASSIFIER_PKL[rt]
    with pkl_path.open("wb") as f:
        pickle.dump(clf, f)

    eval_dir = Path(config.BLOCK3_EVAL_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title(f"{rt} confusion matrix")
    plt.ylabel("true")
    plt.xlabel("pred")
    plt.tight_layout()
    png_path = eval_dir / f"{fname}_confusion_matrix.png"
    plt.savefig(png_path)
    plt.close()

    print(f"[{rt}] accuracy={acc:.4f} f1_weighted={f1w:.4f} → {pkl_path}")
    return {"accuracy": round(acc, 6), "f1_weighted": round(f1w, 6), "skipped": False}


def main() -> None:
    report: dict = {}
    for rt in BLOCK3_TYPES:
        report[rt] = train_one(rt)

    eval_dir = Path(config.BLOCK3_EVAL_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = eval_dir / "classifier_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nclassifier_report 저장: {out_json}")


if __name__ == "__main__":
    main()
