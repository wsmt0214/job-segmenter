"""Phase 6: 세그먼트 라벨 기준 TF-IDF 베이스라인 vs KLUE-BERT 학습 및 배포 규칙 저장"""
from __future__ import annotations

import argparse
import inspect
import json
import pickle
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
import seaborn as sns
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


def prepare_data():
    seg = pd.read_csv(f"{config.DATA_DIR}/segment_assignments.csv")
    raw = pd.read_csv(f"{config.DATA_DIR}/raw_recruits.csv")
    df = seg.merge(raw[["recruitId", "title", "content"]], on="recruitId")
    df["text"] = df["title"].fillna("").astype(str) + " " + df["content"].fillna("").astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    X = df["text"].to_numpy()
    y = df["segment_id"].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_cm(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.ylabel("실제")
    plt.xlabel("예측")
    plt.tight_layout()
    plt.savefig(f"{config.DATA_DIR}/{name}.png")
    plt.close()


def train_baseline(X_tr, X_te, y_tr, y_te):
    print("\n=== 베이스라인: TF-IDF + 로지스틱 회귀 ===")
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(vec.fit_transform(X_tr), y_tr)
    y_pred = clf.predict(vec.transform(X_te))

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    print(f"정확도: {acc:.4f} / 매크로 F1: {f1:.4f}")
    print(classification_report(y_te, y_pred))
    save_cm(y_te, y_pred, "cm_baseline")

    with open(f"{config.MODEL_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(f"{config.MODEL_DIR}/tfidf_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    return f1


def _training_args_compat(**kwargs):
    """transformers 버전에 따라 eval_strategy / evaluation_strategy 선택"""
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments)
    if "eval_strategy" not in sig.parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def train_klue_bert(X_tr, X_te, y_tr, y_te, train_bs: int):
    print("\n=== KLUE-BERT 파인튜닝 ===")
    from datasets import Dataset
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
    )

    model_name = "klue/bert-base"

    unique_labels = sorted(set(y_tr))
    l2i = {l: i for i, l in enumerate(unique_labels)}
    y_tr_i = np.array([l2i[l] for l in y_tr])
    y_te_i = np.array([l2i[l] for l in y_te])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(unique_labels)
    )

    def tokenize(texts, labels):
        enc = tokenizer(list(texts), truncation=True, padding=True, max_length=512)
        enc["labels"] = list(labels)
        return Dataset.from_dict(enc)

    args = _training_args_compat(
        output_dir=f"{config.MODEL_DIR}/klue_bert_classifier",
        num_train_epochs=5,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=min(32, train_bs * 2),
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenize(X_tr, y_tr_i),
        eval_dataset=tokenize(X_te, y_te_i),
    )
    trainer.train()

    out = trainer.predict(tokenize(X_te, y_te_i))
    logits = out.predictions
    y_pred_i = np.argmax(logits, axis=1)
    proba = softmax(logits, axis=1)
    conf = proba.max(axis=1)
    print(f"신뢰도 0.7 이상 비율: {(conf >= 0.7).mean() * 100:.1f}%")

    acc = accuracy_score(y_te_i, y_pred_i)
    f1 = f1_score(y_te_i, y_pred_i, average="macro")
    print(f"정확도: {acc:.4f} / 매크로 F1: {f1:.4f}")
    print(classification_report(y_te_i, y_pred_i))
    save_cm(y_te_i, y_pred_i, "cm_bert")
    return float(f1)


def decide_and_save(baseline_f1: float, bert_f1: float | None, skipped_bert: bool):
    if skipped_bert or bert_f1 is None:
        result = {
            "baseline_macro_f1": round(baseline_f1, 4),
            "bert_macro_f1": None,
            "improvement": None,
            "deploy_bert": False,
            "deployed_model": "tfidf",
            "skipped_bert": True,
        }
    else:
        threshold = baseline_f1 + 0.05
        deploy_bert = bert_f1 >= threshold and (bert_f1 - baseline_f1) >= 0.05
        result = {
            "baseline_macro_f1": round(baseline_f1, 4),
            "bert_macro_f1": round(bert_f1, 4),
            "improvement": round(bert_f1 - baseline_f1, 4),
            "deploy_bert": deploy_bert,
            "deployed_model": "klue_bert" if deploy_bert else "tfidf",
            "skipped_bert": False,
        }

    with open(f"{config.DATA_DIR}/classifier_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== 배포 결정 ===")
    print(f"베이스라인 F1: {baseline_f1:.4f}")
    if skipped_bert:
        print("KLUE-BERT: 스킵 → TF-IDF 배포")
    else:
        print(f"KLUE-BERT F1:  {bert_f1:.4f} (개선 {bert_f1 - baseline_f1:.4f})")
        print(f"→ {'KLUE-BERT 배포' if result['deploy_bert'] else 'TF-IDF 배포'}")


def run(skip_bert: bool, train_bs: int) -> None:
    X_tr, X_te, y_tr, y_te = prepare_data()
    baseline_f1 = train_baseline(X_tr, X_te, y_tr, y_te)
    if skip_bert:
        decide_and_save(baseline_f1, None, skipped_bert=True)
        return
    bert_f1 = train_klue_bert(X_tr, X_te, y_tr, y_te, train_bs)
    decide_and_save(baseline_f1, bert_f1, skipped_bert=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6 세그먼트 분류기")
    ap.add_argument(
        "--skip-bert",
        action="store_true",
        help="BERT 학습 생략 (베이스라인만 학습·평가)",
    )
    ap.add_argument(
        "--bert-batch-size",
        type=int,
        default=16,
        help="VRAM 부족 시 8 등으로 감소",
    )
    args = ap.parse_args()
    run(skip_bert=args.skip_bert, train_bs=args.bert_batch_size)


if __name__ == "__main__":
    main()
