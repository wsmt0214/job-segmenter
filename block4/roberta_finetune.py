"""KLUE-RoBERTa 타입별 파인튜닝·테스트 평가·채택 여부 산출 (Trainer 미사용 — accelerate 불필요)"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
    set_seed,
)

from block4.hf_logging import suppress_transformers_load_report

MODEL_NAME = "klue/roberta-base"
MACRO_F1_ADOPTION_MARGIN = 0.05


def _stratified_split_indices(
    y_idx: np.ndarray,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """인덱스 기준 80/20 — 베이스라인과 동일 sklearn 분할 재현"""
    n = len(y_idx)
    indices = np.arange(n)
    uniq = np.unique(y_idx)
    strat = y_idx if len(uniq) > 1 else None
    try:
        tr, te = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        tr, te = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )
    return tr, te


def _tokenize_dataset(
    texts: list[str],
    labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 256,
) -> Dataset:
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def _tok(batch: dict) -> dict:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["labels"]
        return enc

    return ds.map(_tok, batched=True, remove_columns=["text"])


def _cluster_metrics(
    y_true_cluster: np.ndarray,
    y_pred_cluster: np.ndarray,
) -> dict[str, float]:
    acc = float(accuracy_score(y_true_cluster, y_pred_cluster))
    macro = float(
        f1_score(y_true_cluster, y_pred_cluster, average="macro", zero_division=0)
    )
    weighted = float(
        f1_score(y_true_cluster, y_pred_cluster, average="weighted", zero_division=0)
    )
    return {
        "accuracy": round(acc, 6),
        "macro_f1": round(macro, 6),
        "weighted_f1": round(weighted, 6),
    }


def finetune_roberta_for_type(
    *,
    recruit_type: str,
    dataset_csv: Path,
    models_dir: Path,
    baseline_macro_f1: float | None,
    random_state: int = 42,
    test_size: float = 0.2,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> dict:
    """
    단일 급종 CSV 학습
    반환: 메트릭·adopted·저장 경로(채택 시에만 실디렉터리 존재)
    """
    df = pd.read_csv(dataset_csv, dtype={"recruitId": np.int64})
    if df.empty:
        return {
            "skipped": True,
            "reason": "empty_csv",
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "adopted": "baseline",
            "saved_model_dir": None,
            "baseline_macro_f1": baseline_macro_f1,
            "macro_f1_gain": None,
        }

    texts_all = df["text"].fillna("").astype(str).tolist()
    cluster_ids = df["cluster_id"].to_numpy(dtype=np.int64)

    if len(df) < 5:
        return {
            "skipped": True,
            "reason": "too_few_samples",
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "adopted": "baseline",
            "saved_model_dir": None,
            "baseline_macro_f1": baseline_macro_f1,
            "macro_f1_gain": None,
        }

    sorted_clusters = sorted(np.unique(cluster_ids).tolist())
    num_labels = len(sorted_clusters)
    cid_to_idx = {int(c): i for i, c in enumerate(sorted_clusters)}
    idx_to_cid = {i: int(c) for i, c in enumerate(sorted_clusters)}

    y_idx = np.array([cid_to_idx[int(c)] for c in cluster_ids], dtype=np.int64)
    tr_pos, te_pos = _stratified_split_indices(y_idx, random_state=random_state, test_size=test_size)

    def pick_rows(pos: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        t = [texts_all[i] for i in pos]
        yi = y_idx[pos]
        yc = cluster_ids[pos]
        return t, yi, yc

    texts_tr, y_tr_idx, _y_tr_cid = pick_rows(tr_pos)
    texts_te, y_te_idx, y_te_cid = pick_rows(te_pos)

    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    eff_batch = batch_size if use_cuda else min(batch_size, 8)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"[{recruit_type}] 학습·평가용 토크나이즈 중…")
    train_ds = _tokenize_dataset(texts_tr, y_tr_idx.tolist(), tokenizer)
    eval_ds = _tokenize_dataset(texts_te, y_te_idx.tolist(), tokenizer)

    classes = np.arange(num_labels, dtype=np.int64)
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_tr_idx,
    )
    loss_weights = torch.tensor(cw, dtype=torch.float32)

    id2lab = {str(i): str(idx_to_cid[i]) for i in range(num_labels)}
    lab2id = {str(idx_to_cid[i]): i for i in range(num_labels)}

    suppress_transformers_load_report()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2lab,
        label2id=lab2id,
        ignore_mismatched_sizes=True,
    )
    print(
        f"[{recruit_type}] 사전학습 가중치 로드 완료 — 분류 헤드(classifier)는 "
        f"체크포인트에 없어 새로 초기화된 뒤 파인튜닝됨 (정상)"
    )
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=eff_batch,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, (len(train_ds) + eff_batch - 1) // eff_batch)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    loss_fct = nn.CrossEntropyLoss(weight=loss_weights.to(device))

    print(
        f"[{recruit_type}] 학습 시작 ({epochs}에폭, 배치 {eff_batch}, "
        f"steps/epoch≈{steps_per_epoch}) — 진행바가 움직일 때까지 대기"
    )
    model.train()
    for epoch_i in range(epochs):
        bar = tqdm(
            train_loader,
            desc=f"[{recruit_type}] epoch {epoch_i + 1}/{epochs}",
            leave=False,
        )
        for batch in bar:
            batch_tensors = {k: v.to(device) for k, v in batch.items()}
            labels_b = batch_tensors.pop("labels")
            outputs = model(**batch_tensors)
            logits = outputs.logits
            loss = loss_fct(
                logits.view(-1, num_labels),
                labels_b.view(-1).long(),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    print(f"[{recruit_type}] 테스트 셋 추론 중…")
    model.eval()
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eff_batch,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    pred_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in eval_loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred_chunks.append(logits.argmax(dim=-1).cpu().numpy())
    pred_idx = np.concatenate(pred_chunks)
    y_pred_cid = np.array([idx_to_cid[int(i)] for i in pred_idx], dtype=np.int64)

    metrics = _cluster_metrics(y_te_cid, y_pred_cid)

    baseline_f1 = baseline_macro_f1
    roberta_f1 = metrics["macro_f1"]
    gain = None if baseline_f1 is None else round(roberta_f1 - baseline_f1, 6)

    if baseline_f1 is None:
        adopted_roberta = True
    else:
        adopted_roberta = roberta_f1 >= baseline_f1 + MACRO_F1_ADOPTION_MARGIN

    stem = "model" if recruit_type == "model" else recruit_type
    save_dir = models_dir / f"roberta_{stem}"

    result = {
        "skipped": False,
        **metrics,
        "adopted": "roberta" if adopted_roberta else "baseline",
        "saved_model_dir": None,
        "baseline_macro_f1": baseline_f1,
        "macro_f1_gain": gain,
        "num_labels": num_labels,
        "device": str(device),
        "batch_size_used": eff_batch,
    }

    if adopted_roberta:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        result["saved_model_dir"] = str(save_dir)
        print(
            f"[{recruit_type}] RoBERTa 채택 macro_f1={roberta_f1:.4f} "
            f"(baseline {baseline_f1}, Δ={gain}) → {save_dir}"
        )
    else:
        print(
            f"[{recruit_type}] 베이스라인 유지 macro_f1={roberta_f1:.4f} "
            f"(baseline {baseline_f1}, Δ={gain}, 필요 Δ≥{MACRO_F1_ADOPTION_MARGIN})"
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
