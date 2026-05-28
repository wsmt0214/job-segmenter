"""타입별 ids·labels·DB 텍스트 정렬 및 CSV 저장"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from block3.constants import BLOCK3_TYPES

import config
from block4.recruit_text_repository import fetch_title_content_by_ids
from block4.text_prep import build_training_text


def npy_stem(recruit_type: str) -> str:
    """encoded/clustering 파일 접두사 (model 타입은 model)"""
    rt = str(recruit_type)
    return "model" if rt == "model" else rt


def load_ids_and_labels(recruit_type: str) -> tuple[np.ndarray, np.ndarray]:
    stem = npy_stem(recruit_type)
    ids_path = Path(config.BLOCK3_ENCODED_DIR) / f"{stem}_ids.npy"
    labels_path = Path(config.BLOCK3_CLUSTERING_DIR) / f"{stem}_labels.npy"
    ids = np.load(ids_path)
    labels = np.load(labels_path)
    if ids.shape != labels.shape:
        raise ValueError(
            f"{recruit_type}: ids와 labels 길이 불일치 "
            f"{ids.shape} vs {labels.shape}"
        )
    return ids.astype(np.int64, copy=False), labels.astype(np.int64, copy=False)


def assemble_frame(
    ids: np.ndarray,
    labels: np.ndarray,
    texts_by_id: dict[int, tuple[str, str]],
) -> tuple[pd.DataFrame, list[int]]:
    """ids·labels 순서 유지하며 text 열 결합"""
    texts: list[str] = []
    missing: list[int] = []
    for rid in ids.tolist():
        rid_i = int(rid)
        pair = texts_by_id.get(rid_i)
        if pair is None:
            missing.append(rid_i)
            texts.append(build_training_text("", ""))
            continue
        texts.append(build_training_text(pair[0], pair[1]))

    df = pd.DataFrame(
        {
            "recruitId": ids,
            "text": texts,
            "cluster_id": labels,
        }
    )
    return df, missing


def build_typed_frame(
    recruit_type: str,
    texts_by_id: dict[int, tuple[str, str]],
) -> tuple[pd.DataFrame, list[int]]:
    ids, labels = load_ids_and_labels(recruit_type)
    return assemble_frame(ids, labels, texts_by_id)


def save_type_csv(recruit_type: str, df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npy_stem(recruit_type)
    path = out_dir / f"{stem}_dataset.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def print_type_stats(recruit_type: str, df: pd.DataFrame, missing_ids: list[int]) -> None:
    print(f"\n=== [{recruit_type}] ===")
    print(f"행 수: {len(df)}")
    if missing_ids:
        print(f"  경고: DB 미조회 recruitId {len(missing_ids)}건 (빈 텍스트로 저장)")
        head = missing_ids[:8]
        print(f"  예시 id: {head}")
    empty_text = int((df["text"].str.len() == 0).sum())
    if empty_text:
        print(f"  빈 text 행: {empty_text}")
    if len(df):
        dist = df["cluster_id"].value_counts().sort_index()
        parts = ", ".join(f"c{k}:{int(v)}" for k, v in dist.items())
        print(f"  군집 분포: {parts}")
