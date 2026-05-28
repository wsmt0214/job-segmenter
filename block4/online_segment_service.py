"""model_comparison.json 기준 TF-IDF 또는 RoBERTa 로더·추론"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import load_schema

import config
from block4.cluster_catalog import load_cluster_catalog
from block4.hf_logging import suppress_transformers_load_report
from block4.text_dataset import npy_stem
from block4.text_prep import build_training_text


SEGMENT_VERSION = "v1.0"


class SegmentPredictor(Protocol):
    model_type: str

    def predict(self, title: str, content: str) -> tuple[int, float]:
        """(cluster_id, confidence)"""


class TfidfSegmentPredictor:
    model_type = "tfidf"

    def __init__(self, pkl_path: Path) -> None:
        self._pipe: Any = joblib.load(pkl_path)

    def predict(self, title: str, content: str) -> tuple[int, float]:
        text = build_training_text(title, content)
        proba = self._pipe.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        cid = int(self._pipe.classes_[idx])
        return cid, float(proba[idx])


class RobertaSegmentPredictor:
    model_type = "roberta"

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        suppress_transformers_load_report()
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )
        self._model.eval()
        self._model.to(device)
        self._device = device

    def predict(self, title: str, content: str) -> tuple[int, float]:
        text = build_training_text(title, content)
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)
        idx = int(pred_idx.item())
        lab = self._model.config.id2label
        raw = lab.get(str(idx), lab.get(idx))
        if raw is None:
            raise RuntimeError(f"id2label 에 예측 인덱스 {idx} 없음")
        return int(raw), float(conf.item())


class OnlineSegmentService:
    """타입별 예측기·군집 메타 로드"""

    def __init__(self) -> None:
        self._predictors: dict[str, SegmentPredictor] = {}
        self._catalogs: dict[str, dict[int, dict]] = {}

    def load(self) -> None:
        eval_dir = Path(config.BLOCK3_EVAL_DIR)
        cmp_path = eval_dir / "model_comparison.json"
        if not cmp_path.is_file():
            raise FileNotFoundError(
                f"{cmp_path} 없음 — b4_03_finetune.py 실행 후 서버 기동"
            )

        with cmp_path.open(encoding="utf-8") as f:
            comparison = json.load(f)

        schema = load_schema()
        cl_dir = Path(config.BLOCK3_CLUSTERING_DIR)
        models_dir = Path(config.BLOCK3_MODELS_DIR)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for rt in BLOCK3_TYPES:
            self._catalogs[rt] = load_cluster_catalog(rt, cl_dir, schema)

            row = comparison.get(rt)
            if not row:
                raise KeyError(f"model_comparison.json 에 {rt} 항목 없음")

            adopted = str(row.get("adopted") or "baseline").lower()
            stem = npy_stem(rt)

            if adopted == "roberta":
                rdir = models_dir / f"roberta_{stem}"
                if not rdir.is_dir():
                    raise FileNotFoundError(
                        f"채택 RoBERTa 디렉터리 없음: {rdir} — 재학습 또는 adopted 확인"
                    )
                self._predictors[rt] = RobertaSegmentPredictor(rdir, device)
            else:
                pkl = models_dir / f"baseline_{stem}.pkl"
                if not pkl.is_file():
                    raise FileNotFoundError(
                        f"베이스라인 피클 없음: {pkl} — b4_02_baseline.py 실행"
                    )
                self._predictors[rt] = TfidfSegmentPredictor(pkl)

    def predict(
        self,
        recruit_type: str,
        title: str,
        content: str,
    ) -> dict[str, Any]:
        rt = str(recruit_type)
        if rt not in BLOCK3_TYPES:
            raise ValueError(f"지원하지 않는 recruit_type: {rt}")

        pred = self._predictors[rt]
        cid, conf = pred.predict(title or "", content or "")
        meta = self._catalogs[rt].get(cid, {})
        name = str(meta.get("name") or f"세그먼트{cid}")

        return {
            "cluster_id": cid,
            "name": name,
            "confidence": round(conf, 6),
            "segment_version": SEGMENT_VERSION,
            "model_type": pred.model_type,
        }

    def clusters_for_type(self, recruit_type: str) -> list[dict[str, Any]]:
        rt = str(recruit_type)
        if rt not in BLOCK3_TYPES:
            raise ValueError(f"지원하지 않는 recruit_type: {rt}")

        cat = self._catalogs[rt]
        rows = [
            {
                "cluster_id": cid,
                "name": info["name"],
                "size": info["size"],
            }
            for cid, info in sorted(cat.items(), key=lambda x: x[0])
        ]
        return rows
