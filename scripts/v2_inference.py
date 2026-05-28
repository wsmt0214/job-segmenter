"""v2 Task 6 — RF 추론·정보부족(C-1) 판정"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config
from schema_v2 import CLUSTERING_4DIM, UNCLEAR_VALUE, load_schema, phase3_attr_names
from v2_segment_ops import info_density_row, is_explicit

logger = logging.getLogger(__name__)

CLUSTER_VERSION = config.CLUSTER_VERSION
INFO_POOR_SEGMENT_ID = -1
INFO_POOR_THRESHOLD = 1  # Fix C-1: 4축 score < 1

INFO_DENSITY_COLS = list(CLUSTERING_4DIM)
SEGMENT_CATALOG_PATH = Path(config.V2_DATA_DIR) / "segment_catalog_v21.json"
RF_EVAL_PATH = Path(config.V2_DATA_DIR) / "rf_eval_v21.json"


def group_key(pg: str, rt: str) -> str:
    return f"{pg}_{rt}"


def is_info_poor(attrs: dict[str, str]) -> bool:
    """Fix C-1 — 4축 전부 불명확(score=0)이면 정보부족"""
    row = pd.Series({c: attrs.get(c, UNCLEAR_VALUE) for c in INFO_DENSITY_COLS})
    score = info_density_row(row, INFO_DENSITY_COLS)
    return score < INFO_POOR_THRESHOLD


def load_segment_catalog(path: Path = SEGMENT_CATALOG_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"세그먼트 카탈로그 없음: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_rf_eval(path: Path = RF_EVAL_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"RF 평가 JSON 없음: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def segment_key_for(catalog: dict, pg: str, rt: str, cluster_id: int) -> str | None:
    cell = catalog.get("cells", {}).get(group_key(pg, rt), {})
    seg = cell.get("segments", {}).get(str(cluster_id))
    if seg:
        return str(seg.get("segment_key", ""))
    return None


class V2Classifier:
    """(recruitType × payment_group) 셀별 RF 분류기"""

    def __init__(
        self,
        model_dir: Path | None = None,
        catalog: dict | None = None,
        rf_eval: dict | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or config.V2_MODEL_DIR)
        self.catalog = catalog or load_segment_catalog()
        self.rf_eval = rf_eval or load_rf_eval()
        self.models: dict[str, object] = {}
        self.encoders: dict[str, dict] = {}
        self.skipped_cells: set[str] = set()
        self._load_all()

    def _load_all(self) -> None:
        cells = self.rf_eval.get("cells", {})
        for rt in config.RECRUIT_TYPES:
            for pg in config.PAYMENT_GROUPS:
                key = group_key(pg, rt)
                cell_eval = cells.get(key, {})
                if cell_eval.get("skipped"):
                    self.skipped_cells.add(key)
                    logger.info(
                        "RF SKIP [%s] — %s",
                        key,
                        cell_eval.get("reason", "skipped"),
                    )
                    continue

                clf_path = self.model_dir / f"rf_{key}.pkl"
                enc_path = self.model_dir / f"enc_{key}.pkl"
                if not clf_path.is_file() or not enc_path.is_file():
                    self.skipped_cells.add(key)
                    logger.warning("모델 파일 없음: %s", key)
                    continue

                with clf_path.open("rb") as f:
                    self.models[key] = pickle.load(f)
                with enc_path.open("rb") as f:
                    self.encoders[key] = pickle.load(f)
                logger.info("RF 로드: %s", key)

    def has_model(self, pg: str, rt: str) -> bool:
        return group_key(pg, rt) in self.models

    def predict(
        self,
        payment_group: str,
        recruit_type: str,
        attributes: dict[str, str],
    ) -> dict:
        """
        Phase3 속성 → cluster_id 예측

        반환: cluster_id, confidence, segment_key, status, cluster_version
        """
        pg = str(payment_group)
        rt = str(recruit_type)
        key = group_key(pg, rt)

        if rt not in config.RECRUIT_TYPES:
            return {
                "cluster_id": None,
                "confidence": 0.0,
                "segment_key": None,
                "status": "invalid_recruit_type",
                "cluster_version": CLUSTER_VERSION,
            }

        if is_info_poor(attributes):
            return {
                "cluster_id": INFO_POOR_SEGMENT_ID,
                "confidence": 1.0,
                "segment_key": "정보부족",
                "status": "info_poor",
                "cluster_version": CLUSTER_VERSION,
            }

        if key in self.skipped_cells or key not in self.models:
            return {
                "cluster_id": INFO_POOR_SEGMENT_ID,
                "confidence": 0.0,
                "segment_key": "정보부족",
                "status": "no_rf_model",
                "cluster_version": CLUSTER_VERSION,
            }

        enc_bundle = self.encoders[key]
        enc = enc_bundle["encoder"]
        attr_names = enc_bundle["attr_names"]
        row = {name: attributes.get(name, UNCLEAR_VALUE) for name in attr_names}
        X = enc.transform(pd.DataFrame([row]))

        clf = self.models[key]
        proba = clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        cluster_id = int(clf.classes_[idx])
        confidence = float(proba[idx])
        seg_key = segment_key_for(self.catalog, pg, rt, cluster_id)

        return {
            "cluster_id": cluster_id,
            "confidence": confidence,
            "segment_key": seg_key,
            "status": "ok",
            "cluster_version": CLUSTER_VERSION,
        }


def build_classifier() -> tuple[V2Classifier, dict, list[str]]:
    """schema + classifier 한 번에 로드"""
    schema = load_schema()
    attr_names = phase3_attr_names(schema)
    classifier = V2Classifier()
    return classifier, schema, attr_names
