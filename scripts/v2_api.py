"""v2 클러스터 배정 API (FastAPI)"""
from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from v2_inference import (
    CLUSTER_VERSION,
    INFO_POOR_SEGMENT_ID,
    V2Classifier,
    build_classifier,
    group_key,
    load_segment_catalog,
)
from v2_phase3_core import extract_attributes

LOG_DIR = Path(config.PROJECT_DIR) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "v2_api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8766
BATCH_MAX = 50

_classifier: V2Classifier | None = None
_schema: dict = {}
_attr_names: list[str] = []
_catalog: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _schema, _attr_names, _catalog
    logger.info("v2 API 시작: 모델 로딩...")
    _classifier, _schema, _attr_names = build_classifier()
    _catalog = load_segment_catalog()
    logger.info("RF 로드 완료 (cells=%d)", len(_classifier.models))
    yield
    logger.info("v2 API 종료")


app = FastAPI(
    title="job-segmenter v2 API",
    version=CLUSTER_VERSION,
    lifespan=lifespan,
)


class SegmentRequest(BaseModel):
    recruit_id: int | None = Field(None, description="recruitId (로깅용, 필수 아님)")
    recruit_type: str = Field(..., description="model | beauty | photo")
    title: str = Field("", description="구인글 제목")
    content: str = Field("", description="구인글 본문")
    categories: list[str] = Field(default_factory=list, description="카테고리 목록")
    payment: int = Field(0, description="payment 값 (-2=상호무페이, -3=협찬, else=페이)")


class BatchSegmentRequest(BaseModel):
    items: list[SegmentRequest] = Field(..., max_length=BATCH_MAX)


class PredictOnlyRequest(BaseModel):
    """Phase3 속성을 이미 갖고 있을 때 RF만 호출"""
    recruit_type: str = Field(..., description="model | beauty | photo")
    payment: int = Field(0, description="payment 값")
    attributes: dict[str, str] = Field(..., description="Phase3 7개 속성 dict")


class SegmentResponse(BaseModel):
    recruit_id: int | None = None
    cluster_id: int | None = None
    segment_key: str | None = None
    confidence: float = 0.0
    cluster_version: str = CLUSTER_VERSION
    payment_group: str = ""
    recruit_type: str = ""
    status: str = ""
    attributes: dict[str, str] | None = None
    elapsed_ms: float = 0.0


class BatchSegmentResponse(BaseModel):
    results: list[SegmentResponse]
    total: int
    success: int
    failed: int
    elapsed_ms: float


def _process_one(req: SegmentRequest) -> SegmentResponse:
    t0 = time.time()
    rt = req.recruit_type.strip().lower()
    pg = config.get_payment_group(req.payment)

    if rt not in config.RECRUIT_TYPES:
        return SegmentResponse(
            recruit_id=req.recruit_id,
            status="invalid_recruit_type",
            recruit_type=rt,
            payment_group=pg,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    attrs = extract_attributes(
        _schema, req.categories, req.title, req.content, retries=2
    )
    if attrs is None:
        return SegmentResponse(
            recruit_id=req.recruit_id,
            status="phase3_fail",
            recruit_type=rt,
            payment_group=pg,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    pred = _classifier.predict(pg, rt, attrs)
    elapsed = (time.time() - t0) * 1000

    return SegmentResponse(
        recruit_id=req.recruit_id,
        cluster_id=pred["cluster_id"],
        segment_key=pred.get("segment_key"),
        confidence=pred["confidence"],
        cluster_version=pred["cluster_version"],
        payment_group=pg,
        recruit_type=rt,
        status=pred["status"],
        attributes=attrs,
        elapsed_ms=round(elapsed, 1),
    )


@app.post("/segment", response_model=SegmentResponse)
def segment(body: SegmentRequest) -> SegmentResponse:
    """단건 세그먼트 배정 (Phase3 LLM 추출 + RF 예측)"""
    result = _process_one(body)
    if result.status in ("phase3_fail", "invalid_recruit_type"):
        logger.warning(
            "FAIL recruit_id=%s status=%s",
            body.recruit_id,
            result.status,
        )
    return result


@app.post("/segment/batch", response_model=BatchSegmentResponse)
def segment_batch(body: BatchSegmentRequest) -> BatchSegmentResponse:
    """다건 세그먼트 배정 (최대 50건)"""
    if len(body.items) > BATCH_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"최대 {BATCH_MAX}건까지 가능 (요청: {len(body.items)}건)",
        )

    t0 = time.time()
    results: list[SegmentResponse] = []
    success = 0

    for req in body.items:
        r = _process_one(req)
        results.append(r)
        if r.status in ("ok", "info_poor", "no_rf_model"):
            success += 1

    elapsed = (time.time() - t0) * 1000
    return BatchSegmentResponse(
        results=results,
        total=len(results),
        success=success,
        failed=len(results) - success,
        elapsed_ms=round(elapsed, 1),
    )


@app.post("/segment/predict-only", response_model=SegmentResponse)
def segment_predict_only(body: PredictOnlyRequest) -> SegmentResponse:
    """Phase3 속성이 이미 있을 때 RF 예측만 수행"""
    t0 = time.time()
    rt = body.recruit_type.strip().lower()
    pg = config.get_payment_group(body.payment)

    if rt not in config.RECRUIT_TYPES:
        return SegmentResponse(
            status="invalid_recruit_type",
            recruit_type=rt,
            payment_group=pg,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    pred = _classifier.predict(pg, rt, body.attributes)
    elapsed = (time.time() - t0) * 1000

    return SegmentResponse(
        cluster_id=pred["cluster_id"],
        segment_key=pred.get("segment_key"),
        confidence=pred["confidence"],
        cluster_version=pred["cluster_version"],
        payment_group=pg,
        recruit_type=rt,
        status=pred["status"],
        attributes=body.attributes,
        elapsed_ms=round(elapsed, 1),
    )


@app.get("/segments")
def get_all_segments() -> dict:
    """전체 세그먼트 카탈로그 (셀별 segment_id → segment_key)"""
    cells = _catalog.get("cells", {})
    summary: dict = {
        "cluster_version": CLUSTER_VERSION,
        "info_poor_segment_id": INFO_POOR_SEGMENT_ID,
        "total_cells": len(cells),
        "cells": {},
    }
    for key, cell in cells.items():
        summary["cells"][key] = {
            "label": cell.get("label"),
            "recruit_type": cell.get("recruit_type"),
            "payment_group": cell.get("payment_group"),
            "n_segments": cell.get("n_segments"),
            "segments": cell.get("segments"),
        }
    return summary


@app.get("/segments/{cell_key}")
def get_cell_segments(cell_key: str) -> dict:
    """특정 셀(예: n2_model)의 세그먼트 목록"""
    cells = _catalog.get("cells", {})
    if cell_key not in cells:
        available = list(cells.keys())
        raise HTTPException(
            status_code=404,
            detail=f"셀 '{cell_key}' 없음. 가능한 셀: {available}",
        )
    cell = cells[cell_key]
    return {
        "cell_key": cell_key,
        "label": cell.get("label"),
        "recruit_type": cell.get("recruit_type"),
        "payment_group": cell.get("payment_group"),
        "cluster_version": CLUSTER_VERSION,
        "info_poor_segment_id": INFO_POOR_SEGMENT_ID,
        "n_segments": cell.get("n_segments"),
        "segments": cell.get("segments"),
    }


@app.get("/stats")
def get_stats() -> dict:
    """DB 기준 배정 통계"""
    import pymysql

    conn = pymysql.connect(**config.DB_CONFIG)
    cur = conn.cursor()
    t = config.JOB_POSTINGS_TABLE

    cur.execute(f"""
        SELECT
            {config.SEGMENT_VERSION_COL},
            COUNT(*) AS total,
            SUM(CASE WHEN {config.SEGMENT_ID_COL} IS NOT NULL THEN 1 ELSE 0 END) AS assigned,
            SUM(CASE WHEN {config.SEGMENT_ID_COL} = -1 THEN 1 ELSE 0 END) AS info_poor,
            AVG({config.SEGMENT_CONFIDENCE_COL}) AS avg_conf
        FROM {t}
        WHERE {config.active_jobs_sql()}
        GROUP BY {config.SEGMENT_VERSION_COL}
    """)
    version_rows = cur.fetchall()

    cur.execute(f"""
        SELECT COUNT(*)
        FROM {t}
        WHERE {config.active_jobs_sql()} AND {config.SEGMENT_ID_COL} IS NULL
    """)
    unassigned = cur.fetchone()[0]

    cur.execute(f"""
        SELECT COUNT(*)
        FROM {t}
        WHERE {config.active_jobs_sql()}
    """)
    total_recruits = cur.fetchone()[0]
    conn.close()

    versions = []
    for version, total, assigned, info_poor, avg_conf in version_rows:
        versions.append({
            "cluster_version": version or "NULL",
            "total": total,
            "assigned": assigned or 0,
            "info_poor": info_poor or 0,
            "avg_confidence": round(float(avg_conf or 0), 4),
        })

    return {
        "total_recruits": total_recruits,
        "unassigned": unassigned,
        "current_version": CLUSTER_VERSION,
        "by_version": versions,
    }


@app.get("/health")
def health() -> dict:
    """서비스 상태 확인"""
    return {
        "status": "ok",
        "cluster_version": CLUSTER_VERSION,
        "rf_cells_loaded": len(_classifier.models) if _classifier else 0,
        "skipped_cells": sorted(_classifier.skipped_cells) if _classifier else [],
        "timestamp": datetime.now().isoformat(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "내부 서버 오류", "error": str(exc)},
    )


def assert_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError as e:
            raise RuntimeError(f"포트 {port} 이미 사용 중 ({e})") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 클러스터 배정 API")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    assert_port_free(args.host, args.port)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
