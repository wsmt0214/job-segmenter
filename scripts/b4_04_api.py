#!/usr/bin/env python3
"""Block 4 Step 4 — 세그먼트 배정 FastAPI (포트 8765)"""

from __future__ import annotations

import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from block3.constants import BLOCK3_TYPES

from block4.online_segment_service import OnlineSegmentService

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
API_VERSION = "1.0"

_service = OnlineSegmentService()


def assert_port_free(host: str, port: int) -> None:
    """리슨 직전 포트 점유 여부 확인"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError as e:
            raise RuntimeError(f"포트 {port} 이미 사용 중 — 종료 ({e})") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    _service.load()
    yield


app = FastAPI(title="job-segmenter Block4", lifespan=lifespan)


class SegmentRequest(BaseModel):
    recruit_type: str = Field(..., description="model | beauty | photo")
    title: str = ""
    content: str = ""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": API_VERSION}


@app.post("/segment")
def segment(body: SegmentRequest) -> dict:
    rt = body.recruit_type.strip()
    if rt not in BLOCK3_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"recruit_type 은 {list(BLOCK3_TYPES)} 중 하나",
        )
    try:
        out = _service.predict(rt, body.title, body.content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    # 응답 스키마는 지시문 필드만 (model_type은 디버그용 제외 가능 — 로깅 테이블과 정합 위해 유지 가능)
    return {
        "cluster_id": out["cluster_id"],
        "name": out["name"],
        "confidence": out["confidence"],
        "segment_version": out["segment_version"],
    }


@app.get("/clusters/{recruit_type}")
def clusters(recruit_type: str) -> list[dict]:
    rt = recruit_type.strip()
    if rt not in BLOCK3_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"recruit_type 은 {list(BLOCK3_TYPES)} 중 하나",
        )
    try:
        return _service.clusters_for_type(rt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def main() -> None:
    assert_port_free(DEFAULT_HOST, DEFAULT_PORT)
    import uvicorn

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
