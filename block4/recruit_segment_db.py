"""phase4·encoded·clustering 기준 recruitId→segment_id 매핑 및 DB 반영"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymysql

from block3.constants import BLOCK3_TYPES
from block3.io_phase4 import iter_phase4_ok

import config
from block4.text_dataset import npy_stem


SEGMENT_VERSION = "v1.0"


def _rid_cluster_maps() -> dict[str, dict[int, int]]:
    """급종별 recruitId → cluster_id"""
    enc = Path(config.BLOCK3_ENCODED_DIR)
    cl = Path(config.BLOCK3_CLUSTERING_DIR)
    out: dict[str, dict[int, int]] = {}
    for rt in BLOCK3_TYPES:
        stem = npy_stem(rt)
        ids = np.load(enc / f"{stem}_ids.npy")
        labels = np.load(cl / f"{stem}_labels.npy")
        if len(ids) != len(labels):
            raise ValueError(f"{rt}: ids/labels 길이 불일치")
        d: dict[int, int] = {}
        for rid, lab in zip(ids.tolist(), labels.tolist()):
            d[int(rid)] = int(lab)
        out[rt] = d
    return out


def build_update_plan() -> tuple[dict[int, int], int]:
    """
    반환: (recruitId → segment_id), 스킵 건수(encoded 에 없는 phase4 행)
    """
    maps = _rid_cluster_maps()
    target: dict[int, int] = {}
    skipped = 0
    for obj in iter_phase4_ok():
        rt = str(obj.get("recruitType", ""))
        if rt not in BLOCK3_TYPES:
            continue
        rid = int(obj["recruitId"])
        cid = maps.get(rt, {}).get(rid)
        if cid is None:
            skipped += 1
            continue
        target[rid] = int(cid)
    return target, skipped


def _table_columns(conn: pymysql.Connection, db_name: str, table: str) -> set[str]:
    sql = """
        SELECT COLUMN_NAME FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (db_name, table))
        rows = cur.fetchall()
    return {str(r[0]) for r in rows}


def ensure_recruit_segment_columns(
    conn: pymysql.Connection,
    *,
    db_name: str,
    table: str | None = None,
) -> list[str]:
    if table is None:
        table = config.JOB_POSTINGS_TABLE
    """없는 컬럼만 추가, 실행한 DDL 문구 목록 반환"""
    cols = _table_columns(conn, db_name, table)
    ddl_done: list[str] = []
    alters: list[str] = []
    if "segment_id" not in cols:
        alters.append(
            f"ALTER TABLE `{table}` ADD COLUMN `segment_id` INT NULL "
            f"COMMENT 'Block3 군집 cluster_id'"
        )
    if "segment_version" not in cols:
        alters.append(
            f"ALTER TABLE `{table}` ADD COLUMN `segment_version` VARCHAR(16) NULL "
            f"COMMENT '세그먼트 정의 버전'"
        )
    with conn.cursor() as cur:
        for stmt in alters:
            cur.execute(stmt)
            ddl_done.append(stmt)
    conn.commit()
    return ddl_done


def sync_segments_to_db(
    *,
    dry_run: bool,
    ensure_columns: bool,
    batch_size: int,
    eval_report_path: Path,
) -> dict:
    """배치 UPDATE 후 리포트 dict 반환·JSON 저장"""
    cfg = dict(config.DB_CONFIG)
    db_name = str(cfg.get("db") or cfg.get("database") or "")

    rid_to_cluster, skipped_phase4 = build_update_plan()
    pairs = list(rid_to_cluster.items())
    report: dict = {
        "dry_run": dry_run,
        "ensure_columns": ensure_columns,
        "segment_version": SEGMENT_VERSION,
        "planned_unique_recruits": len(pairs),
        "skipped_no_encoded_match": skipped_phase4,
        "updated": 0,
        "failed": 0,
        "skipped": skipped_phase4,
        "ddl_executed": [],
    }

    eval_report_path.parent.mkdir(parents=True, exist_ok=True)

    if not pairs:
        with eval_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    if dry_run:
        sample = pairs[:3]
        report["sample_updates"] = [
            {"recruitId": rid, "segment_id": cid, "segment_version": SEGMENT_VERSION}
            for rid, cid in sample
        ]
        report["note"] = "dry_run — DB 미연결·미반영"
        with eval_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    conn = pymysql.connect(**cfg)
    try:
        if ensure_columns:
            report["ddl_executed"] = ensure_recruit_segment_columns(conn, db_name=db_name)

        sql = (
            f"UPDATE {config.JOB_POSTINGS_TABLE} "
            f"SET {config.SEGMENT_ID_COL} = %s, {config.SEGMENT_VERSION_COL} = %s "
            f"WHERE {config.JOB_POSTING_ID_COL} = %s"
        )

        updated = 0
        failed = 0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            rows_params = [(cid, SEGMENT_VERSION, rid) for rid, cid in chunk]
            try:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows_params)
                conn.commit()
                updated += len(chunk)
            except pymysql.Error:
                conn.rollback()
                failed += len(chunk)

        report["updated"] = updated
        report["failed"] = failed

    finally:
        conn.close()

    with eval_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
