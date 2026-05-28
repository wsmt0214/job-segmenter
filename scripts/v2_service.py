"""v2 Task 6 — 신규 구인글 자동 클러스터 배정 서비스"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pymysql

import config
from v2_inference import CLUSTER_VERSION, V2Classifier, build_classifier
from v2_phase3_core import extract_attributes

LOG_DIR = Path(config.PROJECT_DIR) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "v2_service.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

LOW_CONF_THRESHOLD = 0.7
BATCH_SLEEP_SEC = 0.3


def get_db_connection():
    return pymysql.connect(**config.DB_CONFIG)


def get_categories_for_recruit(recruit_id: int, conn) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT c.{config.CATEGORY_NAME_COL}
        FROM {config.CATEGORY_JOIN_TABLE} rc
        JOIN {config.CATEGORY_TABLE} c ON c.{config.CATEGORY_ID_COL} = rc.{config.CATEGORY_ID_COL}
        WHERE rc.{config.CATEGORY_JOIN_JOB_ID_COL} = %s
        ORDER BY c.{config.CATEGORY_ID_COL}
        """,
        (recruit_id,),
    )
    rows = cur.fetchall()
    return [str(r[0]) for r in rows] if rows else []


def update_db_assignment(
    conn,
    recruit_id: int,
    cluster_id: int,
    confidence: float,
    version: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE {config.JOB_POSTINGS_TABLE}
        SET {config.SEGMENT_ID_COL} = %s,
            {config.SEGMENT_VERSION_COL} = %s,
            {config.SEGMENT_CONFIDENCE_COL} = %s,
            {config.SEGMENT_ASSIGNED_AT_COL} = NOW()
        WHERE {config.JOB_POSTING_ID_COL} = %s
        """,
        (cluster_id, version, confidence, recruit_id),
    )
    conn.commit()


def log_failure(recruit_id: int, reason: str, extra: dict | None = None) -> None:
    payload = {
        "ts": datetime.now().isoformat(),
        "recruitId": recruit_id,
        "reason": reason,
    }
    if extra:
        payload.update(extra)
    logger.warning("FAIL recruitId=%s reason=%s", recruit_id, reason)
    with (LOG_DIR / "v2_failures.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def process_single(
    recruit_id: int,
    title: str,
    content: str,
    categories: list[str],
    payment: int,
    recruit_type: str,
    schema: dict,
    classifier: V2Classifier,
) -> dict:
    """Phase3 추출 → RF 예측"""
    pg = config.get_payment_group(payment)
    rt = str(recruit_type or "").strip().lower()

    if rt not in config.RECRUIT_TYPES:
        return {
            "cluster_id": None,
            "confidence": 0.0,
            "segment_key": None,
            "status": "invalid_recruit_type",
            "payment_group": pg,
            "recruit_type": rt,
        }

    attrs = extract_attributes(schema, categories, title, content, retries=2)
    if attrs is None:
        return {
            "cluster_id": None,
            "confidence": 0.0,
            "segment_key": None,
            "status": "phase3_fail",
            "payment_group": pg,
            "recruit_type": rt,
        }

    pred = classifier.predict(pg, rt, attrs)
    pred["payment_group"] = pg
    pred["recruit_type"] = rt
    pred["attributes"] = attrs
    return pred


def fetch_unassigned(conn, limit: int) -> list[tuple]:
    cur = conn.cursor()
    t = config.JOB_POSTINGS_TABLE
    cur.execute(
        f"""
        SELECT {config.JOB_POSTING_ID_COL}, {config.TITLE_COL}, {config.CONTENT_COL},
               {config.PAYMENT_COL}, {config.JOB_TYPE_COL}
        FROM {t}
        WHERE {config.active_jobs_sql()}
          AND {config.SEGMENT_ID_COL} IS NULL
        ORDER BY {config.JOB_POSTING_ID_COL}
        LIMIT %s
        """,
        (limit,),
    )
    return list(cur.fetchall())


def batch_mode(limit: int = 500) -> None:
    logger.info("=== v2.1 배치 배정 시작 (limit=%s) ===", limit)

    classifier, schema, _ = build_classifier()
    conn = get_db_connection()
    rows = fetch_unassigned(conn, limit)

    if not rows:
        logger.info("배정할 구인글 없음")
        conn.close()
        return

    logger.info("배정 대상: %s건", len(rows))
    success, fail, low_conf, info_poor = 0, 0, 0, 0

    for recruit_id, title, content, payment, recruit_type in rows:
        categories = get_categories_for_recruit(int(recruit_id), conn)
        result = process_single(
            int(recruit_id),
            title or "",
            content or "",
            categories,
            payment or 0,
            recruit_type or "",
            schema,
            classifier,
        )

        status = result["status"]
        if status in ("ok", "info_poor", "no_rf_model"):
            cid = int(result["cluster_id"])
            conf = float(result["confidence"])
            update_db_assignment(conn, int(recruit_id), cid, conf, CLUSTER_VERSION)
            success += 1
            if status == "info_poor":
                info_poor += 1
            elif status == "ok" and conf < LOW_CONF_THRESHOLD:
                low_conf += 1
                logger.info(
                    "  LOW_CONF recruitId=%s cell=%s×%s cluster=%s conf=%.3f seg=%s",
                    recruit_id,
                    result.get("recruit_type"),
                    result.get("payment_group"),
                    cid,
                    conf,
                    result.get("segment_key"),
                )
        else:
            fail += 1
            log_failure(
                int(recruit_id),
                status,
                {
                    "recruit_type": result.get("recruit_type"),
                    "payment_group": result.get("payment_group"),
                },
            )

        time.sleep(BATCH_SLEEP_SEC)

    conn.close()
    logger.info(
        "완료: 성공 %s건 / 실패 %s건 / 정보부족 %s건 / 저신뢰도 %s건",
        success,
        fail,
        info_poor,
        low_conf,
    )


def single_mode(recruit_id: int) -> None:
    classifier, schema, _ = build_classifier()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {config.TITLE_COL}, {config.CONTENT_COL}, {config.PAYMENT_COL}, {config.JOB_TYPE_COL}
        FROM {config.JOB_POSTINGS_TABLE}
        WHERE {config.JOB_POSTING_ID_COL} = %s
        """,
        (recruit_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"recruitId={recruit_id} 없음")
        conn.close()
        return

    title, content, payment, recruit_type = row
    categories = get_categories_for_recruit(recruit_id, conn)
    conn.close()

    pg = config.get_payment_group(payment)
    rt = str(recruit_type or "").strip().lower()
    print(f"recruitId={recruit_id}")
    print(f"recruitType: {rt}")
    print(f"카테고리: {categories}")
    print(f"payment: {payment} → {pg} ({config.PAYMENT_GROUPS.get(pg, pg)})")
    print(f"RF 모델: {'있음' if classifier.has_model(pg, rt) else '없음(SKIP)'}")

    result = process_single(
        recruit_id,
        title or "",
        content or "",
        categories,
        payment or 0,
        recruit_type or "",
        schema,
        classifier,
    )

    print(f"status: {result['status']}")
    print(f"cluster_id: {result['cluster_id']}")
    print(f"confidence: {result.get('confidence', 0):.4f}")
    print(f"segment_key: {result.get('segment_key')}")
    print(f"cluster_version: {result.get('cluster_version', CLUSTER_VERSION)}")
    if result.get("attributes"):
        print("attributes:")
        for k, v in result["attributes"].items():
            print(f"  {k}: {v}")


def stats_mode() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    t = config.JOB_POSTINGS_TABLE
    cur.execute(
        f"""
        SELECT
            {config.SEGMENT_VERSION_COL},
            COUNT(*) AS total,
            COUNT({config.SEGMENT_ID_COL}) AS assigned,
            SUM(CASE WHEN {config.SEGMENT_ID_COL} = -1 THEN 1 ELSE 0 END) AS info_poor,
            AVG({config.SEGMENT_CONFIDENCE_COL}) AS avg_conf
        FROM {t}
        WHERE {config.active_jobs_sql()}
        GROUP BY {config.SEGMENT_VERSION_COL}
        """
    )
    rows = cur.fetchall()

    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM {t}
        WHERE {config.active_jobs_sql()}
          AND {config.SEGMENT_ID_COL} IS NULL
        """
    )
    unassigned = cur.fetchone()[0]
    conn.close()

    print("=== v2 배정 통계 ===")
    print(f"미배정({config.SEGMENT_ID_COL} IS NULL): {unassigned}건")
    for version, total, assigned, info_poor, avg_conf in rows:
        version = version or "미배정(NULL version)"
        avg_conf = avg_conf or 0
        info_poor = info_poor or 0
        print(
            f"  [{version}] 전체 {total}건 / 배정 {assigned}건 / "
            f"정보부족(-1) {info_poor}건 / 평균 신뢰도 {avg_conf:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="v2.1 클러스터 배정 서비스")
    parser.add_argument(
        "--mode",
        choices=["batch", "single", "stats"],
        default="batch",
    )
    parser.add_argument("--id", type=int, default=None, help="single 모드 recruitId")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="batch 모드 1회 처리 상한",
    )
    args = parser.parse_args()

    if args.mode == "batch":
        batch_mode(limit=args.limit)
    elif args.mode == "single":
        if args.id is None:
            print("--id 필요")
            sys.exit(1)
        single_mode(args.id)
    elif args.mode == "stats":
        stats_mode()


if __name__ == "__main__":
    main()
