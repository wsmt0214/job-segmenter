import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymysql
import pandas as pd

import config


def extract():
    conn = pymysql.connect(**config.DB_CONFIG)
    t = config.JOB_POSTINGS_TABLE
    id_col = config.JOB_POSTING_ID_COL
    query = f"""
        SELECT
            r.{id_col} AS recruitId,
            r.{config.TITLE_COL} AS title,
            r.{config.CONTENT_COL} AS content,
            r.{config.JOB_TYPE_COL} AS recruitType,
            r.{config.PAYMENT_COL} AS payment,
            r.{config.CREATED_AT_COL} AS createdAt
        FROM {t} r
        WHERE {config.active_jobs_sql("r")}
          AND r.{config.CREATED_AT_COL} >= '2025-07-01'
          AND r.{config.CONTENT_COL} IS NOT NULL
          AND CHAR_LENGTH(TRIM(r.{config.CONTENT_COL})) > 10
        ORDER BY r.{id_col}
    """
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"추출 건수: {len(df)}건 (2025-07-01 이후 생성, 삭제·만료 제외, 본문 조건 충족)")
    print(f"recruitType 분포:\n{df['recruitType'].value_counts()}")

    out = f"{config.DATA_DIR}/raw_recruits.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out}")


if __name__ == "__main__":
    extract()
