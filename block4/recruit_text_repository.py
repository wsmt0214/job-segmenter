"""job_postings 테이블에서 제목·본문 배치 조회"""

from __future__ import annotations

from typing import Iterable

import pymysql
import pymysql.cursors

import config


def _chunks(ids: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(ids), size):
        yield ids[i : i + size]


def fetch_title_content_by_ids(
    db_config: dict,
    recruit_ids: list[int],
    *,
    chunk_size: int = 500,
) -> dict[int, tuple[str, str]]:
    """
    job posting id → (title, content) 원본 문자열
    조회 누락 id는 결과에서 생략 → 호출측에서 빈 텍스트로 채움
    """
    if not recruit_ids:
        return {}

    out: dict[int, tuple[str, str]] = {}
    cfg = dict(db_config)
    id_col = config.JOB_POSTING_ID_COL
    t = config.JOB_POSTINGS_TABLE

    with pymysql.connect(
        **cfg,
        cursorclass=pymysql.cursors.DictCursor,
    ) as conn:
        with conn.cursor() as cur:
            for chunk in _chunks(recruit_ids, chunk_size):
                placeholders = ",".join(["%s"] * len(chunk))
                sql = (
                    f"SELECT {id_col}, {config.TITLE_COL}, {config.CONTENT_COL} "
                    f"FROM {t} WHERE {id_col} IN ({placeholders})"
                )
                cur.execute(sql, chunk)
                rows = cur.fetchall()
                for row in rows:
                    rid = int(row[id_col])
                    title = row.get(config.TITLE_COL)
                    content = row.get(config.CONTENT_COL)
                    t_s = "" if title is None else str(title)
                    c_s = "" if content is None else str(content)
                    out[rid] = (t_s, c_s)

    return out
