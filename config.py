"""공통 설정. DB 비밀값은 local_config 또는 환경변수로만 둠."""
import os
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent)
DATA_DIR = f"{PROJECT_DIR}/data"
MODEL_DIR = f"{PROJECT_DIR}/data/models"
BLOCK3_MODELS_DIR = f"{PROJECT_DIR}/models"
BLOCK3_ENCODED_DIR = f"{DATA_DIR}/encoded"
BLOCK3_CLUSTERING_DIR = f"{DATA_DIR}/clustering"
BLOCK3_EVAL_DIR = f"{DATA_DIR}/evaluation"

BLOCK4_TEXT_DIR = f"{DATA_DIR}/text"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

# --- 공개용 DB 스키마 (README §7.4). 운영 환경은 APP_* 환경변수로 재정의 ---
JOB_POSTINGS_TABLE = os.environ.get("APP_JOB_TABLE", "job_postings")
JOB_POSTING_ID_COL = os.environ.get("APP_JOB_ID_COL", "id")
JOB_TYPE_COL = os.environ.get("APP_JOB_TYPE_COL", "job_type")
PAYMENT_COL = os.environ.get("APP_PAYMENT_COL", "payment")
TITLE_COL = "title"
CONTENT_COL = "content"
CREATED_AT_COL = os.environ.get("APP_CREATED_AT_COL", "created_at")
IS_DELETED_COL = os.environ.get("APP_IS_DELETED_COL", "is_deleted")
IS_EXPIRED_COL = os.environ.get("APP_IS_EXPIRED_COL", "is_expired")

SEGMENT_ID_COL = "segment_id"
SEGMENT_VERSION_COL = "segment_version"
SEGMENT_CONFIDENCE_COL = "segment_confidence"
SEGMENT_ASSIGNED_AT_COL = "segment_assigned_at"

CATEGORY_JOIN_TABLE = os.environ.get("APP_CATEGORY_JOIN_TABLE", "job_posting_categories")
CATEGORY_TABLE = os.environ.get("APP_CATEGORY_TABLE", "categories")
CATEGORY_ID_COL = os.environ.get("APP_CATEGORY_ID_COL", "category_id")
CATEGORY_NAME_COL = "name"
CATEGORY_JOIN_JOB_ID_COL = os.environ.get(
    "APP_CATEGORY_JOIN_JOB_ID_COL", "job_posting_id"
)

# 하위 호환 alias (데이터프레임·JSON 필드명)
RECRUIT_ID_COL = JOB_POSTING_ID_COL
RECRUIT_TYPE_COL = JOB_TYPE_COL


def active_jobs_sql(alias: str = "r") -> str:
    """활성 구인글 WHERE 조건 (삭제·만료 제외)."""
    a = alias
    return (
        f"{a}.{IS_DELETED_COL} = 0 "
        f"AND {a}.{IS_EXPIRED_COL} = 0"
    )


def _load_db_config() -> dict:
    try:
        from local_config import DB_CONFIG as cfg

        return dict(cfg)
    except ImportError:
        pass
    return {
        "host": os.environ.get("APP_MYSQL_HOST", "localhost"),
        "user": os.environ.get("APP_MYSQL_USER", ""),
        "password": os.environ.get("APP_MYSQL_PASSWORD", ""),
        "db": os.environ.get("APP_MYSQL_DB", "your_database"),
        "charset": "utf8mb4",
    }


DB_CONFIG = _load_db_config()

V2_DATA_DIR = f"{PROJECT_DIR}/data/v2"
V2_MODEL_DIR = f"{PROJECT_DIR}/data/v2/models"

PAYMENT_GROUPS = {
    "n2": "상호무페이",
    "n3": "협찬",
    "pay": "페이",
}

# Task 4 클러스터링 분리 기준 (v1 급종과 동일)
RECRUIT_TYPES = ("model", "beauty", "photo")
RECRUIT_TYPE_LABELS = {
    "model": "모델",
    "beauty": "뷰티",
    "photo": "포토",
}

CLUSTER_VERSION = "v2.1-tune"
V2_CLUSTER_ASSIGNMENTS_CSV = f"{V2_DATA_DIR}/cluster_assignments_v21_tune.csv"

CATEGORY_JOIN_SQL = f"""
    SELECT
        r.{JOB_POSTING_ID_COL},
        r.{PAYMENT_COL},
        GROUP_CONCAT(c.{CATEGORY_NAME_COL} ORDER BY c.{CATEGORY_ID_COL} SEPARATOR '|||') AS categories
    FROM {JOB_POSTINGS_TABLE} r
    LEFT JOIN {CATEGORY_JOIN_TABLE} rc ON r.{JOB_POSTING_ID_COL} = rc.{CATEGORY_JOIN_JOB_ID_COL}
    LEFT JOIN {CATEGORY_TABLE} c ON c.{CATEGORY_ID_COL} = rc.{CATEGORY_ID_COL}
    WHERE {active_jobs_sql("r")}
    GROUP BY r.{JOB_POSTING_ID_COL}, r.{PAYMENT_COL}
"""


def get_payment_group(payment_value) -> str:
    """payment 값을 n2/n3/pay 그룹 키로 변환"""
    val = int(payment_value) if payment_value is not None else 0
    if val == -2:
        return "n2"
    elif val == -3:
        return "n3"
    return "pay"
