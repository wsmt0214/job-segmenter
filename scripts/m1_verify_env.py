# DB·Ollama·GPU(선택) 사전 점검
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymysql
import requests


def check_ollama():
    try:
        import config

        resp = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": "안녕",
                "stream": False,
            },
            timeout=60,
        )
        print(f"[OK] Ollama: {resp.json()['response'][:30]}")
        return True
    except Exception as e:
        print(f"[FAIL] Ollama: {e}")
        print("        → curl -fsSL https://ollama.com/install.sh | sh 후 ollama pull qwen2.5:14b")
        return False


def check_mysql():
    try:
        import config

        t = config.JOB_POSTINGS_TABLE
        conn = pymysql.connect(**config.DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        total = cur.fetchone()[0]
        cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE {config.active_jobs_sql()}"
        )
        active = cur.fetchone()[0]
        cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE {config.active_jobs_sql()} "
            f"AND {config.CREATED_AT_COL} >= '2025-07-01'"
        )
        active_2025h2 = cur.fetchone()[0]
        conn.close()
        print(
            f"[OK] MySQL: 전체 {total}건, "
            f"활성(미삭제·미만료) {active}건, "
            f"활성+2025-07-01 이후 {active_2025h2}건"
        )
        return True
    except Exception as e:
        print(f"[FAIL] MySQL: {e}")
        return False


def check_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}, VRAM {vram:.1f}GB")
        else:
            print("[WARN] CUDA 없음 — CPU 모드")
        return True
    except ImportError:
        print("[WARN] torch 미설치 — venv에서 pip install -r requirements.txt 후 재실행")
        return True


if __name__ == "__main__":
    ok_mysql = check_mysql()
    ok_ollama = check_ollama()
    check_gpu()
    sys.exit(0 if (ok_mysql and ok_ollama) else 1)
