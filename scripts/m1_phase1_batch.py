import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import time

import pandas as pd
from tqdm import tqdm

import config
from m1_phase1_sample import call_llm


def run():
    df = pd.read_csv(f"{config.DATA_DIR}/raw_recruits.csv")
    out_path = f"{config.DATA_DIR}/phase1_results.jsonl"

    done_ids = set()
    try:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["recruitId"])
        print(f"이미 처리: {len(done_ids)}건, 이어서 처리합니다")
    except FileNotFoundError:
        pass

    remaining = df[~df["recruitId"].isin(done_ids)]
    print(f"처리 대상: {len(remaining)}건")

    with open(out_path, "a", encoding="utf-8") as f:
        for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
            res = call_llm(
                str(row["title"]) if pd.notna(row["title"]) else "",
                str(row["content"]) if pd.notna(row["content"]) else "",
            )
            entry = {
                "recruitId": int(row["recruitId"]),
                "recruitType": row["recruitType"],
                "llm_result": res,
                "ok": res is not None,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # 중단·wc -l 모니터링용 즉시 디스크 반영
            time.sleep(0.3)

    results = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    total = len(results)
    success = sum(1 for r in results if r["ok"])
    recruit = sum(1 for r in results if r["ok"] and r["llm_result"]["글_유형"] == "구인글")
    apply_ = sum(1 for r in results if r["ok"] and r["llm_result"]["글_유형"] == "지원글")
    print(f"\n완료: 전체 {total} / 성공 {success} / 구인글 {recruit} / 지원글 {apply_} / 실패 {total - success}")


if __name__ == "__main__":
    run()
