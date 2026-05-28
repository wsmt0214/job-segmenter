#!/usr/bin/env python3
"""v2 배정 API 스모크·정합성 검증 (TestClient, Ollama 불필요)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import pandas as pd
from fastapi.testclient import TestClient
import v2_api
from v2_inference import build_classifier, group_key

ASSIGN_CSV = Path(config.V2_CLUSTER_ASSIGNMENTS_CSV)
PHASE3_PATH = Path(config.V2_DATA_DIR) / "phase3_results.jsonl"


def load_phase3() -> dict[int, dict]:
    out: dict[int, dict] = {}
    with PHASE3_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("attributes"):
                out[int(r["recruitId"])] = r
    return out


def check_endpoints(client: TestClient) -> list[str]:
    errors: list[str] = []
    h = client.get("/health")
    if h.status_code != 200 or h.json().get("status") != "ok":
        errors.append(f"/health 실패: {h.status_code} {h.text[:200]}")
    elif h.json().get("rf_cells_loaded", 0) < 1:
        errors.append("RF 모델 0개 로드")

    s = client.get("/segments")
    if s.status_code != 200:
        errors.append(f"/segments 실패: {s.status_code}")

    c = client.get("/segments/n2_model")
    if c.status_code != 200:
        errors.append(f"/segments/n2_model 실패: {c.status_code}")

    bad = client.post(
        "/segment/predict-only",
        json={
            "recruit_type": "invalid",
            "payment": -2,
            "attributes": {
                "촬영 장소": "불명확",
                "촬영 목적": "불명확",
                "촬영 주제": "불명확",
                "시술 종류": "불명확",
                "경력 조건": "불명확",
                "작업 지속성": "불명확",
                "긴급도": "불명확",
            },
        },
    )
    if bad.json().get("status") != "invalid_recruit_type":
        errors.append("invalid_recruit_type 미반환")

    unclear = client.post(
        "/segment/predict-only",
        json={
            "recruit_type": "model",
            "payment": -2,
            "attributes": {
                "촬영 장소": "불명확",
                "촬영 목적": "불명확",
                "촬영 주제": "불명확",
                "시술 종류": "불명확",
                "경력 조건": "경력 무관",
                "작업 지속성": "1회성",
                "긴급도": "일반",
            },
        },
    )
    d = unclear.json()
    if d.get("status") != "info_poor" or d.get("cluster_id") != -1:
        errors.append("Fix C-1 info_poor 미동작")

    skip = client.post(
        "/segment/predict-only",
        json={
            "recruit_type": "photo",
            "payment": -3,
            "attributes": {
                "촬영 장소": "스튜디오",
                "촬영 목적": "포트폴리오",
                "촬영 주제": "스냅",
                "시술 종류": "불명확",
                "경력 조건": "경력 무관",
                "작업 지속성": "1회성",
                "긴급도": "일반",
            },
        },
    )
    if skip.json().get("status") != "no_rf_model":
        errors.append("SKIP 셀(n3_photo) no_rf_model 미반환")

    return errors


def rf_agreement(assign: pd.DataFrame, phase3: dict[int, dict]) -> dict:
    clf, _, _ = build_classifier()
    ok_match = ok_total = 0
    poor_match = poor_total = 0
    by_cell: dict[str, list[int]] = {}

    for _, row in assign.iterrows():
        rid = int(row.recruitId)
        if rid not in phase3:
            continue
        rt, pg = row.recruitType, row.payment_group
        expected = int(row.merged_segment_id)
        key = group_key(pg, rt)
        pred = clf.predict(pg, rt, phase3[rid]["attributes"])
        st = pred["status"]
        pred_id = pred.get("cluster_id")

        if st == "info_poor":
            poor_total += 1
            if expected == -1:
                poor_match += 1
        elif st == "ok" and expected != -1:
            ok_total += 1
            if pred_id == expected:
                ok_match += 1
            by_cell.setdefault(key, [0, 0])
            by_cell[key][1] += 1
            if pred_id == expected:
                by_cell[key][0] += 1

    return {
        "rf_ok_match_pct": 100 * ok_match / max(ok_total, 1),
        "rf_ok_match": ok_match,
        "rf_ok_total": ok_total,
        "info_poor_match_pct": 100 * poor_match / max(poor_total, 1),
        "info_poor_match": poor_match,
        "info_poor_total": poor_total,
        "by_cell": {
            k: {"match": v[0], "total": v[1], "pct": round(100 * v[0] / max(v[1], 1), 1)}
            for k, v in sorted(by_cell.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-rf-match-pct",
        type=float,
        default=75.0,
        help="운영 세그 RF 일치율 하한 (기본 75%%)",
    )
    args = ap.parse_args()

    assign = pd.read_csv(ASSIGN_CSV, dtype={"recruitId": int})
    phase3 = load_phase3()

    with TestClient(v2_api.app) as client:
        errs = check_endpoints(client)
        h = client.get("/health").json()
        agree = rf_agreement(assign, phase3)

    print("=== v2 배정 API 검증 ===")
    print(f"cluster_version: {h.get('cluster_version')}")
    print(f"rf_cells_loaded: {h.get('rf_cells_loaded')} skipped: {h.get('skipped_cells')}")
    print(f"배정 CSV: {ASSIGN_CSV.name}")
    print(
        f"RF 일치(운영 세그): {agree['rf_ok_match']}/{agree['rf_ok_total']} "
        f"= {agree['rf_ok_match_pct']:.1f}%"
    )
    print(
        f"정보부족 일치: {agree['info_poor_match']}/{agree['info_poor_total']} "
        f"= {agree['info_poor_match_pct']:.1f}%"
    )
    for k, v in agree["by_cell"].items():
        print(f"  {k}: {v['pct']}% ({v['match']}/{v['total']})")

    if errs:
        print("\n[FAIL] 엔드포인트:")
        for e in errs:
            print(f"  - {e}")
    else:
        print("\n[PASS] 엔드포인트 스모크")

    if agree["rf_ok_match_pct"] < args.min_rf_match_pct:
        print(
            f"\n[WARN] RF 일치율 {agree['rf_ok_match_pct']:.1f}% "
            f"< 기준 {args.min_rf_match_pct}%"
        )
        return 1

    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
