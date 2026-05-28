"""Phase 4 정식 구조 추출: schema_definition.json 기반 Ollama JSON 추출"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import config
from schema_attrs import attrs_for_type

SCHEMA_PATH = Path(config.DATA_DIR) / "schema_definition.json"
PHASE1_PATH = Path(config.DATA_DIR) / "phase1_results.jsonl"
RAW_CSV = Path(config.DATA_DIR) / "raw_recruits.csv"
OUT_PHASE4 = Path(config.DATA_DIR) / "phase4_results.jsonl"
OUT_EVIDENCE = Path(config.DATA_DIR) / "phase4_evidence_sample.jsonl"
CHK_SUFFIX = ".checkpoint"

BASE_PROMPT = """당신은 구인 플랫폼 구인글 분석기다.
[규칙]
- 출력은 유효한 JSON **한 개**만. 앞뒤 설명·마크다운 금지
{top_level_rule}
- attributes 의 키는 아래 속성 name 과 정확히 동일하게
- 텍스트에 근거가 없으면 해당 속성 값은 반드시 "불명확"
- 허용 값 목록에만 맞춰 출력 (동의어를 값으로 바꿀 수 없음)

[추출 속성]
{attr_block}

[구인글]
제목: {title}
본문: {body}
"""

EVIDENCE_BLOCK = """
[evidence 규칙]
- "evidence" 객체의 키는 attributes 와 동일
- 각 값은 본문에서 인용한 근거 문구(80자 이내). 불명확이면 ""
"""


def _parse_json_object(text: str) -> dict | None:
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for i, chunk in enumerate(parts):
            c = chunk.strip()
            if i % 2 == 1:
                c = re.sub(r"^json\s*", "", c, flags=re.I).strip()
            if c.startswith("{"):
                t = c
                break
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = t.find("{", i)
        if j < 0:
            return None
        try:
            obj, _ = dec.raw_decode(t[j:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = j + 1


def _load_schema() -> dict:
    if not SCHEMA_PATH.is_file():
        raise SystemExit(
            f"schema_definition.json 이 없습니다. Step 2에서 수동 작성 후 저장하세요.\n경로: {SCHEMA_PATH}"
        )
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _attr_block(attrs: list[dict]) -> str:
    lines = []
    for a in attrs:
        name = a.get("name")
        desc = a.get("description", "")
        values = a.get("values") or []
        if not name or not isinstance(values, list):
            continue
        vals = ", ".join(str(v) for v in values)
        lines.append(f"- name `{name}`\n  허용 값: {vals}\n  설명: {desc}")
    return "\n".join(lines)


def _expected_attr_keys(attrs: list[dict]) -> list[str]:
    return [str(a["name"]) for a in attrs if a.get("name")]


def _call_ollama_json(prompt: str, temperature: float = 0.1) -> dict | None:
    try:
        r = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 1024},
            },
            timeout=240,
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        obj = _parse_json_object(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _build_prompt(
    attrs: list[dict],
    title: str,
    body: str,
    evidence: bool,
) -> str:
    block = _attr_block(attrs)
    tail = EVIDENCE_BLOCK if evidence else ""
    top = (
        '- 최상위 키는 "attributes", "evidence" 두 개 (evidence 값은 근거 문자열)'
        if evidence
        else '- 최상위 키는 "attributes" 하나만'
    )
    return BASE_PROMPT.format(
        top_level_rule=top,
        attr_block=block + tail,
        title=title[:300],
        body=body[:8000],
    )


def _try_payload(
    attrs_def: list[dict],
    obj: dict | None,
    keys: list[str],
    allowed_sets: dict[str, set[str]],
    evidence: bool,
) -> tuple[bool, dict, dict | None]:
    if not isinstance(obj, dict):
        return False, {}, None
    cand: dict | None = None
    if isinstance(obj.get("attributes"), dict):
        cand = obj["attributes"]
    elif not evidence and _validate_attributes(attrs_def, obj, allowed_sets):
        cand = {k: obj[k] for k in keys if k in obj}
    if cand is None or not _validate_attributes(attrs_def, cand, allowed_sets):
        return False, {}, None
    out_a = {k: cand[k] for k in keys}
    if evidence:
        ev = obj.get("evidence")
        if not isinstance(ev, dict):
            return False, {}, None
        out_e = {k: str(ev.get(k, ""))[:200] for k in keys}
        return True, out_a, out_e
    return True, out_a, None


def _validate_attributes(attrs_def: list[dict], got: dict, allowed_sets: dict[str, set[str]]) -> bool:
    if not isinstance(got, dict):
        return False
    for a in attrs_def:
        name = a.get("name")
        if not name or name not in got:
            return False
        v = got[name]
        if v not in allowed_sets.get(name, set()):
            return False
    return True


def _allowed_maps(attrs: list[dict]) -> dict[str, set[str]]:
    out = {}
    for a in attrs:
        name = str(a["name"])
        vs = a.get("values") or []
        out[name] = set(str(x) for x in vs)
    return out


def _load_phase1_recruits() -> list[dict]:
    rows: list[dict] = []
    with PHASE1_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if not o.get("ok"):
                continue
            lr = o.get("llm_result") or {}
            if lr.get("글_유형") != "구인글":
                continue
            rows.append(o)
    return rows


def _load_raw_texts() -> dict[int, tuple[str, str]]:
    if not RAW_CSV.is_file():
        raise SystemExit(f"없음: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, dtype={"recruitId": int})
    out = {}
    for _, row in df.iterrows():
        rid = int(row["recruitId"])
        t = str(row["title"]) if pd.notna(row["title"]) else ""
        c = str(row["content"]) if pd.notna(row["content"]) else ""
        out[rid] = (t, c)
    return out


def _length_bin(content_len: int, p33: float, p66: float) -> str:
    if content_len <= p33:
        return "short"
    if content_len <= p66:
        return "medium"
    return "long"


def _stratified_300(
    recruits: list[dict],
    texts: dict[int, tuple[str, str]],
    rng: random.Random,
    n_total: int = 300,
) -> list[dict]:
    types = ("model", "beauty", "photo")
    bins = ("short", "medium", "long")
    cells: dict[tuple[str, str], list[dict]] = {(rt, b): [] for rt in types for b in bins}

    for rt in types:
        sub = [r for r in recruits if str(r.get("recruitType")) == rt]
        lens: list[int] = []
        for r in sub:
            rid = int(r["recruitId"])
            L = len(texts.get(rid, ("", ""))[1])
            lens.append(L)
        arr = np.array(lens, dtype=float)
        if len(arr) >= 3:
            p33, p66 = [float(x) for x in np.percentile(arr, [100 / 3, 200 / 3])]
        else:
            p33 = float(np.max(arr)) if len(arr) else 0.0
            p66 = float(np.max(arr)) if len(arr) else 0.0
        for r, L in zip(sub, lens, strict=True):
            b = _length_bin(int(L), p33, p66)
            cells[(rt, b)].append(r)

    keys_order = [(rt, b) for rt in types for b in bins]
    per = n_total // len(keys_order)
    rem = n_total % len(keys_order)
    chosen: list[dict] = []
    used_ids: set[int] = set()

    for i, key in enumerate(keys_order):
        need = per + (1 if i < rem else 0)
        pool = [r for r in cells[key] if int(r["recruitId"]) not in used_ids]
        rng.shuffle(pool)
        take = pool[:need]
        for r in take:
            used_ids.add(int(r["recruitId"]))
        chosen.extend(take)

    deficit = n_total - len(chosen)
    if deficit > 0:
        rest = [r for r in recruits if int(r["recruitId"]) not in used_ids]
        rng.shuffle(rest)
        chosen.extend(rest[:deficit])

    return chosen[:n_total]


def run_batch(evidence: bool) -> None:
    schema = _load_schema()
    recruits = _load_phase1_recruits()
    texts = _load_raw_texts()
    out_path = OUT_EVIDENCE if evidence else OUT_PHASE4

    if evidence:
        rng = random.Random(42)
        work = _stratified_300(recruits, texts, rng, 300)
    else:
        work = recruits

    done: set[int] = set()
    if out_path.is_file() and not evidence:
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["recruitId"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        work = [r for r in work if int(r["recruitId"]) not in done]

    n_ok = n_fail = 0
    by_type_ok: dict[str, int] = defaultdict(int)
    by_type_fail: dict[str, int] = defaultdict(int)

    mode = "evidence 300" if evidence else f"전체 {len(work)} (이미 완료 {len(done)} 제외)"
    print(f"대상: {mode} -> {out_path}")

    open_mode = "w" if evidence else ("a" if out_path.exists() else "w")
    with out_path.open(open_mode, encoding="utf-8") as fout:
        for i, row in enumerate(
            tqdm(work, desc="structured_extract", total=len(work)),
            start=1,
        ):
            rid = int(row["recruitId"])
            rt = str(row.get("recruitType", ""))
            title, body = texts.get(rid, ("", ""))
            attrs = attrs_for_type(schema, rt)
            allowed = _allowed_maps(attrs)
            keys = _expected_attr_keys(attrs)
            prompt = _build_prompt(attrs, title, body, evidence=evidence)

            ok, attrs_out, evid_out = _try_payload(
                attrs, _call_ollama_json(prompt), keys, allowed, evidence
            )
            if not ok:
                ok, attrs_out, evid_out = _try_payload(
                    attrs, _call_ollama_json(prompt, temperature=0.25), keys, allowed, evidence
                )

            rec = {
                "recruitId": rid,
                "recruitType": rt,
                "attributes": attrs_out if ok else {},
                "ok": ok,
            }
            if evidence:
                rec["evidence"] = evid_out if ok else {}

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if ok:
                n_ok += 1
                by_type_ok[rt] += 1
            else:
                n_fail += 1
                by_type_fail[rt] += 1

            if not evidence and i > 0 and i % 1000 == 0:
                shutil.copy2(out_path, out_path.with_name(out_path.name + CHK_SUFFIX))

            time.sleep(0.15)

    total = n_ok + n_fail
    print(
        f"\n완료: 전체 처리 {total} / 성공 {n_ok} / 실패 {n_fail}\n"
        f"  타입별 성공: {dict(by_type_ok)}\n"
        f"  타입별 실패: {dict(by_type_fail)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--evidence",
        action="store_true",
        help="검증 샘플 300건만 근거 포함 추출 -> phase4_evidence_sample.jsonl",
    )
    args = ap.parse_args()
    run_batch(evidence=args.evidence)


if __name__ == "__main__":
    main()
