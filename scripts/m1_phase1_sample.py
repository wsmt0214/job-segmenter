import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

import config

# 잘못된 JSON 템플릿(예: "구인글" 또는 "지원글")은 모델이 그대로 따라 해 깨진 출력을 유발함 — 유효 JSON 예시만 제시
PROMPT = """당신은 구인 플랫폼의 분석 전문가입니다.
아래 텍스트를 읽고 세 가지를 수행하세요.

첫째, 이 글이 구인글(모델/스태프를 구하는 글)인지,
지원글(본인이 지원하는 글)인지 판단하세요.

둘째, 이 글이 어떤 세부 도메인에 속하는지 분류하세요.
- 헤어: 헤어컬러, 펌, 커트, 두피 케어, 헤어 익스텐션 등
- 메이크업: 웨딩 메이크업, 무대 메이크업, 뷰티 메이크업, 네일 등
- 스냅: 웨딩스냅, 야외스냅, 인물스냅, 일상스냅, 여행스냅, 스냅촬영, 화보, 프로필, 제품촬영 등
  (텍스트에 '스냅'이라는 단어가 포함되어 있으면 다른 도메인의 명확한 근거가 없는 한 스냅으로 분류)
- 기타: 위 세 도메인에 명확히 속하지 않거나 복합적인 경우

셋째, 구인글인 경우에만, 지원자 입장에서 이 기회를 특징짓는
핵심 키워드를 추출하세요.

[조건]
- 키워드는 지원자가 이 기회를 어떻게 경험하는지를 나타내야 합니다
- 키워드는 반드시 한국어로 작성하세요
- 촬영 종류나 시술 유형(헤어, 메이크업, 스냅 등)은 키워드에 넣지 마세요
- 키워드는 명사구 형태로, 텍스트에서 읽히는 특징만 최대 5개
- "일정 협의 가능", "연락주세요", "문의 환영", "시간 맞춰요", "꾸준히 함께해요" 처럼
  거의 모든 구인글에 공통으로 쓰이는 표현은 키워드에서 제외하세요.
  이 구인글만의 구체적인 특징에 집중하세요
- 지원글이면 keywords는 빈 배열 []

[출력 규칙]
- 반드시 유효한 JSON 객체 한 개만 출력합니다. 앞뒤 설명·마크다운 금지
- 키 이름은 정확히 "글_유형", "도메인", "keywords" 만 사용
- 글_유형 값은 문자열 "구인글" 또는 "지원글" 중 하나
- 도메인 값은 문자열 "헤어", "메이크업", "스냅", "기타" 중 하나
- keywords는 문자열 배열

올바른 출력 예:
{{"글_유형": "구인글", "도메인": "스냅", "keywords": ["상호무페이", "야외"]}}

[텍스트]
제목: {title}
본문: {body}
"""


_DOMAINS = ("헤어", "메이크업", "스냅", "기타")
_TYPES = ("구인글", "지원글")

# Ollama 출력이 잘리거나 `]}` 대신 `]]` 로 끝나는 경우
_REPAIR_TAIL = re.compile(r"\]\]\s*$")

_REPAIR_PROMPT = """아래는 구인글 분류 결과 JSON인데 형식이 깨졌다.
같은 의미로 유효한 JSON 객체 한 개만 출력하라. 설명·코드펜스 금지.
키는 정확히 "글_유형", "도메인", "keywords" 만 사용.
도메인은 "헤어", "메이크업", "스냅", "기타" 중 하나. 글_유형은 "구인글" 또는 "지원글".

깨진 출력:
{broken}
"""


def _repair_json_tail(s: str) -> str:
    s = s.strip()
    if _REPAIR_TAIL.search(s):
        s = _REPAIR_TAIL.sub("]}", s)
    return s


def _parse_json_from_response(text: str) -> dict | None:
    """코드펜스·앞뒤 잡담이 있어도 첫 번째 JSON 객체를 파싱"""
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


def _normalize_llm_dict(d: dict) -> dict | None:
    typ = d.get("글_유형")
    if typ not in _TYPES and isinstance(typ, str):
        s = re.sub(r"\s+", "", typ)
        if s in ("구인글", "구인", "모집", "리크루트") or (s.endswith("구인") and "지원글" not in s):
            typ = "구인글"
        elif s in ("지원글", "지원"):
            typ = "지원글"
    if typ not in _TYPES:
        return None

    dom = d.get("도메인")
    if dom not in _DOMAINS:
        dom_s = str(dom).strip() if dom is not None else ""
        # 모델 이형 변형 (예: 스nap)
        if dom_s and "스" in dom_s and "nap" in dom_s.lower():
            dom = "스냅"
        else:
            dom = None
            for a in _DOMAINS:
                if a == dom_s or a in dom_s:
                    dom = a
                    break
            if dom is None:
                dom = "기타"

    kws = d.get("keywords", [])
    if isinstance(kws, str):
        kws = [x.strip() for x in re.split(r"[,，]", kws) if x.strip()]
    elif not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()][:5]
    if typ == "지원글":
        kws = []

    return {"글_유형": typ, "도메인": dom, "keywords": kws}


def _llm_repair_json(broken: str) -> dict | None:
    """깨진 분류 JSON만 짧게 재생성해 파싱"""
    prompt = _REPAIR_PROMPT.format(broken=broken.strip()[:900])
    for _ in range(2):
        try:
            r = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.05, "num_predict": 384},
                },
                timeout=120,
            )
            raw = r.json()["response"].strip()
            for candidate in (raw, _repair_json_tail(raw)):
                data = _parse_json_from_response(candidate)
                if data is not None:
                    normalized = _normalize_llm_dict(data)
                    if normalized is not None:
                        return normalized
        except Exception:
            pass
        time.sleep(1)
    return None


def call_llm(title: str, body: str, retries: int = 5) -> dict | None:
    prompt = PROMPT.format(title=title[:200], body=body[:1000])
    last_raw = ""
    for _ in range(retries):
        try:
            r = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    # 키워드 배열이 잘리며 JSON 깨짐이 나와 여유 확보
                    "options": {"temperature": 0.1, "num_predict": 768},
                },
                timeout=180,
            )
            last_raw = r.json()["response"].strip()
            data = _parse_json_from_response(last_raw)
            if data is None:
                data = _parse_json_from_response(_repair_json_tail(last_raw))
            if data is None:
                raise ValueError("no json object")
            normalized = _normalize_llm_dict(data)
            if normalized is None:
                raise ValueError("normalize failed")
            return normalized
        except Exception:
            time.sleep(2)
    if last_raw:
        return _llm_repair_json(last_raw)
    return None


def run(n=100):
    df = pd.read_csv(f"{config.DATA_DIR}/raw_recruits.csv").sample(n=n, random_state=42)
    results, fails = [], 0

    for _, row in tqdm(df.iterrows(), total=n):
        res = call_llm(
            str(row["title"]) if pd.notna(row["title"]) else "",
            str(row["content"]) if pd.notna(row["content"]) else "",
        )
        results.append(
            {
                "recruitId": row["recruitId"],
                "title": row["title"],
                "preview": str(row["content"])[:80],
                "result": res,
                "ok": res is not None,
            }
        )
        if res is None:
            fails += 1
        time.sleep(0.5)

    out = f"{config.DATA_DIR}/phase1_sample.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rate = (n - fails) / n * 100
    print(f"\n파싱 성공률: {rate:.1f}% (기준: 95%)")
    print(f"결과 저장: {out}")
    print("\n⚠️  phase1_sample.jsonl을 열어 20건 이상 수동으로 확인하세요.")
    print("통과 시 m1_phase1_batch.py 실행 / 미통과 시 PROMPT 수정 후 재실행")


if __name__ == "__main__":
    run()
