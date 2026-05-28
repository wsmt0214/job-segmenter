"""v2.0 Phase 2 — 유사어 정규화 후 LLM 차원 후보 발견"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

import config

INPUT_PATH = Path(config.V2_DATA_DIR) / "phase1_with_category.jsonl"
RAW_FREQ_PATH = Path(config.V2_DATA_DIR) / "all_signals_freq_raw.txt"
NORM_FREQ_PATH = Path(config.V2_DATA_DIR) / "all_signals_freq.txt"
NORM_MAP_PATH = Path(config.V2_DATA_DIR) / "signal_normalization.json"

THRESHOLDS = [0.003, 0.005, 0.01, 0.02, 0.03]
NORM_BATCH_SIZE = 60


def is_korean(text: str) -> bool:
    """대표어에 한글(가-힣) 포함 여부 — 영어/중국어/일본어 대표어 차단용"""
    return any("가" <= c <= "힣" for c in text)


def threshold_label(pct: float) -> str:
    """0.003 → 0.3pct, 0.01 → 1pct"""
    label_val = pct * 100
    if label_val == int(label_val):
        return f"{int(label_val)}pct"
    return f"{label_val:g}pct"


def threshold_display(pct: float) -> str:
    """0.003 → 0.3%, 0.01 → 1%"""
    label_val = pct * 100
    if label_val == int(label_val):
        return f"{int(label_val)}%"
    return f"{label_val:g}%"


def _parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"JSON 없음: {raw[:200]}")
    return json.loads(match.group())


def collect_raw_signals() -> tuple[Counter, list[list[str]], int]:
    """원본 all_signals 수집. 카테고리 시그널은 정규화 대상에서 제외"""
    all_raw: list[list[str]] = []
    n_posts = 0

    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_raw.append(row.get("all_signals", []))
            n_posts += 1

    flat = [sig for signals in all_raw for sig in signals]
    raw_freq = Counter(flat)
    kw_count = sum(1 for s in raw_freq if not s.startswith("카테고리:"))
    print(f"구인글: {n_posts}건 / 고유 시그널: {len(raw_freq)}개 (키워드 {kw_count}개)")
    return raw_freq, all_raw, n_posts


def save_raw_freq_txt(raw_freq: Counter, n_posts: int) -> None:
    with RAW_FREQ_PATH.open("w", encoding="utf-8") as f:
        f.write(f"[정규화 전 원본 빈도]\n총 구인글: {n_posts}건 / 고유 시그널: {len(raw_freq)}개\n\n")
        f.write(f"{'시그널':<30}{'빈도':>8}{'비율':>8}\n")
        f.write("-" * 50 + "\n")
        for sig, cnt in raw_freq.most_common():
            f.write(f"{sig:<30}{cnt:>8}{cnt / n_posts * 100:>7.1f}%\n")
    print(f"원본 빈도 저장: {RAW_FREQ_PATH}")


def build_normalization_map_with_llm(raw_freq: Counter) -> dict[str, str]:
    """자유 키워드 유사어 정규화 매핑 생성 (배치당 LLM 1회)"""
    keywords = [s for s in raw_freq if not s.startswith("카테고리:")]
    keywords.sort(key=lambda s: -raw_freq[s])

    norm_map: dict[str, str] = {}
    batches = [keywords[i : i + NORM_BATCH_SIZE] for i in range(0, len(keywords), NORM_BATCH_SIZE)]
    print(f"  유사어 정규화: 총 {len(keywords)}개 키워드를 {len(batches)}개 배치로 처리")

    for i, batch in enumerate(batches):
        kw_list = "\n".join(f"  - {kw} ({raw_freq[kw]}회)" for kw in batch)
        prompt = f"""당신은 구인 플랫폼 데이터 분석 전문가입니다.
아래는 구인글에서 추출된 키워드 목록입니다.

[키워드 목록]
{kw_list}

다음을 수행하세요:
1. 같은 의미이거나 매우 유사한 키워드끼리 하나의 그룹으로 묶으세요.
2. 각 그룹에서 가장 짧고 명확한 단어를 "대표어"로 지정하세요.
3. 각 키워드에 대해 {{원래 키워드: 대표어}} 형태로 매핑을 만드세요.
   - 그룹에 속하지 않는 단어(독자적 의미)는 자기 자신을 대표어로 씁니다.

🚫 절대 규칙 — 위반 시 전체 출력이 무효입니다:
  - 대표어는 오직 한국어(한글)만 사용합니다.
  - 중국어(漢字), 영어(alphabet), 일본어(かな) 사용 금지입니다.
  - 영어 키워드 → 한국어로 의미 번역 후 대표어 작성
  - 중국어 키워드 → 한국어로 의미 번역 후 대표어 작성
  - 일본어 키워드 → 한국어로 의미 번역 후 대표어 작성

  ✓ 올바른 예:
    "courteous behavior" → "예의바른태도"  (영어→한국어 ✓)
    "crying" → "우는표정"                 (영어→한국어 ✓)
    "after photo" → "사후사진"            (영어→한국어 ✓)
    "工作合作" → "작업협업"               (중국어→한국어 ✓)
    "眉毛を整えても" → "눈썹정리가능"     (일본어→한국어 ✓)

  ✗ 잘못된 예:
    "courteous behavior" → "礼貌行为"     (중국어 금지 ✗)
    "crying" → "哭泣"                     (중국어 금지 ✗)
    "after photo" → "after photo"         (영어 금지 ✗)

예시:
  "급구", "당일가능", "즉시가능", "바로가능", "급촬영" → 대표어: "긴급"
  "경력무관", "초보가능", "초보환영" → 대표어: "경력무관"
  "레이어드컷", "레이어드 컷" → 대표어: "레이어드컷"
  "매주화요일", "월2회 이상", "정기적" → 대표어: "지속정기"
  "인스타 업로드", "콘텐츠 플랫폼 업로드", "sns업로드" → 대표어: "sns업로드"
  "비포 애프터 촬영", "비포 앤드 애프터" → 대표어: "비포애프터"

반드시 아래 JSON만 출력하세요 (다른 설명 없이):
{{
  "원래키워드1": "대표어1",
  "원래키워드2": "대표어2"
}}"""

        print(f"    배치 {i + 1}/{len(batches)} 처리 중...")
        try:
            resp = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 2000},
                },
                timeout=180,
            )
            resp.raise_for_status()
            batch_map = _parse_llm_json(resp.json().get("response", ""))
            if not isinstance(batch_map, dict):
                raise ValueError("매핑 JSON이 dict가 아님")

            # 한글 아닌 대표어는 원본 키워드로 되돌림
            fallback_count = 0
            sanitized: dict[str, str] = {}
            for kw, canon in batch_map.items():
                canon_str = str(canon)
                if not is_korean(canon_str):
                    sanitized[str(kw)] = str(kw)
                    fallback_count += 1
                else:
                    sanitized[str(kw)] = canon_str
            if fallback_count > 0:
                print(f"    ⚠️ 한글 아닌 대표어 {fallback_count}개 → 원본으로 대체")

            norm_map.update(sanitized)
        except Exception as e:
            print(f"    ⚠️ 배치 {i + 1} 실패: {e} — 해당 키워드는 원본 그대로 사용")
            for kw in batch:
                norm_map.setdefault(kw, kw)

        time.sleep(1)

    # LLM이 누락한 키워드는 자기 자신으로 보완
    for kw in keywords:
        norm_map.setdefault(kw, kw)

    print(f"  정규화 매핑 완성: {len(norm_map)}개 키워드")
    return norm_map


def load_or_build_normalization_map(raw_freq: Counter, *, rebuild: bool = False) -> dict[str, str]:
    if NORM_MAP_PATH.is_file() and not rebuild:
        with NORM_MAP_PATH.open(encoding="utf-8") as f:
            norm_map = json.load(f)
        print(f"  유사어 매핑 재사용: {NORM_MAP_PATH} ({len(norm_map)}개)")
        return norm_map

    print("  signal_normalization.json 없음 → LLM으로 생성" if not NORM_MAP_PATH.is_file() else "  --rebuild-norm: LLM 재생성")
    norm_map = build_normalization_map_with_llm(raw_freq)
    with NORM_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(norm_map, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"  정규화 매핑 저장: {NORM_MAP_PATH}")
    print("  ⚠️  파일을 열어 매핑 확인·수동 수정 가능 (재실행 시 재사용)")
    return norm_map


def apply_normalization(all_raw: list[list[str]], norm_map: dict[str, str]) -> tuple[Counter, int]:
    """정규화 적용 후 빈도 재집계"""
    normalized_flat: list[str] = []
    n_posts = len(all_raw)

    for signals in all_raw:
        for sig in signals:
            if sig.startswith("카테고리:"):
                normalized_flat.append(sig)
            else:
                normalized_flat.append(norm_map.get(sig, sig))

    norm_freq = Counter(normalized_flat)
    kw_before = len(norm_map)
    kw_after = sum(1 for s in norm_freq if not s.startswith("카테고리:"))
    print(f"  정규화 결과: 키워드 {kw_before}개 → 고유 표현 {kw_after}개로 축소")
    return norm_freq, n_posts


def save_norm_freq_txt(norm_freq: Counter, n_posts: int) -> None:
    with NORM_FREQ_PATH.open("w", encoding="utf-8") as f:
        f.write(f"[정규화 후 빈도]\n총 구인글: {n_posts}건 / 고유 시그널: {len(norm_freq)}개\n\n")
        f.write(f"{'시그널':<30}{'빈도':>8}{'비율':>8}\n")
        f.write("-" * 50 + "\n")
        for sig, cnt in norm_freq.most_common():
            f.write(f"{sig:<30}{cnt:>8}{cnt / n_posts * 100:>7.1f}%\n")
    print(f"정규화 후 빈도 저장: {NORM_FREQ_PATH}")


def filter_candidates(freq: Counter, n_posts: int, threshold_pct: float) -> list[tuple[str, int]]:
    threshold = n_posts * threshold_pct
    candidates = [(sig, cnt) for sig, cnt in freq.most_common() if cnt >= threshold]
    cat_count = sum(1 for s, _ in candidates if s.startswith("카테고리:"))
    kw_count = len(candidates) - cat_count
    print(
        f"  빈도 {threshold_display(threshold_pct)} 이상: 총 {len(candidates)}개 "
        f"(키워드 {kw_count}개 + 카테고리 {cat_count}개)"
    )
    return candidates


def llm_group_signals(candidates: list[tuple[str, int]], threshold_pct: float) -> dict:
    cat_signals = [(s, c) for s, c in candidates if s.startswith("카테고리:")]
    kw_signals = [(s, c) for s, c in candidates if not s.startswith("카테고리:")]

    cat_text = "\n".join(f"  {s} ({c}회)" for s, c in cat_signals[:30])
    kw_text = "\n".join(f"  {s} ({c}회)" for s, c in kw_signals[:100])

    prompt = f"""당신은 구인 플랫폼 데이터 분석 전문가입니다.
아래는 구인글 10,450건에서 추출한 시그널(키워드 + 카테고리)의 빈도 목록입니다.
키워드는 유사어가 이미 하나의 대표어로 정규화되어 있습니다.
(빈도 {threshold_display(threshold_pct)} 이상 시그널만 포함)

[카테고리 시그널 (사용자가 직접 선택한 값)]
{cat_text}

[자유 키워드 시그널 (정규화된 대표어 + 합산 빈도)]
{kw_text}

다음을 수행하세요:

1. 자유 키워드 시그널을 관련 있는 것끼리 묶어 "차원"을 정의합니다.
   각 차원은 서로 다른 속성을 나타내야 합니다. 중복 차원을 만들지 마세요.
2. 각 차원에 대해:
   - 카테고리 시그널이 이미 이 차원을 커버하면 is_redundant: true
   - payment(보상 유형)와 중복되는 차원도 is_redundant: true
   - 클러스터링에 유의미한 차원은 is_useful: true

반드시 아래 JSON만 출력하세요:
{{
  "threshold_pct": {threshold_pct},
  "signal_count": {len(candidates)},
  "dimensions": [
    {{
      "dimension_name": "차원명 (짧은 명사구, 예: 긴급도)",
      "signals": ["포함된 키워드1", "포함된 키워드2"],
      "possible_values": ["값1", "값2", "불명확"],
      "is_useful": true,
      "is_redundant": false,
      "redundant_reason": ""
    }}
  ]
}}"""

    print(f"  LLM 차원 그룹화 요청 중 (임계값 {threshold_display(threshold_pct)})...")
    resp = requests.post(
        config.OLLAMA_URL,
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 3000},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return _parse_llm_json(resp.json().get("response", ""))


def print_result_summary(result: dict) -> None:
    dims = result.get("dimensions", [])
    useful = [d for d in dims if d.get("is_useful") and not d.get("is_redundant")]
    removed = [d for d in dims if not d.get("is_useful") or d.get("is_redundant")]
    print(f"\n  → 유효 차원 {len(useful)}개 / 제외 차원 {len(removed)}개")
    for d in useful:
        print(f"     ✓ {d['dimension_name']}: {d.get('possible_values', [])}")
    for d in removed:
        print(f"     ✗ {d['dimension_name']} [{d.get('redundant_reason', '')}]")


def print_comparison_summary(all_results: dict[float, dict]) -> None:
    print("\n" + "=" * 60)
    print("임계값별 차원 발견 결과 비교")
    print("=" * 60)
    for pct, result in all_results.items():
        dims = result.get("dimensions", [])
        useful = sum(1 for d in dims if d.get("is_useful") and not d.get("is_redundant"))
        sig_count = result.get("signal_count", "?")
        print(f"\n  [{threshold_display(pct)} 임계값] 시그널 {sig_count}개 → 유효 차원 {useful}개 / 제외 {len(dims) - useful}개")
        for d in dims:
            if d.get("is_useful") and not d.get("is_redundant"):
                print(f"    ✓ {d['dimension_name']}")
    print("=" * 60)


def _print_manual_guide() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  수동 작업 — Task 2                                      ║
╠══════════════════════════════════════════════════════════════╣
║  1. signal_normalization.json 검토·수정 (가장 먼저)          ║
║  2. all_signals_freq.txt — 정규화 후 빈도 확인               ║
║  3. dimension_candidates_v2_0.3/0.5/1/2/3pct.json 비교       ║
║  4. schema_definition_v2.json 작성                           ║
╚══════════════════════════════════════════════════════════════╝
""")


def run(*, rebuild_norm: bool = False, thresholds: list[float] | None = None, norm_only: bool = False) -> None:
    if not INPUT_PATH.is_file():
        raise SystemExit(f"입력 없음: {INPUT_PATH}")

    pct_list = thresholds or THRESHOLDS
    Path(config.V2_DATA_DIR).mkdir(parents=True, exist_ok=True)

    title = "유사어 정규화까지" if norm_only else "유사어 정규화 + 임계값 실험"
    print(f"=== v2.0 Phase 2 — 차원 재발견 ({title}) ===\n")

    print("--- STEP 1. 원본 시그널 수집 ---")
    raw_freq, all_raw, n_posts = collect_raw_signals()
    save_raw_freq_txt(raw_freq, n_posts)

    print("\n--- STEP 2. 유사어 정규화 ---")
    norm_map = load_or_build_normalization_map(raw_freq, rebuild=rebuild_norm)

    print("\n--- STEP 3. 정규화 후 빈도 재집계 ---")
    norm_freq, _ = apply_normalization(all_raw, norm_map)
    save_norm_freq_txt(norm_freq, n_posts)

    if norm_only:
        print("""
╔══════════════════════════════════════════════════════════════╗
║  STEP 1~3 완료 — normalization까지만 실행됨                  ║
╠══════════════════════════════════════════════════════════════╣
║  1. signal_normalization.json 검토·수정                      ║
║  2. all_signals_freq.txt — 정규화 후 빈도 확인               ║
║  3. 검토 후 차원 그룹화: python3 scripts/v2_phase2_discovery.py ║
╚══════════════════════════════════════════════════════════════╝
""")
        return

    all_results: dict[float, dict] = {}
    for pct in pct_list:
        label = threshold_label(pct)
        print(f"\n--- STEP 4-{label}: 임계값 {threshold_display(pct)} ---")
        candidates = filter_candidates(norm_freq, n_posts, pct)
        result = llm_group_signals(candidates, pct)
        print_result_summary(result)

        out = Path(config.V2_DATA_DIR) / f"dimension_candidates_v2_{label}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  저장: {out}")
        all_results[pct] = result
        time.sleep(2)

    print_comparison_summary(all_results)
    _print_manual_guide()


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Phase 2 차원 재발견")
    parser.add_argument("--rebuild-norm", action="store_true", help="signal_normalization.json LLM 재생성")
    parser.add_argument("--norm-only", action="store_true", help="STEP 1~3(normalization)까지만 실행")
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds", help="임계값 (기본 0.3/0.5/1/2/3%%)")
    args = parser.parse_args()
    run(rebuild_norm=args.rebuild_norm, thresholds=args.thresholds, norm_only=args.norm_only)


if __name__ == "__main__":
    main()
