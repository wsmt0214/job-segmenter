"""Phase 2 차원 발견: 구인글 키워드 임베딩·군집화 후 차원 후보·리포트 생성"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

PHASE1_PATH = Path(config.DATA_DIR) / "phase1_results.jsonl"
EMBED_PATH = Path(config.DATA_DIR) / "keyword_embeddings.npy"
VOCAB_PATH = Path(config.DATA_DIR) / "keyword_embedding_vocab.json"
OUT_JSON = Path(config.DATA_DIR) / "dimension_candidates.json"
OUT_REPORT = Path(config.DATA_DIR) / "dimension_report.md"

EMBED_MODEL_ID = "jhgan/ko-sroberta-multitask"
DOMAIN_EXCLUDE = frozenset({"헤어", "메이크업", "스냅", "기타"})

COVERAGE_MIN = 0.05
DOMINANCE_MAX = 0.90
CLUSTER_DIST_THRESHOLD = 0.35
MIN_CLUSTER_KEYWORDS = 5
TOP_REPRESENTATIVE = 10

# Qwen 계열이 차원 이름에 한자·중국어 식 표기를 섞는 경우가 있어 출력 검증·재시도함
_CJK_IDEOGRAPH = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

NAME_PROMPT = """아래 키워드들은 구인 플랫폼 구인글에서 유사하게 함께 등장하는 묶음이다.

이 묶음이 드러내는 **하나의 의미 차원**을 이름 지어라.

[이름 규칙 — 반드시 지킬 것]
- 한글과 필요 시 숫자·공백만 사용 (2~10자 정도의 짧은 명사구)
- 한자(CJK 한자)·중국어·일본 한자·번체·간체 **절대 사용 금지**
- 영어 단어는 쓰지 말 것
- 부가 설명·따옴표·콜론·번호 목록 금지, 이름 한 줄만 출력

키워드: {keywords}
"""

NAME_RETRY_PROMPT = """동일 키워드 묶음에 대한 차원 이름을 다시 지어라.
방금 출력은 한자 등 금지 문자가 포함되어 **무효**다. **순수 한글만**으로 2~10자 명사구 한 줄만 출력하라.

이전 무효 출력: {bad}

키워드: {keywords}
"""


def _forbidden_in_dimension_name(s: str) -> bool:
    """한자·CJK 호환 한자만 차단 (한글·라틴·숫자는 허용)"""
    if not s or not s.strip():
        return True
    return bool(_CJK_IDEOGRAPH.search(s))


def _load_phase1_recruits() -> tuple[list[dict], int]:
    """구인글·ok 만 반환. (행 목록, 원본 전체 구인글 수는 별도 집계)"""
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
    return rows, len(rows)


def _flatten_keywords(rows: list[dict]) -> tuple[list[tuple[str, int, str]], Counter]:
    """(키워드, recruitId, recruitType), 키워드 전역 빈도"""
    occ: list[tuple[str, int, str]] = []
    kw_count: Counter = Counter()
    for o in rows:
        rid = int(o["recruitId"])
        rt = str(o.get("recruitType", ""))
        kws = (o.get("llm_result") or {}).get("keywords") or []
        if not isinstance(kws, list):
            continue
        for raw in kws:
            k = str(raw).strip()
            if not k:
                continue
            occ.append((k, rid, rt))
            kw_count[k] += 1
    return occ, kw_count


def _vocab_key(keywords: list[str]) -> str:
    h = hashlib.sha256()
    for k in sorted(keywords):
        h.update(k.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _load_or_encode_embeddings(keywords: list[str]) -> np.ndarray:
    """keyword_embeddings.npy + vocab 캐시 재사용"""
    if EMBED_PATH.is_file() and VOCAB_PATH.is_file():
        with VOCAB_PATH.open(encoding="utf-8") as f:
            meta = json.load(f)
        cached = meta.get("keywords")
        digest = meta.get("sha256")
        if (
            isinstance(cached, list)
            and len(cached) == len(keywords)
            and digest == _vocab_key(keywords)
            and list(cached) == keywords
        ):
            return np.load(EMBED_PATH)

    model = SentenceTransformer(EMBED_MODEL_ID, cache_folder=str(config.MODEL_DIR))
    vecs = model.encode(
        keywords,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(EMBED_PATH, vecs.astype(np.float32))
    with VOCAB_PATH.open("w", encoding="utf-8") as f:
        json.dump({"keywords": keywords, "sha256": _vocab_key(keywords)}, f, ensure_ascii=False)
    return vecs


def _cluster_keywords(vectors: np.ndarray) -> np.ndarray:
    # 코사인 거리(1-유사도) 기준, 평균 연결
    clu = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DIST_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    return clu.fit_predict(vectors)


def _ollama_dimension_name(keyword_sample: list[str]) -> str:
    joined = ", ".join(keyword_sample[:20])
    last_line = ""
    for attempt in range(3):
        if attempt == 0:
            prompt = NAME_PROMPT.format(keywords=joined)
        else:
            prompt = NAME_RETRY_PROMPT.format(
                keywords=joined, bad=last_line[:120] if last_line else "(없음)"
            )
        r = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.15 if attempt == 0 else 0.05, "num_predict": 80},
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        text = re.sub(r'^["\'「]|[」"\'\']$', "", text.strip())
        line = (text.splitlines()[0].strip()[:80] if text else "") or ""
        # 접두어 제거 (모델이 "이름: OO" 형태로 줄 때)
        line = re.sub(r"^(이름|차원)\s*[:：]\s*", "", line).strip()
        if line and not _forbidden_in_dimension_name(line):
            return line
        last_line = line

    # LLM 재시도 후에도 실패 시 대표 키워드 기반 한글 폴백
    for w in keyword_sample:
        w = str(w).strip()
        if w and not _forbidden_in_dimension_name(w):
            return f"{w[:12]} 관련" if len(w) > 12 else f"{w} 관련"
    return "의미군집 미명명"


def _hits_for_cluster(cluster_keywords: set[str], occ: list[tuple[str, int, str]]) -> set[int]:
    return {rid for kw, rid, _rt in occ if kw in cluster_keywords}


def _recruit_type_totals(rows: list[dict]) -> dict[str, int]:
    c = Counter(str(o.get("recruitType", "")) for o in rows)
    return dict(c)


def run() -> None:
    if not PHASE1_PATH.is_file():
        raise SystemExit(f"없음: {PHASE1_PATH}")

    print("Phase1 구인글 로드 …")
    rows, n_rec = _load_phase1_recruits()
    if n_rec == 0:
        raise SystemExit("구인글(ok) 데이터 없음")

    occ, kw_counter = _flatten_keywords(rows)
    n_occurrences = len(occ)
    unique_keywords = sorted(kw_counter.keys())
    n_unique = len(unique_keywords)
    print(f"구인글 {n_rec}건, 키워드 출현 {n_occurrences}회, 고유 {n_unique}개")

    print("임베딩 …")
    vectors = _load_or_encode_embeddings(unique_keywords)

    print("군집화 …")
    labels = _cluster_keywords(vectors)
    n_clusters = int(labels.max()) + 1 if len(labels) else 0

    cluster_to_words: dict[int, list[str]] = defaultdict(list)
    for w, lab in zip(unique_keywords, labels, strict=True):
        cluster_to_words[int(lab)].append(w)

    # 군집별 대표 키워드·도메인 단어 제외
    cluster_meta: list[dict] = []
    rid_by_type = defaultdict(set)
    for o in rows:
        rid_by_type[str(o.get("recruitType", ""))].add(int(o["recruitId"]))

    pbar = tqdm(range(n_clusters), desc="군집 메트릭·이름")
    for cid in pbar:
        words = cluster_to_words[cid]
        domain_hits = sorted([w for w in words if w in DOMAIN_EXCLUDE])
        rep_candidates = [w for w in words if w not in DOMAIN_EXCLUDE]
        rep_sorted = sorted(rep_candidates, key=lambda x: -kw_counter[x])[:TOP_REPRESENTATIVE]

        cset = set(words)
        hit_rids = _hits_for_cluster(cset, occ)
        n_hit = len(hit_rids)
        cov = n_hit / n_rec if n_rec else 0.0

        type_cov: dict[str, float] = {}
        for rt, all_r in rid_by_type.items():
            denom = len(all_r)
            type_cov[rt] = (len(hit_rids & all_r) / denom) if denom else 0.0

        # 변별력: 군집 내 출현 비율(max share)
        counts_in_c = [kw_counter[w] for w in words]
        total_in_c = sum(counts_in_c) if counts_in_c else 1
        max_share = max(counts_in_c) / total_in_c if counts_in_c else 0.0
        discriminative_ok = max_share < DOMINANCE_MAX
        coverage_ok = cov >= COVERAGE_MIN

        is_noise = len(words) < MIN_CLUSTER_KEYWORDS
        if is_noise:
            dim_name = "(소형·노이즈) 검토 필요"
        else:
            name_src = rep_sorted if rep_sorted else words[:TOP_REPRESENTATIVE]
            if not name_src:
                name_src = words
            try:
                dim_name = _ollama_dimension_name(name_src)
            except Exception as e:
                dim_name = f"OLLAMA 실패: {e}"

        type_cov_values = list(type_cov.values())
        if type_cov_values:
            lo, hi = min(type_cov_values), max(type_cov_values)
            uneven = (hi - lo) > 0.15 and cov >= COVERAGE_MIN
        else:
            uneven = False

        flags = []
        if not coverage_ok and not is_noise:
            flags.append("coverage_low")
        if not discriminative_ok:
            flags.append("low_discrimination")
        if uneven:
            flags.append("type_imbalance")

        if is_noise:
            verdict = "⚠ 소형 군집(노이즈)"
        elif not coverage_ok:
            verdict = "⚠ 커버리지 미달"
        elif not discriminative_ok:
            verdict = "⚠ 변별력 낮음"
        elif uneven:
            verdict = "⚠ 타입 편차 (참고)"
        else:
            verdict = "✓ 기준 통과"

        cluster_meta.append(
            {
                "cluster_id": cid,
                "llm_dimension_name": dim_name,
                "is_noise_cluster": is_noise,
                "n_keywords": len(words),
                "keywords_in_cluster": sorted(words),
                "domain_keywords_found": domain_hits,
                "representative_keywords": rep_sorted,
                "coverage": {
                    "fraction": round(cov, 4),
                    "recruit_count": n_hit,
                    "recruit_total": n_rec,
                    "by_type_fraction": {k: round(v, 4) for k, v in sorted(type_cov.items())},
                },
                "discrimination": {
                    "max_keyword_share_in_cluster": round(max_share, 4),
                    "pass": discriminative_ok,
                },
                "gates": {
                    "coverage_min_5pct": coverage_ok,
                    "discriminative_ok": discriminative_ok,
                    "uneven_type_coverage": uneven,
                },
                "flags": flags,
                "report_verdict": verdict,
            }
        )

    passed = [
        m
        for m in cluster_meta
        if not m["is_noise_cluster"]
        and m["gates"]["coverage_min_5pct"]
        and m["gates"]["discriminative_ok"]
    ]

    out = {
        "meta": {
            "phase1_path": str(PHASE1_PATH),
            "embedding_model": EMBED_MODEL_ID,
            "clustering": {
                "algorithm": "AgglomerativeClustering",
                "metric": "cosine",
                "linkage": "average",
                "distance_threshold": CLUSTER_DIST_THRESHOLD,
            },
            "n_recruit_posts_analyzed": n_rec,
            "n_keyword_occurrences": n_occurrences,
            "n_unique_keywords": n_unique,
            "n_clusters": n_clusters,
            "gates": {
                "coverage_min_fraction": COVERAGE_MIN,
                "dominance_max_fraction": DOMINANCE_MAX,
                "min_cluster_size_keywords": MIN_CLUSTER_KEYWORDS,
            },
        },
        "recruit_type_totals": _recruit_type_totals(rows),
        "dimensions": sorted(cluster_meta, key=lambda x: x["cluster_id"]),
        "summary_passing_gates": len(passed),
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = [
        "# Phase 2 차원 발견 결과",
        "",
        "## 전체 통계",
        f"- 분석 구인글: {n_rec:,}건",
        f"- 추출 키워드 총 수: {n_occurrences:,}건 (고유: {n_unique:,}건)",
        f"- 발견 군집 수: {n_clusters}개",
        "",
        "## 차원 후보 목록",
        "",
    ]
    for m in sorted(cluster_meta, key=lambda x: x["cluster_id"]):
        cid = m["cluster_id"]
        name = m["llm_dimension_name"]
        c = m["coverage"]
        pct = 100.0 * c["fraction"]
        rep = ", ".join(m["representative_keywords"]) or "(대표 없음 — 도메인 키워드만 등)"
        if m["domain_keywords_found"]:
            lines.append(
                f"*(도메인 키워드 군집 내 포함·대표 제외: {', '.join(m['domain_keywords_found'])})*"
            )
        byt = c.get("by_type_fraction") or {}
        tb = " / ".join(f"{k} {100*byt[k]:.1f}%" for k in sorted(byt.keys()))
        lines.extend(
            [
                f"### 차원 {cid + 1}: {name}",
                f"- 커버리지: {pct:.1f}% ({c['recruit_count']:,}건 / {c['recruit_total']:,}건)",
                f"- 대표 키워드: {rep}",
                f"- 타입별 분포(해당 타입 내 비율): {tb}",
                f"- 판정: {m['report_verdict']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 기준 통과 차원 요약",
            f"- 통과 개수(非노이즈·커버리지≥{int(100*COVERAGE_MIN)}%·변별력 양호): **{len(passed)}**개",
        ]
    )
    for m in sorted(passed, key=lambda x: -x["coverage"]["fraction"]):
        lines.append(
            f"  - **{m['llm_dimension_name']}** (군집 {m['cluster_id']}, 커버 {100*m['coverage']['fraction']:.1f}%)"
        )
    lines.append("")
    with OUT_REPORT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"저장: {OUT_JSON}")
    print(f"저장: {OUT_REPORT}")


if __name__ == "__main__":
    run()
