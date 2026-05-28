"""v2.0 Phase 3 — LLM 구조화 속성 추출 공통 로직"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

import config
from schema_v2 import (
    P3_SNAP_TIER1_ONLY,
    P3_SNAP_TIER2_BODY_SNAP,
    P3_SNAP_TIER2_CONDITIONAL,
    P3_SNAP_TIER_DEFAULT,
    P3_SNAP_TIER_FULL,
    UNCLEAR_VALUE,
    load_schema,
    phase3_attr_names,
    phase3_extract_attrs,
    valid_values_map,
)

TITLE_MAX = 200
BODY_MAX = 1000

# 펌 후처리 — 제목+본문 키워드 검증용 (LLM 환각 차단)
PERM_EVIDENCE_WORDS = [
    "펌", "빈티지펌", "히피펌", "레이어드펌", "다운펌", "셋팅펌", "젤리펌",
    "볼륨매직", "윈드펌", "보더펌", "perm", "파마", "매직",
]

# P3-헤어: 헤어 카테고리 + 시술 미기재 → 헤어 시술 미언급 (LLM 추론 아님, guard_perm 이후)
GENERIC_HAIR_VALUE = "헤어 시술 미언급"

# P3-스냅: 카테고리·본문 신호 → 촬영 주제 스냅 (스키마 extraction_guide 결정적 적용)
TOPIC_SNAP_VALUE = "스냅"
SNAP_TOPIC_CATEGORIES_HIGH: frozenset[str] = frozenset({"일상", "여행", "우정"})
SNAP_TOPIC_CATEGORY_PORTRAIT = "인물스냅"
CONFLICT_TOPIC_CATEGORIES: frozenset[str] = frozenset({"웨딩", "커플", "한복"})
SNAP_BODY_RE = re.compile(
    r"스냅|인물\s*스냅|데일리\s*스냅|여행\s*스냅|우정\s*사진|일상\s*사진",
    re.I,
)
BODY_TOPIC_CONFLICT_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"웨딩|스드메|신부|예복"), "웨딩"),
    (re.compile(r"커플|연인"), "커플"),
    (re.compile(r"한복"), "한복"),
)

# P3-목적 / P3-장소 — schema extraction_guide 결정적 적용
PURPOSE_BEAUTY = "뷰티·메이크업"
PURPOSE_FASHION = "패션·룩북·화보"
PURPOSE_PORTFOLIO = "포트폴리오"
PURPOSE_PROFILE = "프로필·증명사진"
PURPOSE_BA = "비포애프터"
PLACE_STUDIO = "스튜디오"
PLACE_OUTDOOR = "야외"
PLACE_HOME = "홈스냅"

BEAUTY_CATEGORIES: frozenset[str] = frozenset({"뷰티", "메이크업"})
FASHION_CATEGORIES: frozenset[str] = frozenset({"패션"})
STUDIO_CATEGORY = "스튜디오"
PROFILE_CATEGORY = "프로필"

PURPOSE_BA_RE = re.compile(r"비포\s*애프터|비포애프터|before\s*after", re.I)
PURPOSE_PROFILE_RE = re.compile(
    r"증명\s*사진|증명사진|여권\s*사진|여권사진|프로필\s*촬영|프로필사진|바디\s*프로필|세미\s*바디",
    re.I,
)
PURPOSE_BEAUTY_RE = re.compile(r"메이크업|메컵|베이스\s*메|헤메|화장|뷰티\s*촬영", re.I)
PURPOSE_FASHION_RE = re.compile(r"룩북|화보|피팅|쇼핑몰|룩\s*북|룩북\s*촬영", re.I)
PURPOSE_PORTFOLIO_RE = re.compile(
    r"포트폴리오|포폴|연습\s*촬영|작업물|졸업|오디션|상호\s*무페이|상호무페이|졸작|작품\s*촬영|필름\s*촬영",
    re.I,
)

PLACE_STUDIO_RE = re.compile(
    r"스튜디오|실내\s*촬영|작업실|미용실|헤어\s*샵|헤어샵|미용\s*실|매장|살롱|호리존|\b스튜\b",
    re.I,
)
PLACE_OUTDOOR_RE = re.compile(
    r"야외|자연광|옥외|공원|거리\s*촬영|외부\s*촬영|로케이션|루프탑|야간\s*촬영|야외\s*촬영|옥상",
    re.I,
)
PLACE_HOME_RE = re.compile(r"홈\s*스냅|홈스냅|집에서|우리\s*집|일상\s*공간|자택", re.I)

# 기타 헤어 적용 제외 — 제목·본문에 구체 시술명이 있으면 Phase3 불명확 유지
EXPLICIT_TREATMENT_MARKERS: tuple[str, ...] = (
    *PERM_EVIDENCE_WORDS,
    "컷", "커트", "레이어드컷", "허쉬컷", "드라이", "스타일링",
    "염색", "탈색", "컬러", "톤다운", "블랙",
    "두피케어", "두피스케일링",
    "속눈썹", "속눈썹펌", "속눈썹연장",
    "눈썹 타투", "눈썹문신", "입술 타투", "입술문신",
)

BASE_PROMPT = """당신은 구인 플랫폼 구인글 분석기다.
아래 구인글을 읽고 각 속성을 분류하라.
텍스트에 명확한 근거가 없으면 반드시 "{unclear}"을 선택하라.
허용 값 목록에만 맞춰 출력하고, 동의어를 값으로 바꿀 수 없다.
출력은 유효한 JSON 한 개만. 앞뒤 설명·마크다운 금지.

[시술 종류 추출 규칙 — 반드시 준수]
시술 종류는 제목 또는 본문에 시술명이 단어 그대로 등장할 때만 추출한다.
카테고리가 '헤어'인 것, 헤어 관련 언급이 있는 것, 촬영 스타일 단어(빈티지·레트로·컨셉 등)는
시술 종류 추출의 근거가 되지 않는다.
제목·본문에 펌·컷·컬러·두피케어·속눈썹·눈썹 타투·입술 타투 중 하나가 명시되지 않으면
시술 종류는 반드시 "{unclear}"으로 출력한다.

[시술 종류 — 절대 규칙, 다른 차원 규칙에 영향받지 않음]
다른 차원에서 카테고리·맥락을 참조해도 된다는 규칙이 있더라도,
시술 종류만큼은 제목 또는 본문에 시술명이 직접 명시된 경우에만 분류한다.
카테고리 '헤어', '헤메', '헤어메이크업', '스타일리스트' 단독 → 시술 종류 불명확.
'빈티지', '레트로', '컨셉', '웨이브', '볼륨' 단독 → 시술 종류 불명확.
펌·파마라는 단어가 제목·본문에 없으면 펌으로 분류하지 않는다.

[촬영 목적 추출 규칙 — 시술 종류 규칙과 별개, 시술 규칙을 촬영 목적에 적용하지 마라]
촬영 목적은 카테고리와 맥락으로 분류한다.
카테고리 메이크업·뷰티 또는 본문 메이크업·베이스·헤메 언급 → 뷰티·메이크업
본문 포폴·연습·작업물·영화·드라마·웹드라마·졸업·오디션·상호무페이·스냅 언급 → 포트폴리오
목적이 암시되어 있으면 '포트폴리오'라는 단어 없이도 해당 값 선택 가능

[촬영 장소 추출 규칙 — 시술 종류 규칙과 별개, 시술 규칙을 촬영 장소에 적용하지 마라]
카테고리 '스튜디오' 포함 → 스튜디오
본문 스튜디오·실내·작업실·샵·미용실·헤어샵·매장·살롱·호리존 언급 → 스튜디오
본문 야외·자연광·공원·거리·외부·로케이션·루프탑·야외 촬영 언급 → 야외
집·일상 공간·홈스냅 언급 → 홈스냅

[촬영 주제 추출 규칙 — v3 방식, 시술 종류 규칙과 별개]
시술 종류 규칙의 엄격함을 촬영 주제에 적용하지 마라. 카테고리와 본문 맥락으로 분류한다.
카테고리 웨딩 또는 본문 웨딩·스드메·신부·예복 → 웨딩
카테고리 커플 또는 본문 커플·연인·친구+커플 → 커플
카테고리 한복 또는 본문 한복 → 한복
카테고리 인물스냅·일상·여행·우정 또는 본문 스냅·여행·우정·일상·인물 스냅 → 스냅
주제가 암시되어 있으면 해당 값 선택 가능

[카테고리] (사용자 등록 시 선택한 값)
{categories}

[속성 목록]
{attr_block}

[구인글]
제목: {title}
본문: {body}

반드시 아래 JSON 형식만 출력:
{example_json}
"""


def parse_json_object(text: str) -> dict | None:
    """LLM 응답에서 첫 번째 JSON 객체 추출 — v1 Phase4 파서 패턴"""
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


def _attr_block(attrs: list[dict]) -> str:
    lines: list[str] = []
    for a in attrs:
        name = a.get("name")
        desc = a.get("description", "")
        values = a.get("values") or []
        guide = a.get("extraction_guide", "")
        signals = a.get("known_signals") or []
        vals = " / ".join(repr(str(v)) for v in values)
        lines.append(f'  - {name} ({desc}): {vals}')
        if guide:
            lines.append(f"    추출 기준: {guide}")
        if signals:
            hint = ", ".join(str(s) for s in signals[:12])
            lines.append(f"    참고 키워드: {hint}")
    return "\n".join(lines)


def build_prompt(
    attrs: list[dict],
    categories: list[str],
    title: str,
    body: str,
) -> str:
    """카테고리 + 제목 + 본문 기반 Phase3 프롬프트"""
    cat_str = ", ".join(categories) if categories else "없음"
    example = {str(a["name"]): str((a.get("values") or [UNCLEAR_VALUE])[0]) for a in attrs}
    return BASE_PROMPT.format(
        unclear=UNCLEAR_VALUE,
        categories=cat_str,
        attr_block=_attr_block(attrs),
        title=title[:TITLE_MAX],
        body=body[:BODY_MAX],
        example_json=json.dumps(example, ensure_ascii=False),
    )


def normalize_attributes(
    raw: dict | None,
    attr_names: list[str],
    allowed: dict[str, set[str]],
) -> dict[str, str]:
    """파싱된 JSON을 허용값 기준으로 정규화 — 범위 밖은 불명확"""
    out: dict[str, str] = {}
    src = raw if isinstance(raw, dict) else {}
    for name in attr_names:
        val = src.get(name)
        if val is None or str(val) not in allowed.get(name, set()):
            out[name] = UNCLEAR_VALUE
        else:
            out[name] = str(val)
    return out


def has_explicit_treatment_text(title: str, body: str) -> bool:
    """제목·본문에 구체 시술명이 등장하는지 — 기타 헤어 제외 판단용"""
    full_text = f"{title} {body}"
    return any(m in full_text for m in EXPLICIT_TREATMENT_MARKERS)


def guard_perm_hallucination(
    attrs: dict[str, str],
    title: str,
    body: str,
) -> dict[str, str]:
    """펌 분류인데 제목+본문에 펌·파마 근거 없으면 불명확으로 교정"""
    if attrs.get("시술 종류") != "펌":
        return attrs
    full_text = f"{title} {body}"
    if any(w in full_text for w in PERM_EVIDENCE_WORDS):
        return attrs
    corrected = dict(attrs)
    corrected["시술 종류"] = UNCLEAR_VALUE
    return corrected


def apply_generic_hair_treatment(
    treatment: str,
    categories: list[str],
    *,
    title: str = "",
    body: str = "",
    skip_when_treatment_text: bool = True,
) -> str:
    """
    guard_perm 이후 후처리 — LLM에 헤어 시술 미언급을 시키지 않고 결정적으로 치환

    조건 (모두):
      - categories에 '헤어' 포함
      - 시술 종류 == 불명확
      - (선택) 제목·본문에 구체 시술 키워드 없음
    """
    if treatment != UNCLEAR_VALUE:
        return treatment
    if "헤어" not in categories:
        return treatment
    if skip_when_treatment_text and has_explicit_treatment_text(title, body):
        return treatment
    return GENERIC_HAIR_VALUE


def apply_generic_hair_treatment_attrs(
    attrs: dict[str, str],
    categories: list[str],
    title: str,
    body: str,
    **kwargs,
) -> dict[str, str]:
    """속성 dict에 기타 헤어 후처리 적용"""
    out = dict(attrs)
    out["시술 종류"] = apply_generic_hair_treatment(
        out.get("시술 종류", UNCLEAR_VALUE),
        categories,
        title=title,
        body=body,
        **kwargs,
    )
    return out


def load_recruit_texts(recruit_ids: set[int]) -> dict[int, tuple[str, str]]:
    """raw_recruits — 기타 헤어 후처리용 제목·본문"""
    path = Path(config.DATA_DIR) / "raw_recruits.csv"
    if not path.exists() or not recruit_ids:
        return {}
    raw = pd.read_csv(path, usecols=["recruitId", "title", "content"])
    raw = raw[raw["recruitId"].isin(recruit_ids)]
    return {
        int(row.recruitId): (str(row.title or ""), str(row.content or ""))
        for _, row in raw.iterrows()
    }


def apply_generic_hair_phase3_frame(
    phase_df: pd.DataFrame,
    categories: dict[int, list[str]],
    texts: dict[int, tuple[str, str]],
    *,
    treatment_col: str = "시술 종류",
) -> pd.DataFrame:
    """Phase3 프레임에 P3-헤어 보정 — 클러스터링 직전"""
    out = phase_df.copy()
    for idx, row in out.iterrows():
        rid = int(row["recruitId"])
        title, body = texts.get(rid, ("", ""))
        out.at[idx, treatment_col] = apply_generic_hair_treatment(
            str(row.get(treatment_col, UNCLEAR_VALUE)),
            categories.get(rid, []),
            title=title,
            body=body,
        )
    return out


def body_has_snap_signal(title: str, body: str) -> bool:
    return bool(SNAP_BODY_RE.search(f"{title} {body}"))


def has_conflicting_topic_signal(
    categories: list[str],
    title: str,
    body: str,
) -> bool:
    if set(categories) & CONFLICT_TOPIC_CATEGORIES:
        return True
    text = f"{title} {body}"
    return any(pat.search(text) for pat, _ in BODY_TOPIC_CONFLICT_RES)


def should_skip_snap_profile_only(
    categories: list[str],
    title: str,
    body: str,
) -> bool:
    """프로필 카테고리 + 본문 스냅 없음 → 주제 스냅 강제 skip"""
    cat_set = set(categories)
    if PROFILE_CATEGORY not in cat_set:
        return False
    if body_has_snap_signal(title, body):
        return False
    if cat_set & (SNAP_TOPIC_CATEGORIES_HIGH | {SNAP_TOPIC_CATEGORY_PORTRAIT}):
        return False
    return True


def body_has_portfolio_snap_context(title: str, body: str) -> bool:
    """Tier2 조건부 — 본문 포트폴리오·TFP 등 스냅 맥락"""
    return bool(PURPOSE_PORTFOLIO_RE.search(f"{title} {body}"))


def portrait_snap_tier_allowed(
    categories: list[str],
    title: str,
    body: str,
    *,
    snap_tier: str = P3_SNAP_TIER_DEFAULT,
) -> bool:
    """Tier2(인물스냅) → 스냅 허용 여부"""
    if snap_tier == P3_SNAP_TIER1_ONLY:
        return False
    if snap_tier in (P3_SNAP_TIER2_BODY_SNAP, P3_SNAP_TIER2_CONDITIONAL):
        if body_has_snap_signal(title, body):
            return True
        if snap_tier == P3_SNAP_TIER2_CONDITIONAL:
            return body_has_portfolio_snap_context(title, body)
        return False
    return snap_tier == P3_SNAP_TIER_FULL


def apply_snap_topic_correction(
    topic: str,
    categories: list[str],
    *,
    title: str = "",
    body: str = "",
    snap_tier: str = P3_SNAP_TIER_DEFAULT,
) -> str:
    """
    P3-스냅 — Phase3 촬영 주제=불명확을 스키마 규칙으로 스냅 채움

    Tier 1: 카테고리 일상·여행·우정 OR 본문 스냅 키워드
    Tier 2: 카테고리 인물스냅 (snap_tier로 범위 조절)
    웨딩·커플·한복 카테/본문 충돌 시 skip
    """
    if topic != UNCLEAR_VALUE:
        return topic
    if has_conflicting_topic_signal(categories, title, body):
        return topic
    if should_skip_snap_profile_only(categories, title, body):
        return topic
    cat_set = set(categories)
    if cat_set & SNAP_TOPIC_CATEGORIES_HIGH or body_has_snap_signal(title, body):
        return TOPIC_SNAP_VALUE
    if SNAP_TOPIC_CATEGORY_PORTRAIT in cat_set:
        if portrait_snap_tier_allowed(
            categories, title, body, snap_tier=snap_tier
        ):
            return TOPIC_SNAP_VALUE
    return topic


def apply_purpose_correction(
    purpose: str,
    categories: list[str],
    *,
    title: str = "",
    body: str = "",
) -> str:
    """P3-목적 — 불명확만 · 구체 목적 우선 · 스키마 extraction_guide"""
    if purpose != UNCLEAR_VALUE:
        return purpose
    text = f"{title} {body}"
    cat_set = set(categories)
    if PURPOSE_BA_RE.search(text):
        return PURPOSE_BA
    if PROFILE_CATEGORY in cat_set or PURPOSE_PROFILE_RE.search(text):
        return PURPOSE_PROFILE
    if cat_set & BEAUTY_CATEGORIES or PURPOSE_BEAUTY_RE.search(text):
        return PURPOSE_BEAUTY
    if cat_set & FASHION_CATEGORIES or PURPOSE_FASHION_RE.search(text):
        return PURPOSE_FASHION
    if PURPOSE_PORTFOLIO_RE.search(text):
        return PURPOSE_PORTFOLIO
    return purpose


def body_has_studio_signal(text: str) -> bool:
    return bool(PLACE_STUDIO_RE.search(text))


def body_has_outdoor_signal(text: str) -> bool:
    return bool(PLACE_OUTDOOR_RE.search(text))


def apply_place_correction(
    place: str,
    categories: list[str],
    *,
    title: str = "",
    body: str = "",
) -> str:
    """P3-장소 — 불명확만 · 스튜디오/야외 동시 신호 시 skip"""
    if place != UNCLEAR_VALUE:
        return place
    text = f"{title} {body}"
    cat_set = set(categories)
    has_studio = STUDIO_CATEGORY in cat_set or body_has_studio_signal(text)
    has_outdoor = body_has_outdoor_signal(text)
    has_home = bool(PLACE_HOME_RE.search(text))
    if has_studio and has_outdoor:
        return place
    if has_home and (has_studio or has_outdoor):
        return place
    if STUDIO_CATEGORY in cat_set:
        return PLACE_STUDIO
    if has_outdoor:
        return PLACE_OUTDOOR
    if has_home:
        return PLACE_HOME
    if body_has_studio_signal(text):
        return PLACE_STUDIO
    return place


def apply_p3_corrections_to_row(
    row: dict[str, str],
    categories: list[str],
    title: str,
    body: str,
    *,
    apply_hair: bool = True,
    apply_snap: bool = True,
    apply_purpose: bool = True,
    apply_place: bool = True,
    snap_tier: str = P3_SNAP_TIER_DEFAULT,
) -> dict[str, str]:
    """단일 행 P3 보정 — 헤어 → 스냅 → 목적 → 장소"""
    out = dict(row)
    if apply_hair:
        out["시술 종류"] = apply_generic_hair_treatment(
            out.get("시술 종류", UNCLEAR_VALUE),
            categories,
            title=title,
            body=body,
        )
    if apply_snap:
        out["촬영 주제"] = apply_snap_topic_correction(
            out.get("촬영 주제", UNCLEAR_VALUE),
            categories,
            title=title,
            body=body,
            snap_tier=snap_tier,
        )
    if apply_purpose:
        out["촬영 목적"] = apply_purpose_correction(
            out.get("촬영 목적", UNCLEAR_VALUE),
            categories,
            title=title,
            body=body,
        )
    if apply_place:
        out["촬영 장소"] = apply_place_correction(
            out.get("촬영 장소", UNCLEAR_VALUE),
            categories,
            title=title,
            body=body,
        )
    return out


def apply_p3_corrections_phase3_frame(
    phase_df: pd.DataFrame,
    categories: dict[int, list[str]],
    texts: dict[int, tuple[str, str]],
    *,
    apply_hair: bool = True,
    apply_snap: bool = True,
    apply_purpose: bool = True,
    apply_place: bool = True,
    snap_tier: str = P3_SNAP_TIER_DEFAULT,
) -> pd.DataFrame:
    """Phase3 프레임 전체 P3 보정 — 클러스터링 직전"""
    out = phase_df.copy()
    dim_cols = [
        c
        for c in out.columns
        if c in ("시술 종류", "촬영 주제", "촬영 목적", "촬영 장소")
    ]
    for idx, row in out.iterrows():
        rid = int(row["recruitId"])
        title, body = texts.get(rid, ("", ""))
        patched = apply_p3_corrections_to_row(
            {c: str(row.get(c, UNCLEAR_VALUE)) for c in dim_cols},
            categories.get(rid, []),
            title,
            body,
            apply_hair=apply_hair,
            apply_snap=apply_snap,
            apply_purpose=apply_purpose,
            apply_place=apply_place,
            snap_tier=snap_tier,
        )
        for c, v in patched.items():
            out.at[idx, c] = v
    return out


def apply_snap_topic_attrs(
    attrs: dict[str, str],
    categories: list[str],
    title: str,
    body: str,
) -> dict[str, str]:
    out = dict(attrs)
    out["촬영 주제"] = apply_snap_topic_correction(
        out.get("촬영 주제", UNCLEAR_VALUE),
        categories,
        title=title,
        body=body,
        snap_tier=P3_SNAP_TIER_DEFAULT,
    )
    return out


def apply_purpose_correction_attrs(
    attrs: dict[str, str],
    categories: list[str],
    title: str,
    body: str,
) -> dict[str, str]:
    out = dict(attrs)
    out["촬영 목적"] = apply_purpose_correction(
        out.get("촬영 목적", UNCLEAR_VALUE),
        categories,
        title=title,
        body=body,
    )
    return out


def apply_place_correction_attrs(
    attrs: dict[str, str],
    categories: list[str],
    title: str,
    body: str,
) -> dict[str, str]:
    out = dict(attrs)
    out["촬영 장소"] = apply_place_correction(
        out.get("촬영 장소", UNCLEAR_VALUE),
        categories,
        title=title,
        body=body,
    )
    return out


def apply_all_p3_correction_attrs(
    attrs: dict[str, str],
    categories: list[str],
    title: str,
    body: str,
) -> dict[str, str]:
    out = apply_generic_hair_treatment_attrs(attrs, categories, title, body)
    out = apply_snap_topic_attrs(out, categories, title, body)
    out = apply_purpose_correction_attrs(out, categories, title, body)
    return apply_place_correction_attrs(out, categories, title, body)


def apply_snap_topic_phase3_frame(
    phase_df: pd.DataFrame,
    categories: dict[int, list[str]],
    texts: dict[int, tuple[str, str]],
    *,
    topic_col: str = "촬영 주제",
) -> pd.DataFrame:
    """Phase3 프레임에 P3-스냅 보정 — P3-헤어 다음, 클러스터링 직전"""
    out = phase_df.copy()
    for idx, row in out.iterrows():
        rid = int(row["recruitId"])
        title, body = texts.get(rid, ("", ""))
        out.at[idx, topic_col] = apply_snap_topic_correction(
            str(row.get(topic_col, UNCLEAR_VALUE)),
            categories.get(rid, []),
            title=title,
            body=body,
        )
    return out


def extract_attributes(
    schema: dict,
    categories: list[str],
    title: str,
    body: str,
    retries: int = 2,
) -> dict[str, str] | None:
    """
    Ollama로 Phase3 속성 추출.
    성공 시 7개 키 dict 반환, JSON 파싱 완전 실패 시 None
    """
    attrs = phase3_extract_attrs(schema)
    attr_names = phase3_attr_names(schema)
    allowed = valid_values_map(attrs)
    prompt = build_prompt(attrs, categories, title, body)

    for attempt in range(retries + 1):
        try:
            r = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                },
                timeout=120,
            )
            r.raise_for_status()
            raw_text = r.json().get("response", "").strip()
            parsed = parse_json_object(raw_text)
            if parsed is not None:
                attrs = normalize_attributes(parsed, attr_names, allowed)
                attrs = guard_perm_hallucination(attrs, title, body)
                return apply_all_p3_correction_attrs(
                    attrs, categories, title, body
                )
        except Exception:
            pass
        if attempt < retries:
            time.sleep(2)

    return None
