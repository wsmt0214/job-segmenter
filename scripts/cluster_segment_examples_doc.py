"""급종별 cluster 예시 MD/HTML 생성기 (beauty / photo / model)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ONE_LINERS: dict[str, dict[int, str]] = {
    "beauty": {
        0: "실내·스튜디오 포폴용 일 회 시술·모델 협업 줄이면서 신입 허용 톤이 강함 보상·긴급도는 속성에 잘 안 잡히는 공고 무리에 가까움",
        1: "실내 스튜디오 포폴 헤어메 성격 줄인데 업무 장기 여부와 경력은 불명확이 많고 마감만 일반 톤처럼 잡히는 혼합",
        2: "소정의 페이를 전제로 한 신입 헤어모델 줄이나 촬영 장소·작업 길이는 표기 속성 불명확 비중 큼",
        3: "실내 중심이면서 정기·장기 협업처럼 같은 라인을 채우는 글이 많음 보상·경력 속성 불명확 잔존 큼",
        4: "실내 포폴 헤어메·모델 + 신입 허용 줄인데 일회인지 장기인지 속성에서는 불명확이 크게 남음",
        5: "급구 속성 전부 해당하는 일 회 실내 헤어메 협업 형태로 특정 마감에 맞춰 급히 인력 채우는 공고 줄",
        6: "무페이로 포폴 교환하는 헤어모델·시술 공고에 가깝고 장소·지속성은 불명확으로 남는 경우 많음",
        7: "일 회 실내 헤어메 협업으로 읽히나 마감·보상·경력은 속성 불명확 비중이 큰 혼합",
        8: "급구 즉시 + 신입 허용 + 실내 일 회 포폴 모집처럼 마감 임박 인력 보충형에 가까운 클러스터",
        9: "구조 속성이 대부분 불명확인 짧은 문의·혼합 텍스트 덩어리로 활용처까지 라벨이 잘 안 남는 무리",
    },
    "photo": {
        0: "일 회·실내 줄에 급구 비중 크며 자연연출 속성 붙음 보상·활용·경력은 속성 불명 많은 스냅 모집형",
        1: "속성 라벨 전반이 불명확하게 남은 짧은 문의 줄 작가·모델 섞인 사진 모집 혼합",
        2: "실내·포폴·신입 허용 줄에 마감은 일반 톤이나 작업 장기성·보상은 속성 불명 많음",
        3: "포폴·신입 허용·일상 연출 줄이 강함 야외·혼합 촬영도 섞임 환경·지속성은 불명 많음",
    },
    "model": {
        0: "포폴만 뚜렷 나머지 장소·기간·보상·마감·경력은 속성 불명 많음 신체 요건 줄 자주 명시되는 무리",
        1: "실내 일 회 급구 줄 경력 불명 많음 신체 조건 줄 자주 박힘",
        2: "포폴·신입 허용·신체 제약 없음 줄 강 장소·지속·긴급도는 속성 불명 많음",
        3: "실내·포폴·신입 허용 신체 조건 명시 강 지속성은 속성 불명 많음",
        4: "SNS 온라인 홍보 실내 중심 장기·상시 모집 혼재 경력 불명 신체 자유 혼재",
        5: "실내·포폴·일 회·신입 허용·신체 자유 마감은 불명 많음",
        6: "기술연습·테스트 목적 실내 일 회·신입 허용·신체 조건 명시 강",
        7: "실내·포폴·신입 허용·일반 마감 톤 지속성 불명·신체 자유 혼재",
        8: "포폴만 뚜렷 나머지 장소·지속·보상·마감·경력 대부분 불명·신체 제약 없음",
        9: "구조 속성 대체로 불명 짧거나 모호한 모델 모집 문의 혼합",
        10: "소정의 페이·실내·일 회·포폴·경력 불명·신체 조건 명시 강",
        11: "SNS 온라인·실내·일 회·급구·경력 불명·신체 자유",
    },
}

KIND_META: dict[str, dict[str, Any]] = {
    "beauty": {
        "md_h1": "# 뷰티 구인글 클러스터별 예시 (클러스터당 3건)",
        "dataset": ROOT / "data/text/beauty_dataset.csv",
        "profile": ROOT / "data/clustering/beauty_profiles.json",
        "out_md": ROOT / "docs/cluster_segment_examples_beauty.md",
        "html_title": "뷰티 구인글 클러스터별 예시",
    },
    "photo": {
        "md_h1": "# 사진(포토) 구인글 클러스터별 예시 (클러스터당 3건)",
        "dataset": ROOT / "data/text/photo_dataset.csv",
        "profile": ROOT / "data/clustering/photo_profiles.json",
        "out_md": ROOT / "docs/cluster_segment_examples_photo.md",
        "html_title": "사진(포토) 구인글 클러스터별 예시",
    },
    "model": {
        "md_h1": "# 모델 구인글 클러스터별 예시 (클러스터당 3건)",
        "dataset": ROOT / "data/text/model_dataset.csv",
        "profile": ROOT / "data/clustering/model_profiles.json",
        "out_md": ROOT / "docs/cluster_segment_examples_model.md",
        "html_title": "모델 구인글 클러스터별 예시",
    },
}


def _dominant_kv_line(dom: dict[str, Any]) -> str:
    if not dom:
        return "(프로파일에 dominant_values 없음)"
    return ", ".join(f"{k}={v}" for k, v in dom.items())


def build_markdown_for_kind(kind: str) -> Path:
    cfg = KIND_META[kind]
    liners = ONE_LINERS[kind]

    df = pd.read_csv(cfg["dataset"])
    raw = pd.read_csv(ROOT / "data/raw_recruits.csv", usecols=["recruitId", "title", "content"])
    merged = df.merge(raw, on="recruitId", how="left")

    with open(cfg["profile"], encoding="utf-8") as f:
        profile = json.load(f)

    size_by_cid = {c["cluster_id"]: c.get("size", 0) for c in profile["clusters"]}
    dom_by_cid = {
        c["cluster_id"]: dict(c.get("dominant_values") or {}) for c in profile["clusters"]
    }

    rel_ds = cfg["dataset"].relative_to(ROOT).as_posix()
    rel_pr = cfg["profile"].relative_to(ROOT).as_posix()
    k = profile.get("k")
    silhouette = float(profile.get("silhouette_score", 0.0))

    lines_out: list[str] = [
        cfg["md_h1"],
        "",
        f"- 데이터: `{rel_ds}` 의 `cluster_id` 와 `data/raw_recruits.csv` 의 제목·본문 조인",
        "",
        f"- 프로파일 요약: `{rel_pr}` (K={k}, 실루엣 {silhouette:.6f})",
        "",
        "- 예시 선정: `recruitId` 오름차순으로 클러스터당 상위 3건 (재현 가능)",
        "",
    ]

    for cid in sorted(merged["cluster_id"].unique()):
        sub = merged[merged["cluster_id"] == cid].sort_values("recruitId")
        n_cluster = len(sub)
        prof_n = size_by_cid.get(int(cid), n_cluster)
        lines_out.append(f"## 세그먼트 (클러스터) {int(cid)}")
        lines_out.append("")
        lines_out.append(f"- 본 데이터 기준 건수: **{n_cluster}건** (프로파일 JSON 기준 {prof_n}건)")
        lines_out.append("")
        lines_out.append("- 지배적 속성(프로파일): " + _dominant_kv_line(dom_by_cid.get(int(cid), {})))
        lines_out.append("")
        lines_out.append(
            "- 한 줄 요약: "
            + liners.get(int(cid), f"(한 줄 요약 미작성 cluster_id={int(cid)})")
        )
        lines_out.append("")
        for i, row in enumerate(sub.head(3).itertuples(index=False), 1):
            title = (getattr(row, "title") or "").strip()
            body = (getattr(row, "content") or "").strip().replace("```", "'''")
            title_safe = title.replace("```", "'''")
            rid = int(getattr(row, "recruitId"))
            lines_out.append(f"### 예시 {i} · recruitId `{rid}`")
            lines_out.append("")
            lines_out.append("**제목**")
            lines_out.append("")
            lines_out.append(title_safe if title_safe else "(제목 없음)")
            lines_out.append("")
            lines_out.append("**본문**")
            lines_out.append("")
            lines_out.append("```")
            lines_out.append(body if body else "(본문 없음)")
            lines_out.append("```")
            lines_out.append("")

    out_path = cfg["out_md"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out), encoding="utf-8")
    return out_path


def _load_md_to_html() -> Any:
    script_path = ROOT / "scripts/cluster_examples_md_to_html.py"
    spec = importlib.util.spec_from_file_location("cluster_examples_md_to_html", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cluster_examples_md_to_html 로드 실패")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_html_for_kind(kind: str) -> Path:
    cfg = KIND_META[kind]
    md_path = cfg["out_md"]
    html_path = md_path.with_suffix(".html")
    mod = _load_md_to_html()
    mod.convert(
        md_path,
        html_path,
        page_title=cfg["html_title"],
        md_filename=md_path.name,
    )
    return html_path


def run(kind: str, *, want_md: bool, want_html: bool) -> None:
    if want_md:
        print("MD:", build_markdown_for_kind(kind))
    if want_html:
        print("HTML:", build_html_for_kind(kind))


def main() -> None:
    ap = argparse.ArgumentParser(description="클러스터 예시 MD/HTML (beauty / photo / model)")
    ap.add_argument("--kind", choices=["beauty", "photo", "model", "all"], default="all")
    ap.add_argument("--no-md", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()
    kinds = list(KIND_META) if args.kind == "all" else [args.kind]
    for k in kinds:
        run(k, want_md=not args.no_md, want_html=not args.no_html)


if __name__ == "__main__":
    main()
