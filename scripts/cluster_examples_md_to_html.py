"""cluster_segment_examples_beauty.md → 단일 HTML (브라우저에서 파일 열기용)."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MD_DEFAULT = ROOT / "docs/cluster_segment_examples_beauty.md"
HTML_DEFAULT = ROOT / "docs/cluster_segment_examples_beauty.html"
DEFAULT_HTML_PAGE_TITLE = "뷰티 구인글 클러스터별 예시"

# 인라인: **bold**, `mono` (` 먼저 처리 후 ** 처리)
_INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE = re.compile(r"`([^`]+)`")


def _fmt_inline(s: str) -> str:
    s = html.escape(s)
    s = _INLINE_CODE.sub(lambda m: "<code>" + m.group(1) + "</code>", s)
    s = _INLINE_BOLD.sub(lambda m: "<strong>" + m.group(1) + "</strong>", s)
    return s


def _md_section_to_html(text: str) -> str:
    """펜스 밖 구간만; 블록 단위 헤더·리스트·문단 처리"""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    para: list[str] = []
    in_ul = False

    def flush_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def flush_para():
        nonlocal para
        if not para:
            return
        inner = "<br />\n".join(_fmt_inline(p) for p in para)
        out.append(f'<p class="md-p">{inner}</p>')
        para = []

    while i < len(lines):
        line = lines[i]
        st = line.strip()

        if not st:
            flush_para()
            flush_ul()
            i += 1
            continue

        if st.startswith("### "):
            flush_para()
            flush_ul()
            out.append(f'<h3 class="md-h3">{_fmt_inline(st[4:])}</h3>')
            i += 1
            continue

        if st.startswith("## "):
            flush_para()
            flush_ul()
            out.append(f'<h2 class="md-h2">{_fmt_inline(st[3:])}</h2>')
            i += 1
            continue

        if st.startswith("# "):
            flush_para()
            flush_ul()
            out.append(f'<h1 class="md-h1">{_fmt_inline(st[2:])}</h1>')
            i += 1
            continue

        if st.startswith("- "):
            flush_para()
            if not in_ul:
                out.append('<ul class="md-ul">')
                in_ul = True
            body = st[2:]
            out.append(f'<li class="md-li">{_fmt_inline(body)}</li>')
            i += 1
            continue

        flush_ul()
        para.append(st)
        i += 1

    flush_para()
    flush_ul()
    return "\n".join(out)


def markdown_to_standalone_html(md_text: str, *, page_title: str, md_basename: str) -> str:
    parts = md_text.split("```")
    if len(parts) % 2 == 0:
        raise ValueError("펜스 ``` 짝 불일치 (열 수 닫히지 않았을 수 있음)")

    body_chunks: list[str] = []
    for idx, chunk in enumerate(parts):
        if idx % 2 == 0:
            body_chunks.append(_md_section_to_html(chunk))
        else:
            inner = chunk.strip("\n")
            body_chunks.append(
                '<figure class="code-block"><figcaption>본문</figcaption>'
                "<pre><code>"
                + html.escape(inner)
                + "</code></pre></figure>"
            )

    inner_html = "\n".join(body_chunks)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --card: #fff;
      --text: #1a1a1a;
      --muted: #586069;
      --border: #e1e4e8;
      --accent: #0969da;
      --pre-bg: #f6f8fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
        "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      font-size: 15px;
      line-height: 1.6;
      color: var(--text);
      background: var(--bg);
    }}
    .wrap {{
      max-width: 56rem;
      margin: 0 auto;
      padding: 2rem 1.25rem 4rem;
    }}
    header.page-head {{
      margin-bottom: 2rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }}
    header.page-head h1 {{ margin: 0 0 0.5rem; font-size: 1.55rem; font-weight: 700; }}
    header.page-head p {{
      margin: 0;
      font-size: 0.875rem;
      color: var(--muted);
    }}
    .md-h1 {{ font-size: 1.45rem; margin: 2rem 0 1rem; font-weight: 700; }}
    .md-h2 {{
      font-size: 1.2rem;
      margin: 2.5rem 0 0.75rem;
      padding-bottom: 0.35rem;
      border-bottom: 2px solid var(--accent);
      font-weight: 700;
    }}
    .md-h3 {{
      font-size: 1.05rem;
      margin: 1.75rem 0 0.5rem;
      font-weight: 600;
      color: #24292f;
    }}
    .md-p {{ margin: 0.5rem 0 0; }}
    .md-ul {{
      margin: 0.4rem 0 0;
      padding-left: 1.25rem;
    }}
    .md-li {{ margin: 0.2rem 0; }}
    code {{
      font-family: ui-monospace, "SF Mono", Monaco, Menlo, Consolas, monospace;
      font-size: 0.9em;
      background: var(--pre-bg);
      padding: 0.12rem 0.35rem;
      border-radius: 4px;
    }}
    .code-block {{
      margin: 0.65rem 0 1rem;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(30,35,41,0.04);
    }}
    .code-block figcaption {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      padding: 0.45rem 0.85rem;
      background: linear-gradient(#fafbfc, #f6f8fa);
      border-bottom: 1px solid var(--border);
    }}
    .code-block pre {{
      margin: 0;
      padding: 0.85rem 1rem;
      background: var(--pre-bg);
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .code-block pre code {{
      background: transparent;
      padding: 0;
      font-size: 0.8125rem;
      line-height: 1.55;
    }}
    strong {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="page-head">
      <h1>{html.escape(page_title)}</h1>
      <p><code>{html.escape(md_basename)}</code> 에서 생성 · 로컬에서 파일 더블클릭 또는 브라우저로 드래그 가능</p>
    </header>
    <article class="content">
{inner_html}
    </article>
  </div>
</body>
</html>
"""


def strip_leading_document_h1(md: str) -> str:
    """문서 최상단 `# 한 줄 제목`(cluster 예시 리포 패턴 제거 헤더용)"""
    lines_md = md.lstrip("\ufeff").split("\n")
    if lines_md and lines_md[0].startswith("# "):
        drop = 1
        while drop < len(lines_md) and lines_md[drop].strip() == "":
            drop += 1
        return "\n".join(lines_md[drop:]).lstrip("\n")
    return md


def convert(
    md_path: Path,
    html_path: Path,
    *,
    page_title: str | None = None,
    md_filename: str | None = None,
    strip_first_h1: bool = True,
) -> None:
    md = md_path.read_text(encoding="utf-8")
    if strip_first_h1:
        md = strip_leading_document_h1(md)
    title = page_title or "구인글 클러스터별 예시"
    fname = md_filename or md_path.name
    out = markdown_to_standalone_html(md, page_title=title, md_basename=fname)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(out, encoding="utf-8")


def main() -> None:
    convert(
        MD_DEFAULT,
        HTML_DEFAULT,
        page_title=DEFAULT_HTML_PAGE_TITLE,
        md_filename=MD_DEFAULT.name,
    )
    print("작성:", HTML_DEFAULT)


if __name__ == "__main__":
    main()
