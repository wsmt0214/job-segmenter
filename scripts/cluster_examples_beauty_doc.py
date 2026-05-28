"""뷰티 전용 레거시 엔트리 (Markdown 만 갱신)."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster_segment_examples_doc import build_markdown_for_kind  # noqa: E402


def main() -> None:
    path = build_markdown_for_kind("beauty")
    print("갱신:", path)


if __name__ == "__main__":
    main()
