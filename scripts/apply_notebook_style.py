from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path


STYLE_VERSION = "2026-04-v1"
STYLE_MARKER = f'data-notebook-style-version="{STYLE_VERSION}"'


STYLE_CELL_SOURCE = """<style>
/* notebook-style-v1: dark-theme-first markdown presentation */

.jp-MarkdownCell .jp-RenderedHTMLCommon,
.text_cell_render,
.markdown-body {
  color: #d7e0e7;
  line-height: 1.8;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon h1,
.text_cell_render h1,
.markdown-body h1 {
  background: linear-gradient(135deg, #22313d 0%, #284740 100%);
  color: #edf5f3;
  padding: 18px 22px;
  border-radius: 20px;
  border: 1px solid #3a5255;
  border-left: 10px solid #78b0a1;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20);
  margin: 0 0 18px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon h2,
.text_cell_render h2,
.markdown-body h2 {
  background: linear-gradient(90deg, #22313d 0%, #273743 100%);
  border-left: 8px solid #72aa9b;
  color: #e7eff3;
  padding: 11px 16px;
  border-radius: 14px;
  border: 1px solid #3a4b58;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
  margin: 22px 0 14px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon h3,
.text_cell_render h3,
.markdown-body h3 {
  display: inline-block;
  color: #d8e4ea;
  background: #22303a;
  border-left: 5px solid #7fa89e;
  padding: 8px 14px;
  border-radius: 12px;
  border: 1px solid #3a4a52;
  margin: 18px 0 10px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon h4,
.text_cell_render h4,
.markdown-body h4 {
  color: #9fd2c3;
  margin: 16px 0 8px 0;
  font-weight: 700;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon p,
.jp-MarkdownCell .jp-RenderedHTMLCommon li,
.text_cell_render p,
.text_cell_render li,
.markdown-body p,
.markdown-body li {
  color: #d7e0e7;
  line-height: 1.8;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon ul,
.jp-MarkdownCell .jp-RenderedHTMLCommon ol,
.text_cell_render ul,
.text_cell_render ol,
.markdown-body ul,
.markdown-body ol {
  margin: 10px 0 16px 20px;
  padding-left: 18px;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon li,
.text_cell_render li,
.markdown-body li {
  margin: 6px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon blockquote,
.text_cell_render blockquote,
.markdown-body blockquote {
  margin: 14px 0;
  padding: 12px 16px;
  background: #24313d;
  border-left: 5px solid #90b7a2;
  border-radius: 14px;
  border: 1px solid #3c4d5a;
  color: #d9e4ec;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon blockquote p,
.text_cell_render blockquote p,
.markdown-body blockquote p {
  margin: 6px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon pre,
.text_cell_render pre,
.markdown-body pre {
  background: #1b2330;
  color: #e6edf3;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid #334155;
  overflow-x: auto;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon code,
.text_cell_render code,
.markdown-body code {
  background: #27313a;
  color: #e8eef4;
  padding: 2px 6px;
  border-radius: 6px;
  border: 1px solid #3c4a55;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon pre code,
.text_cell_render pre code,
.markdown-body pre code {
  background: transparent;
  border: 0;
  padding: 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon hr,
.text_cell_render hr,
.markdown-body hr {
  border: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, #556977, transparent);
  margin: 20px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon table,
.text_cell_render table,
.markdown-body table {
  border-collapse: collapse;
  background: #ffffff;
  color: #31475b;
  font-size: 0.96rem;
  margin: 16px 0;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon thead th,
.text_cell_render thead th,
.markdown-body thead th {
  background: #f3f6f8;
  color: #24415c;
  padding: 10px 12px;
  text-align: left;
  border: 1px solid #dde6ec;
  font-weight: 700;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon tbody td,
.text_cell_render tbody td,
.markdown-body tbody td {
  padding: 10px 12px;
  border: 1px solid #e4edf2;
  vertical-align: top;
  color: #31475b;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon tbody tr:nth-child(even) td,
.text_cell_render tbody tr:nth-child(even) td,
.markdown-body tbody tr:nth-child(even) td {
  background: #f8fbff;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon a,
.text_cell_render a,
.markdown-body a {
  color: #8fd1ff;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon summary,
.text_cell_render summary,
.markdown-body summary {
  color: #e6edf3;
  cursor: pointer;
  font-weight: 700;
}

.jp-MarkdownCell .jp-RenderedHTMLCommon mjx-container,
.text_cell_render mjx-container,
.markdown-body mjx-container {
  color: #e6edf3 !important;
}
</style>
<div data-notebook-style-version="2026-04-v1" style="display:none;"></div>
"""


@dataclass
class NotebookResult:
    path: Path
    action: str
    markdown_cells: int
    code_cells: int


def build_style_cell() -> dict:
    return {
        "cell_type": "markdown",
        "id": "notebook-style-v1",
        "metadata": {},
        "source": [line + "\n" for line in STYLE_CELL_SOURCE.strip("\n").splitlines()],
    }


def iter_notebooks(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for path in sorted(root.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        notebooks.append(path)
    return notebooks


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(path: Path, notebook: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write("\n")


def find_style_cell(cells: list[dict]) -> int | None:
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if STYLE_MARKER in source:
            return idx
    return None


def apply_style_cell(notebook: dict) -> tuple[dict, str]:
    updated = copy.deepcopy(notebook)
    cells = updated.setdefault("cells", [])
    style_cell = build_style_cell()
    style_index = find_style_cell(cells)

    if style_index is None:
        updated["cells"] = [style_cell, *cells]
        return updated, "inserted"

    cells[style_index] = style_cell
    return updated, "updated"


def validate_unchanged_content(before: dict, after: dict, action: str) -> None:
    before_cells = before.get("cells", [])
    after_cells = after.get("cells", [])

    if action == "inserted":
        if after_cells[1:] != before_cells:
            raise ValueError("Notebook content changed outside the inserted style cell.")
        return

    style_index = find_style_cell(after_cells)
    if style_index is None:
        raise ValueError("Updated notebook is missing the style marker cell.")

    if len(after_cells) != len(before_cells):
        raise ValueError("Updated notebook changed cell count unexpectedly.")

    for idx, (old_cell, new_cell) in enumerate(zip(before_cells, after_cells)):
        if idx == style_index:
            continue
        if old_cell != new_cell:
            raise ValueError(f"Notebook content changed in cell {idx} outside the style cell.")


def summarize_cells(notebook: dict) -> tuple[int, int]:
    markdown_cells = 0
    code_cells = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            markdown_cells += 1
        elif cell.get("cell_type") == "code":
            code_cells += 1
    return markdown_cells, code_cells


def process_notebook(path: Path, write: bool) -> NotebookResult:
    original = load_notebook(path)
    updated, action = apply_style_cell(original)
    validate_unchanged_content(original, updated, action)

    if write:
        save_notebook(path, updated)

    markdown_cells, code_cells = summarize_cells(updated)
    return NotebookResult(path=path, action=action, markdown_cells=markdown_cells, code_cells=code_cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert or update a shared markdown style cell across notebooks.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root to scan for notebooks.")
    parser.add_argument("--check", action="store_true", help="Validate and report without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    notebooks = iter_notebooks(root)
    results: list[NotebookResult] = []

    for path in notebooks:
        result = process_notebook(path, write=not args.check)
        results.append(result)

    inserted = sum(1 for result in results if result.action == "inserted")
    updated = sum(1 for result in results if result.action == "updated")
    print(f"processed={len(results)} inserted={inserted} updated={updated} style_version={STYLE_VERSION}")
    for result in results:
        print(f"{result.action}\t{result.markdown_cells}\t{result.code_cells}\t{result.path.relative_to(root)}")


if __name__ == "__main__":
    main()
