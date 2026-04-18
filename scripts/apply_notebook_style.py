from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "markdown-it-py is required. Use tmp/notebook-style-venv/bin/python to run this script, "
        "or install markdown-it-py in an isolated environment."
    ) from exc


OLD_STYLE_VERSION = "2026-04-v1"
OLD_STYLE_MARKER = f'data-notebook-style-version="{OLD_STYLE_VERSION}"'
CELL_STYLE_VERSION = "2026-04-v2"
CELL_STYLE_MARKER = f'data-notebook-cell-style-version="{CELL_STYLE_VERSION}"'

MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable("table").enable("strikethrough")


ROOT_DIV_STYLE = (
    "color:#d7e0e7;line-height:1.8;font-family:ui-sans-serif,system-ui,-apple-system,"
    "BlinkMacSystemFont,'Segoe UI',sans-serif;"
)
H1_STYLE = (
    "background:linear-gradient(135deg,#22313d 0%,#284740 100%);color:#edf5f3;padding:18px 22px;"
    "border-radius:20px;border:1px solid #3a5255;border-left:10px solid #78b0a1;"
    "box-shadow:0 10px 24px rgba(0,0,0,0.20);margin:0 0 18px 0;"
)
H2_STYLE = (
    "background:linear-gradient(90deg,#22313d 0%,#273743 100%);border-left:8px solid #72aa9b;"
    "color:#e7eff3;padding:11px 16px;border-radius:14px;border:1px solid #3a4b58;"
    "box-shadow:0 6px 18px rgba(0,0,0,0.18);margin:22px 0 14px 0;"
)
H3_STYLE = (
    "display:inline-block;color:#d8e4ea;background:#22303a;border-left:5px solid #7fa89e;"
    "padding:8px 14px;border-radius:12px;border:1px solid #3a4a52;margin:18px 0 10px 0;"
)
H4_STYLE = "color:#9fd2c3;margin:16px 0 8px 0;font-weight:700;"
P_STYLE = "margin:10px 0 15px 0;line-height:1.82;color:#d7e0e7;font-size:1rem;"
LIST_STYLE = "margin:10px 0 16px 20px;padding-left:18px;line-height:1.9;color:#d2dce4;"
LI_STYLE = "margin:6px 0;"
BLOCKQUOTE_STYLE = (
    "margin:14px 0;padding:12px 16px;background:#24313d;border-left:5px solid #90b7a2;"
    "border-radius:14px;border:1px solid #3c4d5a;color:#d9e4ec;"
)
BLOCKQUOTE_P_STYLE = "margin:6px 0;line-height:1.75;color:#d9e4ec;"
PRE_STYLE = (
    "background:#1b2330;color:#e6edf3;padding:14px 16px;border-radius:14px;"
    "border:1px solid #334155;overflow-x:auto;line-height:1.7;margin:14px 0;"
)
PRE_CODE_STYLE = "background:transparent;color:#e6edf3;border:0;padding:0;"
INLINE_CODE_STYLE = (
    "background:#27313a;color:#e8eef4;padding:2px 6px;border-radius:6px;border:1px solid #3c4a55;"
)
HR_STYLE = "border:0;height:1px;background:linear-gradient(90deg,transparent,#556977,transparent);margin:20px 0;"
TABLE_STYLE = (
    "width:100%;border-collapse:collapse;background:#ffffff;font-size:0.96rem;margin:16px 0;"
    "border:1px solid #dde6ec;border-radius:16px;overflow:hidden;"
)
TH_STYLE = (
    "background:#f3f6f8;color:#24415c;padding:10px 12px;text-align:left;border:1px solid #dde6ec;"
    "font-weight:700;"
)
TD_STYLE = "padding:10px 12px;border:1px solid #e4edf2;vertical-align:top;color:#31475b;"
A_STYLE = "color:#8fd1ff;"
SUMMARY_STYLE = "color:#e6edf3;cursor:pointer;font-weight:700;"
CALLOUT_STYLES = {
    "interview tip:": (
        "margin:14px 0;padding:12px 16px;background:#2d251e;border-left:5px solid #caa56c;"
        "border-radius:14px;border:1px solid #56442f;color:#f1e5d0;line-height:1.75;"
    ),
    "interview one-liner:": (
        "margin:14px 0;padding:12px 16px;background:#21352d;border-left:5px solid #72aa9b;"
        "border-radius:14px;border:1px solid #38574d;color:#dff4eb;line-height:1.75;"
    ),
    "purpose:": (
        "margin:14px 0;padding:12px 16px;background:#24313d;border-left:5px solid #90b7a2;"
        "border-radius:14px;border:1px solid #3c4d5a;color:#d9e4ec;line-height:1.75;"
    ),
    "definition:": (
        "margin:14px 0;padding:12px 16px;background:#24313d;border-left:5px solid #6f95b0;"
        "border-radius:14px;border:1px solid #3c4d5a;color:#d9e4ec;line-height:1.75;"
    ),
    "practical note:": (
        "margin:14px 0;padding:12px 16px;background:#24313d;border-left:5px solid #90b7a2;"
        "border-radius:14px;border:1px solid #3c4d5a;color:#d9e4ec;line-height:1.75;"
    ),
}

HTML_DIRECTIVE_RE = re.compile(
    r"<\s*(details|summary|style|script|iframe|img|video|audio|a|div|table|h[1-6]|p)(?:\s+[A-Za-z_:][^>]*)?\s*>",
    re.I,
)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\\n]+\)")
EXTERNAL_MEDIA_RE = re.compile(r"<img\b|!\[[^\]\n]*\]\((?!attachment:)[^)\\n]+\)", re.I)
TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.M)
MATH_BLOCK_RE = re.compile(r"<p>\s*\$\$(.*?)\$\$\s*</p>", re.S)
INLINE_CALLOUT_RE = re.compile(r"<p><strong>([^<:]+:)</strong>\s*(.*?)</p>", re.S)
CODE_BLOCK_RE = re.compile(r"<pre><code(?P<attrs>[^>]*)>(?P<content>.*?)</code></pre>", re.S)
DIV_WRAPPED_FORMULA_RE = re.compile(
    r'<div style="(?P<style>[^"]*)">\s*\$\$\s*(?P<formula>.*?)\s*\$\$\s*</div>',
    re.S,
)
P_WRAPPED_FORMULA_RE = re.compile(
    r'<p style="(?P<style>[^"]*)">(?P<before>.*?)\s*\$\$\s*(?P<formula>.*?)\s*\$\$\s*(?P<after>.*?)</p>',
    re.S,
)


@dataclass
class NotebookResult:
    path: Path
    action: str
    transformed_cells: int
    skipped_cells: int
    removed_style_cells: int


def lines(text: str) -> list[str]:
    return [(line + "\n") for line in text.splitlines()]


def normalize_display_math(source: str) -> str:
    normalized: list[str] = []

    for line in source.splitlines():
        if line.count("$$") != 2:
            normalized.append(line)
            continue

        first = line.find("$$")
        second = line.find("$$", first + 2)
        before = line[:first].rstrip()
        expr = line[first + 2 : second].strip()
        after = line[second + 2 :].lstrip()

        if not expr:
            normalized.append(line)
            continue

        if before and after:
            inline = f"${expr}$"
            joined = before
            if not joined.endswith((" ", "(", "[", "{", "-")):
                joined += " "
            joined += inline
            if after and not after.startswith((".", ",", ";", ":", ")", "]", "}")):
                joined += " "
            joined += after
            normalized.append(joined)
            continue

        if before and not after:
            normalized.extend([before, "", "$$", expr, "$$"])
            continue

        if not before and after:
            normalized.extend(["$$", expr, "$$", "", after])
            continue

        normalized.extend(["$$", expr, "$$"])

    return "\n".join(normalized)


def repair_styled_math_source(source: str) -> tuple[str, bool]:
    changed = False

    def div_repl(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        formula = match.group("formula").strip()
        return f"$$\n{formula}\n$$"

    repaired = DIV_WRAPPED_FORMULA_RE.sub(div_repl, source)

    def p_repl(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        style = match.group("style")
        before = match.group("before").strip()
        formula = match.group("formula").strip()
        after = match.group("after").strip()

        if before and after:
            compact_formula = " ".join(formula.split())
            if "\n" not in formula and len(compact_formula) <= 40:
                return f'<p style="{style}">{before} ${compact_formula}$ {after}</p>'
            return f'<p style="{style}">{before}</p>\n\n$$\n{formula}\n$$\n\n<p style="{style}">{after}</p>'

        if before and not after:
            return f'<p style="{style}">{before}</p>\n\n$$\n{formula}\n$$'

        if not before and after:
            return f'$$\n{formula}\n$$\n\n<p style="{style}">{after}</p>'

        return f'$$\n{formula}\n$$'

    repaired = P_WRAPPED_FORMULA_RE.sub(p_repl, repaired)
    return repaired, changed


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


def is_old_style_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    return OLD_STYLE_MARKER in "".join(cell.get("source", []))


def is_already_styled_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    source = "".join(cell.get("source", []))
    stripped = source.lstrip()
    return CELL_STYLE_MARKER in source or bool(
        re.match(r"<(?:div|p|ul|ol|pre|blockquote|h[1-6])\s+style=", stripped)
    )


def has_attachment(cell: dict, source: str) -> bool:
    return bool(cell.get("attachments")) or "attachment:" in source


def has_directive_or_raw_html(source: str) -> bool:
    return bool(HTML_DIRECTIVE_RE.search(source))


def classify_markdown_cell(cell: dict) -> str:
    source = "".join(cell.get("source", []))
    stripped = source.strip()

    if not stripped:
        return "skip"
    if has_attachment(cell, source):
        return "skip"
    if MARKDOWN_IMAGE_RE.search(source) or EXTERNAL_MEDIA_RE.search(source):
        return "skip"
    if is_already_styled_cell(cell):
        return "skip"
    if has_directive_or_raw_html(source):
        return "skip"
    return "transform"


def style_heading(tag: str, style: str, html: str) -> str:
    return html.replace(f"<{tag}>", f'<{tag} style="{style}">')


def style_tag(tag: str, style: str, html: str) -> str:
    return re.sub(rf"<{tag}>", f'<{tag} style="{style}">', html)


def transform_code_blocks(html: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def repl(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        content = match.group("content")
        block = f'<pre style="{PRE_STYLE}"><code{attrs} style="{PRE_CODE_STYLE}">{content}</code></pre>'
        blocks.append(block)
        return f"__CODE_BLOCK_{len(blocks) - 1}__"

    transformed = CODE_BLOCK_RE.sub(repl, html)
    return transformed, blocks


def restore_code_blocks(html: str, blocks: list[str]) -> str:
    for idx, block in enumerate(blocks):
        html = html.replace(f"__CODE_BLOCK_{idx}__", block)
    return html


def extract_formula_blocks(html: str) -> tuple[str, list[str]]:
    formulas: list[str] = []

    def repl(match: re.Match[str]) -> str:
        formulas.append(match.group(1).strip())
        return f"__FORMULA_BLOCK_{len(formulas) - 1}__"

    transformed = MATH_BLOCK_RE.sub(repl, html)
    return transformed, formulas


def restore_formula_blocks(html: str, formulas: list[str]) -> str:
    for idx, formula in enumerate(formulas):
        html = html.replace(f"__FORMULA_BLOCK_{idx}__", f"\n$$\n{formula}\n$$\n")
    return html


def apply_callout_styles(html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        label = match.group(1)
        body = match.group(2)
        style = CALLOUT_STYLES.get(label.strip().lower())
        if style is None:
            return match.group(0)
        return (
            f'<div style="{style}"><p style="margin:6px 0;line-height:1.75;">'
            f"<strong>{label}</strong> {body}</p></div>"
        )

    return INLINE_CALLOUT_RE.sub(repl, html)


def style_table_rows(html: str) -> str:
    row_index = {"count": 0}

    def repl(match: re.Match[str]) -> str:
        row_index["count"] += 1
        if row_index["count"] % 2 == 0:
            return '<tr style="background:#f8fbff;">'
        return "<tr>"

    return re.sub(r"<tr>", repl, html)


def style_inline_html(html: str) -> str:
    html, code_blocks = transform_code_blocks(html)
    html, formula_blocks = extract_formula_blocks(html)
    html = apply_callout_styles(html)

    html = style_heading("h1", H1_STYLE, html)
    html = style_heading("h2", H2_STYLE, html)
    html = style_heading("h3", H3_STYLE, html)
    html = style_heading("h4", H4_STYLE, html)
    html = style_tag("p", P_STYLE, html)
    html = style_tag("ul", LIST_STYLE, html)
    html = style_tag("ol", LIST_STYLE, html)
    html = style_tag("li", LI_STYLE, html)
    html = style_tag("blockquote", BLOCKQUOTE_STYLE, html)
    html = re.sub(r"<blockquote>\s*<p style=\"[^\"]*\">", f'<blockquote style="{BLOCKQUOTE_STYLE}"><p style="{BLOCKQUOTE_P_STYLE}">', html)
    html = re.sub(r"<hr\s*/>", f'<hr style="{HR_STYLE}" />', html)
    html = style_tag("table", TABLE_STYLE, html)
    html = style_tag("th", TH_STYLE, html)
    html = style_tag("td", TD_STYLE, html)
    html = re.sub(r"<a href=", f'<a style="{A_STYLE}" href=', html)
    html = re.sub(r"<summary>", f'<summary style="{SUMMARY_STYLE}">', html)
    html = re.sub(r"<code>(.*?)</code>", f'<code style="{INLINE_CODE_STYLE}">\\1</code>', html, flags=re.S)
    html = style_table_rows(html)

    html = restore_code_blocks(html, code_blocks)
    html = restore_formula_blocks(html, formula_blocks)
    return html


def transform_markdown_source(source: str) -> str:
    html = MARKDOWN.render(source)
    html = style_inline_html(html)
    return f'<!-- {CELL_STYLE_MARKER} -->\n{html}'


def process_notebook(path: Path, write: bool) -> NotebookResult:
    notebook = load_notebook(path)
    updated = copy.deepcopy(notebook)
    original_cells = updated.get("cells", [])

    cleaned_cells: list[dict] = []
    removed_style_cells = 0
    transformed_cells = 0
    skipped_cells = 0

    for cell in original_cells:
        if is_old_style_cell(cell):
            removed_style_cells += 1
            continue

        if cell.get("cell_type") != "markdown":
            cleaned_cells.append(cell)
            continue

        classification = classify_markdown_cell(cell)

        if classification != "transform":
            if is_already_styled_cell(cell):
                source = "".join(cell.get("source", []))
                repaired_source, repaired = repair_styled_math_source(source)
                if repaired:
                    transformed = copy.deepcopy(cell)
                    transformed["source"] = lines(repaired_source)
                    transformed_cells += 1
                    cleaned_cells.append(transformed)
                    continue
            skipped_cells += 1
            cleaned_cells.append(cell)
            continue

        source = "".join(cell.get("source", []))
        source = normalize_display_math(source)
        transformed = copy.deepcopy(cell)
        transformed["source"] = lines(transform_markdown_source(source))
        transformed_cells += 1
        cleaned_cells.append(transformed)

    updated["cells"] = cleaned_cells

    if write:
        save_notebook(path, updated)

    action = "changed" if transformed_cells or removed_style_cells else "unchanged"
    return NotebookResult(
        path=path,
        action=action,
        transformed_cells=transformed_cells,
        skipped_cells=skipped_cells,
        removed_style_cells=removed_style_cells,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply direct inline styling to notebook markdown cells.")
    parser.add_argument("paths", nargs="*", help="Optional notebook paths. If omitted, scan the repo root.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root to scan for notebooks.")
    parser.add_argument("--check", action="store_true", help="Report planned changes without writing.")
    return parser.parse_args()


def resolve_paths(root: Path, paths: list[str]) -> list[Path]:
    if paths:
        return [Path(path).resolve() if Path(path).is_absolute() else (root / path).resolve() for path in paths]
    return [path.resolve() for path in iter_notebooks(root)]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    targets = resolve_paths(root, args.paths)

    if not targets:
        print("No notebooks found.", file=sys.stderr)
        raise SystemExit(1)

    results: list[NotebookResult] = []
    for path in targets:
        result = process_notebook(path, write=not args.check)
        results.append(result)

    changed = sum(1 for result in results if result.action == "changed")
    transformed = sum(result.transformed_cells for result in results)
    removed = sum(result.removed_style_cells for result in results)
    skipped = sum(result.skipped_cells for result in results)
    print(
        f"processed={len(results)} changed={changed} transformed_cells={transformed} "
        f"skipped_markdown_cells={skipped} removed_style_cells={removed} style_version={CELL_STYLE_VERSION}"
    )
    for result in results:
        rel = result.path.relative_to(root)
        print(
            f"{result.action}\t{result.transformed_cells}\t{result.skipped_cells}\t"
            f"{result.removed_style_cells}\t{rel}"
        )


if __name__ == "__main__":
    main()
