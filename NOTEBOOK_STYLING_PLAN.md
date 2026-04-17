# Notebook Styling Plan

## Goal

Apply the same notebook redesign approach across the notebooks in this folder without:

- losing original teaching context
- breaking image attachments
- turning markdown into unreadable mixed syntax
- making notebooks unreadable in dark theme

Reference notebook:

- `1. Stastics.ipynb`

---

## Design Direction

The target look is:

- easy to read in dark theme
- elegant, not overly colorful
- strong section hierarchy
- light visual accents, not harsh black panels everywhere
- good scanning for interview prep notes

Keep these visual rules:

- `h1`: strong but not heavy; dark-slate panel with soft accent border
- `h2`: indexed section headers should remain obvious, for example `1.`, `2.`, `9.`
- `h3`: lighter subsection chips, clearly lower priority than `h2`
- body text: light enough for dark theme
- bullets and numbered steps: high contrast and roomy spacing
- callouts: dark-muted cards, not pale cards
- code/formula blocks: dark-surface blocks with readable light text
- tables: current table treatment is acceptable, do not redesign unless necessary

---

## Non-Negotiable Safety Rules

### 1. Never start from an already-transformed notebook if you can avoid it

Always prefer the original notebook content as the source of truth.

Why:

- repeated transformations compound formatting errors
- mixed HTML + markdown becomes hard to recover
- inline formulas and lists break more easily on second pass

Safe rule:

- read the original notebook JSON
- transform from original markdown content
- then write the styled notebook once

### 2. Do not touch image attachment cells

Leave these exactly as-is:

- cells that only contain `![...](attachment:...)`
- cells that mix attachment markdown with explanatory text unless there is a very specific reason to restyle them

Why:

- attachment rendering breaks easily if converted to raw HTML
- notebook trust/render behavior is more fragile for attachment-backed images

### 3. Preserve code cells exactly unless the user asks otherwise

Do not:

- rewrite code
- clear outputs unless asked
- change execution order unless necessary

### 4. Preserve teaching meaning

Do not shorten, paraphrase, or “summarize” the notes unless requested.

Allowed changes:

- presentation
- spacing
- header hierarchy
- list rendering
- table styling
- callout styling

Not allowed by default:

- rewriting explanations
- changing examples
- changing formulas
- removing sections

---

## Cell Classification Before Editing

For every notebook, classify markdown cells into these buckets first.

### A. Attachment-only markdown cells

Examples:

```md
![image.png](attachment:image.png)
```

Action:

- leave untouched

### B. Attachment + text markdown cells

Examples:

- image followed by explanation
- explanation followed by image

Action:

- default: leave untouched
- only style later if needed and only after confirming render safety

### C. Text-only markdown cells

Examples:

- notes
- theory
- formulas
- Q&A
- tables

Action:

- safe to convert fully into styled HTML blocks

### D. Code cells

Action:

- leave content untouched

---

## Safe Transformation Order

Use this order every time.

1. Load notebook JSON.
2. Identify attachment-containing markdown cells.
3. Exclude attachment-containing cells from styling.
4. Convert text-only markdown cells from original markdown source.
5. Apply only the approved style system.
6. Save notebook.
7. Re-check that attachment cells are still plain attachment markdown.
8. Re-check that no raw markdown syntax is leaking inside transformed text cells.

---

## What To Convert In Text-Only Cells

These should render as styled HTML equivalents:

- `#`, `##`, `###`, `####`
- normal paragraphs
- unordered lists
- ordered lists
- blockquotes
- fenced code blocks
- inline code
- markdown tables
- interview tips / one-liners / practical notes

These require extra care:

- LaTeX blocks like `$$ ... $$`
- inline formulas in backticks
- checklist items like `- [ ]`
- long case-study cells with equations and nested bullets

---

## Current Style Rules To Reuse

### Headers

- `h1`: dark-slate gradient panel, soft green accent, light text
- `h2`: indexed section banner with strong left accent
- `h3`: lighter dark chip, softer border
- `h4`: simple accent text

### Text

- body text should be light gray-blue, not dark navy
- list text should match body contrast
- keep line-height around `1.8`

### Callouts

- interview tips: warm dark panel
- explanatory notes: cool dark-slate panel
- one-liners: green-tinted dark panel
- formulas: muted violet/slate panel

### Tables

- leave the current table visual style unless the notebook has a specific issue

### Code / Formulas

- dark block
- light text
- no heavy black unless needed

---

## Problems We Already Hit

These are the failure modes to avoid on the next notebooks.

### 1. Adding extra theme cells

This made the notebook depend on a separate execution step.

Avoid:

- inserting a style-loader cell unless the user explicitly wants that model

Prefer:

- direct cell-level rendering for text-only markdown cells

### 2. Converting attachment images to raw HTML

This broke image rendering.

Avoid:

- replacing `attachment:` markdown with `<img>` HTML

### 3. Mixing HTML headers with raw markdown bullets

This caused half-rendered notes.

Avoid:

- styling only headings while leaving the rest of the same cell as raw markdown

Prefer:

- full cell conversion for text-only cells

### 4. Re-running transformations on already-styled cells

This caused malformed emphasis and formula rendering.

Avoid:

- incremental restyling from transformed notebook output

Prefer:

- rebuild from original markdown

### 5. Using light-theme text colors in dark theme

This made the notebook unreadable.

Avoid:

- dark body text on transparent notebook background

Prefer:

- light body text for transformed text cells

---

## Validation Checklist Per Notebook

Before considering a notebook done, verify all of this:

- attachment-only cells still contain plain attachment markdown
- mixed image cells still render as original markdown
- no raw `**`, `- `, `1. `, or `> ` syntax remains in transformed text-only sections where it should have been rendered
- formulas still look correct
- section numbers remain visible in major headers
- body text is readable in dark theme
- tables are still readable
- code cells are unchanged

---

## Batch Rollout Plan

### Phase 1: High-value notebooks

Apply first to broad theory/reference notebooks:

- `1.1 Stastic General ML.ipynb`
- `1.5 Practical Stastics.ipynb`
- `General ML.ipynb`
- `General LLM QnA.ipynb`
- `python-for-dsa-interviews.ipynb`

### Phase 2: Core ML topic notebooks

- `4.1  Classification Problem, Logistic Regression and Gradient Descent.ipynb`
- `4.3 Decision Tree Understanding.ipynb`
- `4.4 PCA_Understanding.ipynb`
- `5.1 NLP.ipynb`
- `6.1 All About DL.ipynb`

### Phase 3: LLM and RAG notebooks

- `9.1 LLM.ipynb`
- `10. 1 RAG.ipynb`
- `10.2 RAG_Agent.ipynb`
- `10.3 RAG_EVALUATION.ipynb`
- `11.1 Agentic Systems Architecture & Design.ipynb`

### Phase 4: Smaller notebooks and cleanup

- short notebooks
- project notebooks
- any notebook with mixed image+text cells that needs hand review

---

## Recommended Execution Pattern

For each notebook:

1. Inspect the cell map first.
2. Count attachment-only and mixed attachment cells.
3. Transform only text-only markdown cells.
4. Keep tables unless they are clearly broken.
5. Validate dark-theme readability.
6. Move to the next notebook.

Do not batch-edit all notebooks blindly in one pass.

---

## What To Reuse From This Notebook

Use `1. Stastics.ipynb` as the style reference for:

- section hierarchy
- dark-theme body text contrast
- header accent treatment
- callout tone
- readable spacing

Do not reuse from it blindly:

- any image-cell handling logic
- any earlier theme-loader-cell approach

---

## If We Automate Later

If we later create a script, it should:

- parse notebook JSON
- detect attachment-containing markdown cells
- skip attachment cells
- transform only text-only markdown cells
- preserve code cells exactly
- preserve notebook order
- write a new notebook safely

Nice-to-have script outputs:

- number of markdown cells transformed
- number of attachment cells skipped
- list of mixed image+text cells requiring manual review

---

## Practical Next Step

Best next notebook to test this process:

- `1.1 Stastic General ML.ipynb`

Reason:

- likely similar structure
- good signal for whether the workflow generalizes
- lower risk than jumping directly into a large image-heavy notebook
