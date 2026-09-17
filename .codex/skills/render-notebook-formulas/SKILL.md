---
name: render-notebook-formulas
description: Convert computational formulas, mathematical notation, and structured process explanations in Jupyter Notebook Markdown cells into reliable LaTeX while preserving ordinary prose, code, outputs, and notebook structure. Use for .ipynb rendering fixes involving equations, tensor shapes, variables, matrix derivations, module pipelines, step-by-step flows, formula-like text fences, and audits for unrendered notebook math or process diagrams.
---

# Render Notebook Formulas

Convert three content classes in notebook Markdown cells to Jupyter-renderable LaTeX: computational formulas, mathematical notation, and structured process explanations. Then verify both the notebook structure and rendered export.

## Workflow

1. Locate the requested `.ipynb` files and inspect nearby notebooks to match their style.
2. Check the worktree before editing. Preserve unrelated and pre-existing changes.
3. Inventory formula-like content in Markdown cells:
   - Computational formulas such as residual equations, normalization formulas, weighted sums, and matrix products.
   - ASCII dimensions such as `N x D` or `B x h x N x d_head`.
   - Transposes, subscripts, functions, and variables such as `QK^T`, `d_head`, `q_i`, `W_Q`, `sqrt(d_k)`, `score`, and `alpha`.
   - Matrix-shape derivations using `@` or arrows.
   - Structured process explanations such as `Attention -> Add & Norm -> FFN`, input/output routes, ordered transformations, module comparisons, and branching relationships.
   - Formula tables, process diagrams, or definitions trapped inside fenced `text` blocks or inline code.
4. Edit contextually. Do not perform an unreviewed repository-wide replacement.
5. Run `scripts/audit_notebook_math.py` on the edited path.
6. Parse every edited notebook as JSON, export it with `jupyter nbconvert`, and inspect the diff.

## LaTeX Rules

- Use `$...$` for variables or short expressions embedded in prose.
- Use `$$...$$` for important standalone formulas. Put each `$$` delimiter on its own line.
- Use `\begin{aligned}...\end{aligned}` for multi-line shape derivations.
- Use `\begin{array}` or `\begin{bmatrix}` for mathematical tables and matrices.
- Use `\rightarrow`, `\xrightarrow{...}`, `\begin{aligned}`, and `\text{...}` for structured process explanations.
- Keep Chinese explanations outside math mode. When text must appear inside display math, wrap it with `\text{...}`.
- Remove formulas, mathematical notation, and structured process explanations from code fences before adding LaTeX; MathJax does not render inside fenced code.
- Leave only ordinary narrative prose outside math mode. Do not preserve a structured flow as a text fence merely because it contains Chinese words.

Apply this scope explicitly:

- Computational formula: `y = x + F(x)` -> display LaTeX.
- Mathematical notation: `B x N x D`, `d_ff`, or `x_i` -> inline or display LaTeX according to context.
- Process explanation: `Attention -> Add & Norm -> FFN` -> a LaTeX flow using `\rightarrow` and `\operatorname`/`\text`.

Normalize common notation:

- `N x D` -> `$N \times D$`
- `QK^T` -> `$QK^{\top}$`
- `d_head` -> `$d_{\mathrm{head}}$`
- `sqrt(d_k)` -> `$\sqrt{d_k}$`
- `alpha_i` -> `$\alpha_i$`
- `softmax` inside a formula -> `\operatorname{softmax}`

## Shape-Safety Rules

Distinguish tensor dimensions from matrix multiplication.

- A tensor shape is one dimension chain: `$B \times h \times N \times d_{\mathrm{head}}$`.
- A matrix-shape multiplication is two grouped shapes: `$(N \times d_k)\cdot(d_k \times N) \rightarrow N \times N$`.
- Never rewrite a four-axis tensor shape as `(B \times h)\cdot(N \times d_{\mathrm{head}})`.
- Preserve the mathematical meaning of `@`; do not flatten both operand shapes into one ambiguous chain of `\times` symbols.

## Editing Boundaries

- Modify Markdown cell sources only unless the user explicitly requests code changes.
- Preserve code cells, outputs, execution counts, notebook metadata, lesson order, and ordinary prose.
- Keep source arrays valid JSON and retain UTF-8 Chinese text.
- Avoid adding notebook cell IDs or normalizing unrelated metadata solely to silence existing warnings.

## Validation

Run the bundled audit:

```powershell
python .codex/skills/render-notebook-formulas/scripts/audit_notebook_math.py <notebook-or-directory>
```

Then perform structural and export checks:

```powershell
python -c "import json, pathlib; [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('<directory>').glob('*.ipynb')]"
jupyter nbconvert --to html --stdout --log-level ERROR <notebook.ipynb> | Out-Null
git diff --check -- <edited-notebooks>
```

Treat audit hits as review candidates, not automatic proof of an error. Inspect surrounding prose before changing them. Finish only after all requested notebooks parse and export successfully and the diff contains no unrelated edits.
