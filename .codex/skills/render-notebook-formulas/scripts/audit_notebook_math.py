#!/usr/bin/env python3
"""Audit Jupyter Markdown cells for common unrendered or malformed math."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DISPLAY_RE = re.compile(r"\$\$[\s\S]*?\$\$")
INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(?:\\.|[^$\n])+\$")
FENCE_RE = re.compile(r"```[\s\S]*?```")
TEXT_RE = re.compile(r"\\text\{[^{}]*\}")
FLOW_FENCE_RE = re.compile(
    r"(?:->|→|Add\s*&\s*Norm|(?:^|\n)\s*(?:输入|输出|第\s*\d+\s*步|第一段|第二段|先|再|最后)\s*[：:]?)",
    re.MULTILINE,
)

RAW_MATH_RE = re.compile(
    r"""(?x)
    (?:
        \bQK\^T\b
      | \bK\^TQ\b
      | \bsqrt\(d_k\)
      | \bd_head\b
      | \bW_[QKVO](?:\^\d+)?\b
      | \b[qkvcsoax]_[A-Za-z0-9]+(?:,\d+)?\b
      | (?<![A-Za-z0-9_])
        (?:\d+|[BNDQKVAOh]|d_(?:k|v|head))
        (?:\s+x\s+(?:\d+|[BNDQKVAOh]|d_(?:k|v|head)))+
        (?![A-Za-z0-9_])
    )
    """
)


@dataclass(frozen=True)
class Finding:
    path: Path
    cell: int
    kind: str
    excerpt: str

    def format(self) -> str:
        return f"{self.path}:cell {self.cell} [{self.kind}] {self.excerpt}"


def compact(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def discover(targets: list[Path]) -> list[Path]:
    notebooks: set[Path] = set()
    for target in targets:
        if target.is_file() and target.suffix.lower() == ".ipynb":
            notebooks.add(target.resolve())
        elif target.is_dir():
            notebooks.update(p.resolve() for p in target.rglob("*.ipynb"))
        else:
            print(f"warning: skipped missing or unsupported target: {target}", file=sys.stderr)
    return sorted(notebooks)


def strip_protected_math(text: str) -> str:
    text = DISPLAY_RE.sub("", text)
    text = INLINE_RE.sub("", text)
    return text


def audit_markdown(path: Path, cell_index: int, source: str) -> list[Finding]:
    findings: list[Finding] = []

    if source.count("$$") % 2:
        findings.append(Finding(path, cell_index, "unbalanced-display", "odd number of $$ delimiters"))

    single_dollars = re.findall(r"(?<!\\)(?<!\$)\$(?!\$)", source)
    if len(single_dollars) % 2:
        findings.append(Finding(path, cell_index, "unbalanced-inline", "odd number of inline $ delimiters"))

    for line in source.splitlines():
        if "$$" in line and line.strip() != "$$":
            findings.append(Finding(path, cell_index, "display-delimiter", compact(line)))

    for fence in FENCE_RE.findall(source):
        match = RAW_MATH_RE.search(fence)
        if match:
            findings.append(Finding(path, cell_index, "math-in-code-fence", compact(fence)))
        elif FLOW_FENCE_RE.search(fence):
            findings.append(Finding(path, cell_index, "flow-in-code-fence", compact(fence)))

    plain = FENCE_RE.sub("", strip_protected_math(source))
    match = RAW_MATH_RE.search(plain)
    if match:
        findings.append(Finding(path, cell_index, "raw-math", compact(match.group(0))))

    for display in DISPLAY_RE.findall(source):
        body = display[2:-2]
        without_text = TEXT_RE.sub("", body)
        suspicious = re.search(r"\s[x@]\s|\^T|\bd_head\b", without_text)
        if suspicious:
            findings.append(Finding(path, cell_index, "raw-token-in-latex", compact(suspicious.group(0))))
        if re.search(r"[\u4e00-\u9fff]", without_text):
            findings.append(Finding(path, cell_index, "unwrapped-cjk-in-latex", compact(without_text)))

    return findings


def audit_notebook(path: Path) -> list[Finding]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [Finding(path, -1, "invalid-notebook", compact(str(exc)))]

    findings: list[Finding] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        findings.extend(audit_markdown(path, index, text))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path, help="Notebook files or directories to audit")
    args = parser.parse_args()

    notebooks = discover(args.targets)
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 2

    findings = [finding for path in notebooks for finding in audit_notebook(path)]
    for finding in findings:
        print(finding.format())

    if findings:
        print(f"Found {len(findings)} issue(s) in {len(notebooks)} notebook(s).")
        return 1

    print(f"OK: audited {len(notebooks)} notebook(s); no common raw-math issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
