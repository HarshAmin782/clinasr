"""Deterministically bundle src/clinasr/*.py into kaggle/run_clinasr.py.

Kaggle's kernel push uploads exactly one `code_file`, so the tested
src/clinasr package (which depends on normal intra-package imports) is
concatenated into a single self-contained script rather than duplicated by
hand. This is the ONLY way kaggle/run_clinasr.py should be edited: never
hand-edit the generated file, change src/clinasr/*.py and rebuild.

How bundling stays correct:
- Modules are concatenated in a fixed dependency order (MODULE_ORDER below):
  each module only needs names already defined by an earlier module in that
  order, since after concatenation everything shares one global namespace.
- Each module's own `from clinasr.<x> import ...` / `import clinasr` lines
  are stripped (those names already exist earlier in the bundle), and only
  ONE `from __future__ import annotations` is kept, at the very top, since
  Python requires future-imports to precede all other statements in a file.
- The build is a pure function of src/clinasr/*.py plus the current git
  commit (recorded for traceability only, not used in comparisons): running
  it twice at the same commit with unchanged sources produces byte-identical
  output, which test_build_kaggle_script.py checks directly.

Usage:
    python scripts/build_kaggle_script.py            # write kaggle/run_clinasr.py
    python scripts/build_kaggle_script.py --check     # exit 1 if out of date, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "clinasr"
OUTPUT_PATH = ROOT / "kaggle" / "run_clinasr.py"

# Dependency order: each module must only reference names defined by a
# module earlier in this list (or by itself).
MODULE_ORDER = [
    "normalization",
    "metrics",
    "entities",
    "medical_entities",
    "noise",
    "config",
    "run_manifest",
    "schema_audit",
    "smoke_test",
    "sampling",
    "stage2",
    "stage4",
    "kaggle_entrypoint",
]

_STRIP_PREFIXES = ("from clinasr.", "from clinasr import", "import clinasr")
_FUTURE_IMPORT = "from __future__ import annotations"


def _get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


_MODULE_IMPORT = re.compile(r"^\s*(from clinasr import |import clinasr\.)")


def check_flat_imports(module_name: str, source: str) -> None:
    """Reject `from clinasr import <module>` / `import clinasr.x`: the
    bundle is one flat namespace, so a module-qualified reference such as
    `stage2.select` would raise NameError on Kaggle (it did, Stage 4 v3).
    Import the needed names directly instead."""
    for lineno, line in enumerate(source.splitlines(), 1):
        if _MODULE_IMPORT.match(line):
            raise ValueError(f"src/clinasr/{module_name}.py:{lineno}: module import not allowed in bundled code: {line.strip()}")


def _strip_module_source(source: str) -> str:
    kept_lines: List[str] = []
    in_paren_import = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_paren_import:
            # Continuation of a stripped `from clinasr.x import (` block.
            if ")" in stripped:
                in_paren_import = False
            continue
        if stripped == _FUTURE_IMPORT:
            continue
        if any(stripped.startswith(p) for p in _STRIP_PREFIXES):
            if "(" in stripped and ")" not in stripped:
                in_paren_import = True
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def build(module_order: List[str] = MODULE_ORDER) -> str:
    parts = [
        '"""GENERATED FILE -- DO NOT EDIT DIRECTLY.',
        "",
        "Built by scripts/build_kaggle_script.py from src/clinasr/*.py.",
        "To change behavior, edit the source modules under src/clinasr/ and",
        "rebuild. This file is what Kaggle CLI uploads as the single code_file.",
        '"""',
        "",
        _FUTURE_IMPORT,
    ]
    for module_name in module_order:
        module_path = SRC_DIR / f"{module_name}.py"
        source = module_path.read_text(encoding="utf-8")
        check_flat_imports(module_name, source)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        parts.append("")
        parts.append(f"# ---- begin src/clinasr/{module_name}.py (sha256:{digest}) ----")
        parts.append(_strip_module_source(source))
        parts.append(f"# ---- end src/clinasr/{module_name}.py ----")
    parts.append("")
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if kaggle/run_clinasr.py is out of date; do not write.",
    )
    args = parser.parse_args(argv)

    bundled = build()
    git_commit = _get_git_commit()
    bundled_with_commit = bundled.replace(
        'Built by scripts/build_kaggle_script.py from src/clinasr/*.py.',
        f'Built by scripts/build_kaggle_script.py from src/clinasr/*.py.\n'
        f'Source commit: {git_commit}',
    )

    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"{OUTPUT_PATH} does not exist; run without --check to create it.")
            return 1
        existing = OUTPUT_PATH.read_text(encoding="utf-8")
        # Ignore the commit line when comparing so --check only flags real
        # source drift, not "you are on a different commit".
        existing_no_commit = "\n".join(
            line for line in existing.splitlines() if not line.startswith("Source commit:")
        )
        bundled_no_commit = "\n".join(
            line for line in bundled.splitlines() if not line.startswith("Source commit:")
        )
        if existing_no_commit != bundled_no_commit:
            print(f"{OUTPUT_PATH} is out of date with src/clinasr/*.py. Rebuild it.")
            return 1
        print("kaggle/run_clinasr.py is up to date.")
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(bundled_with_commit, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(bundled_with_commit)} bytes) from commit {git_commit}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
