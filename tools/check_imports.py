"""
check_imports.py

Static verification that every intra-project import resolves to a module that actually
exists. Parses with `ast` -- nothing is executed and no third-party package needs to be
installed, so this covers `kernels/` and `compiler/` (which need CUDA and Triton) on a
laptop with neither.

This is the safety net for the package restructure: a rename or a moved module that
breaks an import shows up here instead of on a rented GPU.

Usage:
    python tools/check_imports.py
    python tools/check_imports.py --root src/routingdrift

Exit code 0 if every intra-project import resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anything importable from PyPI or the stdlib. We only verify imports that refer to code
# inside this repository -- third-party availability is requirements.txt's job.
THIRD_PARTY_ROOTS = {
    "torch", "triton", "transformers", "numpy", "pandas", "matplotlib", "seaborn",
    "scipy", "bitsandbytes", "accelerate", "lm_eval", "datasets", "tabulate", "rich",
    "sklearn", "dotenv", "huggingface_hub", "safetensors", "auto_gptq", "optimum",
    "sentencepiece", "protobuf", "pynvml", "modal", "pytest", "IPython",
}


def _module_index(roots: List[Path]) -> Dict[str, Path]:
    """Map every importable dotted name in the repo to its file."""
    index: Dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT)
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            # Register both the full dotted path from the repo root and the bare module
            # name, so flat/implicit-relative layouts and package layouts both resolve.
            index[".".join(parts)] = path
            index.setdefault(parts[-1], path)
            # Also register paths relative to a `src/` root, which is how the package is
            # imported once installed.
            if parts[0] == "src" and len(parts) > 1:
                index[".".join(parts[1:])] = path
    return index


def _iter_imports(tree: ast.AST) -> List[Tuple[str, int, int]]:
    """Yield (dotted_name, level, lineno). `level` > 0 means an explicit relative import."""
    found: List[Tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, 0, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level, node.lineno))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Directory to scan (repeatable). Defaults to every source dir in the repo.",
    )
    args = ap.parse_args()

    if args.root:
        roots = [REPO_ROOT / r for r in args.root]
    else:
        candidates = ["src", "quantization", "kernels", "kernals", "compiler", "Compiler",
                      "report", "scripts", "tools", "tests"]
        roots = [REPO_ROOT / c for c in candidates if (REPO_ROOT / c).is_dir()]

    index = _module_index(roots)
    local_roots: Set[str] = {name.split(".")[0] for name in index}

    print("=" * 78)
    print("IMPORT GRAPH CHECK")
    print("=" * 78)
    print(f"scanning : {', '.join(str(r.relative_to(REPO_ROOT)) for r in roots)}")
    print(f"modules  : {len({str(p) for p in index.values()})} files, {len(index)} importable names\n")

    problems: List[str] = []
    checked = 0
    implicit_relative: List[str] = []

    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                problems.append(f"{rel}:{exc.lineno}: SYNTAX ERROR: {exc.msg}")
                continue

            for name, level, lineno in _iter_imports(tree):
                if level > 0:
                    continue  # explicit relative import; Python resolves these itself
                if not name:
                    continue
                head = name.split(".")[0]
                if head in THIRD_PARTY_ROOTS or head not in local_roots:
                    continue  # third party or stdlib

                checked += 1
                if name in index:
                    # A bare name that is not a top-level package means the file only
                    # imports because Python injected the script's directory into sys.path.
                    # `tests/context.py` is the one sanctioned exception -- it is the
                    # single, deliberate sys.path shim recommended by the python-guide.
                    if name == "context" and rel.parts[0] == "tests":
                        continue
                    if "." not in name and not (REPO_ROOT / name).is_dir():
                        implicit_relative.append(f"{rel}:{lineno}: from {name} import ...")
                    continue
                problems.append(f"{rel}:{lineno}: UNRESOLVED import {name!r}")

    print(f"intra-project imports checked: {checked}")

    if implicit_relative:
        print(f"\nIMPLICIT-RELATIVE IMPORTS ({len(implicit_relative)})")
        print("  These resolve only because Python puts the script's own directory on")
        print("  sys.path. They break as soon as the code is imported as a package.")
        for item in implicit_relative[:40]:
            print(f"  - {item}")
        if len(implicit_relative) > 40:
            print(f"  ... and {len(implicit_relative) - 40} more")

    print("\n" + "=" * 78)
    if problems:
        print(f"FAILED -- {len(problems)} unresolved import(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASSED -- every intra-project import resolves to a file that exists.")
    if implicit_relative:
        print(f"         ({len(implicit_relative)} are implicit-relative; see above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
