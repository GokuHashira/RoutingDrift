"""
prune_results.py

Make a pulled results directory small enough to commit, without losing anything that a
number depends on.

Two things bloat it:

  * lm-eval writes every individual evaluation sample into its JSON. For MMLU at
    limit=500 that is ~500 MB per run, against a few KB of actual metrics. The per-sample
    records are useful for auditing a specific answer and useless for reproducing a score,
    so they are dropped and the `results` / `configs` / version blocks kept.
  * Route dumps written before the gzip change sit alongside their .json.gz replacements.
    In this repo those leftovers are from a run whose MMLU download silently fell back to
    5 generic prompts, so they are not merely redundant, they are wrong. Nothing reads
    them, since both readers prefer .json.gz, but they should not be committed next to
    real data.

Dry run by default; pass --apply to modify.

    python tools/prune_results.py --results_dir modal_outputs
    python tools/prune_results.py --results_dir modal_outputs --apply
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

KEEP_KEYS = ("results", "groups", "configs", "versions", "n-shot", "config",
             "git_hash", "date", "higher_is_better")


def _mb(n: int) -> str:
    return f"{n / 1024**2:.1f} MB"


def prune_lm_eval(path: Path, apply: bool) -> tuple[int, int]:
    before = path.stat().st_size
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"    skip {path.name}: unreadable ({exc})")
        return before, before
    if not isinstance(data, dict):
        return before, before

    pruned = {k: v for k, v in data.items() if k in KEEP_KEYS}
    dropped = sorted(set(data) - set(pruned))
    text = json.dumps(pruned, indent=2, default=str)
    after = len(text.encode())
    if apply:
        path.write_text(text, encoding="utf-8")
    print(f"    {path.name}: {_mb(before)} -> {_mb(after)}"
          + (f"  (dropped {', '.join(dropped)})" if dropped else ""))
    return before, after


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="modal_outputs")
    ap.add_argument("--apply", action="store_true", help="Actually modify files.")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    print(f"{'APPLYING' if args.apply else 'DRY RUN'} on {root}\n")

    total_before = total_after = 0

    print("  lm-eval JSON (dropping per-sample records):")
    found = False
    for path in sorted(root.glob("**/lm_eval/*.json")):
        found = True
        b, a = prune_lm_eval(path, args.apply)
        total_before += b
        total_after += a
    if not found:
        print("    none")

    print("\n  superseded uncompressed route dumps:")
    found = False
    for gz in sorted(root.glob("**/routes_*.json.gz")):
        plain = gz.with_suffix("")  # routes_x.json.gz -> routes_x.json
        if not plain.is_file():
            continue
        found = True
        try:
            raw = json.loads(plain.read_text(encoding="utf-8"))
            n_plain = len(raw[sorted(raw)[0]])
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                n_gz = len(json.load(f)[sorted(raw)[0]])
        except Exception:  # noqa: BLE001
            n_plain = n_gz = -1
        size = plain.stat().st_size
        total_before += size
        note = f"{n_plain} prompts vs {n_gz} in the .gz" if n_plain >= 0 else "unreadable"
        print(f"    {plain.relative_to(root)}: {_mb(size)}  ({note})")
        if args.apply:
            plain.unlink()
    if not found:
        print("    none")

    print(f"\n  total: {_mb(total_before)} -> {_mb(total_after)}")
    if not args.apply:
        print("\n  dry run. Re-run with --apply to make the changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
