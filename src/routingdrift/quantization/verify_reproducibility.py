"""
verify_reproducibility.py

Recompute the committed drift CSVs from the committed raw route dumps and report any
disagreement. This checks the `routes_*.json -> routing_drift_*.csv` half of the pipeline,
which needs no GPU -- so it runs on a laptop, in CI, or on a reviewer's machine.

It does NOT check `weights -> routes_*.json`; that needs the model. For that half, use
`run_experiment.py`'s built-in double-pass determinism check (on by default).

Usage:
    python -m routingdrift.quantization.verify_reproducibility
    python -m routingdrift.quantization.verify_reproducibility --results_dir results/olmoe_top8
    python -m routingdrift.quantization.verify_reproducibility --baseline fp16 --tolerance 1e-6

Exit code is 0 if everything reproduces within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from typing import List


def _install_torch_shim() -> bool:
    """
    `drift.py` only uses .ndim/.reshape/.shape/.tolist on its inputs, and its annotations
    are lazy (`from __future__ import annotations`), so a numpy-backed stand-in exercises
    the exact same metric code when torch is not installed. Returns True if a shim was used.
    """
    try:
        import torch  # noqa: F401

        return False
    except ImportError:
        pass

    import numpy as np

    shim = types.ModuleType("torch")
    shim.Tensor = np.ndarray
    shim.tensor = np.array
    sys.modules["torch"] = shim
    return True


METRICS = ("routing_similarity_rs", "jaccard_drift", "overlap_at_k", "selection_shift")


def load_routes(results_dir: Path, variant: str):
    import torch

    from routingdrift.quantization.io_utils import load_routes_raw

    raw = load_routes_raw(results_dir / f"routes_{variant}.json")
    return {module: [torch.tensor(call) for call in calls] for module, calls in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results_dir",
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Directory holding routes_*.json and routing_drift_*.csv, e.g. "
            "results/olmoe_top8. REQUIRED: this defaulted to results/olmoe_top2_zaratan, "
            "the retired top-2 run, so an argumentless invocation reported PASSED for "
            "numbers the paper no longer makes. There is no correct default across three "
            "models. `make verify` runs all of them."
        ),
    )
    ap.add_argument("--baseline", type=str, default="fp16", help="Baseline variant name.")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Max allowed |committed - recomputed|. Committed CSVs are rounded to 6 decimals.",
    )
    ap.add_argument(
        "--compare_to",
        "--compare-to",
        type=str,
        default=None,
        help=(
            "Second results directory to diff route-for-route against --results_dir. Use this "
            "to check whether a re-run on new hardware reproduces an earlier run "
            "(the weights -> routes half that the CSV check cannot cover)."
        ),
    )
    args = ap.parse_args()

    used_shim = _install_torch_shim()
    from routingdrift.quantization.drift import compute_layerwise_metrics, summarize_research_metrics

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: no such results directory: {results_dir}")
        return 1

    summary_path = results_dir / "routing_drift_summary.csv"
    if not summary_path.is_file():
        print(f"ERROR: no summary CSV at {summary_path}")
        return 1

    committed = {row["variant"]: row for row in csv.DictReader(summary_path.open(encoding="utf-8"))}
    variants = list(committed)

    print("=" * 78)
    print(f"REPRODUCIBILITY CHECK  --  {results_dir}")
    if used_shim:
        print("(torch not installed; using numpy-backed stand-in for drift.py tensor ops)")
    print("=" * 78)

    try:
        routes = {v: load_routes(results_dir, v) for v in variants}
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    # --- provenance ----------------------------------------------------------
    print("\nRAW ROUTE PROVENANCE")
    for variant in variants:
        modules = sorted(routes[variant])
        calls = routes[variant][modules[0]]
        rows_per_module = sum(c.reshape(-1, c.shape[-1]).shape[0] for c in calls)
        print(
            f"  {variant:22s} modules={len(modules):3d}  calls/module={len(calls):3d}  "
            f"top_k={calls[0].shape[-1]}  rows/module={rows_per_module}  "
            f"total_rows={rows_per_module * len(modules)}"
        )

    failures: List[str] = []

    # --- summary CSV ---------------------------------------------------------
    print("\nSUMMARY TABLE")
    print(f"  {'variant':22s} {'metric':24s} {'committed':>11s} {'recomputed':>11s} {'diff':>10s}")
    print("  " + "-" * 74)
    worst = 0.0
    for variant in variants:
        recomputed = summarize_research_metrics(routes[args.baseline], routes[variant])
        for metric in METRICS:
            if metric not in committed[variant]:
                continue
            want = float(committed[variant][metric])
            got = recomputed[metric]
            diff = abs(want - got)
            worst = max(worst, diff)
            bad = diff > args.tolerance
            if bad:
                failures.append(f"summary {variant}/{metric}: {want} != {got}")
            print(
                f"  {variant:22s} {metric:24s} {want:11.6f} {got:11.6f} {diff:10.2e}"
                f"{'  <-- MISMATCH' if bad else ''}"
            )
    print(f"\n  worst deviation: {worst:.3e}   tolerance: {args.tolerance:.1e}")

    # --- per-layer CSV -------------------------------------------------------
    layers_path = results_dir / "routing_drift_layers.csv"
    if layers_path.is_file():
        print("\nPER-LAYER TABLE")
        rows = list(csv.DictReader(layers_path.open(encoding="utf-8")))
        by_key = {(r["variant"], r["module"]): r for r in rows}
        worst_layer, checked = 0.0, 0
        for variant in variants:
            for module, metrics in compute_layerwise_metrics(routes[args.baseline], routes[variant]).items():
                row = by_key.get((variant, module))
                if row is None:
                    continue  # baseline has no committed per-layer rows; not an error
                for metric in METRICS:
                    diff = abs(float(row[metric]) - metrics[metric])
                    worst_layer = max(worst_layer, diff)
                    checked += 1
                    if diff > args.tolerance:
                        failures.append(f"layer {variant}/{module}/{metric}")
        print(f"  committed rows: {len(rows)}   values checked: {checked}   worst deviation: {worst_layer:.3e}")

    # --- self-consistency guard ---------------------------------------------
    print("\nSELF-CONSISTENCY GUARD")
    self_rs = summarize_research_metrics(routes[args.baseline], routes[args.baseline])["routing_similarity_rs"]
    print(f"  {args.baseline}-vs-{args.baseline} RS = {self_rs:.10f}  (must be exactly 1.0)")
    if abs(self_rs - 1.0) > 1e-12:
        failures.append("baseline self-comparison is not 1.0")

    # --- how much actually moved --------------------------------------------
    print("\nROWS CHANGED VS BASELINE")
    for variant in variants:
        if variant == args.baseline:
            continue
        differing = total = 0
        for module in sorted(routes[args.baseline]):
            for base_call, var_call in zip(routes[args.baseline][module], routes[variant][module]):
                flat_b = base_call.reshape(-1, base_call.shape[-1])
                flat_v = var_call.reshape(-1, var_call.shape[-1])
                for i in range(min(flat_b.shape[0], flat_v.shape[0])):
                    total += 1
                    if set(flat_b[i].tolist()) != set(flat_v[i].tolist()):
                        differing += 1
        pct = (100.0 * differing / total) if total else 0.0
        print(f"  {variant:22s} {differing}/{total} rows differ ({pct:.2f}%)")

    # --- cross-run comparison (weights -> routes) ----------------------------
    if args.compare_to:
        other_dir = Path(args.compare_to)
        print(f"\nCROSS-RUN COMPARISON vs {other_dir}")
        if not other_dir.is_dir():
            print(f"  ERROR: no such directory: {other_dir}")
            failures.append(f"compare_to directory missing: {other_dir}")
        else:
            for variant in variants:
                try:
                    other_routes = load_routes(other_dir, variant)
                except FileNotFoundError:
                    print(f"  {variant:22s} not present in the other run -- skipped")
                    continue

                mine = routes[variant]
                shared = sorted(set(mine) & set(other_routes))
                only_here = sorted(set(mine) - set(other_routes))
                only_there = sorted(set(other_routes) - set(mine))

                differing = total = 0
                for module in shared:
                    for call_a, call_b in zip(mine[module], other_routes[module]):
                        flat_a = call_a.reshape(-1, call_a.shape[-1])
                        flat_b = call_b.reshape(-1, call_b.shape[-1])
                        if flat_a.shape != flat_b.shape:
                            differing += max(flat_a.shape[0], flat_b.shape[0])
                            total += max(flat_a.shape[0], flat_b.shape[0])
                            continue
                        for i in range(flat_a.shape[0]):
                            total += 1
                            if set(flat_a[i].tolist()) != set(flat_b[i].tolist()):
                                differing += 1

                pct = (100.0 * differing / total) if total else 0.0
                note = ""
                if only_here or only_there:
                    note = f"  [module sets differ: +{len(only_here)} / -{len(only_there)}]"
                print(f"  {variant:22s} {differing}/{total} rows differ ({pct:.2f}%){note}")
            print(
                "\n  Interpretation: 0.00% means the two runs are bit-identical. A small nonzero\n"
                "  percentage on different GPUs is expected -- fp16 reduction order shifts gate\n"
                "  logits and flips near-tie top-k picks. Large values mean a real environment\n"
                "  difference (checkpoint revision, bitsandbytes version, tokenizer)."
            )

    # --- manifest ------------------------------------------------------------
    manifest_path = results_dir / "run_manifest.json"
    print("\nPROVENANCE MANIFEST")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gpu = manifest.get("gpu", {})
        devices = gpu.get("devices") or []
        packages = manifest.get("packages", {})
        if devices:
            gpu_desc = f"{devices[0].get('name')} x{gpu.get('device_count')}"
        else:
            gpu_desc = "none (CPU run)"
        print(f"  run at    : {manifest.get('timestamp_utc')}")
        print(f"  git       : {manifest.get('git', {}).get('commit')} (dirty={manifest.get('git', {}).get('dirty')})")
        print(f"  gpu       : {gpu_desc}")
        print(f"  torch     : {packages.get('torch')}   transformers: {packages.get('transformers')}")
        print(f"  bnb       : {packages.get('bitsandbytes')}   lm_eval: {packages.get('lm_eval')}")
    else:
        print(f"  ABSENT ({manifest_path.name}) -- this result predates provenance capture.")
        print("  Not a failure, but its environment cannot be reconstructed.")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} mismatch(es):")
        for failure in failures[:20]:
            print(f"  - {failure}")
        return 1
    print("PASSED -- every committed metric reproduces from the raw route dumps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
