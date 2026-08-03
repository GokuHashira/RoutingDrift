"""
collect_diagnostics.py

Bundle everything needed to diagnose a Thunder run, and print a digest of the specific
questions that cannot be answered from a laptop.

Run this after any GPU stage, then share the printed digest (and the tarball if the digest
is not enough).

    python tools/collect_diagnostics.py
    python tools/collect_diagnostics.py --results_dir results --out diagnostics.tar.gz

Raw route dumps are deliberately excluded from the tarball -- they are the bulk of the
bytes and their shapes are already summarised in the digest.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files small enough to always include.
# Route dumps are gzipped now (~17x smaller), so the bundle can carry the complete record
# rather than a summary of it. Model weights are the only thing deliberately left out.
INCLUDE_GLOBS = (
    "**/run_manifest.json", "**/*.csv", "**/logs/*.log", "**/replay_result.json",
    "**/routes_*.json.gz", "**/summary.md", "**/prompts_used.txt", "**/lm_eval/*.json",
)
EXCLUDE_SUBSTRINGS = ()


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _rows(path: Path) -> List[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def digest(results_dir: Path) -> None:
    print("=" * 78)
    print("DIAGNOSTIC DIGEST")
    print("=" * 78)

    runs = sorted(results_dir.glob("*/run_manifest.json"))
    if not runs:
        print(f"no run_manifest.json under {results_dir} -- has any stage completed?")
        return

    for manifest_path in runs:
        run = manifest_path.parent
        m = _read_json(manifest_path) or {}
        gpu = (m.get("gpu") or {}).get("devices") or [{}]
        pkgs = m.get("packages") or {}
        guards = m.get("guards") or {}

        print(f"\n--- {run.name} " + "-" * max(0, 62 - len(run.name)))
        print(f"  status     : {m.get('status')}   at {m.get('timestamp_utc')}")
        print(f"  gpu        : {gpu[0].get('name', 'none')} ({gpu[0].get('total_memory_gb', '?')} GB)")
        print(f"  versions   : torch {pkgs.get('torch')} | transformers {pkgs.get('transformers')} "
              f"| bnb {pkgs.get('bitsandbytes')} | lm_eval {pkgs.get('lm_eval')}")
        print(f"  revision   : {m.get('resolved_revision') or (m.get('variants') or {})}")
        g = m.get("git") or {}
        print(f"  git        : {(g.get('commit') or 'MISSING')[:12]} "
              f"dirty={g.get('dirty')} {g.get('source', '')}")

        # Drift and the KL control must not be collinear. If they are, the sweep cannot
        # tell "routing fidelity predicts quality" from "the gate got noisier", and the
        # correlation table answers nothing.
        sweep_rows = _rows(run / "sweep_drift.csv")
        pairs = [(float(r["jaccard_drift"]), float(r["gate_kl"]))
                 for r in sweep_rows
                 if r.get("config") != "fp16" and r.get("jaccard_drift") and r.get("gate_kl")]
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            num = sum((a - mx) * (b - my) for a, b in pairs)
            den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
            if den:
                r_dk = num / den
                print(f"  drift~kl   : pearson={r_dk:+.3f} over {len(pairs)} configs")
                if abs(r_dk) > 0.95:
                    print("               ^ COLLINEAR. The sweep cannot separate routing")
                    print("                 fidelity from gate noise; replay carries the")
                    print("                 causal claim instead.")
        if guards:
            print(f"  guards     : self_consistency_rs={guards.get('baseline_self_consistency_rs')} "
                  f"all_deterministic={guards.get('all_variants_deterministic')}")
            for variant, det in (guards.get("determinism_by_variant") or {}).items():
                flag = "" if det.get("identical") else "   <-- NOT DETERMINISTIC"
                print(f"               {variant:<10} identical={det.get('identical')} "
                      f"mismatched_rows={det.get('mismatched_rows')}{flag}")

        # Q1: did llm_int8_skip_modules actually work for 4-bit loads?
        sweep_csv = run / "sweep_drift.csv"
        if sweep_csv.is_file():
            rows = _rows(sweep_csv)
            print(f"\n  SWEEP ({len(rows)} configs)")
            print(f"    {'config':<12} {'drift':>8} {'gate_kl':>11} {'vram':>7} {'sec':>6}  audit")
            for r in rows:
                print(f"    {r.get('config',''):<12} {r.get('jaccard_drift',''):>8} "
                      f"{r.get('gate_kl',''):>11} {r.get('peak_vram_gb',''):>7} "
                      f"{r.get('seconds',''):>6}  {r.get('quant_audit','')[:60]}")
            drifts = {r.get("jaccard_drift") for r in rows}
            print(f"    distinct drift values: {len(drifts)} of {len(rows)}")
            if len(drifts) < len(rows) - 1:
                print("    ^ WARNING: configs are collapsing onto the same drift value.")
                print("      Check the audit column: if the nf4_L* rows all show the same")
                print("      layer span, llm_int8_skip_modules is being ignored on 4-bit.")

        # Q2: does drift beat the gate-noise control?
        corr = run / "sweep_correlations.csv"
        if corr.is_file():
            print("\n  CORRELATIONS (drift vs the gate_kl control)")
            for r in _rows(corr):
                if r.get("group") == "all":
                    print(f"    {r.get('predictor',''):<16} n={r.get('n_points','')} "
                          f"pearson={r.get('pearson','')} spearman={r.get('spearman','')}")

        # Accuracy per variant, and whether drift tracks it. run_experiment writes these
        # under different names than the sweep does, and the digest previously reported
        # only the sweep's -- so a completed task1 showed no accuracy at all.
        eval_csv = run / "lm_eval_scores.csv"
        if eval_csv.is_file():
            rows = _rows(eval_csv)
            tasks = sorted({r["task"] for r in rows})
            variants = sorted({r["variant"] for r in rows}, key=lambda v: (v != "fp16", v))
            print(f"\n  ACCURACY ({len(rows)} measurements)")
            print("    " + "variant".ljust(10) + "".join(task.ljust(14) for task in tasks))
            by = {(r["variant"], r["task"]): r for r in rows}
            for variant in variants:
                cells = ""
                for task in tasks:
                    r = by.get((variant, task))
                    cells += (f"{float(r['accuracy']):.4f}".ljust(14)) if r else "-".ljust(14)
                print(f"    {variant:<10}{cells}")
            # Degradation relative to the fp16 row, which is what drift should predict.
            for task in tasks:
                base = by.get(("fp16", task))
                if not base:
                    continue
                for v in variants:
                    if v == "fp16" or (v, task) not in by:
                        continue
                    row = by[(v, task)]
                    drop = float(base["accuracy"]) - float(row["accuracy"])
                    # Combined SE of a difference of two independent estimates.
                    try:
                        se = (float(base.get("stderr") or 0) ** 2
                              + float(row.get("stderr") or 0) ** 2) ** 0.5
                    except ValueError:
                        se = 0.0
                    if se:
                        sigma = abs(drop) / se
                        verdict = "NOISE" if sigma < 2 else "significant"
                        print(f"    {task:<12} {v:<6} drop={drop:+.4f} +/- {se:.4f} "
                              f"({sigma:.1f} sigma, {verdict})")
                    else:
                        print(f"    {task:<12} {v:<6} drop={drop:+.4f} "
                              f"(no stderr recorded; cannot tell this from noise)")

        corr_re = run / "drift_accuracy_correlations.csv"
        if corr_re.is_file():
            rows = _rows(corr_re)
            print("\n  DRIFT vs ACCURACY DROP")
            for r in rows:
                n = r.get("n_points", "")
                print(f"    {r.get('group',''):<12} n={n:<4} "
                      f"pearson={r.get('pearson','') or 'n/a':<10} "
                      f"spearman={r.get('spearman','') or 'n/a'}")
            # Warn on DISTINCT drift values, not row count. n_points counts
            # variant x task pairs, so three precisions across two tasks reports n=4 while
            # resting on two distinct drift values -- and any two points give pearson=1.0.
            distinct = len({
                r["jaccard_drift"] for r in _rows(run / "drift_vs_accuracy_drop.csv")
                if r.get("jaccard_drift")
            }) if (run / "drift_vs_accuracy_drop.csv").is_file() else 0
            if distinct and distinct < 4:
                print(f"    ^ only {distinct} DISTINCT drift value(s) behind these numbers.")
                print("      Two points always give pearson=1.0; this is a direction, not a")
                print("      correlation. The sweep is what makes it reportable.")

        # Q3: how big is the replay masking artifact on the real model?
        replay = run / "replay_result.json"
        if replay.is_file():
            r = _read_json(replay) or {}
            print("\n  REPLAY")
            print(f"    norm_topk_prob      : {r.get('norm_topk_prob')}")
            print(f"    nll fp16            : {r.get('nll_fp16')}")
            print(f"    nll control         : {r.get('nll_control_fp16_routes')}")
            print(f"    intervention artifact: {r.get('intervention_artifact')}")
            print(f"    nll replay          : {r.get('nll_replay')}")
            print(f"    nll quantized       : {r.get('nll_quantized')}")
            print(f"    attribution         : {r.get('routing_attribution')}")
            art = r.get("intervention_artifact")
            gap = None
            try:
                gap = float(r["nll_quantized"]) - float(r["nll_fp16"])
            except Exception:  # noqa: BLE001
                pass
            if art is not None and gap:
                ratio = abs(float(art) / gap)
                print(f"    artifact / degradation gap = {ratio:.1%}")
                if ratio > 0.25:
                    print("    ^ the masking artifact is large relative to the effect being")
                    print("      measured. Block-level replay is needed for a trustworthy number.")

        # Q4: drift table from a plain run
        summary = run / "routing_drift_summary.csv"
        if summary.is_file():
            print("\n  DRIFT SUMMARY")
            for r in _rows(summary):
                print(f"    {r.get('variant',''):<10} rs={r.get('routing_similarity_rs','')} "
                      f"jaccard={r.get('jaccard_drift','')}")

    print("\n" + "=" * 78)


def bundle(results_dir: Path, out_path: Path) -> None:
    added = 0
    with tarfile.open(out_path, "w:gz") as tar:
        for pattern in INCLUDE_GLOBS:
            for path in results_dir.glob(pattern):
                if EXCLUDE_SUBSTRINGS and any(bad in path.name for bad in EXCLUDE_SUBSTRINGS):
                    continue
                if not path.is_file():
                    continue
                tar.add(path, arcname=str(path.relative_to(REPO_ROOT)))
                added += 1
    size_mb = out_path.stat().st_size / 1024**2
    print(f"bundled {added} files -> {out_path} ({size_mb:.1f} MB)")
    print("Includes gzipped route dumps: every reported metric can be recomputed from this.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="diagnostics.tar.gz")
    ap.add_argument("--no_bundle", action="store_true", help="Print the digest only.")
    args = ap.parse_args()

    results_dir = (REPO_ROOT / args.results_dir) if not Path(args.results_dir).is_absolute() else Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: no such directory: {results_dir}")
        return 1

    digest(results_dir)
    if not args.no_bundle:
        bundle(results_dir, REPO_ROOT / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
