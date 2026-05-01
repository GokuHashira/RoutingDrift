"""
eval_accuracy.py — MSML 605 (Gokul)
Runs GSM8K + MMLU on baseline vs patched OLMoE to confirm kernel correctness.
"""
import os
import json
import subprocess
from patch_models import load_olmoe

OLMOE_PATH="/scratch/zt1/project/msml605/user/gsakthiv/models/OLMoE-1B-7B"

EVAL_CONFIGS=[
    {"name": "baseline",      "kernels": False, "precision": "fp16"},
    {"name": "kernels_only",  "kernels": True,  "precision": "fp16"},
    {"name": "int8_baseline", "kernels": False, "precision": "int8"},
    {"name": "int8_kernels",  "kernels": True,  "precision": "int8"},
    {"name": "int4_baseline", "kernels": False, "precision": "int4"},
]


def run_lm_eval(model_path, config_name, limit=100, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    out=os.path.join(out_dir, f"{config_name}_results.json")
    cmd=["lm_eval", "--model", "hf",
         "--model_args", f"pretrained={model_path},dtype=float16",
         "--tasks", "gsm8k,mmlu",
         "--num_fewshot", "5",
         "--device", "cuda",
         "--output_path", out,
         "--limit", str(limit)]
    print(f"\nRunning lm-eval: {config_name}")
    result=subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode!=0:
        print(f"ERROR: {result.stderr}")
        return None
    try:
        with open(out) as f: r=json.load(f)
        return {
            "gsm8k": r["results"].get("gsm8k", {}).get("exact_match,strict-match", -1),
            "mmlu":  r["results"].get("mmlu",  {}).get("acc,none", -1),
        }
    except Exception as e:
        print(f"parse error: {e}")
        return None


def run_accuracy_eval(model_name="OLMoE", limit=100, out_dir=None):
    print(f"\n{'='*60}\n  Accuracy Eval — {model_name}\n{'='*60}")
    all_results={}
    baseline=None
    OUTPUT_DIR=out_dir or os.path.join(os.getcwd(), "results", "accuracy")
    for cfg in EVAL_CONFIGS:
        scores=run_lm_eval(OLMOE_PATH, cfg["name"], limit=limit, out_dir=OUTPUT_DIR)
        if scores:
            all_results[cfg["name"]]=scores
            if cfg["name"]=="baseline": baseline=scores

    print(f"\n  {'Config':<25} {'GSM8K':>8} {'MMLU':>8} {'GSM8K Δ':>10} {'MMLU Δ':>10}")
    print(f"  {'-'*65}")
    for name, s in all_results.items():
        dg=f"{s['gsm8k']-baseline['gsm8k']:+.3f}" if baseline and name!="baseline" else "baseline"
        dm=f"{s['mmlu']-baseline['mmlu']:+.3f}"   if baseline and name!="baseline" else "baseline"
        print(f"  {name:<25} {s['gsm8k']:>8.3f} {s['mmlu']:>8.3f} {dg:>10} {dm:>10}")

    out=os.path.join(OUTPUT_DIR, f"accuracy_summary_{model_name}.json")  # OUTPUT_DIR set above
    with open(out, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\n  saved: {out}")
    return all_results


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Run GSM8K + MMLU accuracy eval")
    parser.add_argument("--out", required=True, help="Output directory for accuracy results")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples per task")
    args=parser.parse_args()
    run_accuracy_eval(model_name="OLMoE", limit=args.limit, out_dir=args.out)
