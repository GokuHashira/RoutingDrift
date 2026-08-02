"""
modal_app.py

RoutingDrift experiment stages on Modal.

Why Modal rather than a rented instance: billing is per-second of actual function
execution, and the workflow here is deliberately stop-and-inspect -- run a stage, read the
diagnostics, decide whether the next one is worth running. On a wall-clock-billed box that
thinking time either costs money or costs the discipline to remember to shut it down. It
also means a persistent volume holds the checkpoints, so the 117 GB across three models is
downloaded once rather than juggled against a fixed disk.

Every stage is separately invokable. Run them in order and stop after stage 1.

    modal run modal_app.py::smoke                 # ~$0.65  <-- START HERE, then send the digest
    modal run modal_app.py::task1                 # ~$1.90
    modal run modal_app.py::sweep                 # ~$6.90  the correlation
    modal run modal_app.py::replay                # ~$1.00  the causal result
    modal run modal_app.py::deepseek_drift        # ~$0.85
    modal run modal_app.py::qwen_drift            # ~$1.90  needs newer transformers
    modal run modal_app.py::diagnostics           # free-ish, prints the digest

Pull results down to the laptop:

    modal volume get routingdrift-results / ./results_modal

Costs assume A100-80GB at $2.50/hr and are +/-50%; lm-eval under INT4 is the wildcard.
"""

import modal

REPO = "/root/RoutingDrift"
RESULTS = "/results"
GPU = "A100-80GB"

# Two images. OLMoE and DeepSeek run on the transformers 4.46 pin that the committed
# results depend on; Qwen3.6-35B-A3B is an April 2026 architecture that 4.46 cannot load,
# so it gets its own. run_manifest.json records the version actually used, and the paper
# has to note the split.
_COMMON = [
    "accelerate", "bitsandbytes==0.44.1", "safetensors", "sentencepiece", "protobuf",
    "numpy<2", "pandas", "matplotlib", "seaborn", "tabulate", "scipy",
    "lm-eval==0.4.4", "datasets", "huggingface_hub",
]

# Experiment output goes to the volume, not the image. But results/olmoe_top2_zaratan
# (460 KB) MUST ship: it is the reference the smoke test diffs against, and without it
# the one check that validates this GPU against the committed A100 run cannot execute.
# The bulky regenerable artifacts stay out -- results/compiler alone is 33 MB.
_IGNORE = [
    ".git", "hpc_runs", "HPC_Outputs", "temp", "docs", "*.zip",
    "results/kernels", "results/kernels_a100", "results/compiler",
    "results/report_plots", "results/*_rerun",
]


def _image(transformers_pin: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.5.0", transformers_pin, *_COMMON)
        .add_local_dir(".", remote_path=REPO, ignore=_IGNORE)
    )


pinned_image = _image("transformers==4.46.0")
latest_image = _image("transformers>=4.57")

app = modal.App("routingdrift")

# Checkpoints persist here, so a failed stage does not re-pay a 72 GB download.
hf_cache = modal.Volume.from_name("routingdrift-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("routingdrift-results", create_if_missing=True)

VOLUMES = {"/cache": hf_cache, RESULTS: results_vol}
ENV = {
    "HF_HOME": "/cache/huggingface",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
}


def _run(*argv: str) -> None:
    """Execute a module in the shipped repo, streaming output into the Modal log."""
    import os
    import subprocess
    import sys

    env = {**os.environ, **ENV, "PYTHONPATH": f"{REPO}/src"}
    print(f"\n$ {' '.join(argv)}\n", flush=True)
    result = subprocess.run([sys.executable, "-m", *argv], cwd=REPO, env=env)
    # Commit both: results so a failed later step does not lose completed work, and the
    # HF cache so a 14/31/72 GB download is paid for once rather than once per run.
    results_vol.commit()
    hf_cache.commit()
    if result.returncode != 0:
        raise RuntimeError(f"stage failed (exit {result.returncode}): {' '.join(argv)}")


def _gpu_report() -> None:
    import torch

    assert torch.cuda.is_available(), "no CUDA device in this container"
    p = torch.cuda.get_device_properties(0)
    import transformers

    print(f"gpu          : {p.name} {p.total_memory / 1024**3:.1f} GB")
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")


OLMOE = "allenai/OLMoE-1B-7B-0924"
# Set once the smoke test reports the resolved SHA, then everything after is pinned.
REVISION = ""


def _rev():
    return ["--revision", REVISION] if REVISION else []


# ---------------------------------------------------------------------------
# Stage 1 -- smoke test. RUN THIS FIRST AND STOP.
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def smoke():
    """
    Reproduce the committed May-2026 Zaratan A100 run and diff the raw routes.

    Deliberately matches the original: the 5 built-in prompts, top-2, max_length 256, no
    lm-eval. Changing any of those invalidates the comparison.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.run_experiment",
        "--model_name", OLMOE, *_rev(),
        "--precisions", "fp16", "int8", "int4",
        "--target_module", "mlp.gate",
        "--output_dir", f"{RESULTS}/smoke_top2",
        "--top_k", "2", "--max_length", "256", "--skip_heatmaps",
    )
    _run(
        "routingdrift.quantization.verify_reproducibility",
        "--results_dir", f"{RESULTS}/smoke_top2",
        "--compare_to", f"{REPO}/results/olmoe_top2_zaratan",
    )
    print("\nSTOP HERE. Run `modal run modal_app.py::diagnostics` and read four things:")
    print("  1. quant_audit layer spans  -- does llm_int8_skip_modules work on 4-bit?")
    print("  2. cross-run diff %         -- 0.00 ideal; <~2% is fp16 reduction order")
    print("  3. distinct drift values    -- do the INT8 thresholds actually differ?")
    print("  4. peak_vram_gb / seconds   -- is the budget realistic?")


# ---------------------------------------------------------------------------
# Stage 2 -- headline table
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 60 * 60)
def task1(n_prompts: int = 100, lm_eval_limit: int = 500):
    """OLMoE top-8 drift on an MMLU prompt set, with accuracy."""
    _gpu_report()
    _run(
        "routingdrift.quantization.build_mmlu_prompts",
        "--n", str(n_prompts), "--seed", "0",
        "--out", f"{RESULTS}/mmlu_prompts.txt",
    )
    _run(
        "routingdrift.quantization.run_experiment",
        "--model_name", OLMOE, *_rev(),
        "--precisions", "fp16", "int8", "int4",
        "--target_module", "mlp.gate",
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--output_dir", f"{RESULTS}/olmoe_top8",
        "--top_k", "8", "--max_length", "128", "--seed", "0",
        "--run_lm_eval",
        # GSM8K stays here for the descriptive table but is dropped from the sweep.
        "--lm_eval_tasks", "mmlu", "gsm8k", "hellaswag",
        "--lm_eval_num_fewshot", "5", "--lm_eval_batch_size", "auto",
        "--lm_eval_limit", str(lm_eval_limit), "--lm_eval_device", "cuda",
    )
    _run("routingdrift.quantization.verify_reproducibility",
         "--results_dir", f"{RESULTS}/olmoe_top8")


# ---------------------------------------------------------------------------
# Stage 3 -- the correlation. This is the paper.
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, timeout=8 * 60 * 60)
def sweep(lm_eval_limit: int = 200):
    """
    15 quantization configs -> drift and gate-KL against accuracy drop.

    GSM8K is excluded deliberately. It is the only generative task here, so it dominates
    eval time and is 2-3x worse again under INT4 -- and OLMoE scores ~8% on it, close
    enough to the floor that an "accuracy drop" carries no signal to correlate against.
    Measuring degradation on a task the model already fails adds noise, not points.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.sweep",
        "--model_name", OLMOE, *_rev(),
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--output_dir", f"{RESULTS}/olmoe_sweep",
        "--top_k", "8", "--target_module", "mlp.gate",
        "--max_length", "128", "--seed", "0",
        "--run_lm_eval",
        "--lm_eval_tasks", "mmlu", "hellaswag",
        "--lm_eval_num_fewshot", "5", "--lm_eval_batch_size", "auto",
        "--lm_eval_limit", str(lm_eval_limit), "--lm_eval_device", "cuda",
    )
    print("\nRead sweep_correlations.csv: jaccard_drift AND gate_kl are both correlated")
    print("against accuracy_drop. If gate_kl explains as much, routing fidelity is a proxy")
    print("for gate noise rather than a metric, and the paper must say so.")


# ---------------------------------------------------------------------------
# Stage 4 -- causal attribution
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, timeout=2 * 60 * 60)
def replay(quant_precision: str = "int4"):
    """FP16 weights driven by the quantized model's expert selections."""
    _gpu_report()
    _run(
        "routingdrift.quantization.route_replay",
        "--model_name", OLMOE, *_rev(),
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--baseline_routes", f"{RESULTS}/olmoe_top8/routes_fp16.json",
        "--replay_routes", f"{RESULTS}/olmoe_top8/routes_{quant_precision}.json",
        "--quant_precision", quant_precision,
        "--top_k", "8", "--target_module", "mlp.gate",
        "--max_length", "128",
        "--output_dir", f"{RESULTS}/olmoe_replay",
    )
    print("\nCheck intervention_artifact in replay_result.json. OLMoE has")
    print("norm_topk_prob=False, so masking shifts mixing weights as well as selection.")
    print("If the artifact is large next to the degradation gap, the attribution number")
    print("is not trustworthy and block-level replay is needed.")


# ---------------------------------------------------------------------------
# Stage 5 -- second architecture: shared-expert axis
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 60 * 60)
def deepseek_drift():
    """
    DeepSeek-V2-Lite: 64 routed + 2 shared, top-6.

    The cleanest available contrast with OLMoE -- same routed-expert count, shared experts
    added. Its MoEGate returns (topk_idx, topk_weight, aux_loss) rather than logits; the
    adapter in routing_logger rebuilds logits from the gate weight. Route replay is not
    defined for such gates and raises rather than silently no-opping.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.run_experiment",
        "--model_name", "deepseek-ai/DeepSeek-V2-Lite",
        "--precisions", "fp16", "int8", "int4",
        "--target_module", "mlp.gate",
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--output_dir", f"{RESULTS}/deepseek_v2_lite",
        "--top_k", "6", "--max_length", "128", "--seed", "0", "--skip_heatmaps",
    )
    _run("routingdrift.quantization.verify_reproducibility",
         "--results_dir", f"{RESULTS}/deepseek_v2_lite")


# ---------------------------------------------------------------------------
# Stage 6 -- third architecture: granularity axis, 2026 model
# ---------------------------------------------------------------------------
@app.function(image=latest_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 60 * 60)
def qwen_drift():
    """
    Qwen3.6-35B-A3B: 256 routed + 1 shared, top-8, April 2026.

    Runs on the newer-transformers image; 4.46 cannot load this architecture. The manifest
    records which version was used, and the cross-model comparison therefore spans two
    library versions -- state that in the paper rather than hiding it.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.run_experiment",
        "--model_name", "Qwen/Qwen3.6-35B-A3B",
        "--precisions", "fp16", "int8", "int4",
        "--target_module", "mlp.gate",
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--output_dir", f"{RESULTS}/qwen36_moe",
        "--top_k", "8", "--max_length", "128", "--seed", "0", "--skip_heatmaps",
    )
    _run("routingdrift.quantization.verify_reproducibility",
         "--results_dir", f"{RESULTS}/qwen36_moe")


# ---------------------------------------------------------------------------
# Diagnostics -- CPU only, so this costs almost nothing
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, volumes=VOLUMES, timeout=15 * 60)
def diagnostics():
    """Print the digest that answers the open questions. No GPU, so effectively free."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": f"{REPO}/src"}
    subprocess.run(
        [sys.executable, f"{REPO}/tools/collect_diagnostics.py",
         "--results_dir", RESULTS, "--no_bundle"],
        cwd=REPO, env=env, check=False,
    )


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` with no function named: print the intended order."""
    print(__doc__)
