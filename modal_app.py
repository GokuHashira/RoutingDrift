"""
modal_app.py

RoutingDrift experiment stages on Modal.

Why Modal rather than a rented instance: billing is per-second of actual function
execution, and the workflow here is deliberately stop-and-inspect -- run a stage, read the
diagnostics, decide whether the next one is worth running. On a wall-clock-billed box that
thinking time either costs money or costs the discipline to remember to shut it down. It
also means a persistent volume holds the checkpoints, so the 117 GB across three models is
downloaded once rather than juggled against a fixed disk.

Use --detach for anything longer than a couple of minutes:

    modal run --detach modal_app.py::task1

Without it the app dies when your local client disconnects -- closing a laptop lid is
enough, and it killed a paid task1 run mid-eval. Detached runs keep going; follow them in
the dashboard and collect results afterwards with the diagnostics stage.

Every stage is separately invokable. Run them in order and stop after stage 1.

    modal run modal_app.py::smoke                 # ~$0.65  <-- START HERE, then send the digest
    modal run modal_app.py::probe                 # ~$0.60  settles the skip_modules question
    modal run modal_app.py::task1                 # ~$1.90
    modal run modal_app.py::sweep                 # ~$6.90  the correlation
    modal run modal_app.py::replay                # ~$1.00  the causal result
    modal run modal_app.py::deepseek_drift        # ~$0.85
    modal run modal_app.py::qwen_drift            # ~$1.90  needs transformers >=4.51,<5
    modal run modal_app.py::kernel_profile        # ~$0.50  honest Amdahl fractions
    modal run modal_app.py::kernel_benchmark      # ~$0.30  E2E on the same machine
    modal run modal_app.py::compile_benchmark     # ~$1     does fusing the dispatch help?
    modal run modal_app.py::compiler_breaks       # CPU only, real-model graph breaks
    modal run modal_app.py::diagnostics           # free-ish, prints the digest

ORDER AND PARALLELISM
    smoke, probe          independent of everything
    task1                 MUST run before sweep/replay/deepseek/qwen -- it writes
                          mmlu_prompts.txt, which they all read
    sweep, replay         after task1; may run concurrently with each other
    deepseek, qwen        after task1; safe to run concurrently. Each modal run gets its
                          own container and GPU, they write different result directories,
                          and they download into different HF cache subdirectories, which
                          Modal merges per file. Serialise only to keep a failure cheap to
                          diagnose, not for correctness
    diagnostics           last; reloads the volumes before reading

Parallelism is cost-neutral on Modal -- billing is per function-second, so two GPUs for
30 minutes costs what one costs for 60. The reason to stay sequential is not money, it is
that a container sees a volume as of MOUNT time: a stage launched before its predecessor
commits will simply not see the files it needs, and will fail in a way that looks like a
bug rather than a race.

Pull results down to the laptop:

    modal volume get routingdrift-results / ./results_modal

Costs assume A100-80GB at $2.50/hr and are +/-50%; lm-eval under INT4 is the wildcard.
"""

import modal

REPO = "/root/RoutingDrift"
RESULTS = "/results"
GPU = "A100-80GB"

# Two images. OLMoE and DeepSeek run on the transformers 4.46 pin that the committed
# results depend on; Qwen3-30B-A3B needs >=4.51, which 4.46 predates, so it gets its own.
# run_manifest.json records the version actually used, and the paper has to note the split.
# Both are pinned below transformers 5 -- see latest_image for why that bound is not
# optional.
_COMMON = [
    "accelerate", "bitsandbytes==0.44.1", "safetensors", "sentencepiece", "protobuf",
    "numpy<2", "pandas", "matplotlib", "seaborn", "tabulate", "scipy",
    # datasets stays unpinned. An earlier attempt to pin <3 (to restore
    # trust_remote_code for lm-eval) broke build_mmlu_prompts with "must be called with a
    # dataclass type or instance" and did not fix lm-eval either. The kwarg is stripped in
    # harness_eval instead, which is surgical and leaves the rest of the stack alone.
    "lm-eval==0.4.4", "datasets", "huggingface_hub",
]

# Experiment output goes to the volume, not the image. But results/olmoe_top2_zaratan
# (460 KB) MUST ship: it is the reference the smoke test diffs against, and without it
# the one check that validates this GPU against the committed A100 run cannot execute.
# The bulky regenerable artifacts stay out -- results/compiler alone is 33 MB.
_IGNORE = [
    # .env is machine-local and gitignored for that reason. Shipping it let load_dotenv
    # override the model path with a Zaratan mount that does not exist in the container,
    # and a .env can hold tokens, which have no business in a remote image.
    ".env",
    ".git", "hpc_runs", "HPC_Outputs", "temp", "docs", "*.zip",
    "results/kernels", "results/kernels_a100", "results/compiler",
    "results/report_plots", "results/*_rerun",
]


def _image(transformers_pin: str, extra: tuple = ()) -> modal.Image:
    """
    Build an image. `add_local_dir` must come LAST: Modal rejects any build step after
    local files are added, since that would rebuild the image on every source edit.
    Extra packages therefore have to be passed in here rather than chained on afterwards.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.5.0", transformers_pin, *_COMMON, *extra)
        .add_local_dir(".", remote_path=REPO, ignore=_IGNORE)
    )


pinned_image = _image("transformers==4.46.0")

# Upper bound is load-bearing, not caution. This was ">=4.57", which on 2026-08-03
# resolved to transformers 5.14.1 and broke the Qwen run twice over:
#
#   1. Visible failure: transformers 5 requires bitsandbytes>=0.46.1, and this project
#      pins 0.44.1 because OLMoE's committed results depend on it. INT8 raised ImportError.
#   2. Silent failure, which is the real reason for the pin: transformers 5 fuses MoE
#      experts into packed 3D parameters (integrations/moe.py) instead of per-expert
#      nn.Linear. The load audit reported "0 of 193 Linear" for a 48-layer, 128-expert
#      model -- 193 is attention plus lm_head, and all 18,432 expert projections are gone
#      as Linear modules. bitsandbytes only replaces nn.Linear, so simply upgrading it
#      would have quantized ATTENTION ONLY and left every expert in fp16, while OLMoE and
#      DeepSeek had experts quantized. The Qwen drift numbers would have been measuring a
#      different intervention under the same label, and nothing would have errored.
#
# 4.x keeps the per-expert nn.Linear layout that OLMoE and DeepSeek were measured under,
# which is what makes the three models comparable. Qwen3-MoE has been supported since
# 4.51. Do not raise this bound without re-checking the load audit's Linear count.
latest_image = _image("transformers>=4.51,<5")

def _git_env() -> dict:
    """
    Capture the local git state at launch and pass it into the container.

    `.git` is not shipped (10 MB, and useless in the container), so `git rev-parse` inside
    the container fails and run_manifest.json would record commit=null. That defeats the
    manifest's purpose: tracing a reported number back to the code that produced it.

    The dirty flag matters more here than in a normal run. `add_local_dir` uploads the
    working TREE, not a commit, so a run can execute code that corresponds to no commit at
    all. A manifest saying `dirty: true` is the only signal that happened.
    """
    import subprocess

    def _git(*cmd: str) -> str:
        try:
            out = subprocess.run(["git", *cmd], capture_output=True, text=True,
                                 timeout=10, check=False)
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:  # noqa: BLE001
            return ""

    return {
        "ROUTINGDRIFT_GIT_COMMIT": _git("rev-parse", "HEAD"),
        "ROUTINGDRIFT_GIT_BRANCH": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "ROUTINGDRIFT_GIT_DIRTY": "1" if _git("status", "--porcelain") else "0",
        "ROUTINGDRIFT_SOURCE": "modal add_local_dir (working tree, not a clone)",
    }


# This module is re-imported INSIDE the container, where .git does not exist, so a plain
# module-level call recomputes to empty strings -- which is exactly what happened on the
# first two runs (manifest showed git: None). modal.Secret.from_dict is resolved on the
# client and injected into the container, so the values survive the round trip.
GIT_SECRET = modal.Secret.from_dict(_git_env())

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


def _reload_volumes() -> None:
    """
    A container sees the volume as of mount time. Without this, diagnostics run after a
    stage reports the *previous* run's manifest -- which is what happened, and made a
    completed run look like it was still `status: started`.
    """
    results_vol.reload()
    hf_cache.reload()


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
# Resolved by the 2026-08-02 smoke run and pinned here so every later stage sees the same
# weights. An unpinned id resolves to whatever main points at on the day, and the three
# models' results would stop being comparable across sessions.
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"


def _rev():
    return ["--revision", REVISION] if REVISION else []


# ---------------------------------------------------------------------------
# Stage 1 -- smoke test. RUN THIS FIRST AND STOP.
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=60 * 60)
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
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=3 * 60 * 60)
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
        # Modal restarts a preempted container from the top with the same input; without
        # this a preemption during the third precision re-pays for the first two. Route
        # dumps whose call count does not match the current prompt set are ignored, so a
        # stale dump cannot be mixed in.
        "--resume",
        "--run_lm_eval",
        # GSM8K removed. It is the only generative task, and its generate_until phase dies
        # with "cuDNN Frontend error: No execution plans support the graph" on this stack.
        # It was already the weakest column: OLMoE scores ~8%, close enough to the floor
        # that an accuracy drop carries no signal. Re-add with --lm_eval_tasks if the
        # descriptive number is wanted and the cuDNN path is fixed.
        "--lm_eval_tasks", "mmlu", "hellaswag",
        "--lm_eval_num_fewshot", "5", "--lm_eval_batch_size", "auto",
        "--lm_eval_limit", str(lm_eval_limit), "--lm_eval_device", "cuda",
    )
    _run("routingdrift.quantization.verify_reproducibility",
         "--results_dir", f"{RESULTS}/olmoe_top8")


# ---------------------------------------------------------------------------
# Probe -- settles the one assumption gating the expensive sweep. ~$0.60.
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=60 * 60)
def probe(n_prompts: int = 20):
    """
    Does bitsandbytes honour llm_int8_skip_modules on 4-bit loads?

    The smoke test cannot answer this: run_experiment never passes skip_modules, so only
    the sweep's nf4_L* configs exercise it. If the answer is no, five of the fifteen sweep
    configs are silent duplicates of full quantization and the layer dial -- the lever that
    sweeps drift continuously rather than in jumps -- has to be rebuilt before the sweep
    is worth $7.

    Four loads, no lm-eval, 20 prompts. Read the quant_audit line for each: nf4 should
    touch every router-bearing layer, nf4_L4 only the first four, nf4_L8 the first eight.
    If all three report the same span, skip_modules is being ignored.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.build_mmlu_prompts",
        "--n", str(n_prompts), "--seed", "0",
        "--out", f"{RESULTS}/probe_prompts.txt",
    )
    _run(
        "routingdrift.quantization.sweep",
        "--model_name", OLMOE, *_rev(),
        "--prompts_file", f"{RESULTS}/probe_prompts.txt",
        "--output_dir", f"{RESULTS}/probe",
        "--top_k", "8", "--target_module", "mlp.gate",
        "--max_length", "128", "--seed", "0",
        "--configs", "fp16", "nf4", "nf4_L4", "nf4_L8",
    )
    print("\nRead the four quant_audit lines above:")
    print("  nf4     should touch ALL router-bearing layers")
    print("  nf4_L4  should touch layers 0-3 only")
    print("  nf4_L8  should touch layers 0-7 only")
    print("If the spans are identical, skip_modules is ignored on 4-bit -> tell Claude")
    print("before running the sweep.")


# ---------------------------------------------------------------------------
# Stage 3 -- the correlation. This is the paper.
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=8 * 60 * 60)
def sweep(quality: str = "nll"):
    """
    15 quantization configs -> drift and gate-KL against a quality loss.

    Quality is measured as NLL, not benchmark accuracy. The sweep needs to separate 15
    configs whose quality differs by fractions of a point, and MMLU at limit=200 has a
    standard error of ~3.5 points against an INT4 drop of ~1: the ranking would be noise.

    NLL wins on PAIRING rather than sample size. The prompt set is small -- 100 prompts,
    at most 12,700 predicted positions -- but every config sees the same prompts in the
    same order, so prompt difficulty is a common term that cancels in the differences the
    correlation is fitted on. Multiple-choice outcomes cannot cancel: one bit per document,
    and 200 documents bound the resolution however the configs are paired.

    Report differences from the fp16 baseline, not raw NLL, and take intervals from
    bootstrapping prompts (bootstrap.py), never a per-token standard error.

    The saving is in evaluation work, not in anything getting faster: lm-eval on
    mmlu+hellaswag at limit=200 is roughly 46k scored continuations per config -- MMLU is
    57 subtasks, so the limit applies 57 times -- against 100 forward passes for NLL. What
    remains is 15 model loads and quantizations, which is why this is about an hour rather
    than minutes.

    Pass quality="both" to also run lm-eval, which restores the eight hours. Accuracy is
    worth having once, as the anchor that says what a given NLL delta means in points on
    a real benchmark -- but it is the wrong instrument for ranking 15 configs.

    If lm-eval is run: GSM8K stays excluded. It is the only generative task, dominates
    eval time, is 2-3x worse again under INT4, and OLMoE scores ~8% on it -- close enough
    to the floor that an "accuracy drop" carries no signal to correlate against.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.sweep",
        "--model_name", OLMOE, *_rev(),
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--output_dir", f"{RESULTS}/olmoe_sweep",
        "--top_k", "8", "--target_module", "mlp.gate",
        "--max_length", "128", "--seed", "0",
        # 15 configs is long enough that a preemption without --resume could restart the
        # whole sweep repeatedly and never finish.
        "--resume",
        "--quality", quality,
        # sweep.py runs lm-eval off --run_lm_eval, independently of --quality, so the flag
        # has to be added rather than implied. Omitting it while passing quality="both"
        # would silently produce an NLL-only run under a name that promised accuracy.
        *(["--run_lm_eval",
           "--lm_eval_tasks", "mmlu", "hellaswag",
           "--lm_eval_num_fewshot", "5", "--lm_eval_batch_size", "auto",
           "--lm_eval_limit", "200", "--lm_eval_device", "cuda"]
          if quality in ("both", "lm_eval") else []),
    )
    print("\nRead sweep_correlations.csv: jaccard_drift AND gate_kl are both correlated")
    print("against the quality loss. If gate_kl explains as much, routing fidelity is a")
    print("proxy for gate noise rather than a metric, and the paper must say so.")
    print("The drift~gate_kl collinearity row is what decides that.")


# ---------------------------------------------------------------------------
# Stage 4 -- causal attribution
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=2 * 60 * 60)
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
@app.function(image=pinned_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=3 * 60 * 60)
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
        # Drift with nothing to relate it to is half a result. The sweep supplies NLL for
        # OLMoE only, so without this the cross-model table has quality for one of three.
        "--measure_nll",
    )
    _run("routingdrift.quantization.verify_reproducibility",
         "--results_dir", f"{RESULTS}/deepseek_v2_lite")


# ---------------------------------------------------------------------------
# Stage 6 -- third architecture: granularity and the renormalisation axis
# ---------------------------------------------------------------------------
@app.function(image=latest_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=4 * 60 * 60)
def qwen_drift():
    """
    Qwen3-30B-A3B-Base: 128 routed experts, top-8, no shared expert.

    NOT Qwen3.6-35B-A3B, which this stage originally named. That model is real (released
    2026-04-15) but its architecture is Qwen3_5MoeForConditionalGeneration with nested
    text_config and vision_config -- an image-text-to-text model. Three problems: it does
    not load through AutoModelForCausalLM, which is all model_loader knows how to call;
    its router sits under a nested language-model submodule; and a vision tower makes it
    the wrong control for a text-only drift comparison against OLMoE and DeepSeek. The
    modality would be confounded with every other difference.

    Qwen3-30B-A3B-Base earns the slot on its own merits rather than being a fallback:

      * norm_topk_prob = TRUE, where OLMoE's is FALSE. OLMoE mixes with the raw softmax
        over all 64 experts and never renormalises over the top-k, so a token that loses
        an expert to quantization loses that expert's weight outright. Qwen renormalises,
        so the surviving experts absorb it. That is a mechanism for why the same jaccard
        drift should cost different amounts of quality, and it is the sharpest contrast
        available -- worth more to the paper than one more 2026 checkpoint.
      * 128 experts against OLMoE's 64 and DeepSeek's 64, holding top-8, which is the
        granularity axis this stage was for.
      * No shared expert, where DeepSeek has 2, so shared-vs-none is isolated.
      * ~60 GB in bf16 on an 80 GB card, against ~72 GB for the 35B. Real headroom.

    Base, not Instruct: instruction tuning reshapes routing, and the other two models are
    base checkpoints.

    Still needs the newer-transformers image -- 4.46 predates qwen3_moe. The manifest
    records the version, and the cross-model comparison therefore spans two library
    versions. State that in the paper rather than hiding it.
    """
    _gpu_report()
    _run(
        "routingdrift.quantization.run_experiment",
        "--model_name", "Qwen/Qwen3-30B-A3B-Base",
        "--precisions", "fp16", "int8", "int4",
        "--target_module", "mlp.gate",
        "--prompts_file", f"{RESULTS}/mmlu_prompts.txt",
        "--output_dir", f"{RESULTS}/qwen3_30b_a3b",
        "--top_k", "8", "--max_length", "128", "--seed", "0", "--skip_heatmaps",
        # Drift with nothing to relate it to is half a result. The sweep supplies NLL for
        # OLMoE only, so without this the cross-model table has quality for one of three.
        "--measure_nll",
    )
    _run("routingdrift.quantization.verify_reproducibility",
         "--results_dir", f"{RESULTS}/qwen3_30b_a3b")


# ---------------------------------------------------------------------------
# Kernel sub-study -- correctness and an honestly measured Amdahl ceiling
# ---------------------------------------------------------------------------
# Triton and dotenv only the kernel stage needs, so they are not in the shared image.
kernel_image = _image("transformers==4.46.0", extra=("triton==3.1.0", "python-dotenv"))


@app.function(image=kernel_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET],
              timeout=2 * 60 * 60)
def kernel_profile():
    """
    Re-measure the kernel sub-study with the attribution bug fixed.

    The reported ~1.015x Amdahl ceiling rested on a fraction obtained by string-matching
    CUDA kernel names, which counted a residual add as RMSNorm, missed the variance
    reduction, always reported softmax as 0.00%, and matched nothing on Mixtral. Fractions
    are now measured with record_function ranges around the real modules.

    Also runs the correctness suite, which now includes non-unit RMSNorm weights. Every
    previous case used w=ones, so a kernel that ignored the weight entirely would have
    passed.

    The conclusion is expected to hold -- the true fraction really is small -- but it will
    be a measurement rather than a guess.
    """
    _gpu_report()
    # Explicit, so the stage does not depend on whatever a .env might have said.
    # _run merges ENV into the subprocess environment, so setting it here is enough and
    # avoids touching os.environ (which this module does not import at top level).
    ENV["OLMOE_PATH"] = OLMOE
    _run("routingdrift.kernels.validate_olmoe")
    _run(
        "routingdrift.kernels.profile_ops",
        "--model", "OLMoE",
        "--out", f"{RESULTS}/kernels_rerun/olmoe",
    )
    print("\nCompare profile_op_fractions_measured.csv against the committed "
          "profile_amdahl.csv. A softmax_pct that is no longer exactly 0.00 is the "
          "clearest sign the old number was an artifact of scanning only the top 15 ops.")


@app.function(image=kernel_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET],
              timeout=3 * 60 * 60)
def compile_benchmark():
    """
    Test the mechanism the other three sub-studies imply.

    The kernels deliver 0.999x against a 1.07x ceiling, the model is launch-bound, and
    torch.compile fuses none of the expert dispatch because of 23 graph breaks that a
    config flag removes. The inference joining those facts -- that fusing the dispatch
    would unlock the kernel gain -- has not been tested. This times eager, eager+kernels,
    compile, compile+capture, and compile+capture+kernels at identical shapes.

    Budget more than the plain benchmark: compilation is billed, and the dynamic-capture
    configuration is the most likely of the five to be slow or to fail outright.
    """
    _gpu_report()
    ENV["OLMOE_PATH"] = OLMOE
    _run(
        "routingdrift.kernels.compile_benchmark",
        "--out", f"{RESULTS}/kernels_rerun/olmoe",
        "--shapes", "512x4,1024x4",
    )


@app.function(image=kernel_image, gpu=GPU, volumes=VOLUMES, secrets=[GIT_SECRET],
              timeout=2 * 60 * 60)
def kernel_benchmark():
    """
    End-to-end latency, baseline versus kernel-patched, on the same machine as the
    profiling that produced the Amdahl fraction.

    This settles the one question sub-study 1 currently leaves open. Measured op fractions
    put the ceiling at 1.07x; the recorded end-to-end figure is 0.985x. Both are at
    seq=512, batch=4, so the shapes agree -- but the 0.985x came from Zaratan on an older
    stack, and INT4 drift already turned out to differ twofold between that environment
    and this one, so timing cannot be assumed to carry over.

    If the gap survives on one machine, the limit is integration overhead rather than
    Amdahl, and the sub-study's conclusion changes.
    """
    _gpu_report()
    ENV["OLMOE_PATH"] = OLMOE
    _run(
        "routingdrift.kernels.benchmark",
        "--model", "OLMoE",
        "--out", f"{RESULTS}/kernels_rerun/olmoe",
    )
    print("\nRead the seq=512 batch=4 row against the 1.07x ceiling in "
          "profile_amdahl.csv. Those two are directly comparable; the older 0.985x was "
          "not, being from different hardware.")


# ---------------------------------------------------------------------------
# Compiler analysis on the REAL checkpoint -- CPU only, so nearly free
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=60 * 60,
              memory=32768)
def compiler_breaks():
    """
    Graph-break analysis on the actual model instead of 2-layer random-init stubs.

    No GPU: dynamo.explain traces the graph without executing kernels, so the result is
    device-independent and this runs on a CPU container for pennies.

    Replaces the three claims the audit found indefensible. On the real modeling code
    OLMoE breaks roughly four times per layer at aten.nonzero in the expert dispatch,
    whose output shape is data-dependent -- not "exactly 1 graph break" as the paper says.
    Subgraph sizes are also wildly unequal, so 1/(breaks+1) was never a meaningful
    "fraction compiled". The 78.8% routing-overhead figure is not recomputed at all: it
    was a scale artifact of hidden-512 stubs and is retired.
    """
    _run(
        "routingdrift.compiler.real_model_breaks",
        "--model_name", OLMOE, *_rev(),
        "--seq_len", "64",
        "--output_dir", f"{RESULTS}/compiler_real",
    )


# ---------------------------------------------------------------------------
# Diagnostics -- CPU only, so this costs almost nothing
# ---------------------------------------------------------------------------
@app.function(image=pinned_image, volumes=VOLUMES, secrets=[GIT_SECRET], timeout=15 * 60)
def diagnostics():
    """Print the digest that answers the open questions. No GPU, so effectively free."""
    _reload_volumes()
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
