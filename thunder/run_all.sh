#!/usr/bin/env bash
# run_all.sh -- the whole GPU experiment sequence, gated and budgeted.
#
# Budget model: Thunder bills WALL-CLOCK while the instance exists, not compute. Model
# downloads, pip installs, and thinking time all cost money. At $1.09/hr a $20 balance is
# 18.3 hours; the sequence below is planned for ~11.3 hours (~$12.30), leaving real margin
# for the failures a never-executed pipeline will have.
#
#   stage                                       hours   $      what it buys
#   0  CPU gates + environment                   0.1    0.11   fails in seconds, not hours
#   1  Smoke: reproduce committed top-2          0.4    0.44   <-- STOP AND READ THE DIFF
#   2  Task 1: top-8 drift + lm-eval             1.5    1.64   the headline table
#   3  Sweep: 15 configs + lm-eval               4.0    4.36   the correlation (the paper)
#   4  Replay: causal attribution                1.0    1.09   is drift causal?
#   5  DeepSeek-V2-Lite, drift only              0.6    0.65   shared-expert axis
#   6  Qwen3.6-35B-A3B, drift only               1.0    1.10   granularity axis, 2026 model
#   (compiler analysis runs on CPU, free -- see notes at the end)
#
# STOP THE INSTANCE BETWEEN STAGES. That is where budgets die.
#
# Usage:
#     STAGES="0 1"   bash thunder/run_all.sh    # cheap gates, then stop and inspect
#     STAGES="2 3 4" bash thunder/run_all.sh    # the OLMoE core
#     STAGES="5"     bash thunder/run_all.sh    # second model only
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0924}"
REVISION="${REVISION:-}"
TOP_K="${TOP_K:-8}"
N_PROMPTS="${N_PROMPTS:-100}"
STAGES="${STAGES:-0 1 2 3 4 5 6}"

DEEPSEEK_ID="${DEEPSEEK_ID:-deepseek-ai/DeepSeek-V2-Lite}"
DEEPSEEK_TOPK="${DEEPSEEK_TOPK:-6}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen3.6-35B-A3B}"
QWEN_TOPK="${QWEN_TOPK:-8}"

PROMPTS="results/mmlu_prompts.txt"
LOG_DIR="thunder/logs"; mkdir -p "$LOG_DIR" results
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SHELL_LOG="$LOG_DIR/run_all_${RUN_ID}.log"
exec > >(tee -a "$SHELL_LOG") 2>&1

REV_ARG=(); [[ -n "$REVISION" ]] && REV_ARG=(--revision "$REVISION")

banner(){ echo; echo "=================================================================="; echo "$*"; echo "=================================================================="; }
have(){ [[ " $STAGES " == *" $1 "* ]]; }
die(){ echo; echo "FAILED at stage $1: $2"; echo "Log: $SHELL_LOG"; exit 1; }
disk(){ echo "[disk] $(df -h . | awk 'NR==2{print $4" free of "$2}')"; }
# 14 + 31 + 72 GB of checkpoints exceeds a 100 GB disk, so caches are cleared between models.
drop_cache(){ echo "[cache] clearing HF cache for $1"; rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"; disk; }

banner "RoutingDrift full run  ${RUN_ID}"
echo "primary  : $MODEL_ID   revision: ${REVISION:-<unpinned>}"
echo "stages   : $STAGES"
echo "log      : $SHELL_LOG"
disk

# --------------------------------------------------------------------------- 0
if have 0; then
    banner "[0] CPU gates + environment"
    pip install -q -e ".[eval,viz]" || die 0 "dependency install"
    nvidia-smi || die 0 "no GPU visible"
    python - <<'PY' || exit 1
import torch, transformers, bitsandbytes
print(f"torch {torch.__version__} | transformers {transformers.__version__} | bnb {bitsandbytes.__version__}")
assert torch.cuda.is_available(), "CUDA not available"
p = torch.cuda.get_device_properties(0)
print(f"gpu   {p.name}  {p.total_memory/1024**3:.1f} GB")
PY
    python tools/check_imports.py || die 0 "import graph"
    python -m pytest -q                 || die 0 "tests"
    echo "stage 0 OK"
fi

# --------------------------------------------------------------------------- 1
if have 1; then
    banner "[1] Smoke test: reproduce the committed May-2026 Zaratan A100 run"
    # Matches the original exactly: built-in 5 prompts, top-2, max_length 256, no lm-eval.
    python -m routingdrift.quantization.run_experiment \
        --model_name "$MODEL_ID" "${REV_ARG[@]}" \
        --precisions fp16 int8 int4 --target_module mlp.gate \
        --output_dir results/smoke_top2 --top_k 2 --max_length 256 --skip_heatmaps \
        || die 1 "smoke run"
    python -m routingdrift.quantization.verify_reproducibility \
        --results_dir results/smoke_top2 --compare_to results/olmoe_top2_zaratan \
        || die 1 "verification"

    banner "STOP HERE AND READ THE CROSS-RUN DIFF"
    echo "0.00%     bit-identical to the A100 run. Ideal."
    echo "<~2%      fp16 reduction-order differences between this GPU and the A100."
    echo "          Acceptable; name the device in the paper."
    echo "large     a real difference (checkpoint revision, bitsandbytes version,"
    echo "          tokenizer). Investigate before spending on stage 2."
    echo
    echo "Then pin the checkpoint for everything after this:"
    echo "  python -c \"import json;print(json.load(open('results/smoke_top2/run_manifest.json'))['variants'])\""
    echo "  export REVISION=<sha>"
fi

# --------------------------------------------------------------------------- 2
if have 2; then
    banner "[2] Task 1: OLMoE top-8 drift + lm-eval"
    [[ -z "$REVISION" ]] && echo "WARNING: no REVISION pinned; '$MODEL_ID' resolves to today's main."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    [[ -f "$PROMPTS" ]] || python -m routingdrift.quantization.build_mmlu_prompts \
        --n "$N_PROMPTS" --seed 0 --out "$PROMPTS" || die 2 "prompt build"

    # limit 500 keeps this ~1.5h instead of ~5h. Subsampling adds noise but does not bias
    # anything, because every config sees the same documents.
    python -m routingdrift.quantization.run_experiment \
        --model_name "$MODEL_ID" "${REV_ARG[@]}" \
        --precisions fp16 int8 int4 --target_module mlp.gate \
        --prompts_file "$PROMPTS" --output_dir results/olmoe_top8 \
        --top_k "$TOP_K" --max_length 128 --seed 0 \
        --run_lm_eval --lm_eval_tasks mmlu gsm8k hellaswag \
        --lm_eval_num_fewshot 5 --lm_eval_batch_size auto \
        --lm_eval_limit 500 --lm_eval_device cuda \
        || die 2 "Task 1"
    python -m routingdrift.quantization.verify_reproducibility \
        --results_dir results/olmoe_top8 || die 2 "Task 1 verification"
    echo "NOTE: three precision points is a trend. Stage 3 is what makes it a correlation."
fi

# --------------------------------------------------------------------------- 3
if have 3; then
    banner "[3] Sweep: 15 quantization configs -> the drift/quality correlation"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m routingdrift.quantization.sweep \
        --model_name "$MODEL_ID" "${REV_ARG[@]}" \
        --prompts_file "$PROMPTS" --output_dir results/olmoe_sweep \
        --top_k "$TOP_K" --target_module mlp.gate --max_length 128 --seed 0 \
        --run_lm_eval --lm_eval_tasks mmlu gsm8k hellaswag \
        --lm_eval_num_fewshot 5 --lm_eval_batch_size auto \
        --lm_eval_limit 200 --lm_eval_device cuda \
        || die 3 "sweep"
    echo
    echo "Read sweep_correlations.csv: jaccard_drift AND gate_kl are both correlated against"
    echo "accuracy_drop. If gate_kl explains as much, routing fidelity is a proxy for gate"
    echo "noise rather than a metric, and the paper must say so."
fi

# --------------------------------------------------------------------------- 4
if have 4; then
    banner "[4] Route replay: is routing drift causal?"
    python -m routingdrift.quantization.route_replay \
        --model_name "$MODEL_ID" "${REV_ARG[@]}" \
        --prompts_file "$PROMPTS" \
        --baseline_routes results/olmoe_top8/routes_fp16.json \
        --replay_routes  results/olmoe_top8/routes_int4.json \
        --quant_precision int4 --top_k "$TOP_K" --target_module mlp.gate \
        --max_length 128 --output_dir results/olmoe_replay \
        || die 4 "replay"
    echo
    echo "Check intervention_artifact in replay_result.json. OLMoE has norm_topk_prob=False,"
    echo "so masking shifts mixing weights as well as selection; attribution is reported"
    echo "against the control, and a large artifact makes the number less trustworthy."
fi

# --------------------------------------------------------------------------- 5
if have 5; then
    banner "[5] DeepSeek-V2-Lite: drift only (64 routed + 2 shared, top-6)"
    # Closest controlled contrast with OLMoE: same 64 routed experts, shared experts added.
    # Its MoEGate returns (topk_idx, topk_weight, aux_loss); the adapter in routing_logger
    # rebuilds logits from the gate weight. Route replay is NOT defined here and will
    # raise rather than silently no-op.
    disk
    python -m routingdrift.quantization.run_experiment \
        --model_name "$DEEPSEEK_ID" \
        --precisions fp16 int8 int4 --target_module mlp.gate \
        --prompts_file "$PROMPTS" --output_dir results/deepseek_v2_lite \
        --top_k "$DEEPSEEK_TOPK" --max_length 128 --seed 0 --skip_heatmaps \
        || die 5 "DeepSeek drift"
    python -m routingdrift.quantization.verify_reproducibility \
        --results_dir results/deepseek_v2_lite || die 5 "DeepSeek verification"
    drop_cache "$DEEPSEEK_ID"
fi

# --------------------------------------------------------------------------- 6
if have 6; then
    banner "[6] Qwen3.6-35B-A3B: drift only (256 routed + 1 shared, top-8, Apr 2026)"
    echo "This model needs a NEWER transformers than the 4.46 pin used for OLMoE."
    echo "Run it in a separate environment so the OLMoE results stay reproducible;"
    echo "run_manifest.json records both versions, and the paper must note the split."
    echo
    read -r -p "Is this shell using the newer-transformers environment? [y/N] " reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || die 6 "wrong environment; see the note above"
    disk
    python -m routingdrift.quantization.run_experiment \
        --model_name "$QWEN_ID" \
        --precisions fp16 int8 int4 --target_module mlp.gate \
        --prompts_file "$PROMPTS" --output_dir results/qwen36_moe \
        --top_k "$QWEN_TOPK" --max_length 128 --seed 0 --skip_heatmaps \
        || die 6 "Qwen3.6 drift"
    python -m routingdrift.quantization.verify_reproducibility \
        --results_dir results/qwen36_moe || die 6 "Qwen3.6 verification"
    drop_cache "$QWEN_ID"
fi

banner "DONE"
disk
echo "Artifacts under results/ ; full shell log at $SHELL_LOG"
echo "Every results dir carries run_manifest.json (versions, GPU, revision, guards) and logs/."
echo
echo "Still to run, on CPU and free -- do NOT spend GPU time on it:"
echo "  ROUTINGDRIFT_COMPILER_OUT=results/compiler_rerun python -m routingdrift.compiler.main"
echo "  python -m routingdrift.reporting.generate_report --out results/report_plots"
