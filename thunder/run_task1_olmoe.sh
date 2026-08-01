#!/usr/bin/env bash
# Task 1 -- OLMoE top-8 routing drift + INT8/INT4 lm-eval, on a Thunder Compute instance.
# Thunder is NOT a SLURM cluster, so this is a plain bash script (no sbatch).
#
# Run `thunder/00_smoke_test.sh` FIRST. It validates the environment and tells you whether
# this machine reproduces the committed results before you spend hours here.
#
# Target: A100 80GB. OLMoE-1B-7B needs ~14GB in FP16 so a 24GB card also works, but the
# whole sweep must run on ONE device -- gate-logit margins are thin enough that fp16
# reduction-order differences between GPUs flip near-tie top-k picks, which would put a
# hardware confound inside the drift measurement itself.
#
# Usage:  bash thunder/run_task1_olmoe.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
# OLMoE-1B-7B is downloaded from the HF Hub (Thunder has internet -- do NOT set HF offline).
export MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0924}"
export REVISION="${REVISION:-}"      # set from the smoke test's manifest; see 00_smoke_test.sh
export N_PROMPTS="${N_PROMPTS:-100}" # ~100 MMLU questions (vs the old 5 generic ones)

# NOTE: deliberately NOT results_olmoe_datasets. That directory holds the committed
# top-2 baseline and is the only copy -- writing here would destroy the comparison that
# shows what widening top-2 -> top-8 actually changed.
OUT="${OUT:-results/olmoe_top8}"

if [[ -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
    echo "WARNING: $OUT already exists and is non-empty."
    echo "         Existing route dumps and CSVs will be overwritten."
    read -r -p "         Continue? [y/N] " reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "Aborted."; exit 1; }
fi

LOG_DIR="thunder/logs"
mkdir -p "$LOG_DIR"
SHELL_LOG="$LOG_DIR/task1_olmoe_$(date -u +%Y%m%dT%H%M%SZ).log"
# Capture pip/HF-download output too, not just what Python prints.
exec > >(tee -a "$SHELL_LOG") 2>&1
echo "[task1] shell log : $SHELL_LOG"
echo "[task1] output dir: $OUT"
[[ -z "$REVISION" ]] && echo "[task1] WARNING: no REVISION pinned -- '$MODEL_ID' resolves to whatever main points at today."

# ---------------------------------------------------------------------------
# 1. Environment (run once per fresh instance; safe to re-run)
# ---------------------------------------------------------------------------
pip install -q -e ".[eval,viz]"

# ---------------------------------------------------------------------------
# 2. Build the MMLU prompt set (writes results/mmlu_prompts.txt)
# ---------------------------------------------------------------------------
python -m routingdrift.quantization.build_mmlu_prompts --n "${N_PROMPTS}" --seed 0 \
    --out results/mmlu_prompts.txt

# ---------------------------------------------------------------------------
# 3. Run drift at top-8 + accuracy eval (fp16 baseline, int8, int4)
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REV_ARG=()
[[ -n "$REVISION" ]] && REV_ARG=(--revision "$REVISION")

# --lm_eval_limit is intentionally absent: this is the headline accuracy table, so it runs
# the full task sets. The Phase 2 sweep is where subsampling belongs.
python -m routingdrift.quantization.run_experiment \
    --model_name "${MODEL_ID}" \
    "${REV_ARG[@]}" \
    --precisions fp16 int8 int4 \
    --target_module mlp.gate \
    --prompts_file results/mmlu_prompts.txt \
    --output_dir "${OUT}" \
    --top_k 8 \
    --max_length 128 \
    --seed 0 \
    --run_lm_eval \
    --lm_eval_tasks mmlu gsm8k hellaswag \
    --lm_eval_num_fewshot 5 \
    --lm_eval_batch_size auto \
    --lm_eval_device cuda

# ---------------------------------------------------------------------------
# 4. Verify the outputs are internally consistent before trusting them
# ---------------------------------------------------------------------------
python -m routingdrift.quantization.verify_reproducibility --results_dir "${OUT}"

echo "=================================================================="
echo "Task 1 done. Verify these exist and are non-empty:"
echo "  ${OUT}/routing_drift_summary.csv        (top_k should be 8 now)"
echo "  ${OUT}/routing_drift_layers.csv"
echo "  ${OUT}/lm_eval/lm_eval_{fp16,int8,int4}.json"
echo "  ${OUT}/drift_accuracy_correlations.csv  (the drift->quality link)"
echo "  ${OUT}/run_manifest.json                (versions, GPU, revision, guards)"
echo "  ${OUT}/logs/                            (per-run Python logs)"
echo "  ${SHELL_LOG}                            (full shell log incl. pip)"
echo
echo "Note: 3 precision points is a trend, not a correlation. The 12-config sweep in"
echo "Phase 2 is what makes drift->quality statistically reportable."
echo
echo "Then: git add ${OUT} && git commit -m 'Task 1: OLMoE top-8 drift + lm-eval'"
echo "=================================================================="
