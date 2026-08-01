#!/usr/bin/env bash
# Thunder Compute smoke test -- RUN THIS FIRST, before any expensive job.
#
# It costs ~10 minutes and answers two questions that would otherwise cost you hours:
#
#   1. Does the environment work at all? (CUDA, bitsandbytes INT8/INT4, HF download, hooks)
#   2. Does `weights -> routes` reproduce on THIS machine?
#
# It re-runs the exact configuration behind the committed May-2026 Zaratan A100 results
# (5 built-in prompts, top-2, no lm-eval) into a separate directory, then diffs the raw
# routes against the committed ones. The CSV-level half of reproducibility is already
# verified on any laptop by `routingdrift.quantization.verify_reproducibility`; this covers the half
# that needs a GPU.
#
# Expected outcome: 0.00% of rows differ. A small nonzero value means fp16 reduction-order
# differences between this GPU and the A100. A large value means something is actually
# wrong (different checkpoint, different bitsandbytes) -- stop and investigate before
# spending money on the real runs.
#
# Usage:  bash thunder/00_smoke_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0924}"
REVISION="${REVISION:-}"            # leave empty on the first run; pin it afterwards (see below)
SMOKE_OUT="results/smoke_top2"
COMMITTED="results/olmoe_top2_zaratan"

LOG_DIR="thunder/logs"
mkdir -p "$LOG_DIR"
SHELL_LOG="$LOG_DIR/smoke_test_$(date -u +%Y%m%dT%H%M%SZ).log"
# Capture everything, including pip and nvidia-smi output that happens outside Python.
exec > >(tee -a "$SHELL_LOG") 2>&1
echo "[smoke] shell log: $SHELL_LOG"

# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------
echo "===== GPU ====="
nvidia-smi || { echo "FATAL: nvidia-smi failed -- no GPU visible."; exit 1; }

echo "===== Dependencies ====="
pip install -q -e ".[eval,viz]"

python - <<'PY'
import torch
print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device         : {torch.cuda.get_device_name(0)}")
    print(f"capability     : {torch.cuda.get_device_capability(0)}")
import bitsandbytes
print(f"bitsandbytes   : {bitsandbytes.__version__}")
import transformers
print(f"transformers   : {transformers.__version__}")
PY

# ---------------------------------------------------------------------------
# 2. Reproduce the committed top-2 configuration
# ---------------------------------------------------------------------------
echo "===== Reproducing committed top-2 run ====="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REV_ARG=()
[[ -n "$REVISION" ]] && REV_ARG=(--revision "$REVISION")

# Deliberately matches the original: built-in 5 prompts (no --prompts_file), top-2,
# max_length 256, no lm-eval. Changing any of these invalidates the comparison.
python -m routingdrift.quantization.run_experiment \
    --model_name "$MODEL_ID" \
    "${REV_ARG[@]}" \
    --precisions fp16 int8 int4 \
    --target_module mlp.gate \
    --output_dir "$SMOKE_OUT" \
    --top_k 2 \
    --max_length 256 \
    --skip_heatmaps

# ---------------------------------------------------------------------------
# 3. Compare against the committed Zaratan results
# ---------------------------------------------------------------------------
echo "===== Cross-run comparison vs committed A100 results ====="
python -m routingdrift.quantization.verify_reproducibility \
    --results_dir "$SMOKE_OUT" \
    --compare_to "$COMMITTED"

echo
echo "=================================================================="
echo "Smoke test finished."
echo "  results : $SMOKE_OUT"
echo "  logs    : $SHELL_LOG  +  $SMOKE_OUT/logs/"
echo
echo "Now pin the checkpoint for every run that follows. Read the resolved SHA with:"
echo "  python -c \"import json;print(json.load(open('$SMOKE_OUT/run_manifest.json'))['variants'])\""
echo "then re-export it:  export REVISION=<sha>"
echo
echo "Determinism + self-consistency guard results are in:"
echo "  $SMOKE_OUT/run_manifest.json  ->  .guards"
echo "=================================================================="
