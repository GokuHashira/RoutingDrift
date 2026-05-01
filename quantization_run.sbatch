#!/bin/bash
#SBATCH --job-name=quantization_routing_drift
#SBATCH --partition=gpu  # Adjust to your HPC's GPU partition (e.g., gpu, gpuA100)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB  # Adjust based on model size
#SBATCH --time=02:00:00  # 2 hours; increase if needed
#SBATCH --gres=gpu:1  # Request 1 GPU; adjust if multi-GPU
#SBATCH --output=quantization_%j.out  # Job output log
#SBATCH --error=quantization_%j.err   # Job error log

set -euo pipefail

# Load necessary modules (adjust for your HPC)
module load cuda/11.8  # Or your CUDA version
module load python/3.10  # Or your Python module

# Activate your conda environment (replace 'myenv' with your env name)
conda activate myenv

# Navigate to the quantization directory
cd /scratch/zt1/project/msml605/user/gsakthiv/quantization

# Set model path variable (update this to your model location).
# Use --precisions gptq for a pre-quantized GPTQ checkpoint.
MODEL_PATH="/path/to/your/model"

# Install dependencies if not already done (uncomment if needed)
# pip install -r ../requirements.txt

# Run the quantization experiment
# Adjust --precisions, --output_dir, etc., as needed
python run_experiment.py \
    --model_name "$MODEL_PATH" \
    --precisions gptq \
    --target_module block_sparse_moe.gate \
    --output_dir results \
    --top_k 2 \
    --max_length 256

# Optional: Run with lm-eval for accuracy correlation
# Uncomment below if you want to include evaluation (may take longer)
# python run_experiment.py \
#     --model_name /path/to/your/model \
#     --precisions fp16 int8 int4 \
#     --run_lm_eval \
#     --lm_eval_tasks mmlu gsm8k hellaswag \
#     --output_dir results_with_eval

echo "Quantization experiment completed. Check results/ for outputs."
