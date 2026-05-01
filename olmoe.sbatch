#!/bin/bash
#SBATCH --job-name=msml605_quantization_olmoe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=msml605_quantization_olmoe.out
#SBATCH --error=msml605_quantization_olmoe.err

set -euo pipefail

module load cuda/11.8

cd /scratch/zt1/project/msml605/user/gsakthiv/quantization
source /scratch/zt1/project/msml605/user/gsakthiv/compenv/bin/activate

MODEL_PATH="allenai/OLMoE-1B-7B-0125"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_experiment.py \
    --model_name "$MODEL_PATH" \
    --precisions fp16 int8 int4 \
    --target_module mlp.gate \
    --output_dir results_olmoe \
    --top_k 8 \
    --max_length 256 \
    --run_lm_eval \
    --lm_eval_tasks mmlu gsm8k hellaswag \
    --lm_eval_num_fewshot 5 \
    --lm_eval_batch_size auto \
    --lm_eval_device cuda

echo "OLMoE quantization experiment completed. Check results_olmoe/ for outputs."
