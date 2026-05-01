#!/bin/bash
#SBATCH --job-name=msml605_quantization_mixtral
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=msml605_quantization_mixtral.out
#SBATCH --error=msml605_quantization_mixtral.err

module load cuda/11.8

cd /scratch/zt1/project/msml605/user/gsakthiv/quantization
source /scratch/zt1/project/msml605/user/gsakthiv/compenv/bin/activate

# GPTQ-only run. This loads the checkpoint as-is instead of applying
# bitsandbytes fp16/int8/int4 modes, which require the dense checkpoint.
MODEL_PATH="/scratch/zt1/project/msml605/user/gsakthiv/models/Mixtral-8x7B-GPTQ"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_experiment.py \
    --model_name "$MODEL_PATH" \
    --precisions gptq \
    --target_module block_sparse_moe.gate \
    --output_dir results_mixtral_gptq \
    --top_k 2 \
    --max_length 256

echo "Mixtral GPTQ experiment completed. Check results_mixtral_gptq/ for outputs."
