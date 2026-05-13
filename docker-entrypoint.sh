#!/usr/bin/env bash
set -e

USAGE="
Usage: docker run ... routingdrift <command> [args]

Commands:
  validate      Validate Triton kernels on OLMoE (kernals/validate_olmoe.py)
  benchmark     Kernel latency sweep           (kernals/benchmark.py)
  eval          GSM8K + MMLU accuracy eval      (kernals/eval_accuracy.py)
  profile       Per-op profiling                (kernals/profile_ops.py)
  quant         Quantization drift experiment   (quantization/run_experiment.py)
  compile       torch.compile analysis          (Compiler/main.py)
  report        Generate cross-study report     (report/generate_report.py)
  bash          Drop to an interactive shell

All commands forward extra arguments to the underlying script.

Examples:
  docker run ... routingdrift validate
  docker run ... routingdrift benchmark --out ./results --model OLMoE
  docker run ... routingdrift quant --model_name allenai/OLMoE-1B-7B --top_k 8
  docker run ... routingdrift eval --out ./results --limit 200
  docker run ... routingdrift report --out report/plots
  docker run ... routingdrift bash
"

CMD="${1:-}"

if [[ -z "$CMD" ]]; then
    echo "$USAGE"
    exit 0
fi

shift  # remaining args forwarded to the script

case "$CMD" in
    validate)
        exec python kernals/validate_olmoe.py "$@"
        ;;
    benchmark)
        exec python kernals/benchmark.py "$@"
        ;;
    eval)
        exec python kernals/eval_accuracy.py "$@"
        ;;
    profile)
        exec python kernals/profile_ops.py "$@"
        ;;
    quant)
        exec python quantization/run_experiment.py "$@"
        ;;
    compile)
        exec python Compiler/main.py "$@"
        ;;
    report)
        exec python report/generate_report.py "$@"
        ;;
    bash)
        exec /bin/bash "$@"
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "$USAGE"
        exit 1
        ;;
esac
