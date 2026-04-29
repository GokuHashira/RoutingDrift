# Project: configuration
# Author:  Amogh Rajendra

from dataclasses import dataclass, field
from typing import List, Optional
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int   
    intermediate_size: int
    num_heads: int
    num_layers: int            
    vocab_size: int = 32000
    max_seq_len: int = 512
 
OLMOE_CONFIG = ModelConfig(
    name="OLMoE",
    hidden_size=2048,
    num_experts=64,
    num_experts_per_tok=8,
    intermediate_size=1024,
    num_heads=16,
    num_layers=16,
)
 
MIXTRAL_CONFIG = ModelConfig(
    name="Mixtral",
    hidden_size=4096,
    num_experts=8,
    num_experts_per_tok=2,
    intermediate_size=14336,
    num_heads=32,
    num_layers=32,
)
 
ALL_MODELS: List[ModelConfig] = [OLMOE_CONFIG, MIXTRAL_CONFIG]

# Slim configs: same routing structure, small enough for CPU analysis.
# Graph break patterns (topk, for-loop, index_add_, one_hot) are identical
# regardless of hidden size — only parameter count changes.
OLMOE_ANALYSIS_CONFIG = ModelConfig(
    name="OLMoE",
    hidden_size=512,
    num_experts=8,
    num_experts_per_tok=2,
    intermediate_size=256,
    num_heads=8,
    num_layers=2,
)

MIXTRAL_ANALYSIS_CONFIG = ModelConfig(
    name="Mixtral",
    hidden_size=512,
    num_experts=8,
    num_experts_per_tok=2,
    intermediate_size=1024,
    num_heads=8,
    num_layers=2,
)

#Modes
@dataclass
class BenchmarkConfig:
    batch_size: int = 1
    seq_len: int = 128
 
    warmup_iters: int = 10
    timed_iters: int = 100
 
    # Percentiles to report
    percentiles: List[float] = field(default_factory=lambda: [0.50, 0.90, 0.99])
 
    # torch.profiler settings
    profiler_wait: int = 1
    profiler_warmup: int = 2
    profiler_active: int = 5
 
BENCH_CFG = BenchmarkConfig()

OUTPUT_DIR = "outputs"
GRAPH11_PATH   = f"{OUTPUT_DIR}/graph11_graph_breaks.png"
GRAPH12_PATH   = f"{OUTPUT_DIR}/graph12_compile_modes.png"
METRICS_PATH   = f"{OUTPUT_DIR}/metrics_summary.json"
BEST_MODE_PATH = f"{OUTPUT_DIR}/best_compile_mode.txt"
INDUCTOR_IR_PATH = f"{OUTPUT_DIR}/inductor_ir_report.txt"
PROFILER_TRACE_PATH = f"{OUTPUT_DIR}/profiler_trace.json"

LAYER_TYPES = ["attention", "moe_routing", "ffn", "rmsnorm", "embed", "lm_head"]

COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]

TRITON_BLOCK_SIZE = 1024
TRITON_NUM_WARPS  = 8
