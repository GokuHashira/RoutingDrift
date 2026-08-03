"""
make_tiny_moe.py

Build a tiny, randomly-initialized OLMoE checkpoint (a few MB) for CPU testing.

Same architecture family as the real target -- `model.layers.N.mlp.gate` router modules,
top-k expert routing -- just small enough to run the whole pipeline on a laptop in
seconds. Nothing is downloaded; the tokenizer is built locally.

This exists so the first execution of the experiment pipeline is not on a rented GPU.

Usage:
    python tools/make_tiny_moe.py --out /tmp/tiny-olmoe
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_tokenizer(vocab_size: int):
    """A minimal byte-level tokenizer, constructed offline."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # Every id must map to a distinct token, or the saved vocab has holes.
    vocab = {f"<{i}>": i for i in range(4)}
    vocab.update({f"tok{i}": i for i in range(4, vocab_size)})

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<0>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<0>",
        bos_token="<1>",
        eos_token="<2>",
        pad_token="<3>",
    )


def build(out_dir: Path, num_experts: int = 8, top_k: int = 2, num_layers: int = 2) -> Path:
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    vocab_size = 256
    config = OlmoeConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        max_position_embeddings=128,
    )

    torch.manual_seed(0)
    model = OlmoeForCausalLM(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    build_tokenizer(vocab_size).save_pretrained(out_dir)

    params = sum(p.numel() for p in model.parameters())
    routers = [n for n, _ in model.named_modules() if n.endswith("mlp.gate")]
    print(f"[make_tiny_moe] wrote {out_dir}")
    print(f"[make_tiny_moe] {params/1e6:.2f}M params, {num_experts} experts, top-{top_k}, {num_layers} layers")
    print(f"[make_tiny_moe] router modules: {routers}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, required=True, help="Directory to write the checkpoint to.")
    ap.add_argument("--num_experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    args = ap.parse_args()
    build(Path(args.out), args.num_experts, args.top_k, args.num_layers)


if __name__ == "__main__":
    main()
