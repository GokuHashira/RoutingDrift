"""Sample MMLU questions into a newline-separated prompts file for the routing-drift study.

Usage:
    python build_mmlu_prompts.py --n 100 --out mmlu_prompts.txt

Each line is one prompt: the question text followed by its multiple-choice options,
formatted the way a model would see an MMLU item. Requires internet the first time
(downloads `cais/mmlu` via the `datasets` library, already a transitive dependency of
lm-eval). If `datasets` or the download is unavailable, falls back to a small built-in set
so the pipeline still runs.
"""
import argparse
import random
from pathlib import Path

_FALLBACK = [
    "What is the primary function of the mitochondria in a eukaryotic cell?",
    "In classical mechanics, what does Newton's second law relate?",
    "Which amendment to the US Constitution abolished slavery?",
    "What is the time complexity of binary search on a sorted array?",
    "Explain the difference between a stack and a queue data structure.",
]


def build(n: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="test")
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), min(n, len(ds)))
        prompts = []
        for i in idxs:
            row = ds[i]
            choices = row.get("choices", [])
            letters = ["A", "B", "C", "D", "E", "F"]
            opts = " ".join(
                f"({letters[j]}) {c}" for j, c in enumerate(choices)
            )
            # single line: question + options (routing drift only needs the tokens, not answers)
            prompts.append(f"{row['question'].strip()} {opts}".replace("\n", " ").strip())
        return prompts
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        print(f"[build_mmlu_prompts] datasets unavailable ({exc}); using fallback set.")
        return _FALLBACK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Number of MMLU questions to sample.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (for reproducibility).")
    ap.add_argument("--out", type=str, default="mmlu_prompts.txt", help="Output prompts file.")
    args = ap.parse_args()

    prompts = build(args.n, args.seed)
    Path(args.out).write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"[build_mmlu_prompts] Wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
