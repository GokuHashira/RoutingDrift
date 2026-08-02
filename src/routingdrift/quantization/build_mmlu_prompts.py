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


def build(n: int, seed: int, allow_fallback: bool = False) -> list[str]:
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
    except Exception as exc:  # noqa: BLE001 - reported, never silently absorbed
        if not allow_fallback:
            raise RuntimeError(
                f"could not load MMLU ({type(exc).__name__}: {exc}).\n"
                "Refusing to substitute the built-in generic prompts. Doing so silently "
                "would produce a drift measurement over 5 generic sentences while every "
                "downstream file labels it a 100-question MMLU corpus -- which is exactly "
                "the mislabelling the correctness audit found in the original results.\n"
                "Pass --allow_fallback if a smoke test on generic prompts is genuinely "
                "what you want."
            ) from exc
        print(f"[build_mmlu_prompts] WARNING: MMLU unavailable ({exc}); "
              f"using {len(_FALLBACK)} GENERIC fallback prompts, NOT MMLU.")
        return _FALLBACK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Number of MMLU questions to sample.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (for reproducibility).")
    ap.add_argument("--out", type=str, default="mmlu_prompts.txt", help="Output prompts file.")
    ap.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Permit the generic built-in prompts if MMLU cannot be loaded. Off by "
             "default: a silent substitution mislabels the resulting measurement.",
    )
    args = ap.parse_args()

    prompts = build(args.n, args.seed, allow_fallback=args.allow_fallback)
    if len(prompts) < args.n and not args.allow_fallback:
        raise RuntimeError(
            f"asked for {args.n} prompts but only {len(prompts)} were produced; "
            "refusing to write a short prompt set that later stages will treat as complete"
        )
    Path(args.out).write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"[build_mmlu_prompts] Wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
