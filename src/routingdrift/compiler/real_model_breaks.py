"""
real_model_breaks.py

Graph-break analysis on the actual checkpoint, replacing the toy-stub numbers.

Every quantitative claim in the compiler sub-study came from randomly-initialized 2-layer
stubs at hidden 512 with 8 experts. The audit found three consequences:

  * "routing = 78.8% of forward time" is a scale artifact. At hidden 512 the expert
    matmuls are microscopic, so Python-loop and kernel-launch overhead dominates. On the
    real model, with intermediate width 1024 across 64 experts and 16 layers, the expert
    matmuls dominate instead and the dispatch share collapses. The figure should be
    retired, not reproduced, and this module does not compute it.
  * "% compiled" was `1/(breaks+1)`, which assumes every subgraph is the same size. This
    reports the op-weighted fraction instead: how many captured FX nodes sit inside
    compiled subgraphs versus how many the eager fallback covers.
  * Break counts disagreed between artifacts (metrics_summary said 1, the README said 4
    and 3). A real trace settles it.

Runs on CPU. dynamo.explain traces the graph without executing kernels, so no GPU is
needed and the result is device-independent.

    python -m routingdrift.compiler.real_model_breaks \\
        --model_name allenai/OLMoE-1B-7B-0924 --revision <sha> \\
        --output_dir results/compiler_rerun
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch


def _explain(model, inputs, cache_size_limit: int = 256,
             capture_dynamic_output_shape_ops: bool = False) -> Any:
    """
    Trace the model and report where the graph breaks.

    cache_size_limit matters. The default is 8, and every decoder layer recompiles because
    `self_attn.layer_idx` differs, so on a 16-layer model dynamo stops tracing partway:

        torch._dynamo hit config.cache_size_limit (8)
        last reason: L['self']._modules['self_attn'].layer_idx == 0

    A break count collected under that limit is a floor, not a count. Raising it lets every
    layer trace.

    capture_dynamic_output_shape_ops is the flag dynamo itself names in the break message.
    The routing break is `aten.nonzero.default`, whose output shape is data-dependent;
    enabling capture lets dynamo trace it with an unbacked symbolic size instead of giving
    up. Whether that actually removes the break is the interesting question, so it is
    measured rather than assumed.
    """
    import torch._dynamo as dynamo

    dynamo.reset()
    prev_limit = dynamo.config.cache_size_limit
    prev_capture = getattr(dynamo.config, "capture_dynamic_output_shape_ops", None)
    dynamo.config.cache_size_limit = cache_size_limit
    if prev_capture is not None:
        dynamo.config.capture_dynamic_output_shape_ops = capture_dynamic_output_shape_ops
    try:
        return dynamo.explain(lambda **kw: model(**kw))(**inputs)
    finally:
        dynamo.config.cache_size_limit = prev_limit
        if prev_capture is not None:
            dynamo.config.capture_dynamic_output_shape_ops = prev_capture


def _op_weighted_fraction(explanation: Any) -> Dict[str, Any]:
    """
    Fraction of traced FX nodes that live inside compiled subgraphs.

    `1/(breaks+1)` treats a 500-node attention subgraph and a 3-node fallback as equal
    halves. Counting nodes says what proportion of the traced computation Inductor
    actually got to fuse.
    """
    graphs = getattr(explanation, "graphs", []) or []
    node_counts = []
    for graph in graphs:
        try:
            node_counts.append(len(list(graph.graph.nodes)))
        except Exception:  # noqa: BLE001 - shape of the object varies across torch versions
            node_counts.append(0)
    total_nodes = sum(node_counts)
    return {
        "subgraphs": len(graphs),
        "nodes_per_subgraph": node_counts,
        "total_captured_nodes": total_nodes,
        # Every captured node is, by definition, inside a compiled subgraph. What the
        # breaks cost is the gap BETWEEN them, which dynamo does not report as nodes --
        # so this is an upper bound on coverage, and is labelled as such.
        "captured_nodes_note": (
            "Nodes here are those dynamo captured. Work in the eager gaps between "
            "subgraphs is not represented, so subgraph sizes describe the shape of the "
            "capture rather than a compiled-versus-eager ratio."
        ),
    }


def _break_reasons(explanation: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for reason in getattr(explanation, "break_reasons", []) or []:
        entry = {"reason": str(getattr(reason, "reason", reason))}
        frames = getattr(reason, "user_stack", None) or []
        if frames:
            last = frames[-1]
            entry["file"] = str(getattr(last, "filename", ""))
            entry["line"] = str(getattr(last, "lineno", ""))
            entry["code"] = str(getattr(last, "line", "")).strip()
        out.append(entry)
    return out


def analyse(model_name: str, revision: str | None, seq_len: int, output_dir: Path) -> Dict[str, Any]:
    from routingdrift.quantization.model_loader import load_model
    from routingdrift.quantization.repro import (
        collect_run_manifest,
        save_run_manifest,
        set_global_seed,
        start_run_log,
    )
    from routingdrift.quantization.routing_logger import infer_num_experts

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = start_run_log(output_dir, name="real_model_breaks")
    seed_settings = set_global_seed()

    # device_map=None keeps the whole model on CPU; dynamo.explain traces without running
    # kernels, so a GPU buys nothing here.
    model, tokenizer = load_model(
        model_name=model_name, precision="fp16", revision=revision, device_map=None
    )
    model = model.float()  # fp16 on CPU is slow and unnecessary for tracing
    model.eval()

    config = model.config
    print(f"[breaks] {model_name}")
    print(f"[breaks] layers={getattr(config, 'num_hidden_layers', '?')} "
          f"experts={infer_num_experts(config)} "
          f"top_k={getattr(config, 'num_experts_per_tok', '?')} "
          f"hidden={getattr(config, 'hidden_size', '?')} "
          f"intermediate={getattr(config, 'intermediate_size', '?')}")

    text = "The capital of France is Paris, and the capital of Germany is"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)

    print("[breaks] running torch._dynamo.explain on the real checkpoint ...")
    explanation = _explain(model, dict(inputs), cache_size_limit=256)

    break_count = int(getattr(explanation, "graph_break_count", 0) or 0)
    graph_count = int(getattr(explanation, "graph_count", 0) or 0)
    reasons = _break_reasons(explanation)
    weighting = _op_weighted_fraction(explanation)

    where = Counter(
        f"{Path(r.get('file', '')).name}:{r.get('line', '')}" for r in reasons if r.get("file")
    )

    result: Dict[str, Any] = {
        "model_name": model_name,
        "revision": revision,
        "seq_len": int(inputs["input_ids"].shape[-1]),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_experts": infer_num_experts(config),
        "graph_break_count": break_count,
        "graph_count": graph_count,
        "breaks_per_layer": (
            break_count / config.num_hidden_layers
            if getattr(config, "num_hidden_layers", 0) else None
        ),
        "break_reasons": reasons,
        "break_locations": dict(where),
        **weighting,
        "retired_claims": {
            "routing_share_of_forward_time": (
                "NOT COMPUTED. The 78.8% figure came from 2-layer random-init stubs at "
                "hidden 512 where expert matmuls are microscopic and dispatch overhead "
                "dominates. It does not transfer to the real model and is retired rather "
                "than reproduced."
            ),
            "percent_compiled_heuristic": (
                "NOT REPORTED as 1/(breaks+1). That assumes equal-sized subgraphs. "
                "Subgraph node counts are reported instead."
            ),
        },
    }

    print(f"\n[breaks] graph breaks : {break_count}")
    print(f"[breaks] subgraphs    : {graph_count}")
    if getattr(config, "num_hidden_layers", 0):
        print(f"[breaks] per layer    : {result['breaks_per_layer']:.2f}")
    print(f"[breaks] nodes/subgraph: {weighting['nodes_per_subgraph']}")
    if where:
        print("[breaks] locations:")
        for loc, n in where.most_common():
            print(f"           {n:>3}x {loc}")
    for r in reasons[:5]:
        print(f"           reason: {r['reason'][:100]}")

    # Does the flag dynamo suggests actually remove the break? This is the actionable
    # half of the finding: "MoE breaks compilation" is a description, "and here is the
    # config that does or does not fix it" is a result.
    print("\n[breaks] retrying with capture_dynamic_output_shape_ops=True ...")
    try:
        with_capture = _explain(model, dict(inputs), cache_size_limit=256,
                                capture_dynamic_output_shape_ops=True)
        captured_breaks = int(getattr(with_capture, "graph_break_count", 0) or 0)
        result["breaks_with_dynamic_capture"] = captured_breaks
        result["dynamic_capture_reasons"] = _break_reasons(with_capture)[:5]
        removed = break_count - captured_breaks
        print(f"[breaks] with capture : {captured_breaks} breaks "
              f"({removed} of {break_count} removed)")
        if captured_breaks == 0:
            print("[breaks] -> the routing break is REMOVABLE by a config flag, so it is")
            print("            not a structural limitation of MoE routing as the paper")
            print("            frames it. CAVEAT: fewer breaks is not the same as faster.")
            print("            Unbacked symbolic shapes can generate worse code, so this")
            print("            needs a latency measurement before being called a win.")
        elif captured_breaks < break_count:
            print("[breaks] -> partially removable; the remainder has another cause.")
        else:
            print("[breaks] -> the flag does not help; the break is structural after all.")
        result["breaks_removed_by_dynamic_capture"] = removed
    except Exception as exc:  # noqa: BLE001 - the flag is version-dependent
        result["breaks_with_dynamic_capture"] = f"unavailable: {type(exc).__name__}: {exc}"
        print(f"[breaks] capture retry failed: {exc}")

    (output_dir / "real_model_graph_breaks.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[Saved] {output_dir / 'real_model_graph_breaks.json'}")

    save_run_manifest(
        collect_run_manifest(model_name, seed_settings,
                             {"experiment": "real_model_graph_breaks",
                              "log_file": str(log_path), "result": result}),
        output_dir / "run_manifest.json",
    )
    return result


def main() -> int:
    from routingdrift.output_guard import assert_safe_output_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--output_dir", default="results/compiler_rerun")
    args = ap.parse_args()

    output_dir = assert_safe_output_dir(args.output_dir, "compiler analysis")
    analyse(args.model_name, args.revision, args.seq_len, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
