"""GPU: the runtime's fast path (``runtime/fast.py``) reproduces the eager backbone bit for bit.

Run in the pinned image on one leased GPU, with PYTHONPATH at the mirror's
src/training/decision2 and the image's kernel site first on the path:

    python3 -m v2.release.tests.gpu_fast_path --work NEW_DIR [--layers 4]

Builds random backbones at the released sizes' layer dimensions (few layers),
stored as the runtime holds them (BF16-resident Linear weights, everything else
FP32): Qwen3 dense at Kai-0.6B's dims, Qwen3.5 hybrids at Eos-0.8B's and
Nox-4B's (16 / 32 gated-delta value heads) dims, and a Qwen3.5 with an unmerged
PEFT LoRA (rank 16, scaling 2) on every Linear, as Vega-27B. An untouched copy
is the reference. For each backbone the fast path is installed (graphs and
kernels; then kernels only) and right-padded batches of many shapes run three
times each (first use eager, second captured, third replayed), every output
compared with the reference's by ``torch.equal``. Writes RESULT.json; exits
non-zero on any difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

LENGTHS = [
    [8],
    [13],
    [64],
    [347],
    [1000],
    [1024],
    [100, 100],
    [96, 96],
    [37, 200],
    [5, 17, 33],
    [64] * 8,
    [251, 249, 250, 7, 120],
    [512, 512, 300],
]


def rounded(model, torch, lora: bool) -> None:
    """Linear weights rounded to BF16 and held in BF16, as the BF16-resident runtime holds them."""
    with torch.no_grad():
        for name, layer in model.named_modules():
            if type(layer) is not torch.nn.Linear:
                continue
            if lora and ("lora_A" in name or "lora_B" in name):
                continue
            layer.weight.data = layer.weight.data.to(torch.bfloat16)


def qwen3(torch, layers: int):
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=layers,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
        attn_implementation="sdpa",
    )
    model = Qwen3Model(config)
    with torch.no_grad():
        for layer in model.layers:
            for norm in (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            ):
                norm.weight.normal_(1, 0.1)
        model.norm.weight.normal_(1, 0.1)
    return model


def qwen3_5(torch, layers: int, value_heads: int, hidden: int):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    pattern = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    config = Qwen3_5TextConfig(
        vocab_size=151936,
        hidden_size=hidden,
        intermediate_size=3 * hidden + 512,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=value_heads,
        layer_types=[pattern[i % 4] for i in range(layers)],
        max_position_embeddings=8192,
        attn_implementation="sdpa",
    )
    model = Qwen3_5TextModel(config)
    with torch.no_grad():
        for layer in model.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.weight.normal_(0, 0.1)
            if layer.block_type == "full_attention":
                layer.self_attn.q_norm.weight.normal_(0, 0.1)
                layer.self_attn.k_norm.weight.normal_(0, 0.1)
            else:
                layer.linear_attn.norm.weight.normal_(1, 0.1)
        model.norm.weight.normal_(0, 0.1)
    return model


def with_lora(model, torch):
    from peft import LoraConfig, get_peft_model

    names = sorted(
        {
            n.rsplit(".", 1)[-1]
            for n, m in model.named_modules()
            if type(m) is torch.nn.Linear
        }
    )
    peft_model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=names)
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_" in name:
                parameter.normal_(0, 0.02)
    return peft_model


class Holder:
    def __init__(self, backbone):
        self.backbone = backbone


def batch(lengths, torch, generator, device):
    T = math.ceil(max(lengths) / 8) * 8
    ids = torch.randint(0, 151936, (len(lengths), T), generator=generator).to(device)
    mask = torch.zeros(len(lengths), T, dtype=torch.long)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
    return ids, mask.to(device)


def run_case(name, build, torch, fast, layers, graphs) -> dict:
    torch.manual_seed(20261002)
    model = build().float()
    lora = hasattr(model, "get_base_model")
    rounded(model, torch, lora)
    device = torch.device("cuda:0")
    reference = copy.deepcopy(model).to(device).eval()
    candidate = model.to(device).eval()
    holder = Holder(candidate)
    path = fast.install(holder, torch, graphs=graphs, kernels=True)
    if path is None:
        return {"installed": False, "passed": False}
    generator = torch.Generator().manual_seed(7)
    differences, compared = [], 0
    started = time.perf_counter()
    for lengths in LENGTHS:
        for repeat in range(3):
            ids, mask = batch(lengths, torch, generator, device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                want = reference(input_ids=ids, attention_mask=mask, use_cache=False)
                want = want.last_hidden_state.clone()
                with path.forward(lengths):
                    got = candidate(input_ids=ids, attention_mask=mask, use_cache=False)
                got = got.last_hidden_state.clone()
            compared += 1
            if not torch.equal(want, got):
                diff = (want - got).abs()
                differences.append(
                    {
                        "lengths": lengths,
                        "repeat": repeat,
                        "mismatched": int((want != got).sum()),
                        "elements": want.numel(),
                        "max_abs": float(diff.max()),
                    }
                )
    receipt = path.receipt()
    stats = receipt.get("graph_stats") or {}
    checks = {
        "bit_identical": not differences,
        "graphs_replayed": (not graphs) or stats.get("replays", 0) > 0,
        "fused_layers": receipt["fused_layers"] == layers,
        "lean_lora_when_adapter": (not lora) or receipt["lora"]["lean"] > 0,
    }
    return {
        "installed": True,
        "graphs": graphs,
        "layers": layers,
        "compared": compared,
        "differences": differences[:20],
        "fast_path": receipt,
        "seconds": time.perf_counter() - started,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--only")
    args = parser.parse_args()
    import torch

    from v2.release.runtime import fast

    if not torch.cuda.is_available():
        raise SystemExit("needs one GPU")
    args.work.mkdir(parents=True)
    n = args.layers
    cases = {
        "qwen3-0.6b-dims": lambda: qwen3(torch, n),
        "qwen3_5-0.8b-dims": lambda: qwen3_5(torch, n, 16, 1024),
        "qwen3_5-4b-dims": lambda: qwen3_5(torch, n, 32, 2560),
        "qwen3_5-lora": lambda: with_lora(qwen3_5(torch, n, 32, 1024), torch),
    }
    results = {}
    for name, build in cases.items():
        if args.only and args.only not in name:
            continue
        for graphs in (True, False):
            key = f"{name}/{'graphs' if graphs else 'eager'}"
            try:
                results[key] = run_case(name, build, torch, fast, n, graphs)
            except Exception as exc:  # recorded, the run goes on
                results[key] = {"error": repr(exc), "passed": False}
            print(json.dumps({key: results[key].get("passed")}), flush=True)
            torch.cuda.empty_cache()
    passed = bool(results) and all(r["passed"] for r in results.values())
    out = {"cases": results, "passed": passed}
    (args.work / "RESULT.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({"passed": passed}))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
