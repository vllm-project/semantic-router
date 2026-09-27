"""One bounded, no-model 9B ROCm backward operator isolation.

This is a diagnostic for the frozen 9B training HOLD, not a training runner.
The GPU mode uses the exact pinned Qwen3.5 reference gated-delta function and
PyTorch SDPA at the official 9B layer shapes. A successful standalone result
cannot exonerate their interaction in the complete transformer.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 20260927
LENGTHS = (4096, 6144, 6144, 6144)
CONFIG_SHA256 = "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
MODELING_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
REFERENCE_SHA256 = "e4116a769851c6a455d1a277f0b5ad777090d73c9da491a84e4e9773e17c45d6"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_preflight(config_path: Path):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    if digest(config_path.read_bytes()) != CONFIG_SHA256:
        raise ValueError("Official 9B config differs from the frozen source")
    config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        config["num_hidden_layers"] != 32
        or config["layer_types"].count("linear_attention") != 24
        or config["layer_types"].count("full_attention") != 8
        or config["linear_num_key_heads"] != 16
        or config["linear_num_value_heads"] != 32
        or config["linear_key_head_dim"] != 128
        or config["linear_value_head_dim"] != 128
        or config["num_attention_heads"] != 16
        or config["num_key_value_heads"] != 4
        or config["head_dim"] != 256
    ):
        raise ValueError("Official 9B operator shapes differ from the frozen protocol")
    if digest(Path(modeling.__file__).read_bytes()) != MODELING_SHA256:
        raise ValueError("Pinned Qwen3.5 modeling code differs")
    reference = modeling.torch_chunk_gated_delta_rule.__wrapped__
    if digest(inspect.getsource(reference).encode("utf-8")) != REFERENCE_SHA256:
        raise ValueError("Qwen3.5 PyTorch reference function differs")
    return reference


def emit(**fields) -> None:
    print(json.dumps(fields, sort_keys=True, allow_nan=False), flush=True)


def inputs(case: str, length: int, device: torch.device):
    if case == "reference_gated_delta":
        shape = (1, length, 32, 128)
        query = torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_()
        key = torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_()
        value = torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_()
        decay = (-0.5 - torch.rand((1, length, 32), device=device)).requires_grad_()
        beta = torch.rand((1, length, 32), device=device).requires_grad_()
        return query, key, value, decay, beta
    if case == "sdpa":
        query = torch.randn(
            (1, 16, length, 256), device=device, dtype=torch.bfloat16
        ).requires_grad_()
        key = torch.randn(
            (1, 4, length, 256), device=device, dtype=torch.bfloat16
        ).requires_grad_()
        value = torch.randn(
            (1, 4, length, 256), device=device, dtype=torch.bfloat16
        ).requires_grad_()
        return query, key, value
    raise ValueError(f"Unknown isolation case: {case}")


def one_iteration(case: str, length: int, reference, device: torch.device) -> float:
    values = inputs(case, length, device)
    if case == "reference_gated_delta":
        query, key, value, decay, beta = values
        output, final_state = reference(
            query,
            key,
            value,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        if final_state is not None:
            raise RuntimeError("Unexpected cached recurrent state")
    else:
        query, key, value = values
        output = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(4, dim=1),
            value.repeat_interleave(4, dim=1),
            is_causal=True,
            dropout_p=0.0,
        )
    loss = output.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite isolated operator loss")
    emit(event="backward_start", case=case, length=length)
    gradients = torch.autograd.grad(loss, values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Nonfinite isolated operator gradient")
    return float(loss.item())


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    reference = source_preflight(args.config)
    if args.cpu_smoke:
        torch.manual_seed(SEED)
        # A tiny shape validates imports and both autograd paths, not GPU behavior.
        device = torch.device("cpu")
        for case in ("reference_gated_delta", "sdpa"):
            loss = one_iteration(case, 64, reference, device)
            emit(event="cpu_smoke_pass", case=case, loss=loss)
        return
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable accelerator must be visible")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    emit(
        event="start",
        seed=SEED,
        schedule=list(LENGTHS),
        cases=["reference_gated_delta", "sdpa"],
        optimizer=None,
        model_loaded=False,
        torch_version=torch.__version__,
        hip_version=torch.version.hip,
    )
    started = time.monotonic()
    for case in ("reference_gated_delta", "sdpa"):
        for index, length in enumerate(LENGTHS, start=1):
            emit(event="iteration_start", case=case, iteration=index, length=length)
            loss = one_iteration(case, length, reference, device)
            emit(
                event="iteration_complete",
                case=case,
                iteration=index,
                length=length,
                loss=loss,
                peak_allocated_gib=round(
                    torch.cuda.max_memory_allocated(device) / math.pow(2, 30), 3
                ),
            )
        torch.cuda.empty_cache()
    emit(event="complete", iterations=8, seconds=round(time.monotonic() - started, 3))


if __name__ == "__main__":
    main()
