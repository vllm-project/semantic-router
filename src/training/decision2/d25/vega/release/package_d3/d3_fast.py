"""ROCm fast path of the d3 runtime: same function, same weights, less host work.

``d3_runtime.D3`` installs it on AMD GPUs (``torch.version.hip``) for Qwen3.5 backbones with noncausal full
attention; every other device and model keeps the plain path, and ``D3_FAST=0`` turns it off.

- Text forward passes replay HIP graphs captured at warm-up. A pass (one batch of questions, left-padded to
  its longest prompt exactly as in the plain path) is extended on the right with masked padding to a length
  bucket and replays the graph of its (questions, bucket) shape. The real tokens keep their positions,
  chunk boundaries and convolution taps; the right padding is masked out of the full-attention keys and
  comes after every real token in the causal Gated DeltaNet layers; the readout reads each prompt's last
  real token. Passes over the graph budget (``D3_GRAPH_TOKENS`` rows x bucket tokens) run eagerly at their
  own shape.
- Eager passes build both attention masks from the host-known prompt lengths, so nothing inside the forward
  pass waits for the GPU.
"""

from __future__ import annotations

import os
import time
from typing import Any

GRAPH_TOKENS = 0
BUCKET = 64
CAPTURE_WARM_RUNS = 2


def enabled(model: Any) -> str | None:
    """None when the fast path applies to this loaded model, else the reason it does not."""
    torch = model.torch
    if os.environ.get("D3_FAST", "").strip().lower() in ("0", "false", "no", "off"):
        return "D3_FAST=0"
    if model.device.type != "cuda" or not getattr(torch.version, "hip", None):
        return "not a ROCm GPU"
    if model.attention_mode != "noncausal_full_attention":
        return f"attention mode {model.attention_mode}"
    if type(model.backbone).__name__ != "Qwen3_5Model":
        return f"backbone {type(model.backbone).__name__}"
    config = model.backbone.language_model.config
    if set(config.layer_types[: config.num_hidden_layers]) - {
        "full_attention",
        "linear_attention",
    }:
        return "layer types other than full / linear attention"
    return None


def fusable(model: Any) -> str | None:
    """None when the fused decoder-layer kernels (``d3_kernels.py``) apply to this model, else the reason."""
    torch = model.torch
    arch = torch.cuda.get_device_properties(model.device).gcnArchName.split(":")[0]
    if arch != "gfx942":
        return f"fused kernels verified on gfx942 only, not {arch}"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "triton is not installed"
    config = model.backbone.language_model.config
    chunk = model.kernels.get("torch_chunk_gated_delta_rule", "")
    checks = {
        "hidden size a multiple of 256": config.hidden_size % 256 == 0,
        "128-wide gated-delta heads": config.linear_key_head_dim == 128
        and config.linear_value_head_dim == 128,
        "256-wide attention heads": getattr(config, "head_dim", None) == 256,
        "4-tap convolution": config.linear_conv_kernel_dim == 4,
        "SiLU MLP": config.hidden_act == "silu",
        "FLA chunk kernel": chunk.startswith("fla"),
        "SDPA attention": config._attn_implementation == "sdpa",
    }
    missing = [name for name, ok in checks.items() if not ok]
    return "needs " + ", ".join(missing) if missing else None


def kernels_next_to_this_file():
    """``d3_kernels.py`` of this file's directory, imported by path once per file (see ``d3_runtime.fast_module``)."""
    import hashlib
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().with_name("d3_kernels.py")
    name = f"d3_kernels_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


class FusedLayers:
    """The decoder layers and final norm of a Qwen3.5 text model through ``d3_kernels`` (eager values)."""

    def __init__(self, lm: Any, torch: Any):
        import importlib

        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

        try:
            from . import d3_kernels as kernels
        except ImportError:
            try:
                kernels = importlib.import_module("d3_kernels")
            except ImportError:
                kernels = kernels_next_to_this_file()
        self.k = kernels
        self.torch = torch
        self.chunk = modeling.torch_chunk_gated_delta_rule
        self.repeat_kv = modeling.repeat_kv
        config = lm.config
        self.layers = list(lm.layers[: config.num_hidden_layers])
        self.params = []
        with torch.no_grad():
            for layer in self.layers:
                p = {
                    "linear": layer.block_type == "linear_attention",
                    "eps": layer.input_layernorm.eps,
                    "w1_in": (1.0 + layer.input_layernorm.weight.float()).contiguous(),
                    "w1_post": (
                        1.0 + layer.post_attention_layernorm.weight.float()
                    ).contiguous(),
                }
                if p["linear"]:
                    m = layer.linear_attn
                    p.update(
                        conv_w=m.conv1d.weight.squeeze(1).contiguous(),
                        A_log=m.A_log.float().contiguous(),
                        dt_bias=m.dt_bias.float().contiguous(),
                    )
                else:
                    m = layer.self_attn
                    p.update(
                        qw1=(1.0 + m.q_norm.weight.float()).contiguous(),
                        kw1=(1.0 + m.k_norm.weight.float()).contiguous(),
                    )
                self.params.append(p)
            self.final_eps = lm.norm.eps
            self.final_w1 = (1.0 + lm.norm.weight.float()).contiguous()

    def gated_delta(self, m: Any, p: dict[str, Any], x: Any) -> Any:
        k = self.k
        rows, length, _ = x.shape
        q, key, v, g, beta = k.gdn_prep(
            m.in_proj_qkv(x),
            m.in_proj_b(x),
            m.in_proj_a(x),
            p["conv_w"],
            p["A_log"],
            p["dt_bias"],
            m.num_k_heads,
            m.head_k_dim,
        )
        core, _ = self.chunk(
            q,
            key,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            use_cache=False,
        )
        out = k.gated_rmsnorm(
            core.reshape(-1, m.head_v_dim),
            m.in_proj_z(x).reshape(-1, m.head_v_dim),
            m.norm.weight,
            m.norm.variance_epsilon,
        )
        return m.out_proj(out.reshape(rows, length, -1))

    def attention(self, m: Any, p: dict[str, Any], x: Any, rotary, full_mask) -> Any:
        rows, length, _ = x.shape
        hd = m.head_dim
        qp = m.q_proj(x)
        kp = m.k_proj(x)
        heads = qp.shape[-1] // (2 * hd)
        kv_heads = kp.shape[-1] // hd
        cos, sin = rotary
        q, key = self.k.attn_prep(
            qp, kp, p["qw1"], p["kw1"], cos, sin, heads, kv_heads, hd, m.q_norm.eps
        )
        v = self.repeat_kv(
            m.v_proj(x).view(rows, length, kv_heads, hd).transpose(1, 2),
            heads // kv_heads,
        )
        attn = self.torch.nn.functional.scaled_dot_product_attention(
            q,
            key,
            v,
            attn_mask=full_mask,
            dropout_p=0.0,
            scale=m.scaling,
            is_causal=False,
        )
        gate = qp.view(rows, length, heads, 2 * hd)[..., hd:]
        return m.o_proj(self.k.sigmoid_gate(attn.transpose(1, 2), gate))

    def forward(self, embeds: Any, full_mask: Any, rowmask: Any | None, rotary) -> Any:
        """Final-norm hidden states [B, L, D]; ``rowmask`` marks real tokens when the batch is padded."""
        k = self.k
        hidden, delta = embeds, None
        for layer, p in zip(self.layers, self.params):
            hidden, x = k.add_rmsnorm(
                hidden, delta, p["w1_in"], p["eps"], rowmask if p["linear"] else None
            )
            if p["linear"]:
                delta = self.gated_delta(layer.linear_attn, p, x)
            else:
                delta = self.attention(layer.self_attn, p, x, rotary, full_mask)
            hidden, x = k.add_rmsnorm(hidden, delta, p["w1_post"], p["eps"])
            mlp = layer.mlp
            delta = mlp.down_proj(k.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))
        return k.add_rmsnorm(hidden, delta, self.final_w1, self.final_eps)[1]


class FastText:
    """Text passes of one loaded model: HIP graphs per (rows, bucket) plus a synchronization-free eager pass."""

    def __init__(
        self,
        model: Any,
        *,
        graph_tokens: int,
        bucket: int,
        max_rows: int,
        fused: FusedLayers | None = None,
        fused_skipped: str | None = None,
    ):
        torch = model.torch
        self.model = model
        self.torch = torch
        self.fused = fused
        self.fused_skipped = fused_skipped
        self.cpu_threads: dict[str, int] | None = None
        self.lm = model.backbone.language_model
        config = self.lm.config
        self.layers = list(self.lm.layers[: config.num_hidden_layers])
        self.kinds = list(config.layer_types[: config.num_hidden_layers])
        self.graph_tokens = graph_tokens
        self.bucket = bucket
        self.max_rows = max_rows
        self.codes = torch.arange(model.readout.shape[0], device=model.device)
        self.graphs: dict[tuple[int, int], dict[str, Any]] = {}
        self.failed: dict[tuple[int, int], str] = {}
        self.pool = None
        self.stats = {"replays": 0, "eager": 0, "captures": 0, "capture_seconds": 0.0}

    # ------------------------------------------------------------------ forward pieces

    def hidden(self, ids, mask, linear_mask):
        """Final-norm hidden states [B, L, D]: ``Qwen3_5Model.forward`` for text with the d3 masks given."""
        torch = self.torch
        lm = self.lm
        embeds = lm.embed_tokens(ids)
        rows, length = ids.shape
        positions = torch.arange(length, device=ids.device).view(1, 1, -1)
        positions = positions.expand(4, rows, -1)
        text_positions, positions = positions[0], positions[1:]
        rotary = lm.rotary_emb(embeds, positions)
        full = mask.bool()[:, None, None, :]
        if self.fused is not None:
            rowmask = None if linear_mask is None else linear_mask.reshape(-1)
            return self.fused.forward(embeds, full, rowmask, rotary)
        hidden = embeds
        for layer, kind in zip(self.layers, self.kinds):
            hidden = layer(
                hidden,
                position_embeddings=rotary,
                attention_mask=full if kind == "full_attention" else linear_mask,
                position_ids=text_positions,
                past_key_values=None,
                use_cache=False,
            )
        return lm.norm(hidden)

    def image_hidden(self, inputs: dict[str, Any], padded: bool):
        """Final-norm hidden states of an image batch: ``Qwen3_5Model.forward`` with the fused text layers."""
        torch = self.torch
        backbone = self.model.backbone
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        embeds = backbone.get_input_embeddings()(ids)
        features = backbone.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        ).pooler_output
        features = torch.cat(features, dim=0).to(embeds.device, embeds.dtype)
        image_mask, _ = backbone.get_placeholder_mask(
            ids, inputs_embeds=embeds, image_features=features
        )
        embeds = embeds.masked_scatter(image_mask, features)
        positions = backbone.compute_3d_position_ids(
            input_ids=ids,
            image_grid_thw=inputs["image_grid_thw"],
            inputs_embeds=embeds,
            attention_mask=mask,
            past_key_values=None,
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        rotary = self.lm.rotary_emb(embeds, positions)
        full = mask.bool()[:, None, None, :]
        return self.fused.forward(
            embeds, full, mask.reshape(-1) if padded else None, rotary
        )

    def probabilities_of(self, last, counts):
        """``D3.logits`` -> temperature -> softmax on the last-token hidden states [B, D]."""
        model = self.model
        if model.readout_dtype == "float32":
            logits = last.float() @ model.readout.T
        else:
            logits = self.torch.nn.functional.linear(last, model.readout).float()
        invalid = self.codes[None] >= counts[:, None]
        return (logits.masked_fill(invalid, float("-inf")) / model.temperature).softmax(
            -1
        )

    # ------------------------------------------------------------------ passes

    def bucket_of(self, rows: int, width: int) -> int | None:
        size = -(-width // self.bucket) * self.bucket
        if rows > self.max_rows or rows * size > self.graph_tokens:
            return None
        return size

    def probabilities(self, sequences, counts) -> list[list[float]]:
        torch = self.torch
        rows = len(sequences)
        width = max(len(s) for s in sequences)
        size = self.bucket_of(rows, width)
        entry = self.graphs.get((rows, size)) if size is not None else None
        length = size if entry is not None else width
        ids = torch.full((rows, length), self.model.pad_id, dtype=torch.long)
        mask = torch.zeros((rows, length), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            ids[i, width - len(sequence) : width] = torch.as_tensor(
                sequence, dtype=torch.long
            )
            mask[i, width - len(sequence) : width] = 1
        counts_cpu = torch.as_tensor(list(counts), dtype=torch.long)
        if entry is not None:
            entry["ids"].copy_(ids, non_blocking=True)
            entry["mask"].copy_(mask, non_blocking=True)
            entry["counts"].copy_(counts_cpu, non_blocking=True)
            entry["last"].fill_(width - 1)
            entry["graph"].replay()
            self.stats["replays"] += 1
            probs = entry["probs"].cpu().tolist()
        else:
            device = self.model.device
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            padded = any(len(s) != width for s in sequences)
            with torch.inference_mode():
                last = self.hidden(ids, mask, mask if padded else None)[:, -1]
                probs = (
                    self.probabilities_of(last, counts_cpu.to(device)).cpu().tolist()
                )
            self.stats["eager"] += 1
        return [p[:c] for p, c in zip(probs, counts)]

    # ------------------------------------------------------------------ graphs

    def shapes(self) -> list[tuple[int, int]]:
        out = []
        for rows in range(1, self.max_rows + 1):
            size = self.bucket
            while rows * size <= self.graph_tokens:
                out.append((rows, size))
                size += self.bucket
        return out

    def capture(self, rows: int, size: int) -> bool:
        torch = self.torch
        key = (rows, size)
        if key in self.graphs:
            return True
        device = self.model.device
        token = self.model.token_ids[0]
        static = {
            "ids": torch.full((rows, size), token, dtype=torch.long, device=device),
            "mask": torch.ones((rows, size), dtype=torch.long, device=device),
            "counts": torch.full((rows,), 2, dtype=torch.long, device=device),
            "last": torch.full((1,), size - 1, dtype=torch.long, device=device),
        }

        def body():
            hidden = self.hidden(static["ids"], static["mask"], static["mask"])
            last = hidden.index_select(1, static["last"]).squeeze(1)
            return self.probabilities_of(last, static["counts"])

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                if self.pool is None:
                    self.pool = torch.cuda.graph_pool_handle()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(CAPTURE_WARM_RUNS):
                        body()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    probs = body()
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 - the shape then runs eagerly
            torch.cuda.synchronize(device)
            self.failed[key] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return False
        self.graphs[key] = {**static, "graph": graph, "probs": probs}
        self.stats["captures"] += 1
        self.stats["capture_seconds"] += time.perf_counter() - started
        return True

    def capture_all(self) -> float:
        started = time.perf_counter()
        for rows, size in self.shapes():
            self.capture(rows, size)
        return time.perf_counter() - started

    def report(self) -> dict[str, Any]:
        return {
            "fused_layers": len(self.fused.layers) if self.fused is not None else 0,
            "fused_skipped": self.fused_skipped,
            "cpu_threads": self.cpu_threads,
            "graph_tokens": self.graph_tokens,
            "bucket": self.bucket,
            "max_rows": self.max_rows,
            "graphs": len(self.graphs),
            "failed": dict(list(self.failed.items())[:5]),
            "failed_count": len(self.failed),
            **{
                k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in self.stats.items()
            },
        }


def dedupe_image_processing(processor: Any, torch: Any) -> None:
    """Process each distinct image of one processor call once and repeat its rows for the other copies.

    The runtime hands the processor one copy of the request's images per question; every copy gives the same
    pixel rows and grid, so the outputs are identical, only the repeated resizing is skipped.
    """
    original = processor._process_images

    def process_images(images, **kwargs):
        flat = list(images) if isinstance(images, (list, tuple)) else [images]
        index: dict[int, int] = {}
        unique, order = [], []
        for image in flat:
            order.append(index.setdefault(id(image), len(unique)))
            if len(unique) < len(index):
                unique.append(image)
        if len(unique) == len(flat):
            return original(images, **kwargs)
        processed = processor.image_processor(unique, **kwargs)
        if set(processed.keys()) != {"pixel_values", "image_grid_thw"}:
            return original(images, **kwargs)
        grid = processed["image_grid_thw"]
        rows = torch.split(processed["pixel_values"], grid.prod(-1).tolist())
        processed["pixel_values"] = torch.cat([rows[i] for i in order])
        processed["image_grid_thw"] = grid[order]
        replacements = [
            processor.replace_image_token(processed, image_idx=i, **kwargs)
            for i in range(len(flat))
        ]
        return processed, replacements

    processor._process_images = process_images


def cpu_quota() -> int | None:
    """CPUs this process may use: the cgroup CPU quota (containers), else the affinity mask."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as stream:
            quota, period = stream.read().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def cap_cpu_threads(torch: Any) -> dict[str, int] | None:
    """Keep torch's CPU threads within the container's CPU quota when OMP_NUM_THREADS is not set.

    torch sizes its pool by the host's CPU count; in a container with a smaller quota the image
    preprocessing then oversubscribes it and the idle-spinning workers starve the thread that feeds the GPU.
    """
    if "OMP_NUM_THREADS" in os.environ:
        return None
    limit, current = cpu_quota(), torch.get_num_threads()
    if limit is None or current <= limit:
        return None
    torch.set_num_threads(limit)
    return {"from": current, "to": limit}


def install(model: Any) -> FastText | None:
    """The fast path for a loaded ``D3`` model, or None (the reason is in ``model.fast_skipped``)."""
    reason = enabled(model)
    if reason is not None:
        model.fast_skipped = reason
        return None
    graph_tokens = int(os.environ.get("D3_GRAPH_TOKENS", GRAPH_TOKENS))
    bucket = int(os.environ.get("D3_GRAPH_BUCKET", BUCKET))
    if model.processor is not None and hasattr(model.processor, "_process_images"):
        dedupe_image_processing(model.processor, model.torch)
    fused, skipped = None, None
    if os.environ.get("D3_FUSED", "").strip().lower() in ("0", "false", "no", "off"):
        skipped = "D3_FUSED=0"
    else:
        skipped = fusable(model)
    if skipped is None:
        fused = FusedLayers(model.backbone.language_model, model.torch)
    fast = FastText(
        model,
        graph_tokens=graph_tokens,
        bucket=bucket,
        max_rows=model.batch_size,
        fused=fused,
        fused_skipped=skipped,
    )
    fast.cpu_threads = cap_cpu_threads(model.torch)
    return fast
