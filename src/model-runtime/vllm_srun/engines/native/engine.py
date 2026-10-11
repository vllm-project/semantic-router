"""The native PyTorch engine: builds a backbone from a ``ModelSpec`` and runs it on one device."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar, cast

import torch
from torch import nn

from ...accel import onednn
from ...accel.kernels import CONTIGUOUS
from ...plugins.base import (
    Accelerator,
    BackboneSpec,
    DeviceInfo,
    EncoderBatch,
    EncoderOutput,
    Engine,
    EngineModel,
    EngineOptions,
    ForwardBatch,
    ForwardOutput,
    ModelSpec,
    ModuleT,
    TreeBatch,
    TreeOutput,
)
from ...scheduler.planner import padded
from . import fast, models
from . import threads as adaptive_threads
from .encoder import EncoderGraphs
from .models.forest import ForestShape
from .models.lora import attach
from .models.tree import Tree
from .reduced import REDUCED_AUTOCAST, linear_bytes, reduced_view, unavailable
from .weights import (
    cast_parameters,
    keep_linear_bf16,
    lay_out_linears,
    load_adapter,
    load_backbone,
)

_T = TypeVar("_T")

# FLA's gated-delta kernels index q / k / v with 32-bit offsets: one forward's stay within 2**30 elements.
GATED_DELTA_ELEMENTS = 2**30 - 1


def graph_receipt(graphs: dict[str | None, EncoderGraphs]) -> dict[str, Any]:
    """The backbone's graph statistics, each branch's under ``branches``."""
    out = graphs[None].receipt()
    branches = {name: g.receipt() for name, g in graphs.items() if name is not None}
    if branches:
        out["branches"] = branches
    return out


class NativeEngineModel(EngineModel):
    spec: ModelSpec

    def __init__(
        self,
        backbone: nn.Module,
        accelerator: Accelerator,
        device_info: DeviceInfo,
        spec: ModelSpec,
        options: EngineOptions,
        residency: dict[str, int] | None,
        branches: dict[str, nn.Module] | None = None,
        reduced_kind: str | None = None,
        towers: dict[str, nn.Module] | None = None,
    ):
        """``reduced_kind`` asks for a reduced copy of every layer stack, which approximate batches run (``reduced.py``).

        A copy the device cannot run is skipped, and the receipt says why.
        ``towers`` are the model's other backbones (``ModelSpec.towers``).
        """
        self.backbone = backbone
        self.branches = branches or {}
        self.towers = towers or {}
        self.accelerator = accelerator
        self.device_info = device_info
        self.device = accelerator.torch_device(device_info)
        self.spec = spec
        self.options = options
        self.residency = residency
        self.reduced_kind = reduced_kind
        self.reduced_skipped = (
            None
            if reduced_kind is None
            else unavailable(reduced_kind, self.device, device_info.bf16)
        )
        # Views are taken before the linears are laid out: they copy nn.Linear layers.
        self.reduced: dict[str | None, nn.Module] = (
            {}
            if reduced_kind is None or self.reduced_skipped
            else {
                name: reduced_view(module, reduced_kind)
                for name, module in self.stacks().items()
            }
        )
        self.kernels = accelerator.kernels(device_info)
        self.kernels.allow_approximate = (
            not options.exact_kernels_only and spec.dtype.approximate_kernels
        )
        self.kernels.use_variants(spec.kernel_variants)
        self.linear = self.kernels.select("linear")
        if self.linear.variant is not None:
            for module in (backbone, *self.branches.values(), *self.towers.values()):
                lay_out_linears(module, self.linear.fn)
        # Packed linears, contiguous GeGLU, norms and unpadded attention grids
        # (``packed(uniform=True)``) compute each row the same in any batch.
        self.batch_invariant = (
            spec.encoder
            and self.device.type == "cpu"
            and self.linear.variant == onednn.PACKED
            and self.kernels.select("geglu").variant == CONTIGUOUS
        )
        for module in (
            *self.stacks().values(),
            *self.reduced.values(),
            *self.towers.values(),
        ):
            cast(models.Backbone, module).kernels = self.kernels
        self.fast: dict[str, Any] = {}
        self.masks: fast.Masks | None = None
        self.graphs: fast.Graphs | None = None
        self._cpu_threads = options.threads or torch.get_num_threads()
        self.threads_resolver = (
            adaptive_threads.AdaptiveThreads(self._cpu_threads)
            if self.device.type == "cpu" and adaptive_threads.enabled()
            else None
        )
        self.encoder_graphs: dict[str | None, EncoderGraphs] = {}
        self.reduced_graphs: dict[str | None, EncoderGraphs] = {}
        if self.device.type == "cuda" and not spec.encoder:
            self._install_fast()
        if self.device.type == "cuda" and spec.encoder and options.graphs:
            self.encoder_graphs = {
                name: EncoderGraphs(cast(models.EncoderBackbone, module), self.device)
                for name, module in self.stacks().items()
            }
            self.reduced_graphs = {
                name: EncoderGraphs(cast(models.EncoderBackbone, view), self.device)
                for name, view in self.reduced.items()
            }

    def stacks(self) -> dict[str | None, nn.Module]:
        """Every layer stack by branch name; ``None`` is the backbone's own."""
        return {None: self.backbone} | self.branches

    def _install_fast(self) -> None:
        """The exact GPU fast path (``fast.py``): fused layers, lean LoRA, host masks and graphs."""
        options, backbone = self.options, self.backbone
        if options.fused_kernels:
            reason = fast.fused_unavailable(backbone, self.kernels)
            self.fast["fused_layers"] = 0 if reason else fast.install_fused(backbone)
            if reason:
                self.fast["fused_skipped"] = reason
        if self.spec.backbone.lora is not None and self.residency is not None:
            self.fast["lora"] = fast.install_lean_lora(backbone)
        if options.fused_kernels or options.graphs:
            self.masks = fast.Masks()
        if options.graphs:
            assert self.masks is not None
            self.graphs = fast.Graphs(backbone, self.masks)

    def receipt(self) -> dict[str, Any]:
        """What runs this model: kernels, fast-path pieces and graph statistics."""
        out: dict[str, Any] = {"kernels": self.kernels.describe(), **self.fast}
        if self.graphs is not None:
            out["graphs"] = self.graphs.receipt()
        if self.encoder_graphs:
            out["encoder_graphs"] = graph_receipt(self.encoder_graphs)
        if self.reduced:
            out["reduced"] = {
                "kind": self.reduced_kind,
                "bytes": sum(linear_bytes(view) for view in self.reduced.values()),
                **(
                    {"encoder_graphs": graph_receipt(self.reduced_graphs)}
                    if self.reduced_graphs
                    else {}
                ),
            }
        elif self.reduced_skipped:
            out["reduced"] = {
                "kind": self.reduced_kind,
                "skipped": self.reduced_skipped,
            }
        return out

    def autocast(self) -> AbstractContextManager[Any]:
        if self.device.type == "cpu":
            from contextlib import nullcontext

            return nullcontext()
        return self.accelerator.autocast(self.device_info, self.spec.dtype.autocast)

    def reduced_autocast(self) -> AbstractContextManager[Any]:
        """The reduced copy's compute dtype: BF16 autocast for BF16 copies, none otherwise."""
        dtype = REDUCED_AUTOCAST.get(self.reduced_kind or "")
        if dtype is None:
            from contextlib import nullcontext

            return nullcontext()
        if self.device.type == "cpu":
            return torch.autocast("cpu", dtype=dtype)
        return self.accelerator.autocast(self.device_info, "bfloat16")

    supports_shared_context = True

    @property
    def replays_graphs(self) -> bool:
        return self.graphs is not None

    def _run_tree(
        self, prefix: torch.Tensor, suffixes: list[torch.Tensor], padded_exact: bool
    ) -> torch.Tensor:
        """The prefix once and every suffix from it, packed in one row; the suffix rows ``[n, width, H]``."""
        lengths = [len(suffix) for suffix in suffixes]
        packed = torch.cat([prefix, *suffixes])[None].to(self.device)
        tree = Tree(
            len(prefix), lengths, padded(max(lengths)), self.device, padded_exact
        )
        hidden = cast(models.TreeBackbone, self.backbone).forward_tree(packed, tree)
        return tree.rows(hidden[0, len(prefix) :])

    def _forward_tree(self, batch: ForwardBatch) -> ForwardOutput:
        """The batch as one shared-context tree: its first ``shared_prefix`` tokens computed once."""
        prefix, lengths = batch.shared_prefix, batch.lengths
        ids = batch.input_ids
        gather = (batch.gather - prefix).clamp(min=0).to(self.device)
        query = (batch.query - prefix).to(self.device)
        with torch.inference_mode(), self.autocast():
            rows_hidden = self._run_tree(
                ids[0, :prefix],
                [ids[row, prefix:length] for row, length in enumerate(lengths)],
                padded_exact=any(length != padded(max(lengths)) for length in lengths),
            )
            rows = torch.arange(rows_hidden.shape[0], device=rows_hidden.device)
            gathered = rows_hidden[rows[:, None], gather]
            queried = rows_hidden[rows, query]
        return ForwardOutput(gathered=gathered, query=queried)

    def tree(self, batch: TreeBatch) -> TreeOutput:
        method = "forward_forest" if batch.layout == "rows" else "forward_tree"
        if not hasattr(self.backbone, method):
            raise NotImplementedError(
                f"the native {self.spec.backbone.model_type!r} backbone has no {batch.layout} tree forward"
            )
        width = max(len(block) for block in batch.blocks)
        with torch.inference_mode(), self.autocast():
            if batch.layout == "rows":
                return TreeOutput(hidden=self._tree_rows(batch))
            hidden = None
            for owner, prefix in enumerate(batch.prefixes):
                members = [i for i, o in enumerate(batch.owners) if o == owner]
                if not members:
                    continue
                rows = self._run_tree(
                    torch.tensor(prefix, dtype=torch.long),
                    [torch.tensor(batch.blocks[i], dtype=torch.long) for i in members],
                    padded_exact=False,
                )
                if hidden is None:
                    hidden = rows.new_zeros((len(batch.blocks), width, rows.shape[-1]))
                span = min(width, rows.shape[1])
                hidden[members, :span] = rows[:, :span]
        assert hidden is not None
        return TreeOutput(hidden=hidden)

    def _tree_rows(self, batch: TreeBatch) -> torch.Tensor:
        """``layout: rows``: left-padded prefix rows and right-padded blocks (``models/forest.py``)."""

        def padded_rows(
            sequences: list[list[int]], left: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            width = max(len(sequence) for sequence in sequences)
            ids = torch.zeros((len(sequences), width), dtype=torch.long)
            mask = torch.zeros((len(sequences), width), dtype=torch.long)
            for row, sequence in enumerate(sequences):
                cut = (
                    slice(width - len(sequence), width)
                    if left
                    else slice(0, len(sequence))
                )
                ids[row, cut] = torch.tensor(sequence, dtype=torch.long)
                mask[row, cut] = 1
            return ids.to(self.device), mask.to(self.device)

        prefix_ids, prefix_mask = padded_rows(batch.prefixes, left=True)
        block_ids, block_mask = padded_rows(batch.blocks, left=False)
        width = prefix_ids.shape[1]
        shape = ForestShape(
            tuple(width - len(prefix) for prefix in batch.prefixes),
            tuple(len(block) for block in batch.blocks),
            tuple(batch.owners),
        )
        owner = torch.tensor(batch.owners, dtype=torch.long, device=self.device)
        _, blocks = cast(models.ForestBackbone, self.backbone).forward_forest(
            prefix_ids, prefix_mask, block_ids, block_mask, owner, shape
        )
        return blocks

    def begin_traffic(self) -> None:
        """Start thread exploration; called once the startup golden check has passed."""
        if self.threads_resolver is not None:
            self.threads_resolver.start()

    @staticmethod
    def _bit_equal(a: Any, b: Any) -> bool:
        """Bit-for-bit equality over the engines' output dataclasses."""
        if type(a) is not type(b):
            return False
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(
                NativeEngineModel._bit_equal(x, y) for x, y in zip(a, b, strict=True)
            )
        fields = getattr(a, "__dataclass_fields__", None)
        if fields:
            return all(
                NativeEngineModel._bit_equal(getattr(a, f), getattr(b, f))
                for f in fields
            )
        return bool(a == b)

    def _with_threads(self, tokens: int, run: Callable[[], _T]) -> _T:
        """Run one batch under the resolver's thread discipline.

        Exploration never changes an answer: the batch is served by the
        configured count, and any other allowed count only runs as a shadow
        pass that has to reproduce the served answer bit for bit before its
        timing counts. Once the resolver has adopted counts, the learned
        count serves the batch itself — only counts that kept the answers
        identical on this host were adopted.
        """
        resolver = self.threads_resolver
        if resolver is None:
            if self.device.type == "cpu":
                adaptive_threads.CpuTeam.apply(self._cpu_threads)
            return run()
        count = resolver.pick(tokens)
        if resolver.adopted:
            adaptive_threads.CpuTeam.apply(count)
            return run()
        base = resolver.base
        # Exploring: the configured count serves; another count only shadows.
        adaptive_threads.CpuTeam.apply(base)
        started = time.perf_counter()
        try:
            result = run()
        finally:
            base_ms = (time.perf_counter() - started) * 1000.0
        if count == base:
            resolver.record(tokens, base_ms)
            return result
        adaptive_threads.CpuTeam.apply(count)
        started = time.perf_counter()
        try:
            shadow = run()
        finally:
            shadow_ms = (time.perf_counter() - started) * 1000.0
        resolver.record(tokens, shadow_ms, exact=self._bit_equal(result, shadow))
        return result

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        return self._with_threads(batch.input_ids.numel(), lambda: self._forward(batch))

    def _forward(self, batch: ForwardBatch) -> ForwardOutput:
        if batch.shared_prefix:
            return self._forward_tree(batch)
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        gather = batch.gather.to(self.device)
        query = batch.query.to(self.device)
        with torch.inference_mode(), self.autocast():
            if self.graphs is not None:
                hidden = self.graphs(input_ids, attention_mask, batch.lengths)
            elif self.masks is not None:
                padded = any(n != input_ids.shape[1] for n in batch.lengths)
                hidden = self.backbone(
                    input_ids,
                    attention_mask,
                    masks=self.masks.build(attention_mask, padded),
                )
            else:
                hidden = self.backbone(input_ids, attention_mask)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            gathered = hidden[rows[:, None], gather]
            queried = hidden[rows, query]
        return ForwardOutput(gathered=gathered, query=queried)

    def encode(self, batch: EncoderBatch) -> EncoderOutput:
        """Hidden states at the batch's exits, through its branch if it names one.

        Packed rows stay packed and padded rows padded. A reduced batch runs its
        stack's reduced copy when one is loaded, and every stack has its graphs.
        A batch that names a tower runs it on its named inputs.
        """
        return self._with_threads(batch.input_ids.numel(), lambda: self._encode(batch))

    def _encode(self, batch: EncoderBatch) -> EncoderOutput:
        if batch.tower is not None:
            return self._encode_tower(batch)
        stack = self.backbone if batch.branch is None else self.branches[batch.branch]
        if not hasattr(stack, "encode"):
            return self._encode_decoder(batch)
        reduced = batch.reduced and batch.branch in self.reduced
        if reduced:
            stack = self.reduced[batch.branch]
        backbone = cast(models.EncoderBackbone, stack)
        graphs = (self.reduced_graphs if reduced else self.encoder_graphs).get(
            batch.branch
        )
        context = self.reduced_autocast if reduced else self.autocast
        exits = tuple(batch.layers) or (backbone.num_layers,)
        if batch.lengths is not None:
            if sum(batch.lengths) != batch.input_ids.numel():
                raise ValueError("packed lengths do not cover the input IDs")
            if graphs is not None:
                with torch.inference_mode(), context():
                    hidden = graphs(
                        batch.input_ids, batch.lengths, exits, batch.normalize_exits
                    )
                return EncoderOutput(hidden=hidden)
            input_ids = batch.input_ids.to(self.device)
            layout = backbone.packed(
                batch.lengths, self.device, uniform=self.batch_invariant
            )
        else:
            input_ids = batch.input_ids.to(self.device)
            rows, width = input_ids.shape
            layout = backbone.padded(batch.attention_mask, rows, width, self.device)
        with torch.inference_mode(), context():
            hidden = backbone.encode(input_ids, layout, exits, batch.normalize_exits)
        return EncoderOutput(hidden=hidden)

    def _encode_tower(self, batch: EncoderBatch) -> EncoderOutput:
        """One tower's named outputs (all of them unless ``batch.outputs`` names some)."""
        assert batch.tower is not None
        tower = self.towers.get(batch.tower)
        if tower is None:
            raise ValueError(
                f"no tower {batch.tower!r} is loaded; loaded: {sorted(self.towers)}"
            )
        inputs = {
            name: value.to(self.device) for name, value in batch.graph_inputs.items()
        }
        with torch.inference_mode(), self.autocast():
            outputs = tower(**inputs)
        if batch.outputs:
            outputs = {name: outputs[name] for name in batch.outputs}
        return EncoderOutput(outputs=outputs)

    def _encode_decoder(self, batch: EncoderBatch) -> EncoderOutput:
        """A decoder's last layer (embedders such as Qwen3-Embedding): right-padded causal rows.

        Causal attention never reads a later position, so padding changes no
        real token; packed requests come back packed.
        """
        last = len(cast(nn.ModuleList, self.backbone.layers))
        if set(batch.layers) - {last}:
            raise ValueError(f"a decoder backbone serves its last layer ({last}) only")
        if batch.lengths is not None:
            width = max(batch.lengths)
            mask = torch.arange(width)[None, :] < torch.tensor(batch.lengths)[:, None]
            input_ids = torch.zeros(mask.shape, dtype=torch.long)
            input_ids[mask] = batch.input_ids.cpu()
        else:
            assert batch.attention_mask is not None
            input_ids, mask = batch.input_ids, batch.attention_mask.bool()
        with torch.inference_mode(), self.autocast():
            hidden = self.backbone(
                input_ids.to(self.device), mask.to(self.device, torch.long)
            )
        if batch.lengths is not None:
            hidden = hidden[mask.to(hidden.device)]
        return EncoderOutput(hidden={last: hidden})

    def _modules(self) -> tuple[nn.Module, ...]:
        return (self.backbone, *self.branches.values(), *self.towers.values())

    def _parameters(self) -> list[nn.Parameter]:
        """Every parameter once (branches share the backbone's embedding)."""
        unique = {
            id(parameter): parameter
            for module in self._modules()
            for parameter in module.parameters()
        }
        return list(unique.values())

    def _packed(self) -> list[onednn.PackedLinear]:
        return [
            layer
            for module in self._modules()
            for layer in module.modules()
            if isinstance(layer, onednn.PackedLinear)
        ]

    def max_forward_tokens(self) -> int | None:
        """Tokens one GPU forward may hold before a gated-delta q / k / v outgrows FLA's offsets."""
        config = self.spec.backbone.config
        heads = config.get("linear_num_value_heads")
        if self.device.type == "cpu" or not heads:
            return None
        width: int = heads * max(
            config["linear_key_head_dim"], config["linear_value_head_dim"]
        )
        return GATED_DELTA_ELEMENTS // width

    def parameter_count(self) -> int:
        packed = sum(layer.weight_elements for layer in self._packed())
        return packed + sum(parameter.numel() for parameter in self._parameters())

    def memory_bytes(self) -> int:
        copy_bytes = sum(linear_bytes(view) for view in self.reduced.values())
        packed = sum(
            layer.packed.numel() * layer.packed.element_size()
            for layer in self._packed()
        )
        return (
            copy_bytes
            + packed
            + sum(p.numel() * p.element_size() for p in self._parameters())
        )

    def place(self, module: ModuleT) -> ModuleT:
        """A head on this device, its linear layers laid out like the backbone's."""
        module = module.to(self.device)
        if self.linear.variant is not None:
            lay_out_linears(module, self.linear.fn)
        return module

    def close(self) -> None:
        self.backbone = None  # type: ignore[assignment]  # close() releases the backbone; a closed model runs nothing
        self.branches = {}
        self.towers = {}
        self.reduced = {}
        self.encoder_graphs = {}
        self.reduced_graphs = {}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass
class HostStacks:
    """A backbone, its branches and towers loaded in host memory, in the dtypes the device holds them in."""

    backbone: nn.Module
    branches: dict[str, nn.Module]
    towers: dict[str, nn.Module]
    residency: dict[str, int] | None


class NativeEngine(Engine):
    name = "native"
    auto_priority = 0

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            **super().descriptor(),
            "architectures": sorted(models.ARCHITECTURES),
            "outputs": ["gathered", "hidden"],
            "encoder_layouts": ["padded", "packed"],
            "shared_context": True,
            "lora": "peft-unmerged",
        }

    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        for backbone in (spec.backbone, *spec.towers.values()):
            if backbone.model_type not in models.ARCHITECTURES:
                return f"no native {backbone.model_type!r} backbone"
        if (
            device.accelerator != "cpu"
            and not device.bf16
            and spec.dtype.autocast == "bfloat16"
        ):
            return f"{device.label} has no BF16 support"
        return None

    def read(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> Callable[[], NativeEngineModel]:
        """The checkpoint in host memory, in the dtypes the device holds; copying it there is the device work.

        On the CPU the host copy is the model's weights, and every CPU torch
        op of the process runs on the CPU's device thread (one OpenMP team),
        so all of the load is device work there.
        """
        if device.accelerator == "cpu":
            return partial(self.load, spec, accelerator, device, options)
        stacks = self._host(spec, accelerator, device, options)
        return partial(self._place, stacks, spec, accelerator, device, options)

    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> NativeEngineModel:
        stacks = self._host(spec, accelerator, device, options)
        return self._place(stacks, spec, accelerator, device, options)

    def _host(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> HostStacks:
        if options.threads:
            torch.set_num_threads(options.threads)
        backbone_spec = spec.backbone
        backbone = build_on_meta(backbone_spec)
        load_backbone(backbone, backbone_spec.weight_files, backbone_spec.weight_prefix)
        lora = backbone_spec.lora
        if lora is not None:
            if options.merge_lora:
                raise ValueError(
                    "merged LoRA changes answers; it belongs to the max_speed profile"
                )
            with torch.device("meta"):
                attach(
                    backbone,
                    list(lora.target_modules),
                    lora.rank,
                    lora.alpha / lora.rank,
                )
            load_adapter(backbone, lora.weight_files)
        branches = {
            name: load_branch(backbone, backbone_spec, name)
            for name in backbone_spec.branches
        }
        towers = {name: load_tower(tower) for name, tower in spec.towers.items()}
        modules = (backbone, *branches.values(), *towers.values())
        leftovers = [
            name
            for module in modules
            for name, tensor in (*module.named_parameters(), *module.named_buffers())
            if tensor.is_meta
        ]
        if leftovers:
            raise ValueError(f"backbone parameters were not loaded: {leftovers[:3]}")
        target = accelerator.torch_device(device)
        residency = None
        for module in modules:
            module.float()
            if target.type != "cpu" and spec.dtype.gpu_weights:
                cast_parameters(module, getattr(torch, spec.dtype.gpu_weights))
            elif target.type != "cpu" and spec.dtype.bf16_resident:
                residency = keep_linear_bf16(module)
        return HostStacks(backbone, branches, towers, residency)

    def _place(
        self,
        stacks: HostStacks,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> NativeEngineModel:
        target = accelerator.torch_device(device)
        for module in (
            stacks.backbone,
            *stacks.branches.values(),
            *stacks.towers.values(),
        ):
            module.to(target).eval()
        kind = (
            spec.dtype.reduced_cpu if target.type == "cpu" else spec.dtype.reduced_gpu
        )
        return NativeEngineModel(
            stacks.backbone,
            accelerator,
            device,
            spec,
            options,
            stacks.residency,
            stacks.branches,
            kind if options.reduced_precision and spec.encoder else None,
            stacks.towers,
        )


def build_on_meta(spec: BackboneSpec) -> nn.Module:
    """The architecture without allocating weights; its computed buffers (rotary tables, indices) on the CPU."""
    with torch.device("meta"):
        module = models.build(spec.model_type, spec.config)
    if hasattr(module, "rotary_emb"):
        module.rotary_emb = type(module.rotary_emb)(spec.config)
    if hasattr(module, "computed_buffers"):
        cast(models.ComputedBuffers, module).computed_buffers()
    return module


def load_tower(spec: BackboneSpec) -> nn.Module:
    """A tower (``ModelSpec.towers``): every tensor under its prefix must belong to it."""
    tower = build_on_meta(spec)
    load_backbone(tower, spec.weight_files, spec.weight_prefix, strict=True)
    return tower


def load_branch(backbone: nn.Module, spec: BackboneSpec, name: str) -> nn.Module:
    """A branch of a branched encoder: its own layer stack and final norm over the backbone's embedding."""
    branch = spec.branches[name]
    with torch.device("meta"):
        view = models.build(spec.model_type, spec.config)
    holder = nn.Module()
    holder.layers = view.layers
    holder.final_norm = view.final_norm
    load_backbone(
        holder,
        branch.weight_files,
        renames={
            f"{branch.layers}.": "layers.",
            f"{branch.final_norm}.": "final_norm.",
        },
    )
    view.embeddings = backbone.embeddings
    view.rotary_emb = backbone.rotary_emb
    return view
