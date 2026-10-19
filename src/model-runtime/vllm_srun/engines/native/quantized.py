"""Linear layers held as quantized checkpoint weights (``quantization.py``): int4 (W4A16), FP8 (W8A8) and NVFP4.

The projections a checkpoint's ``quantization_config`` targets become these
layers on the meta backbone before it loads (``install``), and load the
checkpoint's compressed tensors as they are stored:

- ``Int4Linear``: ``weight_packed`` (int32, eight 4-bit values per word),
  ``weight_scale``, ``weight_shape`` and, for asymmetric weights (AWQ),
  ``weight_zero_point`` (packed eight rows per word). A quarter of BF16. Each
  call unpacks the weight to the activation dtype (``accel/triton_int4.py`` on
  GPUs) and multiplies in that dtype, so the layer computes exactly what a BF16
  layer holding the dequantized weight computes. With approximate kernels on
  NVIDIA GPUs, inputs of up to ``FUSED_ROWS`` rows take one fused kernel instead
  (``triton_int4.matmul``), which reads the packed weight without writing it
  out: 0.6x the BF16 GEMM's time at 32 rows and 0.96x the unpacked path's at
  256 on an RTX 4070 Ti SUPER, summed in another order. MI300X reads BF16
  weights fast enough that the fused kernel does not beat its GEMM.
- ``Float8Linear``: ``weight`` in FP8 E4M3 and ``weight_scale`` per output
  channel. Half of BF16. Each call quantizes the activations to FP8 per row
  (``accel/triton_fp8.py``) and multiplies in ``torch._scaled_mm``; it needs a
  GPU with FP8 matrix units (``float8_format``). On MI300 (gfx942), whose units
  multiply ``float8_e4m3fnuz``, ``prepare`` reinterprets the stored bits and
  doubles the scale, which keeps every value. On Ada (compute capability 8.9)
  the scaled product with per-row scales takes a slow kernel from 256 rows on
  (1.4x the BF16 GEMM on an RTX 4070 Ti SUPER, where one scale per operand runs
  at half of it), so there approximate kernels take the product of
  ``UNSCALED_ROWS`` rows or more unscaled in FP32 and apply the scales after,
  as the per-row epilogue applies them; exact kernels keep the scaled product.
  Before PyTorch 2.12 Ada's per-row kernel cannot run in a graph, so approximate
  kernels take every product unscaled there and exact kernels run without graphs
  (``capture_safe``).
  The scaled product writes BF16 whatever the dtype (hipBLASLt's row-wise
  scaling writes nothing else), cast to the activation dtype after, and a cast
  of the whole model to another dtype leaves the FP8 weight as stored.
- ``NVFP4Linear``: ``weight_packed`` (FP4 E2M1 codes, two per byte),
  ``weight_scale`` (FP8 E4M3 per 16 input columns) and the FP32
  ``weight_global_scale`` and ``input_global_scale``. An eighth of BF16 plus a
  sixteenth for the scales. Each call quantizes the activations to NVFP4 under
  the calibrated input global scale (``accel/triton_nvfp4.py``, dynamic 16-value
  block scales as compressed-tensors computes them) and multiplies on the FP4
  tensor cores in ``torch.nn.functional.scaled_mm`` with 1x16 block scales and a
  tensor scale per operand, writing BF16. It needs a Blackwell GPU
  (``accel/gpu.py``, ``nvfp4_unavailable``), which placement checks before
  loading; ``prepare`` lays the weight scales out as cuBLAS reads them
  (``accel/nvfp4.py``, ``swizzle``) in place of the stored ones. Inputs of more
  than ``NVFP4_ROWS`` rows multiply in slices: rows quantize independently, so
  the slices compute the same values while holding less at once.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ...errors import PackageError
from ...quantization import FLOAT8, FLOAT8_DEVICES, NVFP4, PACK, QuantizedWeights

# e4m3fn negative zero, whose bits are NaN in e4m3fnuz.
E4M3FN_NEGATIVE_ZERO = -128
# Rows up to which approximate kernels multiply int4 weights in one fused kernel (NVIDIA GPUs).
FUSED_ROWS = 256
# Rows from which approximate kernels take the FP8 product unscaled on Ada, where the per-row scaled one turns slow.
UNSCALED_ROWS = 256
# PyTorch before 2.12 zeroes the stream-K workspace of Ada's per-row FP8 kernel on the
# default stream (pytorch/pytorch#177573), so that product hangs in a graph or on another stream.
ROWWISE_CAPTURE_SAFE = torch.torch_version.TorchVersion(torch.__version__) >= "2.12"
# Rows per NVFP4 product: longer inputs multiply in slices (their FP4 copy and output stay bounded).
NVFP4_ROWS = 8192
# The FP8 format each ROCm architecture's matrix units multiply.
ROCM_FLOAT8 = {"gfx942": torch.float8_e4m3fnuz, "gfx950": torch.float8_e4m3fn}


def float8_format(device: torch.device) -> torch.dtype | None:
    """The FP8 format ``device`` multiplies in ``torch._scaled_mm``; None where it has none."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if torch.version.hip is not None:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        return ROCM_FLOAT8.get(arch)
    if torch.cuda.get_device_capability(device) >= (8, 9):
        return torch.float8_e4m3fn
    return None


class Int4Linear(nn.Module):
    """A linear layer over packed int4 weights, unpacked to the activation dtype per call."""

    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_shape: torch.Tensor
    weight_zero_point: torch.Tensor | None
    fused: bool = False  # approximate kernels on an NVIDIA GPU (``FUSED_ROWS``)

    def __init__(self, linear: nn.Linear, group_size: int, symmetric: bool = True):
        super().__init__()
        out_features, in_features = linear.out_features, linear.in_features
        if in_features % group_size:
            raise PackageError(
                f"{in_features} input features do not split into groups of {group_size}"
            )
        self.in_features, self.out_features = in_features, out_features
        self.group_size = group_size
        device = linear.weight.device
        self.register_buffer(
            "weight_packed",
            torch.empty(
                out_features, in_features // PACK, dtype=torch.int32, device=device
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, in_features // group_size, device=device),
        )
        self.register_buffer(
            "weight_shape", torch.empty(2, dtype=torch.int64, device=device)
        )
        zero_rows = -(-out_features // PACK)
        self.register_buffer(
            "weight_zero_point",
            (
                None
                if symmetric
                else torch.empty(
                    zero_rows,
                    in_features // group_size,
                    dtype=torch.int32,
                    device=device,
                )
            ),
        )
        self.bias = linear.bias

    @property
    def weight_elements(self) -> int:
        return self.in_features * self.out_features

    def stored_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers(recurse=False))

    def unpacked(self, dtype: torch.dtype) -> torch.Tensor:
        """The dequantized weight [out, in] in ``dtype``: FP32 products rounded once."""
        if self.weight_packed.is_cuda:
            from ...accel import triton_int4

            return triton_int4.unpack(
                self.weight_packed,
                self.weight_scale,
                self.weight_zero_point,
                self.group_size,
                dtype,
            )
        shifts = torch.arange(
            0, 32, 4, dtype=torch.int32, device=self.weight_packed.device
        )
        nibbles = (self.weight_packed.unsqueeze(-1) >> shifts) & 0xF
        values = (nibbles.reshape(self.out_features, -1) - 8).float()
        if self.weight_zero_point is not None:
            # eight rows per word: row r of group g sits in word (r // 8, g)
            zero = (self.weight_zero_point.unsqueeze(1) >> shifts[:, None]) & 0xF
            zero = zero.reshape(-1, zero.shape[-1])[: self.out_features] - 8
            values = values - zero.float().repeat_interleave(self.group_size, dim=1)
        scale = self.weight_scale.float().repeat_interleave(self.group_size, dim=1)
        return (values * scale).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // self.in_features
        if self.fused and rows <= FUSED_ROWS and x.dtype == torch.bfloat16:
            from ...accel import triton_int4

            flat = x.reshape(rows, self.in_features).contiguous()
            out = triton_int4.matmul(
                flat,
                self.weight_packed,
                self.weight_scale,
                self.weight_zero_point,
                self.group_size,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*x.shape[:-1], self.out_features)
        return nn.functional.linear(x, self.unpacked(x.dtype), self.bias)


class Float8Linear(nn.Module):
    """A linear layer over FP8 weights with one scale per output channel, multiplied with FP8 activations."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    unscaled_product: bool = False  # approximate kernels on Ada (``UNSCALED_ROWS``)
    unscaled_rows = UNSCALED_ROWS
    ada = False

    def __init__(self, linear: nn.Linear):
        super().__init__()
        out_features, in_features = linear.out_features, linear.in_features
        if out_features % 16 or in_features % 16:
            # torch._scaled_mm multiplies only multiples of 16
            raise PackageError(
                f"an FP8 layer of {out_features}x{in_features} is not a multiple of 16"
            )
        self.in_features, self.out_features = in_features, out_features
        device = linear.weight.device
        self.register_buffer(
            "weight",
            torch.empty(
                out_features, in_features, dtype=torch.float8_e4m3fn, device=device
            ),
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, device=device)
        )
        self.bias = linear.bias

    @property
    def weight_elements(self) -> int:
        return self.in_features * self.out_features

    def stored_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers(recurse=False))

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Float8Linear:
        # A cast of the whole model (``Module.float()`` at --dtype float32) keeps
        # the FP8 weight and its FP32 scales as stored; moves still apply.
        weight, scale = self.weight, self.weight_scale
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]  # torch leaves _apply unannotated
        if self.weight.dtype != weight.dtype:
            self.weight = weight.to(self.weight.device)
        if self.weight_scale.dtype != scale.dtype:
            self.weight_scale = scale.to(self.weight_scale.device)
        return self

    def prepare(self) -> None:
        """Hold the weight in the format the device multiplies (``float8_format``)."""
        fmt = float8_format(self.weight.device)
        if fmt is None:
            raise ValueError(FLOAT8_DEVICES)
        if torch.version.hip is None:
            capability = torch.cuda.get_device_capability(self.weight.device)
            self.ada = self.unscaled_product = capability == (8, 9)
            self.unscaled_rows = UNSCALED_ROWS if ROWWISE_CAPTURE_SAFE else 0
            self.one = torch.ones((), device=self.weight.device)
        if fmt == torch.float8_e4m3fnuz and self.weight.dtype == torch.float8_e4m3fn:
            # The same bits mean half the value in e4m3fnuz; -128 (negative zero in
            # e4m3fn) is NaN there, so it becomes zero.
            bits = self.weight.view(torch.int8)
            bits = torch.where(
                bits == E4M3FN_NEGATIVE_ZERO, torch.zeros_like(bits), bits
            )
            self.weight = bits.view(torch.float8_e4m3fnuz)
            self.weight_scale = self.weight_scale * 2
        # views and the kernel bound once: without graphs every call's host time counts
        self.weight_t, self.scale_t = self.weight.t(), self.weight_scale.t()
        from ...accel.triton_fp8 import quantize_rows

        self.quantize_rows = quantize_rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        if rows.stride(-1) != 1:
            rows = rows.contiguous()
        q, scale = self.quantize_rows(rows, self.weight.dtype)
        if self.unscaled_product and rows.shape[0] >= self.unscaled_rows:
            out = torch._scaled_mm(
                q,
                self.weight_t,
                scale_a=self.one,
                scale_b=self.one,
                out_dtype=torch.float32,
            )
            out = (out * scale * self.scale_t).to(x.dtype)
        else:
            # row-wise scales write BF16 only (hipBLASLt refuses FP16 and FP32)
            out = torch._scaled_mm(
                q,
                self.weight_t,
                scale_a=scale,
                scale_b=self.scale_t,
                out_dtype=torch.bfloat16,
            ).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


class NVFP4Linear(nn.Module):
    """A linear layer over NVFP4 weights, multiplied with NVFP4 activations on FP4 tensor cores."""

    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    input_global_scale: torch.Tensor

    def __init__(self, linear: nn.Linear):
        super().__init__()
        out_features, in_features = linear.out_features, linear.in_features
        if out_features % 16 or in_features % 32:
            # FP4 products take multiples of 16 rows and 32 columns (two codes per byte, 16 per scale)
            raise PackageError(
                f"an NVFP4 layer of {out_features}x{in_features} is not a multiple of 16x32"
            )
        self.in_features, self.out_features = in_features, out_features
        device = linear.weight.device
        self.register_buffer(
            "weight_packed",
            torch.empty(
                out_features, in_features // 2, dtype=torch.uint8, device=device
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                out_features,
                in_features // 16,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
        )
        self.register_buffer("weight_global_scale", torch.empty(1, device=device))
        self.register_buffer("input_global_scale", torch.empty(1, device=device))
        self.bias = linear.bias

    @property
    def weight_elements(self) -> int:
        return self.in_features * self.out_features

    def stored_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers(recurse=False))

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> NVFP4Linear:
        # A cast of the whole model keeps the FP8 scales and the FP32 global scales as stored.
        kept = {name: b.dtype for name, b in self.named_buffers(recurse=False)}
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]  # torch leaves _apply unannotated
        for name, dtype in kept.items():
            buffer = getattr(self, name)
            if buffer.dtype != dtype:
                setattr(self, name, buffer.to(dtype))
        return self

    def prepare(self) -> None:
        """Lay the weight scales out for cuBLAS and bind the global scales, so a call never syncs the host."""
        from ...accel.gpu import nvfp4_unavailable

        device = self.weight_packed.device
        reason = (
            "they need a CUDA GPU"
            if device.type != "cuda"
            else nvfp4_unavailable(device.index)
        )
        if reason:
            raise ValueError(f"NVFP4 weights: {reason}")
        from torch.nn.functional import ScalingType, SwizzleType

        from ...accel.nvfp4 import swizzle
        from ...accel.triton_nvfp4 import quantize

        self.weight_scale = swizzle(self.weight_scale)
        self.input_global = float(self.input_global_scale.float().item())
        self.input_decode = (1.0 / self.input_global_scale.float()).reshape(1)
        self.weight_decode = (1.0 / self.weight_global_scale.float()).reshape(1)
        self.weight_fp4 = self.weight_packed.view(torch.float4_e2m1fn_x2).t()
        self.scaling = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
        self.swizzling = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]
        self.quantize = quantize

    def _product(self, rows: torch.Tensor) -> torch.Tensor:
        # the kernel writes the activation scales in cuBLAS's layout already
        q, scale = self.quantize(rows, self.input_global)
        return nn.functional.scaled_mm(
            q.view(torch.float4_e2m1fn_x2),
            self.weight_fp4,
            [scale, self.input_decode],
            self.scaling,
            [self.weight_scale, self.weight_decode],
            self.scaling,
            swizzle_a=self.swizzling,
            swizzle_b=self.swizzling,
            output_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        if rows.dtype != torch.bfloat16:
            rows = rows.to(torch.bfloat16)
        if not rows.is_contiguous():
            rows = rows.contiguous()
        if rows.shape[0] > NVFP4_ROWS:
            parts = torch.split(rows, NVFP4_ROWS)
            out = torch.cat([self._product(part) for part in parts])
        else:
            out = self._product(rows)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


def install(module: nn.Module, weights: QuantizedWeights, prefix: str = "") -> int:
    """Swap every linear layer ``weights`` holds (by checkpoint name ``prefix + name``) for its quantized layer."""
    swapped = 0
    for name, layer in list(module.named_modules()):
        if isinstance(layer, nn.Linear) and weights.holds(prefix + name):
            parent, _, leaf = name.rpartition(".")
            owner = module.get_submodule(parent) if parent else module
            new: nn.Module
            if weights.scheme == FLOAT8:
                new = Float8Linear(layer)
            elif weights.scheme == NVFP4:
                new = NVFP4Linear(layer)
            else:
                new = Int4Linear(layer, weights.group_size, weights.symmetric)
            setattr(owner, leaf, new)
            swapped += 1
    if not swapped:
        raise PackageError(
            "quantization_config targets no linear layer of the backbone"
        )
    return swapped


def layers(module: nn.Module) -> list[Int4Linear | Float8Linear | NVFP4Linear]:
    return [
        layer
        for layer in module.modules()
        if isinstance(layer, (Int4Linear, Float8Linear, NVFP4Linear))
    ]


def capture_safe(module: nn.Module) -> bool:
    """Whether graphs may capture ``module``: not with Ada's per-row FP8 product before PyTorch 2.12."""
    return ROWWISE_CAPTURE_SAFE or not any(
        isinstance(layer, Float8Linear) and layer.ada and not layer.unscaled_product
        for layer in layers(module)
    )


def prepare(module: nn.Module) -> None:
    """Ready the placed layers for their device (``Float8Linear.prepare``, ``NVFP4Linear.prepare``)."""
    for layer in layers(module):
        if isinstance(layer, (Float8Linear, NVFP4Linear)):
            layer.prepare()
