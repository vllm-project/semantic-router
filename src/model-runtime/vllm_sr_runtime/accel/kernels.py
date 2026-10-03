"""Kernel registry with pure-torch reference kernels.

Every kernel slot has a reference implementation that reproduces the
Transformers code the released packages were scored with. An accelerator may
register a faster implementation for a slot; it declares whether it is
bit-exact against the reference on that device. A kernel that is not
bit-exact is used only when the active profile allows approximate numerics.
A variant is an alternative a model's released runtime ran for a slot (for
example a different convolution); it runs only for models that name it
(``ModelSpec.kernel_variants``) and is that model's reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Kernel:
    name: str
    fn: Callable
    source: str
    exact: bool
    variant: str | None = None


@dataclass
class KernelSet:
    """The kernels one device uses; ``select`` honours the profile's exactness and the model's variants."""

    device: str
    available: dict[str, list[Kernel]] = field(default_factory=dict)
    allow_approximate: bool = False
    variants: dict[str, str] = field(default_factory=dict)

    def register(self, kernel: Kernel) -> None:
        self.available.setdefault(kernel.name, []).insert(0, kernel)

    def use_variants(self, variants: Mapping[str, str]) -> None:
        """Run each named slot with the given variant where this device registers it."""
        self.variants = dict(variants)

    def select(self, name: str) -> Kernel:
        candidates = self.available.get(name, [])
        for variant in dict.fromkeys((self.variants.get(name), None)):
            for kernel in candidates:
                if kernel.variant == variant and (
                    kernel.exact or self.allow_approximate
                ):
                    return kernel
        raise KeyError(f"no kernel registered for {name!r} on {self.device}")

    def __call__(self, name: str) -> Callable:
        return self.select(name).fn

    def has(self, name: str) -> bool:
        try:
            self.select(name)
        except KeyError:
            return False
        return True

    def describe(self) -> dict[str, str]:
        return {name: self.select(name).source for name in sorted(self.available)}


# ---------------------------------------------------------------------------
# Reference kernels (Transformers 5.17 semantics)
# ---------------------------------------------------------------------------


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def causal_conv1d_ref(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Depthwise causal convolution over [batch, channels, time]."""
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = _activation(activation)(out)
    return out.to(hidden_states.dtype)


def chunk_gated_delta_rule_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The gated delta rule over chunks of the sequence ([batch, time, heads, dim] inputs)."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    strictly_upper_mask = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        ut_system, v_beta, upper=False, unitriangular=True
    )
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True
    )
    if initial_state is None:
        last_recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=new_values.dtype, device=new_values.device
        )
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = (
            last_recurrent_state * chunk_decay[:, :, i]
            + key[:, :, i].transpose(-1, -2) @ v_new
        )
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    return core_attn_out, last_recurrent_state


def rotary_half_ref(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half rotary embedding of ``[B, H, T, D]`` query and key, in FP32 and cast back (ModernBERT).

    ``cos`` and ``sin`` are ``[1, T, D]``.
    """
    dtype = query.dtype
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    query_embed = (query.float() * cos) + (rotate(query.float()) * sin)
    key_embed = (key.float() * cos) + (rotate(key.float()) * sin)
    return query_embed.to(dtype), key_embed.to(dtype)


def sdpa_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    scale: float,
    is_causal: bool,
    enable_gqa: bool,
) -> torch.Tensor:
    extra = {"enable_gqa": True} if enable_gqa else {}
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
        is_causal=is_causal,
        **extra,
    )


ADDITIVE_MASKS = "additive_masks"
MASK_CACHE_ENTRIES = 4


class AdditiveMaskSDPA:
    """SDPA with an additive float mask on every call, as Transformers 4.57's ModernBERT calls it.

    A boolean mask becomes 0 where allowed and the dtype's minimum elsewhere, a
    missing one zeros; on ROCm the inputs are made contiguous first, as that
    code does for the memory-efficient kernel. Both fix which SDPA backend runs.
    The layers of one forward pass the same mask objects, so each is converted
    once (one instance per kernel set, which one worker thread uses).
    """

    def __init__(self) -> None:
        self._masks: list[tuple[object, torch.Tensor]] = []

    def additive(
        self, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            return attn_mask
        source: object = attn_mask
        if attn_mask is None:
            source = (
                query.shape[0],
                query.shape[-2],
                key.shape[-2],
                query.dtype,
                query.device,
            )
        capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()
        for cached, mask in self._masks:
            if not capturing and (
                cached is source if attn_mask is not None else cached == source
            ):
                return mask
        if attn_mask is None:
            mask = query.new_zeros((query.shape[0], 1, query.shape[-2], key.shape[-2]))
        else:
            mask = torch.zeros(
                attn_mask.shape, dtype=query.dtype, device=query.device
            ).masked_fill_(attn_mask.logical_not(), torch.finfo(query.dtype).min)
        if not capturing:
            self._masks = [(source, mask), *self._masks[: MASK_CACHE_ENTRIES - 1]]
        return mask

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        scale: float,
        is_causal: bool,
        enable_gqa: bool,
    ) -> torch.Tensor:
        mask = self.additive(query, key, attn_mask)
        if (
            torch.version.hip is not None
            and query.is_cuda
            and torch.backends.cuda.mem_efficient_sdp_enabled()
        ):
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        return sdpa_ref(
            query,
            key,
            value,
            mask,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name in ("silu", "swish"):
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unsupported activation {name!r}")


# The ``geglu`` variant that gives each row the same result in any batch.
CONTIGUOUS = "contiguous"
# Columns a row is padded to for ``rowwise``: whole unrolled AVX-512 vectors.
ROW_ALIGN = 64


def rowwise(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """An element-wise ``fn`` whose result for a row never depends on where the row sits.

    Vectorized CPU loops round a vector's lanes and its scalar tail apart, and
    a flattened row straddles vectors by its position; rows padded to
    ``ROW_ALIGN`` columns each start a vector and leave no tail.
    """
    width = x.shape[-1]
    pad = -width % ROW_ALIGN
    if pad == 0:
        return fn(x.contiguous())
    return fn(F.pad(x, (0, pad)))[..., :width]


def geglu_ref(
    projected: torch.Tensor, activation: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """GeGLU over ``[..., 2 * I]`` projections: ``activation(value) * gate`` on the halves (Transformers)."""
    value, gate = projected.chunk(2, dim=-1)
    return activation(value) * gate


def geglu_contiguous(
    projected: torch.Tensor, activation: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """``geglu_ref`` with the activation on whole aligned rows of the value half (``rowwise``)."""
    value, gate = projected.chunk(2, dim=-1)
    return rowwise(activation, value) * gate


def linear_ref(linear: torch.nn.Module) -> torch.nn.Module:
    """The ``linear`` slot lays a linear layer out for its device at load; the reference keeps it."""
    return linear


def reference_kernels(device: str) -> KernelSet:
    kernels = KernelSet(device=device)
    for name, fn in (
        ("causal_conv1d", causal_conv1d_ref),
        ("chunk_gated_delta_rule", chunk_gated_delta_rule_ref),
        ("geglu", geglu_ref),
        ("linear", linear_ref),
        ("rotary_half", rotary_half_ref),
        ("sdpa", sdpa_ref),
    ):
        kernels.register(Kernel(name=name, fn=fn, source="torch-reference", exact=True))
    kernels.register(
        Kernel(
            "sdpa",
            AdditiveMaskSDPA(),
            "torch-reference",
            exact=True,
            variant=ADDITIVE_MASKS,
        )
    )
    return kernels
