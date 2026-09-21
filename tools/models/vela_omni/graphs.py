"""Tensor-only readouts; public input validation and preprocessing stay explicit."""

from __future__ import annotations

import copy
import types

import torch
from attention import install_export_attention
from torch import nn
from torch.nn import functional

BICUBIC_TAPS = 4
CLAP_SPEC_SIZE = 256
CLAP_FREQ_RATIO = 4


class TextGraph(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.variant = reference.variant
        model = reference.model
        if self.variant == "nano":
            self.encoder = model.text_encoder.encoder
            self.projection = model.text_encoder.projection
        else:

            self.encoder = model.text_model[0].model
            if self.encoder.has_sliding_layers:
                raise ValueError(
                    "pinned Mini text graph must use full causal attention"
                )
            install_export_attention(self.encoder)

    def forward(self, input_ids, attention_mask):
        # A mask dictionary bypasses Transformers' quadratic mask builder.
        # The tensor graph creates the same causal/key mask per query block;
        # original projections, rotary positions and decoder layers are retained.
        encoder_mask = (
            {"full_attention": attention_mask}
            if self.variant == "mini"
            else attention_mask
        )
        hidden = (
            self.encoder(
                input_ids=input_ids,
                attention_mask=encoder_mask,
                return_dict=False,
                use_cache=False,
            )[0]
            if self.variant == "mini"
            else self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=False
            )[0]
        )
        if self.variant == "nano":
            # Public encode_text normalizes in MultimodalEmbedder, then VelaOmni.
            return functional.normalize(
                functional.normalize(self.projection(hidden[:, 0]), dim=-1).float(),
                dim=-1,
            )
        # Preserve ST's full-width normalization before the native 768 prefix.
        # 0/1 is exactly representable in FP32; ROCm ArgMax supports this type.
        values, indices = attention_mask.to(torch.float32).flip(1).max(1)
        length = hidden.shape[1]
        indices = torch.where(values == 0, length - 1, indices)
        gather = (length - indices - 1).view(-1, 1, 1).expand(-1, 1, 1024)
        masked = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = torch.gather(masked, 1, gather).squeeze(1)
        return functional.normalize(
            functional.normalize(pooled, dim=1)[..., :768].float(), dim=-1
        )


def causal_padding_mask(attention_mask):
    positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    return causal[None, None] & attention_mask[:, None, None, :].bool()


class ImageGraph(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.variant = reference.variant
        if self.variant == "nano":
            self.encoder = reference.model.image_encoder.vision_encoder
            self.projection = reference.model.image_encoder.projection
        else:
            self.encoder = reference.model.image_model.vision_model
            self.projection = reference.model.image_proj

    def forward(self, pixel_values):
        pooled = self.encoder(pixel_values=pixel_values, return_dict=False)[1]
        if self.variant == "mini":
            pooled = functional.normalize(pooled.float(), dim=-1).to(
                self.projection.weight.dtype
            )
        return functional.normalize(
            functional.normalize(self.projection(pooled), dim=-1).float(), dim=-1
        )


class FixedClapResize(nn.Module):
    """Exact fixed bicubic linear operator, represented with portable Gather/Mul/Add.

    Compute coefficients using the original PyTorch operator on basis vectors;
    no interpolation formula approximation and no change of align_corners.
    A reference parity check is mandatory before artifact publication.
    """

    def __init__(self):
        super().__init__()
        basis = torch.eye(1001, dtype=torch.float32).reshape(1001, 1, 1001, 1)
        matrix = functional.interpolate(
            basis, size=(1024, 1), mode="bicubic", align_corners=True
        )[:, 0, :, 0].T
        indices = torch.zeros((1024, 4), dtype=torch.long)
        weights = torch.zeros((1024, 4), dtype=torch.float32)
        for row in range(1024):
            selected = matrix[row].nonzero().flatten()
            if len(selected) > BICUBIC_TAPS:
                raise ValueError("bicubic support exceeded four taps")
            indices[row, : len(selected)] = selected
            weights[row, : len(selected)] = matrix[row, selected]
        self.register_buffer("indices", indices.flatten())
        self.register_buffer("weights", weights.reshape(1, 1, 1024, 4, 1))

    def forward(self, values):
        selected = values.index_select(2, self.indices).reshape(
            values.shape[0], values.shape[1], 1024, 4, 64
        )
        resized = (selected * self.weights).sum(3)
        # Original ClapAudioEncoder.reshape_mel2img, after its fixed resize.
        return (
            resized.reshape(values.shape[0], values.shape[1] * 4, 256, 64)
            .permute(0, 1, 3, 2)
            .reshape(values.shape[0], values.shape[1], 256, 256)
        )


class ClapGraph(nn.Module):
    def __init__(self, reference):
        super().__init__()
        # Keep the reference completely unmodified for end-to-end parity.
        self.encoder = copy.deepcopy(reference.model.audio_residual.clap)
        core = self.encoder.audio_model.audio_encoder
        if (
            core.enable_fusion
            or core.spec_size != CLAP_SPEC_SIZE
            or core.freq_ratio != CLAP_FREQ_RATIO
        ):
            raise ValueError("unsupported CLAP graph geometry")
        self.resize = FixedClapResize()
        core.reshape_mel2img = types.MethodType(
            lambda _self, values: self.resize(values), core
        )

    def forward(self, input_features):
        # is_longer is unused by the pinned unfused architecture. Every public
        # endpoint window is <=10 s, so the processor also sets it to false.
        values = self.encoder(
            input_features=input_features, return_dict=True
        ).audio_embeds
        return functional.normalize(values.float(), dim=-1)


class AudioGraph(nn.Module):
    def __init__(self, reference):
        super().__init__()
        if reference.variant == "nano":
            self.encoder = reference.model.audio_encoder.encoder
            self.projection = reference.model.audio_encoder.projection
        else:
            self.encoder = reference.model.audio_model
            self.projection = reference.model.audio_proj
        residual = reference.model.audio_residual
        self.register_buffer("mean", residual.mean.detach().clone())
        self.register_buffer("scale", residual.scale.detach().clone())
        self.weight = residual.weight
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.scale).all()
            or not torch.all(self.scale > 0)
        ):
            raise ValueError("invalid frozen residual statistics")

    def forward(self, input_features, clap_embedding):
        hidden = self.encoder(input_features=input_features, return_dict=False)[0]
        original_affine = self.projection(hidden.mean(dim=1))
        residual = functional.linear(
            (clap_embedding - self.mean) / self.scale, self.weight
        )
        # Do not normalize the speech branch before adding the learned residual.
        return functional.normalize(
            functional.normalize(original_affine + residual, dim=-1).float(), dim=-1
        )


def graph_modules(reference):
    return {
        "text": TextGraph(reference),
        "image": ImageGraph(reference),
        "clap": ClapGraph(reference),
        "audio": AudioGraph(reference),
    }
