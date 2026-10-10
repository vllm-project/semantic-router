"""Image-aware code-readout engine for Decision 2.5 Omni checkpoints.

``VisionCodeReadoutModel`` loads a code-readout v1 checkpoint (vega/SPEC.md layout with the vision
tower) and returns one probability per option for requests ``{state, question, images}``:

- Transformers ``Qwen3_5Model`` (``Qwen3VLModel`` for d3-edge's Qwen3-VL layout) + ``Qwen3VLProcessor``,
  images capped at 1.6 MP (1,638,400 px) by the processor's resize, placed before the text in the
  user turn;
- BF16 backbone on accelerators (FP32 on CPU), FP32 readout and softmax, temperature from
  ``decision_config.json``; ``causal`` or ``noncausal_full_attention``; last-token (or mean)
  pooling;
- exact per-request token counts (text plus merged image patches) planned before any image is
  decoded; requests above ``max_length``, with more than four images, more than 255 options or
  literal image placeholders in their text are reported unsupported, never truncated;
- left-padded batches sorted by token count under a padded-token budget; text-only requests are
  only batched with text-only requests, so they take exactly the text path of Vega's engine.
"""

from __future__ import annotations

import binascii
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from d25.omni.common import vision_format
from d25.omni.model import checkpoint, inputs
from d25.omni.model.patch_embed import linearize_patch_embed
from d25.omni.model.attention import ATTENTION_MODES, apply_attention_mode
from d25.vega.common import decision_format as text_format

DEFAULT_MAX_LENGTH = 16_384
POOLINGS = ("last", "mean")


class Unsupported(ValueError):
    """A request the checkpoint's contract does not accept (scored as unanswered)."""


@dataclass
class Plan:
    index: int
    n_options: int = 0
    text: str = ""
    images: list[Any] = field(default_factory=list)
    cost: inputs.Cost | None = None
    status: str = "ok"
    reason: str | None = None

    @property
    def tokens(self) -> int:
        return self.cost.tokens if self.cost else 0


@dataclass(frozen=True)
class Outcome:
    status: str
    probabilities: list[float] | None = None
    input_tokens: int = 0
    reason: str | None = None


def resolve_dtype(value: str | torch.dtype | None, device: str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value in (None, "auto"):
        return torch.float32 if device == "cpu" else torch.bfloat16
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[value]


def make_batches(
    plans: Sequence[Plan], token_budget: int, max_batch_size: int
) -> list[list[Plan]]:
    """Longest-first batches of one modality with ``rows * longest <= token_budget``."""
    batches: list[list[Plan]] = []
    for with_images in (True, False):
        group = sorted(
            (p for p in plans if bool(p.images) == with_images),
            key=lambda p: (-p.tokens, p.index),
        )
        current: list[Plan] = []
        for plan in group:
            if current and (
                len(current) >= max_batch_size
                or (len(current) + 1) * current[0].tokens > token_budget
            ):
                batches.append(current)
                current = []
            current.append(plan)
        if current:
            batches.append(current)
    return batches


class VisionCodeReadoutModel:
    def __init__(
        self,
        ckpt_dir: str | Path,
        device: str | None = None,
        max_pixels: int = vision_format.MAX_PIXELS,
        *,
        attention_mode: str | None = None,
        max_length: int | None = None,
        token_budget: int = 65_536,
        max_batch_size: int = 64,
        dtype: str | torch.dtype | None = None,
        readout_dtype: str | torch.dtype = "float32",
        temperature: float | None = None,
        image_root: str | Path | None = None,
        prefetch: bool = True,
    ) -> None:
        from transformers import AutoProcessor

        self.ckpt_dir = Path(ckpt_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = checkpoint.read_decision_config(self.ckpt_dir)
        self.prompt = self.config.get("prompt", "d25-vega")
        if self.prompt not in inputs.PROMPTS:
            raise ValueError(f"unsupported prompt {self.prompt!r}")
        saved_mode = self.config.get("attention_mode", "causal")
        if attention_mode is not None and attention_mode != saved_mode:
            raise ValueError(
                f"checkpoint attention mode is {saved_mode}, requested {attention_mode}"
            )
        if saved_mode not in ATTENTION_MODES:
            raise ValueError(f"unknown attention mode {saved_mode!r}")
        self.attention_mode = saved_mode
        self.pooling = self.config.get("pooling", "last")
        if self.pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {self.pooling!r}")
        self.temperature = float(
            temperature
            if temperature is not None
            else self.config.get("temperature", 1.0)
        )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.max_length = int(
            max_length or self.config.get("max_length") or DEFAULT_MAX_LENGTH
        )
        self.max_pixels = max_pixels
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.image_root = image_root
        self.prefetch = prefetch
        self.codes: list[str] = list(self.config["codes"])
        self.token_ids: list[int] = list(self.config["token_ids"])

        self.processor = inputs.setup_processor(
            AutoProcessor.from_pretrained(str(self.ckpt_dir)), max_pixels
        )
        inputs.check_codes(self.processor.tokenizer, self.codes, self.token_ids)

        if self.device.startswith("cuda"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.dtype = resolve_dtype(dtype, self.device)
        load_kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        if self.device != "cpu":
            load_kwargs["device_map"] = {"": self.device}
        self.backbone, info = checkpoint.backbone_class(self.ckpt_dir).from_pretrained(
            str(self.ckpt_dir), output_loading_info=True, **load_kwargs
        )
        missing = list(info.get("missing_keys") or [])
        self.has_vision = not any(key.startswith("visual.") for key in missing)
        other = [key for key in missing if not key.startswith("visual.")]
        if other:
            raise ValueError(f"{self.ckpt_dir}: backbone weights missing: {other[:5]}")
        self.backbone.to(self.device).eval().requires_grad_(False)
        if self.has_vision and not linearize_patch_embed(self.backbone):
            raise RuntimeError("vision patch embedding not found")
        self.readout_dtype = resolve_dtype(readout_dtype, "cpu")
        self.readout = checkpoint.load_readout(self.ckpt_dir).to(
            self.device, self.readout_dtype
        )
        if self.readout.shape[1] != self.backbone.config.text_config.hidden_size:
            raise ValueError("readout width differs from the backbone hidden size")
        self.hook = apply_attention_mode(self.backbone, self.attention_mode)

    def plan(
        self, rows: Sequence[dict[str, Any]], root: str | Path | None = None
    ) -> list[Plan]:
        """Render and cost every request without decoding images."""
        root = root if root is not None else self.image_root
        return [self._plan_one(index, row, root) for index, row in enumerate(rows)]

    def _plan_one(self, index: int, row: dict[str, Any], root) -> Plan:
        plan = Plan(index, images=list(row.get("images") or []))
        n_images = len(plan.images)
        try:
            keys, _ = text_format.options(row["question"])
        except (KeyError, TypeError, ValueError) as error:
            return self._reject(plan, "unsupported", f"invalid question: {error}")
        plan.n_options = len(keys)
        if not 1 <= plan.n_options <= min(text_format.MAX_OPTIONS, len(self.codes)):
            return self._reject(
                plan, "unsupported", f"{plan.n_options} options; 1 to 255 supported"
            )
        if n_images > vision_format.MAX_IMAGES:
            return self._reject(
                plan,
                "unsupported",
                f"{n_images} images; at most {vision_format.MAX_IMAGES}",
            )
        if n_images and not self.has_vision:
            return self._reject(plan, "unsupported", "checkpoint has no vision tower")
        try:
            plan.text = inputs.render(
                self.processor,
                self.prompt,
                row.get("state"),
                row["question"],
                self.codes,
                n_images,
            )
        except (TypeError, ValueError) as error:
            return self._reject(
                plan, "unsupported", f"prompt rendering failed: {error}"
            )
        if n_images and inputs.placeholder_conflict(plan.text, n_images):
            return self._reject(
                plan,
                "unsupported",
                "literal image or video placeholder in the row text",
            )
        try:
            sizes = [inputs.image_size(ref, root) for ref in plan.images]
        except (OSError, ValueError, binascii.Error) as error:
            return self._reject(plan, "error", f"image unreadable: {error}")
        try:
            plan.cost = inputs.cost(self.processor, plan.text, sizes)
        except ValueError as error:
            return self._reject(
                plan, "unsupported", f"image rejected by the processor: {error}"
            )
        if plan.tokens > self.max_length:
            return self._reject(
                plan,
                "unsupported",
                f"{plan.tokens} input tokens > max_length {self.max_length}",
            )
        if root is not None:
            plan.images = [
                (
                    ref
                    if not isinstance(ref, str)
                    or ref.startswith("data:")
                    or Path(ref).is_absolute()
                    else str(Path(root) / ref)
                )
                for ref in plan.images
            ]
        return plan

    @staticmethod
    def _reject(plan: Plan, status: str, reason: str) -> Plan:
        plan.status, plan.reason = status, reason
        return plan

    def _prepare(self, batch: list[Plan]):
        images = [
            vision_format.load_image(ref) for plan in batch for ref in plan.images
        ]
        encoded = inputs.encode(self.processor, [plan.text for plan in batch], images)
        if encoded["input_ids"].shape[1] != batch[0].tokens:
            raise RuntimeError(
                f"planned {batch[0].tokens} tokens but the processor produced {encoded['input_ids'].shape[1]}"
            )
        return encoded

    @torch.inference_mode()
    def _probabilities(self, encoded, counts: Sequence[int]) -> list[list[float]]:
        tensors = {name: value.to(self.device) for name, value in encoded.items()}
        hidden = self.backbone(**tensors, use_cache=False).last_hidden_state
        if self.pooling == "mean":
            valid = tensors["attention_mask"].bool()
            pooled = (
                hidden.float().masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1)
                / valid.sum(dim=1, keepdim=True)
            ).to(hidden.dtype)
        else:
            pooled = hidden[:, -1]
        logits = torch.nn.functional.linear(
            pooled.to(self.readout_dtype), self.readout
        ).float()
        mask = (
            torch.arange(logits.shape[-1], device=logits.device)[None]
            >= torch.tensor(counts, device=logits.device)[:, None]
        )
        probabilities = (
            (logits.masked_fill(mask, -1e9) / self.temperature).softmax(-1).cpu()
        )
        return [values[:count].tolist() for values, count in zip(probabilities, counts)]

    def score(
        self, rows: Sequence[dict[str, Any]], root: str | Path | None = None
    ) -> list[Outcome]:
        """One ``Outcome`` per request, in input order; unsupported requests are reported, not run."""
        plans = self.plan(rows, root)
        outcomes: list[Outcome | None] = [
            None if p.status == "ok" else Outcome(p.status, None, p.tokens, p.reason)
            for p in plans
        ]
        batches = make_batches(
            [p for p in plans if p.status == "ok"],
            self.token_budget,
            self.max_batch_size,
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = (
                pool.submit(self._prepare, batches[0])
                if batches and self.prefetch
                else None
            )
            for number, batch in enumerate(batches):
                encoded = (
                    pending.result() if pending is not None else self._prepare(batch)
                )
                pending = (
                    pool.submit(self._prepare, batches[number + 1])
                    if self.prefetch and number + 1 < len(batches)
                    else None
                )
                values = self._probabilities(
                    encoded, [plan.n_options for plan in batch]
                )
                for plan, probabilities in zip(batch, values):
                    outcomes[plan.index] = Outcome("ok", probabilities, plan.tokens)
        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("a planned request produced no outcome")
        return outcomes

    def predict(
        self, rows: Sequence[dict[str, Any]], root: str | Path | None = None
    ) -> list[list[float]]:
        """Probabilities in ``options()`` order; raises ``Unsupported`` if any request is rejected."""
        outcomes = self.score(rows, root)
        rejected = [
            (index, o.reason) for index, o in enumerate(outcomes) if o.status != "ok"
        ]
        if rejected:
            raise Unsupported(
                f"{len(rejected)} request(s) rejected, first: {rejected[0]}"
            )
        return [outcome.probabilities for outcome in outcomes]
