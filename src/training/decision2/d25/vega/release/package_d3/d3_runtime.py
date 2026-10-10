"""d3 runtime: System One typed decisions over a code-readout v1 checkpoint.

One question is decided per forward pass. The prompt lists every option under a single-token answer
code; a 255-way readout scores those codes at the last prompt position and the answer is a
probability for every option. No text is generated and no input is truncated.

    from d3_runtime import D3
    model = D3.from_pretrained("<package dir or Hub id>", device="cuda:0")
    model.system_one(state="...", questions={"route": {"type": "choice", ...}})
    # {"model": ..., "answers": {"route": {"type": "choice", "choice": ..., "probabilities": {...},
    #                                     "confidence": ...}}, "usage": {"input_tokens": n, "output_tokens": 0}}

    model.system_one(state="...", questions={...}, images=["photo.png", "label.jpg"])

The checkpoint directory holds ``config.json`` + ``model*.safetensors`` (a transformers
``Qwen3_5Model``), ``readout.safetensors`` (``{"weight": [255, hidden]}``), ``decision_config.json``
(prompt family, answer codes, attention mode, pooling, temperature, input limit) and the tokenizer.
``d3_format.py`` next to this file is the prompt and answer-code contract of the model.

Images: a request takes any number of images (PIL images, local paths, http(s) URLs or base64
``data:image/...`` URLs), shared by all of its questions. They go before the text in the user turn,
one image placeholder per image in list order, and the checkpoint's own processor (``AutoProcessor``,
torchvision backend) resizes each to at most 1,638,400 pixels (1.6 MP) and at least 65,536, keeping
the aspect ratio; each 32 x 32 pixel patch is one input token. The prompt is the text prompt with the
images in front; a request without images takes exactly the text path. Image inputs need a checkpoint
with the vision tower (``visual.*`` weights).

Numerics: BF16 backbone with SDPA attention, FP32 readout and softmax (unless the checkpoint says
otherwise). A request's questions run in request order, ``batch_size`` per forward pass, each batch
left-padded to its longest prompt. ``noncausal_full_attention`` lets the full-attention layers see the
whole prompt while the Gated DeltaNet layers stay causal. The Gated DeltaNet kernels are the ones
transformers binds at import: flash-linear-attention (and causal-conv1d) when installed, its PyTorch
reference implementation otherwise.

``permutation_average=True`` (off by default) also scores every choice question with two or more options
with its options in reversed order, in the same forward passes as the original order, and answers with
the per-option mean of the two distributions. Noul and score questions are scored once. A pass carrying both
orders that would exceed ``MERGE_TOKENS`` padded tokens runs the reversed prompts in a pass of their own, so
peak memory stays that of the original order.

The noncausal attention mask hook is adapted from perplexity-ai/pplx-decider-v1.1-27b, Copyright
Perplexity AI, Apache License 2.0.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .d3_format import (
        MAX_OPTIONS,
        SYSTEM_PROMPT,
        answer_codes,
        describe,
        options,
        render as render_d3,
        to_answer,
        user_prompt,
    )
except ImportError:
    from d3_format import (
        MAX_OPTIONS,
        SYSTEM_PROMPT,
        answer_codes,
        describe,
        options,
        render as render_d3,
        to_answer,
        user_prompt,
    )

RUNTIME = "d3-runtime/1"
FORMAT_VERSION = 1
PROMPTS = ("d3",)
ATTENTION_MODES = ("causal", "noncausal_full_attention")
DEFAULT_BATCH_SIZE = 8
SCORE_LEVELS = (2, 10)
MANIFEST = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "d3-package-manifest/1"
VERIFY_MODES = ("fast", "full", "none")
# "fast" verification hashes every file up to this size and checks the size of larger ones.
FAST_HASH_BYTES = 64 << 20
ERRORS = ("invalid_question", "max_length_exceeded", "invalid_model_output")
IMAGE_MIN_PIXELS = 65_536
IMAGE_MAX_PIXELS = 1_638_400
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
# Checked for every encoded image the server receives (strict loading); in-process inputs are only
# bounded by PIL's decompression-bomb guard.
MAX_IMAGE_BYTES = 8_000_000
MAX_IMAGE_SOURCE_PIXELS = 16_000_000
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 64 << 20
MERGE_TOKENS = 8192


class MaxLengthExceeded(ValueError):
    """A question prompt is longer than the checkpoint's input limit; nothing is truncated."""


# ---------------------------------------------------------------------------------------------
# Questions and prompts
# ---------------------------------------------------------------------------------------------


@dataclass
class Question:
    """One validated question: what the prompt renders and how the answer is reported."""

    kind: str  # choice | noul | score
    original: Mapping[str, Any]
    rendered: dict[str, Any]  # a choice or noul question in the d3_format contract
    keys: list[str]
    descriptions: list[Any]


def _json_value(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not JSON data") from exc


def normalize_question(question: Any) -> Question:
    """Validate one System One question; raises ValueError with the reason."""
    if not isinstance(question, Mapping):
        raise ValueError("a question is an object with a type")
    kind = question.get("type")
    instructions = question.get("instructions")
    if instructions is not None:
        _json_value(instructions)
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(
                f"choice criteria must map 1 to {MAX_OPTIONS} option keys to descriptions"
            )
        if any(not isinstance(k, str) or not k for k in criteria):
            raise ValueError("choice option keys must be nonempty strings")
        _json_value(dict(criteria))
        rendered = {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(criteria),
        }
        keys = list(criteria)
        return Question(kind, question, rendered, keys, [criteria[k] for k in keys])
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}:
                raise ValueError('noul criteria may only describe "false" and "true"')
            _json_value(dict(criteria))
        rendered = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            rendered["criteria"] = dict(criteria)
        keys, texts = options(rendered)
        return Question(kind, question, rendered, keys, texts)
    if kind == "score":
        low, high = SCORE_LEVELS
        if not isinstance(criteria, (list, tuple)) or not low <= len(criteria) <= high:
            raise ValueError(
                f"score criteria must be an ordered list of {low} to {high} levels"
            )
        _json_value(list(criteria))
        levels = [describe(level) for level in criteria]
        if any(not text for text in levels) or len(set(levels)) != len(levels):
            raise ValueError("score levels must be distinct and nonempty")
        # The ordered levels are the options, each shown by its text alone (code order = level order).
        rendered = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {text: None for text in levels},
        }
        return Question(
            kind,
            question,
            rendered,
            [str(i) for i in range(len(levels))],
            list(criteria),
        )
    raise ValueError(f"unsupported question type {kind!r} (choice, noul or score)")


def render(
    tokenizer, prompt: str, state: Any, question: dict[str, Any], codes: Sequence[str]
) -> str:
    if prompt == "d3":
        return render_d3(tokenizer, state, question, codes)
    raise ValueError(f"unknown prompt family {prompt!r}")


def image_messages(
    prompt: str,
    state: Any,
    question: dict[str, Any],
    codes: Sequence[str],
    n_images: int,
) -> list[dict[str, Any]]:
    """The prompt family's messages with one image placeholder per image before the user text."""
    if n_images < 1:
        raise ValueError(f"an image prompt takes at least 1 image, got {n_images}")
    images: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    if prompt == "d3":
        text = {"type": "text", "text": user_prompt(state, question, codes)}
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": images + [text]},
        ]
    raise ValueError(f"unknown prompt family {prompt!r}")


def reversed_question(question: Question) -> dict[str, Any] | None:
    """The rendered choice question with its options in reverse order (None when there is nothing to permute)."""
    if question.kind != "choice" or len(question.keys) < 2:
        return None
    criteria = question.rendered["criteria"]
    return dict(
        question.rendered, criteria={key: criteria[key] for key in reversed(criteria)}
    )


def average_orders(forward: Sequence[float], backward: Sequence[float]) -> list[float]:
    """Per-option mean of the original-order and reversed-order distributions, in original option order."""
    return [(a + b) / 2 for a, b in zip(forward, reversed(backward))]


def _canonical(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )


def _confidence(values: Sequence[float]) -> float:
    """One minus the normalized entropy of the distribution, clipped to [0, 1] (the Decision 2.0 measure)."""
    if len(values) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def product_answer(
    question: Question, probabilities: Sequence[float]
) -> dict[str, Any]:
    """System One answer from probabilities in option order (Decision 2.0 response shapes)."""
    if question.kind in ("choice", "noul"):
        answer = to_answer(question.rendered, probabilities)
        if question.kind == "choice":
            answer["confidence"] = _confidence(list(answer["probabilities"].values()))
        return answer
    values = [float(v) for v in probabilities]
    if (
        len(values) != len(question.keys)
        or any(not math.isfinite(v) or v < 0 for v in values)
        or sum(values) <= 0
    ):
        raise ValueError("need one finite non-negative probability per level")
    total = sum(values)
    values = [v / total for v in values]
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(values)),
        "probabilities": dict(zip(question.keys, values)),
        "confidence": _confidence(values),
        "legend": {
            key: _canonical(level)
            for key, level in zip(question.keys, question.descriptions)
        },
    }


# ---------------------------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------------------------


def _data_url_payload(value: str, strict: bool) -> bytes:
    header, separator, encoded = value.partition(",")
    kind = header.strip().lower()
    if (
        not separator
        or not kind.startswith("data:image/")
        or not kind.endswith(";base64")
    ):
        raise ValueError("a data URL image is data:image/<format>;base64,<data>")
    if strict:
        if kind[len("data:image/") : -len(";base64")] not in (
            "png",
            "jpeg",
            "jpg",
            "webp",
        ):
            raise ValueError("images must be base64 PNG, JPEG or WebP data URLs")
        if len(encoded) > 4 * -(-MAX_IMAGE_BYTES // 3):
            raise ValueError(f"each image must be at most {MAX_IMAGE_BYTES:,} bytes")
    try:
        return base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 image data") from exc


def _download(url: str) -> bytes:
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": RUNTIME})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"the image at {url} is larger than {MAX_DOWNLOAD_BYTES:,} bytes"
        )
    return payload


def load_image(value: Any, *, strict: bool = False):
    """One image input as an RGB PIL image.

    ``value`` is a PIL image, a local path, an http(s) URL or a ``data:image/<format>;base64,`` URL.
    Pixels are decoded by PIL and converted to RGB, nothing else (no EXIF rotation, no resize: the
    processor resizes). ``strict`` (the server) accepts only data URLs of PNG, JPEG or WebP images of at
    most 8,000,000 bytes and 16,000,000 pixels.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(value, Image.Image) and not strict:
        return value.convert("RGB")
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise ValueError(
            "an image is a PIL image, a path, an http(s) URL or a data:image/...;base64 URL"
        )
    if strict and not value.startswith("data:"):
        raise ValueError("images must be base64 PNG, JPEG or WebP data URLs")
    try:
        if value.startswith("data:"):
            payload = _data_url_payload(value, strict)
        elif value.startswith(("http://", "https://")):
            payload = _download(value)
        else:
            payload = Path(value).expanduser().read_bytes()
    except OSError as exc:
        raise ValueError(f"image unreadable ({exc})") from exc
    if strict and len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"each image must be at most {MAX_IMAGE_BYTES:,} bytes")
    try:
        if strict:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in IMAGE_FORMATS:
                    raise ValueError("images must be PNG, JPEG or WebP")
                if image.width * image.height > MAX_IMAGE_SOURCE_PIXELS:
                    raise ValueError(
                        f"each image must have at most {MAX_IMAGE_SOURCE_PIXELS:,} pixels"
                    )
                image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    except (
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"invalid image data ({type(exc).__name__})") from exc


def load_processor(root: Path):
    """The checkpoint's ``AutoProcessor`` with left padding and the 1.6 MP image budget."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(root))
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {
        "shortest_edge": IMAGE_MIN_PIXELS,
        "longest_edge": IMAGE_MAX_PIXELS,
    }
    return processor


def visual_tokens(image_processor, width: int, height: int) -> int:
    """Input tokens of one image after the processor's resize (raises ValueError on aspect ratio > 200)."""
    patches = image_processor.get_number_of_image_patches(height, width, {})
    return patches // image_processor.merge_size**2


def _linear_patch_embed_forward(self, hidden_states):
    import torch.nn.functional as F

    weight = self.proj.weight
    flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    out = F.linear(flat, weight.reshape(weight.shape[0], -1), self.proj.bias)
    return out.view(-1, self.embed_dim)


def linearize_patch_embed(model) -> int:
    """Run the vision patch embedding as the matrix product it equals; returns how many were patched.

    The patch embedding is a Conv3d whose kernel equals its stride over inputs already cut into single
    patches, i.e. a linear map of each flattened patch (same weights, same result up to summation order).
    On ROCm, MIOpen searches a Conv3d kernel for every new patch count, so each new image size would stall
    for minutes; the evaluation engine runs the same matrix product.
    """
    import types

    import torch

    count = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if (
            isinstance(proj, torch.nn.Conv3d)
            and tuple(proj.kernel_size) == tuple(proj.stride)
            and hasattr(module, "embed_dim")
        ):
            module.forward = types.MethodType(_linear_patch_embed_forward, module)
            count += 1
    return count


# ---------------------------------------------------------------------------------------------
# Package files
# ---------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dir(
    name_or_path: str | os.PathLike, revision: str | None = None, **hub: Any
) -> Path:
    """A local checkpoint directory, or a Hub snapshot (one commit) in the Hugging Face cache."""
    path = Path(os.fspath(name_or_path)).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    options = {k: v for k, v in hub.items() if v is not None and v is not False}
    return Path(snapshot_download(str(name_or_path), revision=revision, **options))


def verify_package(root: Path, mode: str = "fast") -> dict[str, Any] | None:
    """Check the files against ``MODEL_MANIFEST.json``; returns the manifest (None without one)."""
    if mode not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}")
    path = root / MANIFEST
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{MANIFEST} is not a {MANIFEST_SCHEMA} manifest")
    if mode == "none":
        return manifest
    sizes = manifest.get("files_bytes") or {}
    problems = []
    for name, digest in sorted((manifest.get("files_sha256") or {}).items()):
        target = root / name
        if not target.is_file():
            problems.append(f"missing {name}")
            continue
        size = target.stat().st_size
        if name in sizes and size != sizes[name]:
            problems.append(f"{name}: {size} bytes, manifest says {sizes[name]}")
            continue
        if mode == "full" or size <= FAST_HASH_BYTES:
            if sha256_file(target) != digest:
                problems.append(f"{name}: SHA-256 differs from {MANIFEST}")
    if problems:
        raise ValueError("package verification failed: " + "; ".join(problems[:8]))
    return manifest


def vision_weights_present(root: Path) -> bool:
    """True when the checkpoint stores the vision tower (``visual.*`` tensors)."""
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open

        names = []
        for path in sorted(root.glob("model*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                names += list(handle.keys())
    return any(name.startswith(("visual.", "model.visual.")) for name in names)


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------


def enable_noncausal_full_attention(text_model) -> None:
    """Let the softmax-attention layers see future tokens; keep padding and the causal recurrence.

    Adapted from perplexity-ai/pplx-decider-v1.1-27b, Copyright Perplexity AI, Apache License 2.0.
    """
    import torch
    from transformers.masking_utils import create_recurrent_attention_mask

    if text_model.config._attn_implementation != "sdpa":
        raise ValueError("noncausal full attention requires SDPA")

    def mask_inputs(module, args, kwargs):
        if args:
            raise ValueError("noncausal full attention requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("noncausal classification does not support a KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(
                embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
            )
        if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
            raise ValueError("expected a 2D padding mask")
        if (
            padding.shape != embeddings.shape[:2]
            or not padding.bool().any(dim=-1).all()
        ):
            raise ValueError("padding mask must match the complete nonempty input")
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def sdpa_backends(device):
    """CUDA builds skip the cuDNN SDPA backend; ROCm keeps the defaults.

    ``D3_CUDNN_SDPA=1`` keeps PyTorch's default backend choice on CUDA too.
    """
    import contextlib

    import torch

    if (
        torch.device(device).type != "cuda"
        or not torch.version.cuda
        or getattr(torch.version, "hip", None)
        or os.environ.get("D3_CUDNN_SDPA") == "1"
    ):
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    )


def kernel_report() -> dict[str, str]:
    """Which implementation transformers bound for the Gated DeltaNet ops (kernel package or PyTorch)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    def bound(fn) -> str:
        seen, stack = set(), [fn]
        while stack:
            f = stack.pop()
            if id(f) in seen or not callable(f):
                continue
            seen.add(id(f))
            module = getattr(f, "__module__", "") or ""
            if module.startswith(("fla", "causal_conv1d", "kernels")):
                return f"{module}.{getattr(f, '__name__', '?')}"
            for cell in getattr(f, "__closure__", None) or ():
                try:
                    stack.append(cell.cell_contents)
                except ValueError:
                    pass
        return "torch-reference"

    names = ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")
    report = {name: bound(getattr(m, name)) for name in names if hasattr(m, name)}
    try:
        import fla

        report["fla"] = getattr(fla, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        report["fla"] = f"missing ({type(exc).__name__})"
    return report


@dataclass
class Prepared:
    """A tokenized request: runnable questions in request order plus the per-question errors.

    A text request holds token ``sequences``; an image request holds the decoded ``images`` (shared by
    every question), the rendered prompt ``texts`` and the planned input ``lengths`` (text tokens plus
    image tokens), and the processor tokenizes it when it runs.
    """

    keys: list[str]
    questions: dict[str, Question] = field(default_factory=dict)
    sequences: dict[str, list[int]] = field(default_factory=dict)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    images: list[Any] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    # Reversed-option prompts of the choice questions (permutation_average only).
    reversed_sequences: dict[str, list[int]] = field(default_factory=dict)
    reversed_texts: dict[str, str] = field(default_factory=dict)
    reversed_lengths: dict[str, int] = field(default_factory=dict)

    @property
    def runnable(self) -> list[str]:
        return [k for k in self.keys if k in self.sequences or k in self.texts]


class D3:
    """A loaded code-readout checkpoint answering System One requests."""

    def __init__(
        self,
        root: Path,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        manifest: dict[str, Any] | None = None,
        max_length: int | None = None,
        readout_dtype: str | None = None,
        model_name: str | None = None,
        permutation_average: bool = False,
    ):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        self.torch = torch
        self.root = Path(root)
        self.manifest = manifest
        started = time.perf_counter()
        config = json.loads(
            (self.root / "decision_config.json").read_text(encoding="utf-8")
        )
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported decision_config.json format_version")
        self.config = config
        self.prompt = config.get("prompt", "d3")
        if self.prompt not in PROMPTS:
            raise ValueError(f"unknown prompt family {self.prompt!r}")
        self.attention_mode = config.get("attention_mode", "causal")
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(f"unknown attention mode {self.attention_mode!r}")
        if config.get("pooling", "last") != "last":
            raise ValueError(f"unsupported pooling {config.get('pooling')!r}")
        self.temperature = float(config.get("temperature", 1.0))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.readout_dtype = readout_dtype or config.get("readout_dtype", "float32")
        if self.readout_dtype not in ("float32", "bfloat16"):
            raise ValueError("readout_dtype must be float32 or bfloat16")
        limit = max_length if max_length is not None else config.get("max_length")
        self.max_length = int(limit) if limit else None
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = (
            model_name or (manifest or {}).get("model_name") or self.root.name
        )
        self.permutation_average = bool(permutation_average)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.root))
        self.tokenizer.padding_side = "left"
        codes, token_ids = answer_codes(self.tokenizer)
        if config.get("codes") != codes or config.get("token_ids") != token_ids:
            raise ValueError("checkpoint answer codes differ from its tokenizer")
        self.codes, self.token_ids = codes, token_ids
        probe = "State:\nA"
        if (
            self.tokenizer(probe)["input_ids"]
            != self.tokenizer(probe, add_special_tokens=False)["input_ids"]
        ):
            raise ValueError(
                "tokenizer adds special tokens; prompts are tokenized as rendered"
            )
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer has no pad token")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.kernels = kernel_report()
        if self.device.type == "cpu" and any(
            v.startswith(("fla", "causal_conv1d"))
            for k, v in self.kernels.items()
            if k != "fla"
        ):
            raise RuntimeError(
                "flash-linear-attention / causal-conv1d kernels are GPU-only; run on a GPU or "
                "use an environment without them for CPU inference"
            )
        torch.manual_seed(20260919)
        self.backbone = Qwen3_5Model.from_pretrained(
            str(self.root),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": str(self.device)},
        )
        weight = load_file(str(self.root / "readout.safetensors"))["weight"]
        hidden = self.backbone.config.text_config.hidden_size
        if tuple(weight.shape) != (MAX_OPTIONS, hidden):
            raise ValueError(
                f"readout weight has shape {tuple(weight.shape)}, expected {(MAX_OPTIONS, hidden)}"
            )
        self.readout = weight.to(self.device, getattr(torch, self.readout_dtype))
        self.backbone.eval().requires_grad_(False)
        if self.attention_mode == "noncausal_full_attention":
            enable_noncausal_full_attention(self.backbone.language_model)
        self.processor = None
        self.image_unavailable: str | None = None
        if not vision_weights_present(self.root):
            self.image_unavailable = (
                "the checkpoint has no vision tower (visual.* weights)"
            )
        elif not linearize_patch_embed(self.backbone):
            self.image_unavailable = "the vision patch embedding was not found"
        else:
            try:
                self.processor = load_processor(self.root)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - text requests do not use the processor
                self.image_unavailable = (
                    f"the image processor failed to load ({type(exc).__name__}: {exc})"
                )
        self.loaded_seconds = time.perf_counter() - started

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        verify: str = "fast",
        token: str | bool | None = None,
        cache_dir: str | os.PathLike | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        max_length: int | None = None,
        readout_dtype: str | None = None,
        model_name: str | None = None,
        permutation_average: bool = False,
    ) -> D3:
        """Load a package directory or Hub repository.

        ``verify``: ``fast`` (default) hashes every file of ``MODEL_MANIFEST.json`` up to 64 MiB and checks
        the size of the weight shards; ``full`` hashes every file; ``none`` skips the check. A checkpoint
        without a manifest (a plain code-readout export) loads unverified. ``permutation_average``: see the
        module docstring.
        """
        root = resolve_dir(
            name_or_path,
            revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        manifest = verify_package(root, verify)
        return cls(
            root,
            device=device,
            batch_size=batch_size,
            manifest=manifest,
            max_length=max_length,
            readout_dtype=readout_dtype,
            model_name=model_name,
            permutation_average=permutation_average,
        )

    # ------------------------------------------------------------------ requests

    def text(self, state: Any, question: dict[str, Any]) -> str:
        return render(self.tokenizer, self.prompt, state, question, self.codes)

    def image_text(self, state: Any, question: dict[str, Any], n_images: int) -> str:
        return self.processor.apply_chat_template(
            image_messages(self.prompt, state, question, self.codes, n_images),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def load_images(self, images: Sequence[Any], *, strict: bool = False) -> list[Any]:
        """Decode a request's images (RGB PIL), any number; raises ValueError for an image it cannot read."""
        if not images:
            return []
        if self.image_unavailable is not None:
            raise ValueError(
                f"image inputs are not available: {self.image_unavailable}"
            )
        decoded = []
        for number, value in enumerate(images):
            try:
                decoded.append(load_image(value, strict=strict))
            except ValueError as exc:
                raise ValueError(f"images[{number}]: {exc}") from exc
        return decoded

    def prepare(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
    ) -> Prepared:
        """Validate and tokenize one request; malformed ``state``, ``questions`` or ``images`` raise ValueError.

        ``images`` (a list of any number of images shared by every question) select the image path. Without
        images the request takes the text path unchanged.
        """
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError(
                "questions must be a nonempty mapping of question IDs to questions"
            )
        if any(not isinstance(key, str) or not key for key in questions):
            raise ValueError("question IDs must be nonempty strings")
        _json_value(state)
        if images is not None and not isinstance(images, (list, tuple)):
            raise ValueError("images must be a list of images")
        if images:
            return self._prepare_images(state, questions, self.load_images(images))
        prepared = Prepared(keys=list(questions))
        texts = []
        for key, question in questions.items():
            try:
                normalized = normalize_question(question)
                texts.append((key, self.text(state, normalized.rendered)))
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, Mapping) else None
                prepared.errors[key] = {
                    "type": kind,
                    "error": "invalid_question",
                    "message": str(exc),
                }
                continue
            prepared.questions[key] = normalized
        if texts:
            ids = self.tokenizer([t for _, t in texts], add_special_tokens=False)[
                "input_ids"
            ]
            for (key, _), sequence in zip(texts, ids):
                if self.max_length is not None and len(sequence) > self.max_length:
                    prepared.errors[key] = {
                        "type": prepared.questions[key].kind,
                        "error": "max_length_exceeded",
                        "message": f"the question prompt has {len(sequence)} tokens, over the maximum context "
                        f"length of {self.max_length} tokens; nothing was truncated",
                    }
                    continue
                prepared.sequences[key] = sequence
        if self.permutation_average:
            self._prepare_reversed(state, prepared)
        return prepared

    def _prepare_reversed(
        self, state: Any, prepared: Prepared, visual: int = 0
    ) -> None:
        """Render and tokenize the reversed-option prompt of every runnable choice question.

        ``visual``: the image tokens of the request (image path), counted like ``_prepare_images`` counts them.
        """
        n_images = len(prepared.images)
        texts = []
        for key in prepared.runnable:
            flipped = reversed_question(prepared.questions[key])
            if flipped is not None:
                texts.append(
                    (
                        key,
                        (
                            self.image_text(state, flipped, n_images)
                            if n_images
                            else self.text(state, flipped)
                        ),
                    )
                )
        if not texts:
            return
        tokenizer = self.processor.tokenizer if n_images else self.tokenizer
        ids = tokenizer([t for _, t in texts], add_special_tokens=False)["input_ids"]
        for (key, text), sequence in zip(texts, ids):
            length = len(sequence) - n_images + visual
            if self.max_length is not None and length > self.max_length:
                for planned in (prepared.sequences, prepared.texts, prepared.lengths):
                    planned.pop(key, None)
                prepared.errors[key] = {
                    "type": prepared.questions[key].kind,
                    "error": "max_length_exceeded",
                    "message": f"the reversed-option prompt has {length} tokens, over the maximum context "
                    f"length of {self.max_length} tokens; nothing was truncated",
                }
            elif n_images:
                prepared.reversed_texts[key] = text
                prepared.reversed_lengths[key] = length
            else:
                prepared.reversed_sequences[key] = sequence

    def _prepare_images(
        self, state: Any, questions: Mapping[str, Any], images: list[Any]
    ) -> Prepared:
        """Render every question with the images in front and plan its input tokens (images not run yet)."""
        try:
            visual = [
                visual_tokens(self.processor.image_processor, *image.size)
                for image in images
            ]
        except ValueError as exc:
            raise ValueError(f"image rejected by the processor: {exc}") from exc
        prepared = Prepared(keys=list(questions), images=images)
        texts = []
        for key, question in questions.items():
            try:
                normalized = normalize_question(question)
                text = self.image_text(state, normalized.rendered, len(images))
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, Mapping) else None
                prepared.errors[key] = {
                    "type": kind,
                    "error": "invalid_question",
                    "message": str(exc),
                }
                continue
            if text.count(IMAGE_TOKEN) != len(images) or VIDEO_TOKEN in text:
                prepared.errors[key] = {
                    "type": normalized.kind,
                    "error": "invalid_question",
                    "message": "the state or question contains a literal image or video placeholder token",
                }
                continue
            prepared.questions[key] = normalized
            texts.append((key, text))
        if texts:
            ids = self.processor.tokenizer(
                [t for _, t in texts], add_special_tokens=False
            )["input_ids"]
            for (key, text), sequence in zip(texts, ids):
                length = len(sequence) - len(images) + sum(visual)
                if self.max_length is not None and length > self.max_length:
                    prepared.errors[key] = {
                        "type": prepared.questions[key].kind,
                        "error": "max_length_exceeded",
                        "message": f"the question prompt has {length} tokens ({sum(visual)} for "
                        f"{len(images)} image(s)), over the maximum context length of {self.max_length} "
                        "tokens; nothing was truncated",
                    }
                    continue
                prepared.texts[key] = text
                prepared.lengths[key] = length
        if self.permutation_average:
            self._prepare_reversed(state, prepared, sum(visual))
        return prepared

    def logits(self, sequences: Sequence[Sequence[int]], counts: Sequence[int]):
        """Masked code logits (FP32, [B, 255]) for one left-padded batch."""
        torch = self.torch
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            ids[i, width - len(sequence) :] = torch.as_tensor(
                sequence, dtype=torch.long
            )
            mask[i, width - len(sequence) :] = 1
        ids, mask = ids.to(self.device, non_blocking=True), mask.to(
            self.device, non_blocking=True
        )
        with torch.inference_mode(), sdpa_backends(self.device):
            hidden = self.backbone(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state[:, -1]
            if self.readout_dtype == "float32":
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            limit = torch.as_tensor(list(counts), device=self.device)[:, None]
            invalid = torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit
            return logits.masked_fill(invalid, float("-inf"))

    def probabilities(
        self, sequences: Sequence[Sequence[int]], counts: Sequence[int]
    ) -> list[list[float]]:
        """Softmax over each prompt's own codes, in option order."""
        probs = (
            (self.logits(sequences, counts) / self.temperature)
            .softmax(-1)
            .cpu()
            .tolist()
        )
        return [p[:c] for p, c in zip(probs, counts)]

    def image_logits(
        self,
        texts: Sequence[str],
        images: Sequence[Any],
        counts: Sequence[int],
        width: int,
    ):
        """Masked code logits (FP32, [B, 255]) for one left-padded batch of prompts that share ``images``.

        The processor expands each prompt's image placeholders, resizes the images and pads the batch to
        ``width`` tokens (the longest planned prompt); the backbone gets every tensor it returns.
        """
        torch = self.torch
        encoded = self.processor(
            text=list(texts),
            images=[image for _ in texts for image in images],
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] != width:
            raise RuntimeError(
                f"planned {width} input tokens, the processor produced {encoded['input_ids'].shape[1]}"
            )
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode(), sdpa_backends(self.device):
            hidden = self.backbone(**inputs, use_cache=False).last_hidden_state[:, -1]
            if self.readout_dtype == "float32":
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            limit = torch.as_tensor(list(counts), device=self.device)[:, None]
            invalid = torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit
            return logits.masked_fill(invalid, float("-inf"))

    def image_probabilities(
        self,
        texts: Sequence[str],
        images: Sequence[Any],
        counts: Sequence[int],
        width: int,
    ) -> list[list[float]]:
        probs = (
            (self.image_logits(texts, images, counts, width) / self.temperature)
            .softmax(-1)
            .cpu()
            .tolist()
        )
        return [p[:c] for p, c in zip(probs, counts)]

    def _run_images(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        keys = prepared.runnable
        out: dict[str, list[float]] = {}
        for start in range(0, len(keys), self.batch_size):
            chunk = keys[start : start + self.batch_size]
            extra = [k for k in chunk if k in prepared.reversed_texts]
            texts = [prepared.texts[k] for k in chunk] + [
                prepared.reversed_texts[k] for k in extra
            ]
            counts = [len(prepared.questions[k].keys) for k in chunk + extra]
            widths = [prepared.lengths[k] for k in chunk] + [
                prepared.reversed_lengths[k] for k in extra
            ]
            n = len(chunk)
            if extra and len(widths) * max(widths) > MERGE_TOKENS:
                probs = self.image_probabilities(
                    texts[:n], prepared.images, counts[:n], max(widths[:n])
                ) + self.image_probabilities(
                    texts[n:], prepared.images, counts[n:], max(widths[n:])
                )
            else:
                probs = self.image_probabilities(
                    texts, prepared.images, counts, max(widths)
                )
            out.update(zip(chunk, probs))
            for key, backward in zip(extra, probs[n:]):
                out[key] = average_orders(out[key], backward)
        return out, sum(prepared.lengths[k] for k in keys) + sum(
            prepared.reversed_lengths.values()
        )

    def run(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        """Probabilities per runnable question (request order, ``batch_size`` per pass) and the input tokens."""
        if prepared.images:
            return self._run_images(prepared)
        keys = prepared.runnable
        out: dict[str, list[float]] = {}
        for start in range(0, len(keys), self.batch_size):
            chunk = keys[start : start + self.batch_size]
            extra = [k for k in chunk if k in prepared.reversed_sequences]
            sequences = [prepared.sequences[k] for k in chunk] + [
                prepared.reversed_sequences[k] for k in extra
            ]
            counts = [len(prepared.questions[k].keys) for k in chunk + extra]
            n = len(chunk)
            if extra and len(sequences) * max(map(len, sequences)) > MERGE_TOKENS:
                probs = self.probabilities(
                    sequences[:n], counts[:n]
                ) + self.probabilities(sequences[n:], counts[n:])
            else:
                probs = self.probabilities(sequences, counts)
            out.update(zip(chunk, probs))
            for key, backward in zip(extra, probs[n:]):
                out[key] = average_orders(out[key], backward)
        return out, sum(len(prepared.sequences[k]) for k in keys) + sum(
            len(s) for s in prepared.reversed_sequences.values()
        )

    def respond(
        self,
        prepared: Prepared,
        probabilities: Mapping[str, Sequence[float]],
        tokens: int,
    ) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for key in prepared.keys:
            if key in prepared.errors:
                answers[key] = prepared.errors[key]
                continue
            question = prepared.questions[key]
            try:
                answers[key] = product_answer(question, probabilities[key])
            except ValueError as exc:
                answers[key] = {
                    "type": question.kind,
                    "error": "invalid_model_output",
                    "message": str(exc),
                }
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Typed Choice / Noul / Score answers about one state: ``{"model", "answers", "usage"}``.

        ``images``: any number of images (PIL images, paths, http(s) or data URLs) that every question sees.
        A question that cannot be answered gets ``{"type", "error", "message"}`` with ``error`` one of
        ``invalid_question``, ``max_length_exceeded`` (never truncated) or ``invalid_model_output``; the
        other questions of the request are still answered.
        """
        prepared = self.prepare(state, questions, images)
        probabilities, tokens = self.run(prepared)
        return self.respond(prepared, probabilities, tokens)

    def warmup(
        self,
        lengths: Sequence[int] = (37, 64, 320, 333, 1000, 1024),
        images: bool = True,
    ) -> float:
        """Compile and autotune the kernels for every batch size up to ``batch_size`` (twice that with
        ``permutation_average``) before serving.

        The Gated DeltaNet kernels take the batch size as a compile-time constant, and Triton specializes their
        length and chunk-count arguments on being 1 or a multiple of 16; these lengths cover every combination
        (chunks of 64 tokens). A fresh process otherwise pays several seconds on the first request of each new
        class. With ``images`` (and a vision tower) three image requests (one small image, one 1.6 MP image,
        four 1.6 MP images) also warm the vision tower. Answers are unchanged.
        """
        started = time.perf_counter()
        widest = self.batch_size * (2 if self.permutation_average else 1)
        for size in range(1, widest + 1):
            for length in lengths:
                sequences = [
                    [
                        self.token_ids[(i + j) % len(self.token_ids)]
                        for j in range(max(8, length - 7 * i))
                    ]
                    for i in range(size)
                ]
                self.probabilities(sequences, [2] * size)
        if images and self.image_unavailable is None:
            from PIL import Image

            small = Image.new("RGB", (448, 336), (128, 128, 128))
            large = Image.new("RGB", (1280, 1280), (96, 160, 224))
            for batch in ([small], [large], [large] * 4):
                self.system_one(
                    state="warm-up", questions={"q": {"type": "noul"}}, images=batch
                )
        self.synchronize()
        return time.perf_counter() - started

    # ------------------------------------------------------------------ device and records

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def to(self, device: str) -> D3:
        target = self.torch.device(device)
        self.backbone.to(target)
        self.readout = self.readout.to(target)
        self.device = target
        return self

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters()) + self.readout.numel()

    def provenance(self) -> dict[str, Any]:
        files = {
            name: sha256_file(self.root / name)
            for name in (
                "decision_config.json",
                "readout.safetensors",
                "config.json",
                MANIFEST,
            )
            if (self.root / name).is_file()
        }
        identity = (self.manifest or {}).get("identity", {})
        record = {
            "kind": "d3-code-readout",
            "runtime": RUNTIME,
            "model_name": self.model_name,
            "repo_id": (self.manifest or {}).get("repo_id"),
            "model_sha256": identity.get("model_sha256"),
            "format_id": self.config.get("format_id"),
            "prompt": self.prompt,
            "attention_mode": self.attention_mode,
            "pooling": "last",
            "temperature": self.temperature,
            "max_length": self.max_length,
            "readout_dtype": self.readout_dtype,
            "backbone_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "batch_size": self.batch_size,
            "files_sha256": files,
            "kernels": self.kernels,
            "policy": "One forward pass per question; options under single-token answer codes; last-token readout "
            "over the question's codes only; argmax choice; no truncation (over-limit questions are "
            "refused); no option filtering; one fixed prompt for every request.",
            "images": self.image_contract(),
        }
        if self.permutation_average:
            record["permutation_average"] = True
            record["policy"] += (
                " Choice questions with two or more options are also scored with their options in reversed "
                "order, in the same forward passes, and answered with the per-option mean of both distributions."
            )
        return record

    def image_contract(self) -> dict[str, Any]:
        """How image inputs are read (or why they are not available)."""
        contract: dict[str, Any] = {
            "supported": self.image_unavailable is None,
            "min_pixels": IMAGE_MIN_PIXELS,
            "max_pixels": IMAGE_MAX_PIXELS,
            "placement": "before the text of the user turn, one placeholder per image, request order",
            "patch_embedding": "matrix product (equal to the Conv3d with kernel = stride)",
        }
        if self.processor is not None:
            contract["image_processor"] = type(self.processor.image_processor).__name__
            try:
                import torchvision

                contract["torchvision"] = torchvision.__version__
            except Exception:  # noqa: BLE001
                contract["torchvision"] = None
        if self.image_unavailable is not None:
            contract["unavailable"] = self.image_unavailable
        return contract

    def runtime_info(self) -> dict[str, Any]:
        import transformers

        torch = self.torch
        info = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "loaded_seconds": round(self.loaded_seconds, 2),
        }
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(self.device)
        return info
