"""Qwen-family Decision 2.0 checkpoints (full or base-bound LoRA) behind System One.

Model construction, prompt encoding, the candidate head and answer
normalization come from the vendored modules that produced the scored
predictions (``_vendor/dev2model``), byte for byte except where the manifest
records an import-only rewrite. A base-bound adapter's source files are pinned
by repository revision and SHA-256; a supplied or downloaded copy is verified
before use. GPU inference runs the backbone under BF16 autocast with its
BF16-exact Linear weights held in BF16 and every other tensor, the head
included, in FP32; CPU uses FP32. A request's questions run as one padded
batch unless, on a GPU, that batch would put more than 2**30 elements in a
gated-delta q / k / v tensor (``forward_token_budget``); then they run as
several batches.
On a ROCm GPU with the verified Transformers release, ``fast.py`` replays each
padded shape's backbone forward as a HIP graph and trims and fuses kernels,
reproducing the eager forward bit for bit (``graphs`` / ``kernels`` switch it off).
``share_context`` (off by default; per runtime or per request) runs the shared
input of a multi-question request once instead of once per question
(``shared_ctx.py``); answers can then differ slightly from the exact path.
A package whose manifest names a weight ``storage`` codec (bf16z) is first restored
to exact safetensors files in a cache directory; the restored checkpoint must
reproduce the scored identity. A checkpoint whose ``decision_config.json`` declares
``readout: label_token`` is read through its tied LM head's label-token logits at
the answer cue (vendored ``label_token.py``) instead of a candidate head.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ._vendor.dev2model.data import file_sha256


def resolve_base(base: dict[str, Any], base_path: str | Path | None) -> Path:
    """Local directory holding exactly the pinned base files (downloaded if absent)."""
    expected = base["files_sha256"]
    if base_path is None:
        from huggingface_hub import snapshot_download

        base_path = snapshot_download(
            base["repo_id"], revision=base["revision"], allow_patterns=sorted(expected)
        )
    root = Path(base_path).resolve(strict=True)
    for name, digest in expected.items():
        if not (root / name).is_file() or file_sha256(root / name) != digest:
            raise ValueError(f"Base file differs from the pinned revision: {name}")
    return root


def load_score_bias_entry(
    root: Path, manifest: dict[str, Any], model_sha256: str
) -> dict[int, list[float]] | None:
    """Per-level Score offsets bound by the manifest; None when it lists none."""
    entry = manifest.get("score_bias")
    if entry is None:
        return None
    from ._vendor.dev2model.score_bias import load_score_bias

    path = root / entry["file"]
    if (
        manifest["files_sha256"].get(entry["file"]) != entry["sha256"]
        or not path.is_file()
        or file_sha256(path) != entry["sha256"]
    ):
        raise ValueError("Score offsets differ from the packaged score_bias file")
    offsets, _ = load_score_bias(path, model_sha256)
    if {str(key): value for key, value in offsets.items()} != entry["offsets"]:
        raise ValueError("Score offsets differ from MODEL_MANIFEST.json")
    return offsets


def keep_linear_bf16(module: Any, torch: Any) -> dict[str, int]:
    """Hold the BF16-exact ``nn.Linear`` weights and biases of ``module`` in BF16.

    BF16 autocast multiplies every Linear with its weight rounded to BF16, so a
    weight that BF16 represents exactly gives the same products whether it is
    kept in FP32 and cast on every call or kept in BF16. Everything else stays
    FP32: embeddings, norms, gated-delta ``A_log`` / ``dt_bias``, conv filters,
    Linear weights BF16 cannot hold exactly (e.g. FP32-trained LoRA factors,
    which stay unmerged) and weights shared with any other kind of module.
    Returns how many Linear modules hold a BF16 or an FP32 weight.
    """
    shared = {
        id(parameter)
        for layer in module.modules()
        if not isinstance(layer, torch.nn.Linear)
        for parameter in layer.parameters(recurse=False)
    }
    counts = {"linear_bf16": 0, "linear_fp32": 0}
    for layer in module.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        for parameter in (layer.weight, layer.bias):
            if (
                parameter is None
                or parameter.dtype != torch.float32
                or id(parameter) in shared
            ):
                continue
            rounded = parameter.data.to(torch.bfloat16)
            if torch.equal(rounded.float(), parameter.data):
                parameter.data = rounded
        counts[
            "linear_bf16" if layer.weight.dtype == torch.bfloat16 else "linear_fp32"
        ] += 1
    return counts


INT32_MAX = 2**31 - 1


def forward_token_budget(config: Any) -> int | None:
    """Most padded tokens one GPU forward may hold; None when nothing limits it.

    The FLA gated-delta kernels of Qwen3.5-family backbones compute some
    element offsets of their [batch, tokens, value heads, head dim] q / k / v
    tensors in 32-bit integers (the L2-norm forward and the fused KKT-solve
    kernel among them). A forward whose tensors exceed 2**31 - 1 elements reads
    and writes the wrong memory for the later batch rows: wrong answers,
    non-finite logits or a GPU memory fault. On DEV2.0-27B, forwards with
    tensors between about 2**30 and 2**31 elements also hung or crashed the
    process, so the budget keeps these tensors within 2**30 elements.
    """
    config = getattr(config, "text_config", None) or config
    heads = getattr(config, "linear_num_value_heads", None)
    if not heads:
        return None
    width = heads * max(config.linear_key_head_dim, config.linear_value_head_dim)
    return (INT32_MAX // 2) // width


def micro_batches(lengths: list[int], budget: int | None) -> list[list[int]]:
    """Indices of each forward: one batch when its padded size fits the budget.

    Otherwise the questions go longest first into batches whose padded size
    (``collate`` pads to the longest, rounded up to 8) stays within the budget,
    each batch listing its indices in request order.
    """

    def padded(length: int) -> int:
        return -(-length // 8) * 8

    if budget is None or padded(max(lengths)) * len(lengths) <= budget:
        return [list(range(len(lengths)))]
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    if padded(lengths[order[0]]) > budget:
        raise ValueError("A single question exceeds the forward token budget")
    groups: list[list[int]] = []
    while order:
        rows = budget // padded(lengths[order[0]])
        groups.append(sorted(order[:rows]))
        order = order[rows:]
    return groups


class QwenDecision:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any,
        temperatures: dict,
        cap: int,
        torch: Any,
        score_bias: dict[int, list[float]] | None = None,
        residency: dict[str, int] | None = None,
        encode_fn: Any = None,
        batch_tokens: int | None = None,
        fast: Any = None,
        share_context: Any = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperatures = temperatures
        self.cap = cap
        self.torch = torch
        self.score_bias = score_bias
        self.residency = residency
        self.encode_fn = encode_fn
        self.batch_tokens = batch_tokens
        self.fast = fast
        self.share_context = share_context

    @classmethod
    def load(
        cls,
        root: Path,
        manifest: dict[str, Any],
        *,
        device: str,
        base_path: str | Path | None,
        threads: int | None,
        bf16_resident: bool = True,
        graphs: bool = True,
        kernels: bool = True,
        share_context: Any = False,
    ) -> QwenDecision:
        import torch

        from ._vendor.dev2model.calibration import load_calibration
        from ._vendor.dev2model.decision_model import DecisionModel
        from ._vendor.dev2model.infer import checkpoint_fingerprint

        if threads:
            torch.set_num_threads(threads)
        source = None
        if manifest["profile"] == "qwen-adapter":
            source = resolve_base(manifest["base"], base_path)
        model_root = root
        if manifest.get("storage"):
            from .bf16z import materialize

            model_root = materialize(root, manifest)
        metadata = json.loads(
            (model_root / "decision_config.json").read_text(encoding="utf-8")
        )
        residual = metadata.get("dec_residual") is not None
        label = metadata.get("readout") == "label_token"
        if label:
            from ._vendor.dev2model.label_token import label_fingerprint

            identity = label_fingerprint(model_root, source)
        elif residual:
            from ._vendor.dev2model.dec_model import dec_fingerprint

            identity = dec_fingerprint(model_root, source)
        else:
            identity = checkpoint_fingerprint(model_root, source)
        if identity["model_sha256"] != manifest["identity"]["model_sha256"]:
            raise ValueError("Model identity differs from the scored checkpoint")
        score_bias = load_score_bias_entry(root, manifest, identity["model_sha256"])
        calibration = manifest.get("calibration")
        if calibration:
            temperatures, report = load_calibration(
                root / calibration["file"], identity["model_sha256"]
            )
            if report.get("inference", {}).get("max_length") not in (
                None,
                manifest["max_input_tokens"],
            ):
                raise ValueError("Calibration context differs from the package limit")
        else:
            temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0}
        target = torch.device(device)
        if target.type == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("A CUDA/ROCm BF16 GPU is required for GPU inference")
        encode_fn = None
        if label:
            from ._vendor.dev2model.label_token import (
                encode_label,
                load_label_checkpoint,
            )

            model, tokenizer = load_label_checkpoint(model_root, source)
            encode_fn = encode_label
        elif residual:
            from ._vendor.dev2model.dec_model import load_dec_checkpoint

            model, tokenizer = load_dec_checkpoint(model_root, source)
        else:
            model, tokenizer = DecisionModel.from_checkpoint(
                model_root, source_path=source
            )
        model = model.float()
        residency = None
        if target.type == "cuda" and bf16_resident:
            residency = keep_linear_bf16(model.backbone, torch)
        model = model.to(target).eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        batch_tokens = (
            forward_token_budget(model.backbone.config)
            if target.type == "cuda"
            else None
        )
        fast = None
        if target.type == "cuda" and bf16_resident:
            from .fast import install

            fast = install(model, torch, graphs=graphs, kernels=kernels)
        return cls(
            model,
            tokenizer,
            target,
            temperatures,
            manifest["max_input_tokens"],
            torch,
            score_bias,
            residency,
            encode_fn,
            batch_tokens,
            fast,
            share_context,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def system_one(
        self, state: Any, questions: dict[str, Any], share_context: Any = None
    ) -> tuple[dict[str, Any], int]:
        from ._vendor.dev2model.data import canonical
        from ._vendor.dev2model.decision_model import collate, encode
        from ._vendor.dev2model.infer import (
            _api_json_payload,
            product_answer,
            question_to_row,
        )

        if not _api_json_payload(state):
            raise ValueError("state must be text, an object, or an array")
        try:
            canonical(state)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must contain JSON data") from exc
        item = {"id": "request", "state": state}
        answers: dict[str, dict[str, Any]] = {}
        jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        tokens = 0
        share = self.share_context if share_context is None else share_context
        policy, tokenizer = None, self.tokenizer
        if share is not None and share is not False:
            from .shared_ctx import resolve, shared_logits, tokenizer_for

            policy = resolve(share)
            tokenizer = tokenizer_for(self.tokenizer, policy, len(questions))
        for qid, question in questions.items():
            try:
                row = question_to_row(item, qid, question)
                encoded = (self.encode_fn or encode)(row, tokenizer, self.cap)
            except ValueError as exc:
                reason = (
                    "max_length_exceeded"
                    if "exceeds max_length" in str(exc)
                    else "invalid_question"
                )
                answers[qid] = {
                    "type": (
                        question.get("type") if isinstance(question, dict) else None
                    ),
                    "error": reason,
                }
                continue
            tokens += len(encoded["ids"])
            jobs.append((qid, row, encoded))
        if not jobs:
            return answers, tokens
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        logits: list[Any] = [None] * len(jobs)
        groups = micro_batches(
            [len(encoded["ids"]) for _, _, encoded in jobs], self.batch_tokens
        )
        if policy is not None:
            shared = shared_logits(self, jobs, pad_id, policy)
            if shared is not None:
                logits, groups = shared, []
        for group in groups:
            batch = {
                key: value.to(self.device) if self.torch.is_tensor(value) else value
                for key, value in collate(
                    [jobs[index][2] for index in group], pad_id
                ).items()
            }
            autocast = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with self.torch.inference_mode(), autocast:
                if self.fast is None:
                    output = self.model(**batch)
                else:
                    with self.fast.forward(
                        [len(jobs[index][2]["ids"]) for index in group]
                    ):
                        output = self.model(**batch)
            if len(output) != len(group):
                raise RuntimeError(
                    "Model returned the wrong number of question answers"
                )
            for index, values in zip(group, output):
                logits[index] = values
        if self.score_bias is not None:
            from ._vendor.dev2model.score_bias import apply as apply_score_bias
        for (qid, row, encoded), values in zip(jobs, logits):
            try:
                values = values[: len(encoded["keys"])].float().cpu().tolist()
                if self.score_bias is not None and row["task_type"] == "score":
                    values = apply_score_bias(
                        self.score_bias, values, len(row["options"])
                    )
                answers[qid] = product_answer(
                    row["task_type"],
                    encoded["keys"],
                    values,
                    self.temperatures[row["task_type"]],
                    [option["description"] for option in row["options"]],
                )
            except ValueError:
                answers[qid] = {
                    "type": row["task_type"],
                    "error": "invalid_model_output",
                }
        ordered = {qid: answers[qid] for qid in questions}
        return ordered, tokens
