"""CPU-only shape prototype for an independent-option decision encoder.

This is not a trained model or a production System One adapter.  It isolates
the rendering and readout invariants before considering GPU training.  The
``score`` callback stands in for a shared backbone and scalar head.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .data import MAX_OPTIONS, canonical


@dataclass(frozen=True)
class CandidatePrompt:
    key: str
    prefix: str
    tail: str
    key_used_as_semantics: bool

    @property
    def text(self) -> str:
        return self.prefix + self.tail


def _payload(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def candidate_prompts(row: dict[str, Any]) -> tuple[CandidatePrompt, ...]:
    """Render every candidate without any other candidate or list position.

    Choice keys are output identifiers when a description is present.  An
    absent/empty description forces the key to be semantic input: arbitrary
    key-renaming invariance cannot then be promised.  Score level indices and
    Noul truth values are semantic, even if physical candidate storage moves.
    """
    task_type = row.get("task_type")
    if task_type not in {"choice", "noul", "score"}:
        raise ValueError("task_type must be choice, noul or score")
    if not isinstance(row.get("instructions"), (str, dict, list)):
        raise ValueError("instructions must be text or structured JSON")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"options must contain 2..{MAX_OPTIONS} candidates")
    keys = [
        option.get("key") if isinstance(option, dict) else None for option in options
    ]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("each candidate needs a nonempty key")
    if len(set(keys)) != len(keys):
        raise ValueError("candidate keys must be unique")
    if task_type == "noul" and set(keys) != {"false", "true"}:
        raise ValueError("noul candidates must have true/false keys")
    if task_type == "score" and set(keys) != {str(i) for i in range(len(keys))}:
        raise ValueError("score keys must identify levels 0..K-1")
    if task_type == "score" and len(options) > 10:
        raise ValueError("score accepts 2..10 ordered levels")

    prefix = (
        f"Context:\n{_payload(row['state'])}\n\n"
        f"Task type: {task_type}\nQuestion:\n{_payload(row['instructions'])}\n"
    )
    prompts = []
    for option in options:
        key = option["key"]
        description = option.get("description")
        if description is not None and not isinstance(description, (str, dict, list)):
            raise ValueError(
                "candidate description must be text, structured JSON or null"
            )
        if task_type == "choice":
            use_key = description is None or description == ""
            semantic = key if use_key else _payload(description)
        elif task_type == "noul":
            use_key = True
            semantic = f"Answer: {key}\nCriterion: {_payload(description) if description is not None else key}"
        else:
            if description is None:
                raise ValueError("score levels need descriptions")
            use_key = True
            semantic = (
                f"Level: {key} of {len(options) - 1}\n"
                f"Criterion: {_payload(description)}"
            )
        prompts.append(
            CandidatePrompt(
                key=key,
                prefix=prefix,
                tail=f"Candidate:\n{semantic}\nDecision score:",
                key_used_as_semantics=use_key,
            )
        )
    return tuple(prompts)


def score_independent(
    row: dict[str, Any], score: Callable[[str], float]
) -> dict[str, Any]:
    """Apply one shared scalar scorer and reconstruct native decision values.

    Confidence is intentionally omitted: the external API does not specify a
    reproducible formula, and this prototype must not pretend to be its full
    runtime adapter.  Calibration remains a separate experiment.
    """
    prompts = candidate_prompts(row)
    logits = {item.key: float(score(item.text)) for item in prompts}
    if not all(math.isfinite(value) for value in logits.values()):
        raise ValueError("candidate scorer returned a non-finite logit")
    maximum = max(logits.values())
    exponentials = {key: math.exp(value - maximum) for key, value in logits.items()}
    # Stable summation order makes the probabilities bit-identical under a
    # permutation of the input option list, not merely close numerically.
    denominator = math.fsum(exponentials[key] for key in sorted(exponentials))
    probabilities = {key: exponentials[key] / denominator for key in sorted(logits)}
    task_type = row["task_type"]
    if task_type == "noul":
        return {"type": "noul", "noul": probabilities["true"]}
    if task_type == "choice":
        winner = max(sorted(probabilities), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": probabilities,
        }
    levels = len(prompts)
    return {
        "type": "score",
        "score": math.fsum(
            level * probabilities[str(level)] for level in range(levels)
        ),
        "legend": {
            str(level): row_description(row, str(level)) for level in range(levels)
        },
        "probabilities": probabilities,
    }


def row_description(row: dict[str, Any], key: str) -> str:
    return _payload(
        next(option["description"] for option in row["options"] if option["key"] == key)
    )


def candidate_token_ids(
    row: dict[str, Any], tokenizer: Any
) -> tuple[tuple[str, list[int]], ...]:
    """Encode one exact shared token prefix and independent option branches."""
    prompts = candidate_prompts(row)
    prefix_ids = tokenizer.encode(prompts[0].prefix, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError("tokenizer returned an empty shared prefix")
    return tuple(
        (
            item.key,
            prefix_ids + tokenizer.encode(item.tail, add_special_tokens=False),
        )
        for item in prompts
    )


def token_work(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Count input tokens only; this is not measured FLOPs or latency.

    The reference path mirrors the current segmented renderer's separate
    prefix/option/suffix tokenizer calls.  Independent prompts likewise use
    separately encoded shared prefixes and branch tails.  This guarantees
    exact prefix token identity; cache reuse still needs a model parity proof.
    """
    prefix = (
        f"Context:\n{_payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n{_payload(row['instructions'])}\nOptions:"
    )
    suffix = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
    reference_options = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in row["options"]
    ]
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    reference_tokens = (
        len(encode(prefix))
        + sum(len(encode(option)) for option in reference_options)
        + len(encode(suffix))
    )
    prompts = candidate_prompts(row)
    prefix_tokens = len(encode(prompts[0].prefix))
    independent_tokens = [prefix_tokens + len(encode(item.tail)) for item in prompts]
    if not all(independent_tokens):
        raise ValueError("tokenizer returned an empty independent prompt")
    # Idealized lower bound only: even correct KV forking still computes every
    # branch and rereads the prefix during branch attention.
    ideal_shared_prefix_tokens = prefix_tokens + sum(
        max(0, value - prefix_tokens) for value in independent_tokens
    )
    return {
        "candidate_count": len(independent_tokens),
        "reference_tokens": reference_tokens,
        "independent_total_tokens": sum(independent_tokens),
        "independent_longest_prompt_tokens": max(independent_tokens),
        "independent_shortest_prompt_tokens": min(independent_tokens),
        "shared_prefix_tokens": prefix_tokens,
        "independent_tail_tokens": [
            value - prefix_tokens for value in independent_tokens
        ],
        "ideal_shared_prefix_tokens": ideal_shared_prefix_tokens,
    }
