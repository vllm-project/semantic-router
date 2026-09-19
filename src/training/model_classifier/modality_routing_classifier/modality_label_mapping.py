"""Checkpoint label mapping for the modality-routing classifiers.

A checkpoint may order its classes differently from the canonical
AR=0 / DIFFUSION=1 / BOTH=2, so anything that reads its logits (evaluation, or
distillation from a teacher) has to go through the checkpoint's label names.
Both helpers fail closed: an incompatible mapping is an error, never a guess.
"""

import numpy as np
from modality_routing_bert_finetuning_lora import MODALITY_LABELS


def build_label_remap(
    id2label: dict[int, str], model_path: str = "<checkpoint>"
) -> dict[int, int]:
    """Map checkpoint class ids to canonical MODALITY_LABELS ids, or raise.

    Raises unless id2label is a one-to-one mapping onto the canonical labels. A
    missing, partial or generic one (for example LABEL_0/LABEL_1/LABEL_2) is an
    error, because falling back to the raw class id would produce valid-looking
    predictions that are scored against the wrong classes.

    Args:
        id2label: The checkpoint's class id to label name mapping.
        model_path: Checkpoint name, used only in error messages.

    Returns:
        Mapping from checkpoint class id to canonical class id.

    Raises:
        ValueError: If id2label is not a one-to-one mapping onto the canonical labels.
    """
    canonical_id = {label: idx for idx, label in enumerate(MODALITY_LABELS)}
    unknown = sorted(
        {str(name) for name in id2label.values() if name not in canonical_id}
    )
    if unknown or sorted(id2label.values()) != sorted(MODALITY_LABELS):
        raise ValueError(
            f"{model_path}: id2label {dict(id2label)} is not a one-to-one mapping onto "
            f"{MODALITY_LABELS}"
            + (f" (unrecognised labels: {unknown})" if unknown else "")
            + "; refusing to guess the class order"
        )
    return {ckpt_id: canonical_id[name] for ckpt_id, name in id2label.items()}


def check_output_size(
    remap: dict[int, int], output_size: int, model_path: str = "<checkpoint>"
) -> None:
    """Check that the classifier has exactly one output per mapped class id.

    Args:
        remap: Mapping from checkpoint class id to canonical class id.
        output_size: Number of outputs the classifier head produces.
        model_path: Checkpoint name, used only in error messages.

    Raises:
        ValueError: If the output size or the mapped ids do not match the canonical labels.
    """
    if output_size != len(MODALITY_LABELS) or set(remap) != set(range(output_size)):
        raise ValueError(
            f"{model_path}: classifier has {output_size} outputs but id2label covers "
            f"ids {sorted(remap)}; expected 0..{len(MODALITY_LABELS) - 1}"
        )


def logits_to_canonical_order(logits: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Reorder the class columns of logits into canonical label order.

    Args:
        logits: Array of shape [N, C] in the checkpoint's class order.
        remap: Mapping from checkpoint class id to canonical class id.

    Returns:
        Array of shape [N, C] whose columns follow MODALITY_LABELS.
    """
    canonical = np.empty_like(logits)
    for ckpt_id, canonical_id in remap.items():
        canonical[:, canonical_id] = logits[:, ckpt_id]
    return canonical
