"""Standard absolute ModernBERT positions for export and ORT verification."""

import numpy as np

_TOKEN_RANK = 2


def ort_inputs(session, ids, mask):
    """Support legacy two-input graphs and the explicit broadcast position row."""
    if ids.ndim != _TOKEN_RANK or ids.shape != mask.shape or not all(ids.shape):
        raise ValueError(
            "ModernBERT inputs require equal nonempty [batch, sequence] shapes"
        )
    if ids.dtype != np.int64 or mask.dtype != np.int64:
        raise ValueError("ModernBERT token IDs and masks must be int64")
    declared = {value.name: value for value in session.get_inputs()}
    if set(declared) not in (
        {"input_ids", "attention_mask"},
        {"input_ids", "attention_mask", "position_ids"},
    ):
        raise ValueError("Unsupported ModernBERT graph inputs")
    for schema in declared.values():
        if schema.type != "tensor(int64)" or len(schema.shape) != _TOKEN_RANK:
            raise ValueError("ModernBERT graph inputs must be rank-two int64 tensors")
    values = {"input_ids": ids, "attention_mask": mask}
    if "position_ids" in declared:
        # Match native default positions even for left padding or internal holes.
        values["position_ids"] = np.arange(ids.shape[1], dtype=np.int64)[None, :]
        if declared["position_ids"].shape[0] != 1:
            raise ValueError("ModernBERT position_ids must declare one broadcast row")
    for name, value in values.items():
        schema = declared[name]
        if any(
            isinstance(wanted, int) and wanted != actual
            for wanted, actual in zip(schema.shape, value.shape, strict=True)
        ):
            raise ValueError(
                "ModernBERT graph input shape differs from execution shape"
            )
    return values
