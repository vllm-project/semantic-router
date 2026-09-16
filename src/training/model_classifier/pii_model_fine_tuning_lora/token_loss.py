"""Explicit document weighting for token classification training."""

LOGIT_RANK = 3
IGNORE_INDEX = -100


def document_mean_loss(logits, labels, attention_mask):
    """Average each document's attended, nonignored token CE, then documents.

    Each document must have supervision. The trainer draws equally sized
    microbatches, so dividing this mean by the accumulation count gives equal
    weight to every document in the logical optimizer batch.
    """
    token_losses, valid = _token_losses(logits, labels, attention_mask)
    return (token_losses.sum(dim=1) / valid.sum(dim=1)).mean()


def _token_losses(logits, labels, attention_mask):
    from torch.nn import functional  # noqa: PLC0415

    if (
        logits.ndim != LOGIT_RANK
        or labels.shape != logits.shape[:2]
        or attention_mask.shape != labels.shape
        or not logits.shape[0]
    ):
        raise ValueError("Expected batch/token logits, labels and attention mask")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("Attention mask must contain only zero or one")
    valid = (labels != IGNORE_INDEX) & (attention_mask == 1)
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every document needs at least one supervised token")
    effective_labels = labels.masked_fill(~valid, IGNORE_INDEX)
    token_losses = functional.cross_entropy(
        logits.float().transpose(1, 2),
        effective_labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    return token_losses, valid


def entity_document_mean_loss(
    logits, labels, attention_mask, entity_ids, *, o_label_id
):
    """Give entities and O tokens half the mass in each positive document.

    Entity IDs are explicit annotation-span IDs, local to a document, and must
    never be inferred from model predictions. Each entity contributes its mean
    subword CE. Fully negative documents retain their complete O-token mean;
    documents without O tokens use only the mean of entity means. Documents
    then receive equal weight. Ignored or masked positions enter neither term.
    """
    import torch  # noqa: PLC0415

    token_losses, valid = _token_losses(logits, labels, attention_mask)
    if entity_ids.shape != labels.shape or entity_ids.dtype != torch.long:
        raise ValueError("Entity IDs must be int64 with the label shape")
    if type(o_label_id) is not int or not 0 <= o_label_id < logits.shape[-1]:
        raise ValueError("O label ID must identify a logit class")
    entity_tokens = valid & (labels != o_label_id)
    if bool((entity_ids[entity_tokens] < 0).any()) or bool(
        (entity_ids[valid & ~entity_tokens] != -1).any()
    ):
        raise ValueError("Entity tokens need span IDs; observed O tokens need -1")
    document_losses = []
    for index in range(logits.shape[0]):
        entity_mask = entity_tokens[index]
        o_mask = valid[index] & ~entity_mask
        terms = []
        if bool(entity_mask.any()):
            _, inverse, counts = entity_ids[index, entity_mask].unique(
                return_inverse=True, return_counts=True
            )
            sums = token_losses.new_zeros(counts.shape).scatter_add(
                0, inverse, token_losses[index, entity_mask]
            )
            terms.append((sums / counts).mean())
        if bool(o_mask.any()):
            terms.append(token_losses[index, o_mask].mean())
        document_losses.append(torch.stack(terms).mean())
    return torch.stack(document_losses).mean()


def pad_entity_ids(features, padded_length, padding_side):
    """Pad loss metadata separately; these IDs are never model inputs."""
    import torch  # noqa: PLC0415

    if padding_side not in {"left", "right"}:
        raise ValueError("Unsupported tokenizer padding side")
    padded = []
    for feature in features:
        ids = feature["entity_ids"]
        if len(ids) != len(feature["input_ids"]) or len(ids) > padded_length:
            raise ValueError("Entity IDs must cover every input token before padding")
        padding = [-1] * (padded_length - len(ids))
        padded.append(padding + ids if padding_side == "left" else ids + padding)
    return torch.tensor(padded, dtype=torch.long)
