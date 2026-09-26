"""LFM2.5-Encoder as a sequence classifier: pooling, head, loading and prediction.

The encoder has no sequence-classification variant, so a masked-mean-pool plus a linear
head is put on top of its body. The pieces are separate functions so the pooling, the
wrapper and the prediction loop can be tested with a fake encoder.
"""

import torch
from exploration_common import LFM25_MODEL_ID, LFM25_REVISION, MODALITY_LABELS
from modality_routing_bert_finetuning_lora import FocalLoss
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

# LoRA target modules, from direct introspection of named_modules() on the correctly
# loaded body. The architecture interleaves two block types:
#   - conv blocks: conv.in_proj / conv.out_proj (short-conv, linear projections)
#   - attention blocks (layers 2,5,8,10,12,14 only): self_attn.q/k/v/out_proj
# plus feed_forward.w1/w2/w3 (gated SwiGLU-style MLP) present on every layer.
LORA_TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "conv.in_proj",
    "conv.out_proj",
]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the token states of each sequence, ignoring padding.

    Args:
        hidden: Token states of shape [batch, tokens, hidden].
        attention_mask: Mask of shape [batch, tokens], 1 for real tokens.

    Returns:
        Pooled states of shape [batch, hidden].
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


def hidden_size_of(body: nn.Module) -> int:
    """Read the hidden size of an encoder body, plain or wrapped by PEFT.

    Args:
        body: The encoder, or a PEFT model around it.

    Returns:
        The hidden size.
    """
    config = body.config if hasattr(body, "config") else body.base_model.config
    return config.hidden_size


class Lfm2ForModalityClassification(nn.Module):
    """Encoder body, masked-mean pooling and a linear classification head."""

    def __init__(
        self,
        body: nn.Module,
        num_labels: int,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        dropout: float = 0.1,
    ):
        """Wrap an encoder body with a pooled classification head.

        Args:
            body: The encoder, possibly with a LoRA adapter applied.
            num_labels: Number of classes.
            class_weights: Per-class weights for the Focal Loss, or None.
            focal_gamma: Focusing parameter of the Focal Loss.
            dropout: Dropout applied to the pooled states.
        """
        super().__init__()
        self.body = body
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size_of(body), num_labels)
        self.num_labels = num_labels
        self.focal_loss = FocalLoss(
            alpha=class_weights, gamma=focal_gamma, reduction="mean"
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Classify a batch and, given labels, compute the Focal Loss.

        Args:
            input_ids: Token ids of shape [batch, tokens].
            attention_mask: Mask of shape [batch, tokens].
            labels: True class ids of shape [batch], or None.
            **kwargs: Ignored; accepted so the Trainer can pass extra columns.

        Returns:
            A SequenceClassifierOutput with the logits and, if labels were given,
            the loss.
        """
        outputs = self.body(input_ids=input_ids, attention_mask=attention_mask)
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        logits = self.classifier(self.dropout(mean_pool(hidden, attention_mask)))
        loss = self.focal_loss(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_lfm25_tokenizer(
    model_id: str = LFM25_MODEL_ID, revision: str = LFM25_REVISION
):
    """Load the LFM2.5-Encoder tokenizer.

    The revision is pinned because the checkpoint runs custom code from the Hub.

    Args:
        model_id: Hub repo id, or a local directory holding a saved tokenizer.
        revision: Hub revision to load; ignored for a local directory.

    Returns:
        The tokenizer.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )


def load_lfm25_body(model_id: str = LFM25_MODEL_ID, revision: str = LFM25_REVISION):
    """Load the pretrained LFM2.5-Encoder body.

    The model card's `AutoModel.from_pretrained` path returns a randomly initialised
    model for this checkpoint, because the real weights sit under an `lfm2.` prefix
    that the bare encoder does not expect. Loading the masked-LM model and taking its
    `.lfm2` attribute gets the pretrained weights. The revision is pinned because the
    checkpoint runs custom code from the Hub.

    Args:
        model_id: Hub repo id.
        revision: Hub revision to load.

    Returns:
        The encoder body.
    """
    from transformers import AutoModelForMaskedLM  # noqa: PLC0415

    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    return mlm_model.lfm2


def predict_labels(
    body: nn.Module,
    head: nn.Module,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[str]:
    """Predict a label name for each text.

    Args:
        body: The encoder, already on device and in eval mode.
        head: The linear classification head, likewise.
        tokenizer: The encoder's tokenizer.
        texts: Prompt texts to classify.
        batch_size: Number of texts per forward pass.
        max_length: Token limit; longer texts are truncated.
        device: Device the model is on.

    Returns:
        Predicted label names, in the order of texts.
    """
    preds: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = body(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).last_hidden_state
            pooled = mean_pool(hidden, enc["attention_mask"])
            preds.extend(
                MODALITY_LABELS[i] for i in head(pooled.float()).argmax(-1).tolist()
            )
    return preds
