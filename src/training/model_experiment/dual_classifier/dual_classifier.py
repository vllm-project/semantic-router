from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class DualClassifier(nn.Module):
    """
    A dual-purpose classifier using DistilBERT for both category classification and PII detection.

    This model uses a shared DistilBERT backbone with two classification heads:
    1. A sequence classification head for category prediction
    2. A token classification head for PII detection
    """

    def __init__(
        self,
        num_categories: int,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
    ):
        super().__init__()

        # Load base DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.base_model = DistilBertModel.from_pretrained(model_name)
        self.max_length = max_length

        # Get the hidden size from the base model config
        hidden_size = self.base_model.config.hidden_size

        # Category classification head
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_categories),
        )

        # PII detection head (binary classification for each token)
        self.pii_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),  # Binary classification: PII or not PII
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention mask

        Returns:
            Tuple containing:
            - category_logits: Logits for category classification
            - pii_logits: Logits for PII detection per token
        """
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence output (last hidden states)
        sequence_output = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_len, hidden_size)

        # For category classification, use the [CLS] token (first token)
        cls_output = sequence_output[:, 0, :]  # Shape: (batch_size, hidden_size)
        category_logits = self.category_classifier(
            cls_output
        )  # Shape: (batch_size, num_categories)

        # For PII detection, classify each token
        pii_logits = self.pii_classifier(
            sequence_output
        )  # Shape: (batch_size, seq_len, 2)

        return category_logits, pii_logits

    def encode_text(
        self, text: Union[str, List[str]], device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text input for the model.

        Args:
            text: Input text or list of texts
            device: Target device for tensors

        Returns:
            Dictionary containing input_ids and attention_mask
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]

        # Tokenize the input
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device if specified
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        return encoded

    def predict(
        self, text: Union[str, List[str]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for both category and PII detection.

        Args:
            text: Input text or list of texts
            device: Target device for computation

        Returns:
            Tuple containing:
            - category_probs: Probabilities for each category
            - pii_probs: Probabilities of PII for each token
        """
        # Encode the input
        encoded = self.encode_text(text, device)

        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            # Get logits
            category_logits, pii_logits = self(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )

            # Convert to probabilities
            category_probs = torch.softmax(category_logits, dim=-1)
            pii_probs = torch.softmax(pii_logits, dim=-1)

        return category_probs, pii_probs

    def save_pretrained(self, path: str):
        """Save the model and tokenizer to the specified path."""
        # Save the model state
        torch.save(self.state_dict(), f"{path}/model.pt")

        # Save the tokenizer
        self.tokenizer.save_pretrained(path)

        # Save the base model configuration
        self.base_model.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, num_categories: int):
        """Load a pretrained model from the specified path."""
        # Create a new instance
        model = cls(num_categories=num_categories)

        # Load the saved state
        state_dict = torch.load(f"{path}/model.pt")
        model.load_state_dict(state_dict)

        return model
