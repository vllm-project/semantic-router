import pytest
import torch
from dual_classifier import DualClassifier


@pytest.fixture
def model():
    """Create a test model instance."""
    return DualClassifier(num_categories=5)  # 5 categories for testing


def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert isinstance(model, DualClassifier)
    assert model.tokenizer is not None
    assert model.base_model is not None
    assert model.category_classifier is not None
    assert model.pii_classifier is not None


def test_encode_text(model):
    """Test text encoding functionality."""
    # Test single text
    text = "This is a test sentence."
    encoded = model.encode_text(text)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape[0] == 1  # batch size 1

    # Test multiple texts
    texts = ["First sentence.", "Second sentence."]
    encoded = model.encode_text(texts)
    assert encoded["input_ids"].shape[0] == 2  # batch size 2


def test_forward_pass(model):
    """Test the forward pass of the model."""
    # Create dummy input
    text = "This is a test sentence."
    encoded = model.encode_text(text)

    # Run forward pass
    category_logits, pii_logits = model(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    )

    # Check shapes
    assert category_logits.shape == (1, 5)  # (batch_size, num_categories)
    assert pii_logits.shape[0] == 1  # batch size
    assert pii_logits.shape[2] == 2  # binary classification


def test_prediction(model):
    """Test the prediction functionality."""
    # Test single text
    text = "This is a test sentence with email john@example.com"
    category_probs, pii_probs = model.predict(text)

    # Check probability distributions
    assert torch.allclose(category_probs.sum(dim=1), torch.tensor([1.0]))
    assert torch.allclose(
        pii_probs.sum(dim=2), torch.tensor([1.0]).expand_as(pii_probs.sum(dim=2))
    )

    # Test multiple texts
    texts = [
        "First sentence with phone 123-456-7890",
        "Second sentence with name John Smith",
    ]
    category_probs, pii_probs = model.predict(texts)

    # Check shapes and probability distributions
    assert category_probs.shape == (2, 5)  # (batch_size, num_categories)
    assert pii_probs.shape[0] == 2  # batch size
    assert torch.allclose(category_probs.sum(dim=1), torch.tensor([1.0, 1.0]))


def test_save_load(model, tmp_path):
    """Test model saving and loading."""
    # Save the model
    save_path = tmp_path / "test_model"
    save_path.mkdir()
    model.save_pretrained(str(save_path))

    # Load the model
    loaded_model = DualClassifier.from_pretrained(str(save_path), num_categories=5)

    # Verify the loaded model works
    text = "Test sentence"
    original_output = model.predict(text)
    loaded_output = loaded_model.predict(text)

    # Check that outputs match
    assert torch.allclose(original_output[0], loaded_output[0])  # category probs
    assert torch.allclose(original_output[1], loaded_output[1])  # pii probs
