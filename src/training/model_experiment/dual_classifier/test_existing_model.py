import torch
from dual_classifier import DualClassifier


def test_existing_model():
    """Test that our existing trained model works correctly."""

    # Test loading our existing trained model
    model = DualClassifier.from_pretrained("trained_model/", num_categories=10)
    print("✅ Successfully loaded existing trained model")

    # Test prediction
    test_texts = [
        "What is the derivative of x^2?",
        "My email is john@test.com. How does DNA work?",
        "Call me at 555-123-4567 for science help.",
    ]

    print("\n🧪 Testing trained model predictions:")
    for i, text in enumerate(test_texts):
        cat_probs, pii_probs = model.predict(text)
        cat_pred = torch.argmax(cat_probs[0]).item()
        confidence = cat_probs[0][cat_pred].item()

        # Check for PII tokens
        tokens = model.tokenizer.tokenize(text)
        pii_preds = torch.argmax(pii_probs[0], dim=-1)
        pii_tokens = [token for token, pred in zip(tokens, pii_preds) if pred == 1]

        print(f"  Test {i+1}: {text}")
        print(f"    Category: {cat_pred} (confidence: {confidence:.3f})")
        print(f'    PII tokens: {pii_tokens if pii_tokens else "None detected"}')

    print("\n✅ All model tests completed successfully!")


if __name__ == "__main__":
    test_existing_model()
