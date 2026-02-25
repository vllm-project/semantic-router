import torch
from dual_classifier import DualClassifier


def main():
    # Initialize model with 10 categories (example)
    model = DualClassifier(num_categories=10)

    # Example texts with different categories and PII
    texts = [
        "What is the derivative of x^2?",  # Math category, no PII
        "My name is John Smith and my email is john@example.com",  # Personal info with PII
        "The Treaty of Versailles was signed in 1919",  # History category, no PII
        "Please contact Sarah at 123-456-7890 for support",  # Contact info with PII
    ]

    # Make predictions
    print("Making predictions...")
    category_probs, pii_probs = model.predict(texts)

    # Process results
    for i, text in enumerate(texts):
        print(f"\nText: {text}")

        # Print category prediction
        category_idx = torch.argmax(category_probs[i]).item()
        category_confidence = category_probs[i][category_idx].item()
        print(f"Category: {category_idx} (confidence: {category_confidence:.2f})")

        # Print PII detection
        # Get token-level predictions
        pii_predictions = torch.argmax(pii_probs[i], dim=-1)

        # Get the tokens
        tokens = model.tokenizer.tokenize(text)

        # Print tokens marked as PII
        pii_tokens = [
            (token, idx)
            for idx, (token, pred) in enumerate(zip(tokens, pii_predictions))
            if pred == 1
        ]
        if pii_tokens:
            print("Detected PII tokens:")
            for token, idx in pii_tokens:
                print(f"  - {token} (position {idx})")
        else:
            print("No PII detected")


if __name__ == "__main__":
    main()
