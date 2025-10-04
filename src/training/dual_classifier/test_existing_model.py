# Copyright 2025 The vLLM Semantic Router Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
