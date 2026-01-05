import json
import os
import traceback

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_category_mapping(path):
    """Load the category mapping from a JSON file"""
    with open(path, "r") as f:
        mapping = json.load(f)
    return mapping


def predict_category(
    model_path,
    question,
    mapping_path="../../../models/mom-domain-classifier/category_mapping.json",
):
    """Predict category for a given question using the trained model"""
    # Load the category mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        idx_to_category = {int(k): v for k, v in mapping["idx_to_category"].items()}

    # Load the model
    model = SentenceTransformer(model_path)

    # Get logits directly from model.encode
    logits_np = model.encode(question, show_progress_bar=False)
    logits_tensor = torch.tensor(logits_np)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits_tensor, dim=0)

    # Get the predicted class index and confidence
    confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx_tensor.item()
    confidence = confidence_tensor.item()

    predicted_category = idx_to_category.get(
        predicted_idx, f"Unknown category {predicted_idx}"
    )

    return predicted_category, confidence


def main():
    print("Domain Classifier Test (Python Model)")
    print("====================================")
    print()

    # Try to load the category mapping
    mapping_path = "../../../models/mom-domain-classifier/category_mapping.json"
    try:
        mapping = load_category_mapping(mapping_path)
        print("Successfully loaded category mapping")
    except Exception as e:
        print(f"Failed to load category mapping: {e}")
        return {}

    # Load the model
    model_path = "../../../models/mom-domain-classifier"
    num_classes = len(mapping["category_to_idx"])
    print(f"Using classifier with {num_classes} classes...")

    # Test queries
    queries = [
        "What is the derivative of x^2?",  # Updated to match ft.py
        "Explain the concept of supply and demand in economics.",
        "How does DNA replication work in eukaryotic cells?",
        "What is the difference between a civil law and common law system?",
        "Explain how transistors work in computer processors.",
        "Why do stars twinkle?",
        "How do I create a balanced portfolio for retirement?",
        "What causes mental illnesses?",
        "How do computer algorithms work?",
        "Explain the historical significance of the Roman Empire.",
    ]

    # Create a dictionary to store results for comparison
    results = {}

    # Process each query
    print("\nClassifying queries:")
    print("==================")

    for i, query in enumerate(queries):
        try:
            # Call the predict_category function
            category_name, confidence = predict_category(
                model_path, query, mapping_path
            )

            # Print the result
            print(f"{i+1}. Query: {query}")
            print(f"   Classified as: {category_name} (Confidence: {confidence:.4f})")
            print()

            # Store the result in the dictionary
            results[query] = {
                "category": category_name,
                "confidence": float(confidence),
                "class_id": next(
                    (
                        int(idx)
                        for idx, name in mapping["idx_to_category"].items()
                        if name == category_name
                    ),
                    None,
                ),
            }

        except Exception as e:
            print(f"Query {i+1}: Classification failed: {e}")
            traceback.print_exc()
            continue

    print("\nTest complete!")

    return results


if __name__ == "__main__":
    # Run the classification
    results = main()

    # Save the results to a file for comparison
    if results:
        try:
            # Save Python results
            with open("python_classification_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to python_classification_results.json")
        except Exception as e:
            print(f"Failed to save results: {e}")
