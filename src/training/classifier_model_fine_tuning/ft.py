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

import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Load the MMLU-Pro dataset
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
print(f"Dataset splits: {dataset.keys()}")

# Extract questions and categories from the test split
questions = dataset["test"]["question"]
categories = dataset["test"]["category"]

# Get unique categories and create a mapping
unique_categories = list(set(categories))
category_to_idx = {category: idx for idx, category in enumerate(unique_categories)}
idx_to_category = {idx: category for category, idx in category_to_idx.items()}

print(f"Found {len(unique_categories)} unique categories: {unique_categories}")

# Convert categories to indices
category_indices = [category_to_idx[category] for category in categories]

# Split the data into train and validation sets
train_questions, val_questions, train_categories, val_categories = train_test_split(
    questions, category_indices, test_size=0.2, stratify=category_indices
)

# Create a custom model with classification head
word_embedding_model = models.Transformer("sentence-transformers/all-MiniLM-L12-v2")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=len(unique_categories),
    activation_function=torch.nn.Identity(),
)

# Create the full model pipeline
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# Prepare training data
train_samples = [
    (question, category)
    for question, category in zip(train_questions, train_categories)
]

# Define the loss function (cross entropy for classification)
train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)

# Configure the training
num_epochs = 2
warmup_steps = int(len(train_samples) * 0.1 / 16)  # 10% of training samples

train_examples = []
for question, category in train_samples:
    train_examples.append(InputExample(texts=[question], label=category))

# Create DataLoader
batch_size = 16
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=None,
    output_path="category_classifier_model",
)

# Save the category mapping
with open("category_classifier_model/category_mapping.json", "w") as f:
    json.dump(
        {
            "category_to_idx": category_to_idx,
            "idx_to_category": {str(k): v for k, v in idx_to_category.items()},
        },
        f,
    )


# Function to predict category using the model
def predict_category(
    model, question, mapping_path="category_classifier_model/category_mapping.json"
):
    # Load the category mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        idx_to_category = {int(k): v for k, v in mapping["idx_to_category"].items()}

    # Get the embedding for the question
    embedding = model.encode(question)

    # Find nearest category using cosine similarity
    # Create a reference set of embeddings for each category
    embeddings_per_category = {}
    for idx, cat_name in idx_to_category.items():
        # Use random samples from training set for this category
        cat_questions = [
            q for q, c in zip(train_questions, train_categories) if c == idx
        ]
        if cat_questions:
            # Take up to 5 samples per category
            samples = cat_questions[: min(5, len(cat_questions))]
            cat_embeddings = model.encode(samples)
            embeddings_per_category[idx] = np.mean(cat_embeddings, axis=0)

    # Calculate similarity to each category
    similarities = {}
    for cat_idx, cat_embedding in embeddings_per_category.items():
        similarity = np.dot(embedding, cat_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(cat_embedding)
        )
        similarities[cat_idx] = similarity

    # Find the most similar category
    predicted_idx = max(similarities, key=similarities.get)
    predicted_category = idx_to_category[predicted_idx]

    return predicted_category, similarities


# Evaluate on validation set
def evaluate_classifier(model, questions, categories):
    correct = 0
    total = len(questions)

    for question, true_category in zip(questions, categories):
        predicted_category, _ = predict_category(model, question)
        if idx_to_category[true_category] == predicted_category:
            correct += 1

    return correct / total


# Evaluate the model
print("\nEvaluating on validation set...")
val_accuracy = evaluate_classifier(model, val_questions, val_categories)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Calculate and save category embeddings for future use
print("Saving category embeddings...")
embeddings_per_category = {}
for idx, cat_name in idx_to_category.items():
    # Use random samples from training set for this category
    cat_questions = [q for q, c in zip(train_questions, train_categories) if c == idx]
    if cat_questions:
        # Take up to 5 samples per category
        samples = cat_questions[: min(5, len(cat_questions))]
        cat_embeddings = model.encode(samples)
        embeddings_per_category[idx] = np.mean(
            cat_embeddings, axis=0
        ).tolist()  # Convert to list for JSON serialization

# Save category embeddings
with open("category_classifier_model/category_embeddings.json", "w") as f:
    json.dump(embeddings_per_category, f)

# Save the model
model.save("category_classifier_model")

# Test the model with examples
print("\nTesting with examples:")
queries = [
    "What is the derivative of x^2?",
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

for i in range(len(queries)):
    question = queries[i]
    predicted_category, confidence = predict_category(model, question)
    print(f"Question: {question[:100]}...")
    print(f"Predicted category: {predicted_category}, Confidence: {confidence}")
    print("---")
