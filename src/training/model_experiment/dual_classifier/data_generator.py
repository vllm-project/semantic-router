import random
import re
from typing import Dict, List, Tuple

import numpy as np


class SyntheticDataGenerator:
    """
    Generate synthetic data for testing dual-task training.
    Creates both category classification and PII detection labels.
    """

    def __init__(self):
        # Category templates
        self.categories = {
            0: "math",
            1: "science",
            2: "history",
            3: "technology",
            4: "literature",
            5: "geography",
            6: "economics",
            7: "medicine",
            8: "sports",
            9: "general",
        }

        # Sample texts for each category
        self.category_templates = {
            0: [  # math
                "What is the derivative of x^2?",
                "Solve for x: 2x + 5 = 15",
                "Calculate the area of a circle with radius 5",
                "Find the limit as x approaches 0",
                "What is the integral of sin(x)?",
                "Prove that the square root of 2 is irrational",
            ],
            1: [  # science
                "Explain photosynthesis in plants",
                "What are the laws of thermodynamics?",
                "How does DNA replication work?",
                "Describe the structure of an atom",
                "What causes gravity?",
                "Explain the theory of evolution",
            ],
            2: [  # history
                "When was the American Civil War?",
                "Who was Napoleon Bonaparte?",
                "What caused World War I?",
                "Describe the Renaissance period",
                "What was the Cold War?",
                "When did the Roman Empire fall?",
            ],
            3: [  # technology
                "How do computer processors work?",
                "What is artificial intelligence?",
                "Explain blockchain technology",
                "How does the internet work?",
                "What is machine learning?",
                "Describe quantum computing",
            ],
            4: [  # literature
                "Who wrote Romeo and Juliet?",
                "What is the theme of Moby Dick?",
                "Analyze the character of Hamlet",
                "Who is the author of 1984?",
                "What is symbolism in literature?",
                "Describe the Romantic period in literature",
            ],
            5: [  # geography
                "What is the capital of France?",
                "Where is the Amazon rainforest?",
                "What are tectonic plates?",
                "Name the largest ocean",
                "What causes earthquakes?",
                "Where is Mount Everest located?",
            ],
            6: [  # economics
                "What is supply and demand?",
                "Explain inflation and deflation",
                "What is GDP?",
                "How do stock markets work?",
                "What is fiscal policy?",
                "Explain the concept of opportunity cost",
            ],
            7: [  # medicine
                "What is diabetes?",
                "How does the immune system work?",
                "What causes heart disease?",
                "Explain how vaccines work",
                "What is cancer?",
                "How do antibiotics work?",
            ],
            8: [  # sports
                "What are the rules of basketball?",
                "Who won the last World Cup?",
                "How long is a marathon?",
                "What is the offside rule in soccer?",
                "Who holds the record for home runs?",
                "What sports are in the Olympics?",
            ],
            9: [  # general
                "What time is it?",
                "How are you today?",
                "What's the weather like?",
                "Tell me a joke",
                "What's for dinner?",
                "How was your day?",
            ],
        }

        # PII patterns to inject
        self.pii_patterns = {
            "email": [
                "john.doe@email.com",
                "sarah.smith@company.org",
                "mike.johnson@university.edu",
                "anna.brown@gmail.com",
                "david.wilson@yahoo.com",
            ],
            "phone": [
                "123-456-7890",
                "(555) 123-4567",
                "800-555-0123",
                "+1-234-567-8900",
                "555.123.4567",
            ],
            "name": [
                "John Smith",
                "Sarah Johnson",
                "Michael Brown",
                "Emily Davis",
                "David Wilson",
                "Anna Garcia",
                "James Miller",
                "Lisa Anderson",
            ],
            "address": [
                "123 Main Street, New York, NY",
                "456 Oak Avenue, Los Angeles, CA",
                "789 Pine Road, Chicago, IL",
                "321 Elm Street, Houston, TX",
                "654 Maple Drive, Phoenix, AZ",
            ],
            "ssn": [
                "123-45-6789",
                "987-65-4321",
                "555-44-3333",
                "111-22-3333",
                "999-88-7777",
            ],
        }

    def generate_sample(
        self, inject_pii_prob: float = 0.3
    ) -> Tuple[str, int, List[int]]:
        """
        Generate a single sample with category and PII labels.

        Args:
            inject_pii_prob: Probability of injecting PII into the text

        Returns:
            (text, category_label, pii_labels)
        """
        # Choose random category
        category = random.randint(0, 9)
        base_text = random.choice(self.category_templates[category])

        # Decide whether to inject PII
        text = base_text
        if random.random() < inject_pii_prob:
            # Choose random PII type and value
            pii_type = random.choice(list(self.pii_patterns.keys()))
            pii_value = random.choice(self.pii_patterns[pii_type])

            # Create text with PII
            pii_templates = [
                f"My {pii_type} is {pii_value}. {base_text}",
                f"{base_text} Contact me at {pii_value}.",
                f"Hi, I'm {pii_value}. {base_text}",
                f"{base_text} You can reach me at {pii_value}.",
                f"From {pii_value}: {base_text}",
            ]
            text = random.choice(pii_templates)

        # Generate PII labels (simplified token-level labeling)
        pii_labels = self._generate_pii_labels(text)

        return text, category, pii_labels

    def _generate_pii_labels(self, text: str) -> List[int]:
        """
        Generate token-level PII labels (simplified).
        0 = not PII, 1 = PII
        """
        # Simple word-level tokenization for demonstration
        words = text.split()
        labels = []

        for word in words:
            # Check if word contains PII patterns
            is_pii = 0

            # Email pattern
            if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", word):
                is_pii = 1
            # Phone pattern
            elif re.search(r"\b\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b", word):
                is_pii = 1
            # SSN pattern
            elif re.search(r"\b\d{3}-\d{2}-\d{4}\b", word):
                is_pii = 1
            # Simple name pattern (capitalized words that might be names)
            elif (
                word[0].isupper()
                and len(word) > 1
                and any(
                    name in text
                    for names in self.pii_patterns["name"]
                    for name in names.split()
                )
            ):
                is_pii = 1

            labels.append(is_pii)

        return labels

    def generate_dataset(
        self, num_samples: int, pii_ratio: float = 0.3
    ) -> Tuple[List[str], List[int], List[List[int]]]:
        """
        Generate a complete dataset.

        Args:
            num_samples: Number of samples to generate
            pii_ratio: Proportion of samples that should contain PII

        Returns:
            (texts, category_labels, pii_labels)
        """
        texts = []
        category_labels = []
        pii_labels = []

        for _ in range(num_samples):
            text, category, pii_label = self.generate_sample(pii_ratio)
            texts.append(text)
            category_labels.append(category)
            pii_labels.append(pii_label)

        return texts, category_labels, pii_labels

    def get_category_info(self) -> Dict[int, str]:
        """Return mapping of category IDs to names."""
        return self.categories.copy()


def create_sample_datasets(
    train_size: int = 100, val_size: int = 20, pii_ratio: float = 0.3
) -> Tuple[
    Tuple[List[str], List[int], List[List[int]]],
    Tuple[List[str], List[int], List[List[int]]],
]:
    """
    Create sample training and validation datasets.

    Returns:
        (train_data, val_data) where each is (texts, category_labels, pii_labels)
    """
    generator = SyntheticDataGenerator()

    # Generate training data
    train_texts, train_categories, train_pii = generator.generate_dataset(
        train_size, pii_ratio
    )

    # Generate validation data
    val_texts, val_categories, val_pii = generator.generate_dataset(val_size, pii_ratio)

    return (train_texts, train_categories, train_pii), (
        val_texts,
        val_categories,
        val_pii,
    )


if __name__ == "__main__":
    # Test the data generator
    generator = SyntheticDataGenerator()

    print("Sample generated data:")
    for i in range(5):
        text, category, pii_labels = generator.generate_sample(inject_pii_prob=0.5)
        print(f"\nSample {i+1}:")
        print(f"Text: {text}")
        print(f"Category: {category} ({generator.categories[category]})")
        print(f"PII Labels: {pii_labels}")
        print(f"Words: {text.split()}")
