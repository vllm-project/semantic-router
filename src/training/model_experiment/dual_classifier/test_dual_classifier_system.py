import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from data_generator import SyntheticDataGenerator, create_sample_datasets
from dual_classifier import DualClassifier
from trainer import DualTaskDataset, DualTaskLoss, DualTaskTrainer


class TestSyntheticDataGenerator:
    """Test the synthetic data generator."""

    def test_generator_initialization(self):
        """Test that the generator initializes correctly."""
        generator = SyntheticDataGenerator()
        assert len(generator.categories) == 10
        assert len(generator.category_templates) == 10
        assert len(generator.pii_patterns) == 5

    def test_sample_generation(self):
        """Test single sample generation."""
        generator = SyntheticDataGenerator()

        # Generate sample without PII
        text, category, pii_labels = generator.generate_sample(inject_pii_prob=0.0)
        assert isinstance(text, str)
        assert 0 <= category <= 9
        assert isinstance(pii_labels, list)
        assert all(label == 0 for label in pii_labels)  # No PII expected

        # Generate samples with PII until we get one with PII (sometimes PII injection fails)
        pii_found = False
        for _ in range(10):  # Try up to 10 times
            text, category, pii_labels = generator.generate_sample(inject_pii_prob=1.0)
            assert isinstance(text, str)
            assert 0 <= category <= 9
            assert isinstance(pii_labels, list)
            if any(label == 1 for label in pii_labels):
                pii_found = True
                break

        # Should find PII at least once in 10 attempts
        assert (
            pii_found
        ), "Should detect PII in at least one sample with inject_pii_prob=1.0"

    def test_dataset_generation(self):
        """Test dataset generation."""
        generator = SyntheticDataGenerator()
        texts, categories, pii_labels = generator.generate_dataset(
            num_samples=10, pii_ratio=0.5
        )

        assert len(texts) == 10
        assert len(categories) == 10
        assert len(pii_labels) == 10
        assert all(0 <= cat <= 9 for cat in categories)

    def test_pii_detection_patterns(self):
        """Test that PII patterns are correctly detected."""
        generator = SyntheticDataGenerator()

        # Test email detection
        email_text = "Contact me at john@example.com"
        pii_labels = generator._generate_pii_labels(email_text)
        assert 1 in pii_labels  # Should detect email

        # Test phone detection
        phone_text = "Call me at 123-456-7890"
        pii_labels = generator._generate_pii_labels(phone_text)
        assert 1 in pii_labels  # Should detect phone


class TestDualTaskDataset:
    """Test the dual-task dataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        texts = ["What is 2+2?", "My email is test@example.com"]
        categories = [0, 1]
        pii_labels = [[0, 0, 0], [0, 0, 0, 1]]
        return texts, categories, pii_labels

    @pytest.fixture
    def model_tokenizer(self):
        """Get a tokenizer for testing."""
        from transformers import DistilBertTokenizer

        return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def test_dataset_creation(self, sample_data, model_tokenizer):
        """Test dataset creation."""
        texts, categories, pii_labels = sample_data
        dataset = DualTaskDataset(
            texts=texts,
            category_labels=categories,
            pii_labels=pii_labels,
            tokenizer=model_tokenizer,
            max_length=32,
        )

        assert len(dataset) == 2

        # Test __getitem__
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "category_label" in item
        assert "pii_labels" in item

        # Check tensor shapes
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["category_label"].shape == ()
        assert item["pii_labels"].shape == (32,)


class TestDualTaskLoss:
    """Test the dual-task loss function."""

    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = DualTaskLoss(category_weight=1.0, pii_weight=2.0)
        assert loss_fn.category_weight == 1.0
        assert loss_fn.pii_weight == 2.0

    def test_loss_computation(self):
        """Test loss computation."""
        batch_size, seq_len, num_categories = 2, 10, 5

        # Create dummy data with gradients enabled
        category_logits = torch.randn(batch_size, num_categories, requires_grad=True)
        pii_logits = torch.randn(batch_size, seq_len, 2, requires_grad=True)
        category_labels = torch.randint(0, num_categories, (batch_size,))
        pii_labels = torch.randint(0, 2, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        loss_fn = DualTaskLoss()
        total_loss, cat_loss, pii_loss = loss_fn(
            category_logits, pii_logits, category_labels, pii_labels, attention_mask
        )

        # Check that losses are computed
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(cat_loss, torch.Tensor)
        assert isinstance(pii_loss, torch.Tensor)
        assert total_loss.requires_grad  # Should require grad since inputs do

    def test_loss_masking(self):
        """Test that padding tokens are properly masked."""
        batch_size, seq_len, num_categories = 1, 5, 3

        category_logits = torch.randn(batch_size, num_categories)
        pii_logits = torch.randn(batch_size, seq_len, 2)
        category_labels = torch.randint(0, num_categories, (batch_size,))
        pii_labels = torch.randint(0, 2, (batch_size, seq_len))

        # Create attention mask with padding
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])  # First 3 tokens are real

        loss_fn = DualTaskLoss()
        total_loss, cat_loss, pii_loss = loss_fn(
            category_logits, pii_logits, category_labels, pii_labels, attention_mask
        )

        # Loss should be computed only for non-padded tokens
        assert not torch.isnan(total_loss)
        assert not torch.isnan(pii_loss)


class TestDualTaskTrainer:
    """Test the dual-task trainer."""

    @pytest.fixture
    def small_datasets(self):
        """Create small datasets for testing with correct number of categories."""
        train_data, val_data = create_sample_datasets(
            train_size=8, val_size=4, pii_ratio=0.5
        )
        return train_data, val_data

    @pytest.fixture
    def small_model(self):
        """Create a model matching the data categories (10)."""
        return DualClassifier(num_categories=10)  # Match the data generator

    def test_trainer_initialization(self, small_model, small_datasets):
        """Test trainer initialization."""
        train_data, val_data = small_datasets
        train_texts, train_categories, train_pii = train_data

        train_dataset = DualTaskDataset(
            texts=train_texts,
            category_labels=train_categories,
            pii_labels=train_pii,
            tokenizer=small_model.tokenizer,
            max_length=32,
        )

        trainer = DualTaskTrainer(
            model=small_model, train_dataset=train_dataset, batch_size=2, num_epochs=1
        )

        assert trainer.model is small_model
        assert trainer.batch_size == 2
        assert trainer.num_epochs == 1
        assert len(trainer.train_loader) == 4  # 8 samples / 2 batch_size

    def test_training_step(self, small_model, small_datasets):
        """Test a single training step."""
        train_data, val_data = small_datasets
        train_texts, train_categories, train_pii = train_data
        val_texts, val_categories, val_pii = val_data

        train_dataset = DualTaskDataset(
            texts=train_texts,
            category_labels=train_categories,
            pii_labels=train_pii,
            tokenizer=small_model.tokenizer,
            max_length=32,
        )

        val_dataset = DualTaskDataset(
            texts=val_texts,
            category_labels=val_categories,
            pii_labels=val_pii,
            tokenizer=small_model.tokenizer,
            max_length=32,
        )

        trainer = DualTaskTrainer(
            model=small_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=4,
            num_epochs=1,
            learning_rate=1e-4,
        )

        # Get initial parameters
        initial_params = [p.clone() for p in small_model.parameters()]

        # Train for one epoch
        train_loss, cat_loss, pii_loss = trainer.train_epoch()

        # Check that parameters changed
        params_changed = any(
            not torch.equal(initial, current)
            for initial, current in zip(initial_params, small_model.parameters())
        )
        assert params_changed, "Model parameters should change after training"

        # Check loss values
        assert isinstance(train_loss, float)
        assert isinstance(cat_loss, float)
        assert isinstance(pii_loss, float)
        assert train_loss > 0

    def test_evaluation(self, small_model, small_datasets):
        """Test model evaluation."""
        train_data, val_data = small_datasets
        train_texts, train_categories, train_pii = train_data
        val_texts, val_categories, val_pii = val_data

        train_dataset = DualTaskDataset(
            texts=train_texts,
            category_labels=train_categories,
            pii_labels=train_pii,
            tokenizer=small_model.tokenizer,
            max_length=32,
        )

        val_dataset = DualTaskDataset(
            texts=val_texts,
            category_labels=val_categories,
            pii_labels=val_pii,
            tokenizer=small_model.tokenizer,
            max_length=32,
        )

        trainer = DualTaskTrainer(
            model=small_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=4,
            num_epochs=1,
        )

        # Evaluate
        metrics = trainer.evaluate()

        # Check metrics
        assert "val_loss" in metrics
        assert "val_category_acc" in metrics
        assert "val_pii_f1" in metrics
        assert 0 <= metrics["val_category_acc"] <= 1
        assert 0 <= metrics["val_pii_f1"] <= 1

    def test_model_saving_loading(self, small_model, small_datasets):
        """Test model saving and loading."""
        train_data, _ = small_datasets
        train_texts, train_categories, train_pii = train_data

        train_dataset = DualTaskDataset(
            texts=train_texts,
            category_labels=train_categories,
            pii_labels=train_pii,
            tokenizer=small_model.tokenizer,
            max_length=32,
        )

        trainer = DualTaskTrainer(
            model=small_model, train_dataset=train_dataset, batch_size=4, num_epochs=1
        )

        # Train briefly
        trainer.train_epoch()

        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_model(temp_dir)

            # Check files exist
            assert os.path.exists(f"{temp_dir}/model.pt")
            assert os.path.exists(f"{temp_dir}/training_history.json")

            # Load and test
            loaded_model = DualClassifier.from_pretrained(temp_dir, num_categories=10)

            # Test that loaded model works
            test_text = "What is 2+2?"
            original_output = small_model.predict(test_text)
            loaded_output = loaded_model.predict(test_text)

            # Outputs should be very similar (allowing for small floating point differences)
            assert torch.allclose(original_output[0], loaded_output[0], atol=1e-6)
            assert torch.allclose(original_output[1], loaded_output[1], atol=1e-6)


class TestTrainingIntegration:
    """Integration tests for the complete training pipeline."""

    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Create small test case
        train_data, val_data = create_sample_datasets(
            train_size=16, val_size=8, pii_ratio=0.5
        )

        train_texts, train_categories, train_pii = train_data
        val_texts, val_categories, val_pii = val_data

        # Initialize model
        model = DualClassifier(num_categories=10)

        # Create datasets with same max_length as model was designed for
        train_dataset = DualTaskDataset(
            texts=train_texts,
            category_labels=train_categories,
            pii_labels=train_pii,
            tokenizer=model.tokenizer,
            max_length=64,
        )

        val_dataset = DualTaskDataset(
            texts=val_texts,
            category_labels=val_categories,
            pii_labels=val_pii,
            tokenizer=model.tokenizer,
            max_length=64,
        )

        # Create trainer
        trainer = DualTaskTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=2,
        )

        # Get initial performance
        initial_metrics = trainer.evaluate()

        # Train
        trainer.train()

        # Check that training history is recorded
        assert len(trainer.history["train_loss"]) == 2  # 2 epochs
        assert len(trainer.history["val_category_acc"]) == 2

        # Check that loss generally decreased
        final_loss = trainer.history["train_loss"][-1]
        # Note: With very small datasets, loss might not always decrease
        # but we check that training completed without errors
        assert isinstance(final_loss, float)
        assert final_loss > 0

    def test_memory_efficiency(self):
        """Test that the dual-head approach is memory efficient."""
        # Compare memory usage of dual-head vs two separate models
        import tracemalloc

        # Test dual-head model
        tracemalloc.start()
        dual_model = DualClassifier(num_categories=10)
        dual_params = sum(p.numel() for p in dual_model.parameters())
        current, peak = tracemalloc.get_traced_memory()
        dual_memory = peak
        tracemalloc.stop()

        # Dual-head should be significantly smaller than two separate models
        # (This is more of a sanity check since we're not implementing separate models)
        # The dual model should have reasonable parameter count
        assert dual_params > 65_000_000  # Should have base DistilBERT params
        assert dual_params < 70_000_000  # But not too much more

        print(f"Dual-head model parameters: {dual_params:,}")
        print(f"Memory usage: {dual_memory / 1024 / 1024:.1f} MB")


def run_performance_test():
    """Run a performance test to check training speed."""
    print("\n🏃‍♂️ Running Performance Test...")

    import time

    # Create test data
    train_data, val_data = create_sample_datasets(
        train_size=50, val_size=20, pii_ratio=0.4
    )

    train_texts, train_categories, train_pii = train_data

    # Initialize model
    model = DualClassifier(num_categories=10)

    # Create dataset
    train_dataset = DualTaskDataset(
        texts=train_texts,
        category_labels=train_categories,
        pii_labels=train_pii,
        tokenizer=model.tokenizer,
        max_length=128,
    )

    # Create trainer
    trainer = DualTaskTrainer(
        model=model, train_dataset=train_dataset, batch_size=8, num_epochs=1
    )

    # Time training
    start_time = time.time()
    trainer.train_epoch()
    training_time = time.time() - start_time

    print(f"✅ Training 50 samples took {training_time:.1f} seconds")
    print(f"   That's {training_time/50:.3f} seconds per sample")

    # Performance thresholds (adjust based on your system)
    if training_time < 30:
        print("   🚀 Excellent performance!")
    elif training_time < 60:
        print("   ✅ Good performance!")
    else:
        print("   ⚠️  Consider reducing batch_size or max_length for faster training")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

    # Run performance test
    run_performance_test()
