import os
import time

import psutil
import torch
from data_generator import create_sample_datasets
from dual_classifier import DualClassifier
from trainer import DualTaskDataset, DualTaskTrainer


def get_system_info():
    """Get system information for performance monitoring."""
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def main():
    print("🚀 Dual-Purpose Classifier Training Example")
    print("=" * 50)

    # System info
    sys_info = get_system_info()
    print(f"💻 System Info:")
    print(f"   CPU cores: {sys_info['cpu_count']}")
    print(f"   Memory: {sys_info['memory_gb']:.1f} GB")
    print(f"   GPU available: {sys_info['gpu_available']}")
    if sys_info["gpu_available"]:
        print(f"   GPU: {sys_info['gpu_name']}")
    print()

    # Configuration
    config = {
        "num_categories": 10,
        "train_size": 50,  # Small for testing
        "val_size": 20,  # Small for testing
        "batch_size": 4,  # Small batch size for laptop compatibility
        "num_epochs": 2,  # Just 2 epochs for testing
        "learning_rate": 2e-5,
        "max_length": 128,  # Shorter sequences for speed
        "pii_ratio": 0.4,  # 40% of samples have PII
    }

    print(f"⚙️  Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    # Generate sample data
    print("📊 Generating synthetic training data...")
    train_data, val_data = create_sample_datasets(
        train_size=config["train_size"],
        val_size=config["val_size"],
        pii_ratio=config["pii_ratio"],
    )

    train_texts, train_categories, train_pii = train_data
    val_texts, val_categories, val_pii = val_data

    print(f"   Train samples: {len(train_texts)}")
    print(f"   Val samples: {len(val_texts)}")

    # Show sample data
    print(f"\n📝 Sample training data:")
    for i in range(min(3, len(train_texts))):
        print(f"   Sample {i+1}:")
        print(f"     Text: {train_texts[i]}")
        print(f"     Category: {train_categories[i]}")
        print(f"     PII labels: {train_pii[i]}")
    print()

    # Initialize model
    print("🧠 Initializing DualClassifier model...")
    model = DualClassifier(num_categories=config["num_categories"])

    # Create datasets
    print("📦 Creating PyTorch datasets...")
    train_dataset = DualTaskDataset(
        texts=train_texts,
        category_labels=train_categories,
        pii_labels=train_pii,
        tokenizer=model.tokenizer,
        max_length=config["max_length"],
    )

    val_dataset = DualTaskDataset(
        texts=val_texts,
        category_labels=val_categories,
        pii_labels=val_pii,
        tokenizer=model.tokenizer,
        max_length=config["max_length"],
    )

    # Setup trainer
    print("🏋️  Setting up trainer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = DualTaskTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        category_weight=1.0,
        pii_weight=1.0,
        device=device,
    )

    # Start training
    print(f"🔥 Starting training...")
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Monitor training time
    start_time = time.time()

    try:
        # Train the model
        trainer.train()

        training_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"   Total time: {training_time:.1f} seconds")
        print(f"   Time per epoch: {training_time/config['num_epochs']:.1f} seconds")

        # Save the model
        model_path = "trained_model"
        print(f"\n💾 Saving model to {model_path}...")
        trainer.save_model(model_path)

        # Test the trained model
        print(f"\n🧪 Testing trained model...")
        test_texts = [
            "What is the derivative of x^2?",
            "My email is john@example.com. How does DNA work?",
            "Contact Sarah at 123-456-7890 for math help.",
        ]

        model.eval()
        for i, text in enumerate(test_texts):
            category_probs, pii_probs = model.predict(text, device=device)

            category_pred = torch.argmax(category_probs[0]).item()
            confidence = category_probs[0][category_pred].item()

            # Check for PII tokens
            tokens = model.tokenizer.tokenize(text)
            pii_preds = torch.argmax(pii_probs[0], dim=-1)
            pii_tokens = [token for token, pred in zip(tokens, pii_preds) if pred == 1]

            print(f"\n   Test {i+1}: {text}")
            print(f"     Category: {category_pred} (confidence: {confidence:.3f})")
            print(f"     PII tokens: {pii_tokens if pii_tokens else 'None detected'}")

        print(f"\n🎉 Task 2 completed successfully!")
        print(f"   Model saved to: {model_path}/")
        print(f"   Training history saved to: {model_path}/training_history.json")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback

        traceback.print_exc()

        # Performance guidance
        print(f"\n💡 Performance Tips:")
        if not torch.cuda.is_available():
            print("   - Consider using a GPU for faster training")
        print("   - Reduce batch_size if you get out-of-memory errors")
        print("   - Reduce train_size for faster testing")
        print("   - Reduce max_length for lower memory usage")


if __name__ == "__main__":
    main()
