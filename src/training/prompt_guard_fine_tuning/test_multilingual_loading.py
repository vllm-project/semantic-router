#!/usr/bin/env python3
"""
Simple test script to verify multilingual dataset loading functionality
"""

import sys
import os

# Add the script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jailbreak_bert_finetuning import Jailbreak_Dataset

def test_multilingual_loading():
    """Test loading multilingual datasets."""
    print("Testing multilingual dataset loading...")
    
    # Test with multilingual languages parameter
    dataset_loader = Jailbreak_Dataset(
        dataset_sources=["multilingual-fr"],  # Test loading French multilingual dataset
        max_samples_per_source=10,
        languages=["fr"]
    )
    
    print(f"Dataset loader languages: {dataset_loader.languages}")
    
    # Check that multilingual configs are available
    multilingual_configs = {k: v for k, v in dataset_loader.dataset_configs.items() 
                          if k.startswith("multilingual-")}
    print(f"Found {len(multilingual_configs)} multilingual dataset configurations")
    
    # Show some multilingual configs
    for config_name, config in list(multilingual_configs.items())[:5]:
        print(f"  {config_name}: {config['description']}")
    
    # Try to load a specific multilingual dataset if it exists
    test_file = "test_multilingual_datasets/prompt_guard_fr.json"
    if os.path.exists(test_file):
        print(f"\nTesting loading from {test_file}")
        texts, labels = dataset_loader.load_multilingual_dataset(test_file, "fr")
        print(f"Successfully loaded {len(texts)} samples")
        
        # Show some examples
        for i in range(min(3, len(texts))):
            print(f"  Example {i+1} ({labels[i]}): {texts[i][:100]}...")
        
        # Show label distribution
        jailbreak_count = sum(1 for label in labels if label == "jailbreak")
        benign_count = sum(1 for label in labels if label == "benign")
        print(f"  Label distribution: {jailbreak_count} jailbreak, {benign_count} benign")
        
        return True
    else:
        print(f"Test dataset not found: {test_file}")
        return False

if __name__ == "__main__":
    success = test_multilingual_loading()
    if success:
        print("✅ Multilingual dataset loading test PASSED")
    else:
        print("❌ Multilingual dataset loading test FAILED")