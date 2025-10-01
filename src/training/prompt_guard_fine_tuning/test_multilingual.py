#!/usr/bin/env python3
"""
Test script for multilingual dataset generation functionality
"""

import os
import sys
import json
from pathlib import Path

# Add the script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multilingual_dataset_generator():
    """Test the multilingual dataset generator with a small sample."""
    print("Testing multilingual dataset generator...")
    
    # Test generating synthetic datasets
    from multilingual_dataset_generator import MultilingualDatasetGenerator
    
    generator = MultilingualDatasetGenerator(cache_dir="./test_cache")
    
    # Generate a small sample of synthetic data
    test_languages = ["fr", "es"]
    synthetic_data = generator.generate_synthetic_prompts(test_languages, num_samples_per_lang=10)
    
    print(f"Generated synthetic data for {len(synthetic_data)} languages")
    
    for lang, (texts, labels) in synthetic_data.items():
        print(f"\n{lang} ({generator.language_codes.get(lang, lang)}):")
        print(f"  Samples: {len(texts)}")
        jailbreak_count = sum(1 for label in labels if label == "jailbreak")
        benign_count = sum(1 for label in labels if label == "benign")
        print(f"  Jailbreak: {jailbreak_count}, Benign: {benign_count}")
        
        # Show some examples
        for i in range(min(3, len(texts))):
            print(f"  Example {i+1} ({labels[i]}): {texts[i][:100]}...")
    
    # Save test data
    output_dir = "./test_multilingual_datasets"
    generator.save_datasets(synthetic_data, output_dir)
    
    print(f"\nTest datasets saved to {output_dir}")
    
    # Test loading the datasets
    print("\nTesting dataset loading...")
    
    # Create a simple test dataset to verify loading
    test_file = Path(output_dir) / "prompt_guard_fr.json"
    if test_file.exists():
        test_data = generator.load_multilingual_dataset(str(test_file), "fr")
        print(f"Successfully loaded {len(test_data[0])} samples from {test_file}")
    
    return True

def test_jailbreak_multilingual_integration():
    """Test the integration with jailbreak classification."""
    print("\nTesting jailbreak classification multilingual integration...")
    
    try:
        from jailbreak_bert_finetuning import Jailbreak_Dataset
        
        # Test with multilingual languages parameter
        dataset_loader = Jailbreak_Dataset(
            dataset_sources=["chatbot-instructions"],  # Use a simple dataset for testing
            max_samples_per_source=10,
            languages=["en", "fr", "es"]
        )
        
        # Check that the languages parameter is set
        print(f"Dataset loader languages: {dataset_loader.languages}")
        
        # Check multilingual configurations are added
        multilingual_configs = {k: v for k, v in dataset_loader.dataset_configs.items() 
                              if k.startswith("multilingual-")}
        print(f"Found {len(multilingual_configs)} multilingual dataset configurations")
        
        for config_name, config in list(multilingual_configs.items())[:3]:
            print(f"  {config_name}: {config['description']}")
        
        print("✓ Jailbreak classification integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Jailbreak classification integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("MULTILINGUAL PROMPT GUARD DATASET TESTING")
    print("="*60)
    
    success = True
    
    try:
        success &= test_multilingual_dataset_generator()
    except Exception as e:
        print(f"✗ Multilingual dataset generator test failed: {e}")
        success = False
    
    try:
        success &= test_jailbreak_multilingual_integration()
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)