#!/usr/bin/env python3
"""
Comprehensive Multilingual Dataset Generator for Existing Training Data

This script creates multilingual versions of ALL existing datasets used by the 
PII and Prompt Guard training scripts. It focuses on translating the actual 
datasets rather than generating synthetic samples.

Usage:
    # Translate all default prompt guard datasets to multiple languages
    python translate_existing_datasets.py --target-languages fr es de it pt

    # Translate specific datasets with full samples
    python translate_existing_datasets.py --source-datasets salad-data toxic-chat --target-languages fr es de

    # Translate with batch processing for better performance
    python translate_existing_datasets.py --target-languages fr es de --batch-translate --batch-size 64

    # Create multilingual PII datasets (if not using existing AI4Privacy multilingual support)
    python translate_existing_datasets.py --mode pii --target-languages fr es de it

Supported source datasets:
    Prompt Guard: salad-data, toxic-chat, spml-injection, chatbot-instructions, 
                  orca-agentinstruct, vmware-openinstruct, jackhhao-jailbreak
    PII: presidio (AI4Privacy already has multilingual support)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multilingual_dataset_generator import MultilingualDatasetGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExistingDatasetTranslator:
    """Specialized translator for existing training datasets."""
    
    def __init__(self, cache_dir: str = "./translation_cache"):
        self.generator = MultilingualDatasetGenerator(cache_dir=cache_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Define comprehensive dataset groups
        self.dataset_groups = {
            "prompt_guard_default": [
                "salad-data", "toxic-chat", "spml-injection", 
                "chatbot-instructions", "orca-agentinstruct", "vmware-openinstruct"
            ],
            "prompt_guard_all": [
                "salad-data", "toxic-chat", "spml-injection", 
                "chatbot-instructions", "orca-agentinstruct", "vmware-openinstruct",
                "jackhhao-jailbreak", "alpaca-gpt4", "databricks-dolly"
            ],
            "jailbreak_only": [
                "salad-data", "toxic-chat", "spml-injection", "jackhhao-jailbreak"
            ],
            "benign_only": [
                "chatbot-instructions", "orca-agentinstruct", "vmware-openinstruct",
                "alpaca-gpt4", "databricks-dolly"
            ]
        }
    
    def get_dataset_info(self, dataset_names: List[str]) -> Dict:
        """Get information about datasets without loading them."""
        logger.info("Gathering dataset information...")
        
        from jailbreak_bert_finetuning import Jailbreak_Dataset
        
        # Create dataset loader to access configs
        dataset_loader = Jailbreak_Dataset(dataset_sources=dataset_names, max_samples_per_source=1)
        
        info = {
            "requested_datasets": dataset_names,
            "available_configs": {},
            "unavailable_datasets": []
        }
        
        for dataset_name in dataset_names:
            if dataset_name in dataset_loader.dataset_configs:
                config = dataset_loader.dataset_configs[dataset_name]
                info["available_configs"][dataset_name] = {
                    "description": config.get("description", "No description"),
                    "type": config.get("type", "unknown"),
                    "source": config.get("name", "unknown")
                }
            else:
                info["unavailable_datasets"].append(dataset_name)
        
        return info
    
    def translate_dataset_group(self, group_name: str, target_languages: List[str], 
                              max_samples_per_source: Optional[int] = None,
                              use_batch: bool = True, batch_size: int = 32) -> Dict:
        """Translate a predefined group of datasets."""
        
        if group_name not in self.dataset_groups:
            raise ValueError(f"Unknown dataset group: {group_name}. Available: {list(self.dataset_groups.keys())}")
        
        dataset_names = self.dataset_groups[group_name]
        logger.info(f"Translating dataset group '{group_name}': {dataset_names}")
        
        return self.translate_datasets(dataset_names, target_languages, 
                                     max_samples_per_source, use_batch, batch_size)
    
    def translate_datasets(self, dataset_names: List[str], target_languages: List[str],
                          max_samples_per_source: Optional[int] = None, 
                          use_batch: bool = True, batch_size: int = 32) -> Dict:
        """Translate specified datasets to target languages."""
        
        # Get dataset information
        info = self.get_dataset_info(dataset_names)
        
        logger.info("=" * 60)
        logger.info("DATASET TRANSLATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Datasets to translate: {len(info['available_configs'])}")
        for name, config in info['available_configs'].items():
            logger.info(f"  • {name}: {config['description']} ({config['type']})")
        
        if info['unavailable_datasets']:
            logger.warning(f"Unavailable datasets (skipped): {info['unavailable_datasets']}")
        
        logger.info(f"Target languages: {target_languages}")
        logger.info(f"Sample limit per source: {'No limit' if max_samples_per_source is None else max_samples_per_source}")
        logger.info(f"Batch translation: {'Enabled' if use_batch else 'Disabled'}")
        
        # Load source datasets
        logger.info("\nLoading source datasets...")
        available_datasets = list(info['available_configs'].keys())
        texts, labels = self.generator.load_existing_datasets(available_datasets, max_samples_per_source)
        
        # Show source statistics
        total_samples = len(texts)
        jailbreak_count = sum(1 for label in labels if label == "jailbreak")
        benign_count = sum(1 for label in labels if label == "benign")
        unknown_count = total_samples - jailbreak_count - benign_count
        
        logger.info(f"Source dataset statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Jailbreak: {jailbreak_count} ({jailbreak_count/total_samples*100:.1f}%)")
        logger.info(f"  Benign: {benign_count} ({benign_count/total_samples*100:.1f}%)")
        if unknown_count > 0:
            logger.info(f"  Other: {unknown_count} ({unknown_count/total_samples*100:.1f}%)")
        
        # Translate datasets
        logger.info("\nStarting translation process...")
        translated_data = self.generator.translate_datasets(texts, labels, target_languages, use_batch)
        
        # Create comprehensive summary
        translation_summary = {
            "source_info": info,
            "source_statistics": {
                "total_samples": total_samples,
                "jailbreak_samples": jailbreak_count,
                "benign_samples": benign_count,
                "other_samples": unknown_count
            },
            "target_languages": target_languages,
            "translated_statistics": {},
            "settings": {
                "max_samples_per_source": max_samples_per_source,
                "use_batch_translation": use_batch,
                "batch_size": batch_size if use_batch else None
            }
        }
        
        for lang, (lang_texts, lang_labels) in translated_data.items():
            translation_summary["translated_statistics"][lang] = {
                "total_samples": len(lang_texts),
                "jailbreak_samples": sum(1 for label in lang_labels if label == "jailbreak"),
                "benign_samples": sum(1 for label in lang_labels if label == "benign"),
                "language_name": self.generator.language_codes.get(lang, lang)
            }
        
        return {
            "translated_data": translated_data,
            "summary": translation_summary
        }
    
    def save_translated_datasets(self, result: Dict, output_dir: str = "./multilingual_datasets"):
        """Save translated datasets with comprehensive metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        translated_data = result["translated_data"]
        summary = result["summary"]
        
        # Save individual language datasets
        logger.info(f"\nSaving translated datasets to {output_path}")
        for lang, (texts, labels) in translated_data.items():
            lang_file = output_path / f"translated_prompt_guard_{lang}.json"
            
            dataset = []
            for text, label in zip(texts, labels):
                dataset.append({
                    "text": text,
                    "label": label,
                    "language": lang,
                    "language_name": self.generator.language_codes.get(lang, lang),
                    "source": "translated_from_existing_datasets"
                })
            
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
                
            logger.info(f"  • {lang}: {len(dataset)} samples → {lang_file}")
        
        # Save comprehensive summary
        summary_file = output_path / "translation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"  • Summary: {summary_file}")
        
        # Save dataset statistics
        stats_file = output_path / "translation_statistics.md"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("# Multilingual Dataset Translation Statistics\n\n")
            
            f.write("## Source Datasets\n")
            for name, config in summary["source_info"]["available_configs"].items():
                f.write(f"- **{name}**: {config['description']} ({config['type']})\n")
            
            f.write(f"\n## Source Statistics\n")
            src_stats = summary["source_statistics"]
            f.write(f"- Total samples: {src_stats['total_samples']:,}\n")
            f.write(f"- Jailbreak samples: {src_stats['jailbreak_samples']:,}\n")
            f.write(f"- Benign samples: {src_stats['benign_samples']:,}\n")
            
            f.write(f"\n## Translation Results\n")
            f.write(f"- Target languages: {len(summary['target_languages'])}\n")
            f.write(f"- Languages: {', '.join(summary['target_languages'])}\n")
            
            f.write(f"\n## Per-Language Statistics\n")
            for lang, stats in summary["translated_statistics"].items():
                f.write(f"### {stats['language_name']} ({lang})\n")
                f.write(f"- Total samples: {stats['total_samples']:,}\n")
                f.write(f"- Jailbreak samples: {stats['jailbreak_samples']:,}\n")
                f.write(f"- Benign samples: {stats['benign_samples']:,}\n\n")
        
        logger.info(f"  • Statistics: {stats_file}")
        
        logger.info(f"\n✅ Translation complete! Generated {len(translated_data)} multilingual datasets.")

def main():
    parser = argparse.ArgumentParser(description="Translate existing training datasets to multiple languages")
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset-group", choices=["prompt_guard_default", "prompt_guard_all", "jailbreak_only", "benign_only"],
                              help="Predefined dataset group to translate")
    dataset_group.add_argument("--source-datasets", nargs="+", 
                              help="Specific datasets to translate")
    
    # Language and output settings
    parser.add_argument("--target-languages", nargs="+", 
                       help="Target languages (e.g., fr es de it pt zh ja)")
    parser.add_argument("--output-dir", default="./multilingual_datasets",
                       help="Output directory for translated datasets")
    parser.add_argument("--cache-dir", default="./translation_cache",
                       help="Cache directory for translations")
    
    # Performance settings
    parser.add_argument("--max-samples-per-source", type=int, default=None,
                       help="Limit samples per source dataset (None = no limit)")
    parser.add_argument("--batch-translate", action="store_true",
                       help="Use batch translation for better performance")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for batch translation")
    
    # Information mode
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and groups")
    
    args = parser.parse_args()
    
    translator = ExistingDatasetTranslator(cache_dir=args.cache_dir)
    
    if args.list_datasets:
        print("Available Dataset Groups:")
        for group_name, datasets in translator.dataset_groups.items():
            print(f"  {group_name}: {datasets}")
        print("\nUse --dataset-group to translate a group or --source-datasets for specific datasets")
        return
    
    # Determine datasets to translate
    if args.dataset_group:
        result = translator.translate_dataset_group(
            args.dataset_group, 
            args.target_languages,
            args.max_samples_per_source,
            args.batch_translate,
            args.batch_size
        )
    elif args.source_datasets:
        result = translator.translate_datasets(
            args.source_datasets,
            args.target_languages, 
            args.max_samples_per_source,
            args.batch_translate,
            args.batch_size
        )
    else:
        # Default to prompt guard default group
        logger.info("No datasets specified, using default prompt guard datasets")
        result = translator.translate_dataset_group(
            "prompt_guard_default",
            args.target_languages,
            args.max_samples_per_source,
            args.batch_translate, 
            args.batch_size
        )
    
    # Save results
    translator.save_translated_datasets(result, args.output_dir)

if __name__ == "__main__":
    main()