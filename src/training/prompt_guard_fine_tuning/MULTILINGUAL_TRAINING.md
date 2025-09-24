# Multilingual Prompt Guard Training

This documentation explains how to use the enhanced semantic-router training scripts to support multilingual PII and Prompt Guard classification models.

## Overview

The semantic-router now supports multilingual training for both PII detection and Prompt Guard classification:

- **PII Detection**: Already supported multilingual training via AI4Privacy dataset (English, French, German, Italian, Dutch, Spanish)
- **Prompt Guard**: Now supports multilingual training with synthetic dataset generation and multilingual dataset integration

## Current Multilingual Support Status

### PII Detection ✅ (Already Implemented)
- Uses AI4Privacy dataset with 6 languages
- Command: `python pii_bert_finetuning.py --dataset ai4privacy --languages all`
- Supported languages: English, French, German, Italian, Dutch, Spanish

### Prompt Guard ✅ (Newly Implemented) 
- Multilingual dataset generation and training
- Pattern-based synthetic data generation
- Support for 10+ languages
- Command: `python jailbreak_bert_finetuning.py --languages fr es de --datasets multilingual-fr`

## Quick Start

### 1. Generate Multilingual Datasets

```bash
# Generate multilingual prompt guard datasets
cd src/training/prompt_guard_fine_tuning

# Quick offline generation (recommended for testing)
python offline_multilingual_generator.py --languages fr es de it pt --samples-per-lang 1000

# Advanced generation with translation (requires internet)
python multilingual_dataset_generator.py --mode generate --languages fr es de --output-size 5000
```

### 2. Train Multilingual Prompt Guard Models

```bash
# Train with specific multilingual datasets
python jailbreak_bert_finetuning.py --mode train --languages fr es de --datasets multilingual-fr multilingual-es multilingual-de

# Train with default English datasets plus multilingual
python jailbreak_bert_finetuning.py --mode train --languages en fr es --datasets default multilingual-fr multilingual-es

# Use auto-optimization for best performance
python jailbreak_bert_finetuning.py --mode train --model modernbert-base --languages fr es de it
```

### 3. Train Multilingual PII Models

```bash
# Train with all supported languages
python pii_bert_finetuning.py --mode train --dataset ai4privacy --languages all

# Train with specific languages
python pii_bert_finetuning.py --mode train --dataset ai4privacy --languages English French German Spanish
```

## Supported Languages

### Prompt Guard Classification
- **fr**: French
- **es**: Spanish  
- **de**: German
- **it**: Italian
- **pt**: Portuguese
- **zh**: Chinese
- **ja**: Japanese
- **ko**: Korean
- **ru**: Russian
- **ar**: Arabic
- **en**: English (default)

### PII Detection (AI4Privacy Dataset)
- English, French, German, Italian, Dutch, Spanish

## Dataset Generation Methodology

Our multilingual dataset generation is inspired by the Qwen3Guard methodology:

### 1. Pattern-Based Synthesis
- Language-specific jailbreak patterns
- Cultural and linguistic variations
- Balanced jailbreak/benign distribution

### 2. Translation-Based Augmentation
- Translates existing English datasets
- Uses NLLB models for high-quality translation
- Maintains semantic meaning across languages

### 3. Quality Control
- Manual pattern curation by native speakers
- Automatic filtering and validation
- Balanced dataset creation

## Advanced Usage

### Custom Dataset Generation

```bash
# Generate datasets with custom patterns
python multilingual_dataset_generator.py --mode combined \
  --source-datasets salad-data toxic-chat \
  --languages fr es de it pt zh ja \
  --output-size 10000
```

### Fine-tuning with Language-Specific Models

```bash
# Use language-specific BERT models
python jailbreak_bert_finetuning.py --mode train \
  --model bert-base-multilingual-cased \
  --languages fr es de \
  --max-epochs 20 \
  --target-accuracy 0.95
```

### Evaluation Across Languages

```bash
# Test model performance across languages
python jailbreak_bert_finetuning.py --mode test \
  --model modernbert-base \
  --languages fr es de it \
  --datasets multilingual-fr multilingual-es multilingual-de multilingual-it
```

## File Structure

```
src/training/prompt_guard_fine_tuning/
├── jailbreak_bert_finetuning.py         # Enhanced with multilingual support
├── multilingual_dataset_generator.py     # Full-featured dataset generator
├── offline_multilingual_generator.py     # Offline pattern-based generator
├── test_multilingual.py                  # Test suite
├── test_multilingual_loading.py          # Dataset loading tests
└── multilingual_datasets/                # Generated datasets
    ├── prompt_guard_fr.json
    ├── prompt_guard_es.json
    ├── prompt_guard_de.json
    └── dataset_summary.json
```

## Performance Considerations

### Memory Usage
- Use `--max-samples-per-source 5000` to limit memory usage
- Start with smaller models like `minilm` for testing
- Use `--batch-size 8` for limited GPU memory

### Training Time
- Multilingual training takes longer due to larger datasets
- Use auto-optimization: `--enable-auto-optimization` (default)
- Consider progressive training: start with 2-3 languages

### Model Selection
- **ModernBERT**: Best performance for multilingual tasks
- **multilingual-BERT**: Good baseline for multilingual
- **XLM-R**: Alternative for cross-lingual tasks

## Troubleshooting

### Dataset Loading Issues
```bash
# Test dataset loading
python test_multilingual_loading.py

# Generate test datasets
python offline_multilingual_generator.py --languages fr --samples-per-lang 50
```

### Memory Issues
```bash
# Use conservative settings
python jailbreak_bert_finetuning.py --mode train \
  --model minilm \
  --batch-size 8 \
  --max-samples-per-source 1000 \
  --languages fr es
```

### Network Issues
```bash
# Use offline generation instead of translation-based
python offline_multilingual_generator.py --languages fr es de
```

## Contributing

To add support for new languages:

1. Add language patterns to `offline_multilingual_generator.py`
2. Add language codes to `multilingual_dataset_generator.py`
3. Test with native speakers
4. Submit PR with examples

## Examples and Use Cases

### Security Research
- Train models to detect jailbreak attempts in multiple languages
- Evaluate cross-lingual transferability of attacks
- Build robust multilingual safety filters

### Content Moderation
- Deploy multilingual prompt guard models in production
- Filter harmful prompts in user's native language
- Maintain safety across diverse user bases

### Academic Research
- Compare multilingual model performance
- Study cross-lingual jailbreak patterns
- Contribute to multilingual AI safety research

## Limitations and Future Work

### Current Limitations
- Pattern-based generation may lack diversity
- Limited cultural context in synthetic data
- Translation quality varies by language pair

### Future Enhancements
- Native speaker validation
- Cultural adaptation of patterns
- Integration with more multilingual datasets
- Real-world evaluation studies

## Citation

If you use this multilingual functionality in your research, please cite:

```bibtex
@software{semantic_router_multilingual,
  title={Multilingual Prompt Guard Training for Semantic Router},
  author={vLLM Project Contributors},
  year={2024},
  url={https://github.com/vllm-project/semantic-router}
}
```