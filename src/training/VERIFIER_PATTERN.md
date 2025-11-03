# Model Verifier Pattern

This directory contains Go verifiers for different model types. While each verifier is specific to its model type, they follow a common pattern.

## Verifier Files

1. **PII Classifier Verifier** (`pii_classifier_verifier.go`)
   - Verifies PII token classification using ModernBERT
   - ~141 lines

2. **Jailbreak Classifier Verifier** (`jailbreak_classifier_verifier.go`)
   - Verifies jailbreak detection using BERT/ModernBERT
   - ~283 lines

3. **LoRA Verifiers** (in `training_lora/` subdirectories)
   - Similar verifiers for LoRA-tuned models
   - Follow the same pattern with LoRA-specific initialization

## Common Pattern

All verifiers follow this structure:

```go
// 1. Model Configuration
type ModelConfig struct {
    ModelPath string
    UseCPU    bool
    // ... model-specific fields
}

// 2. Model Initialization
func initializeModels(config ModelConfig) error {
    // Initialize model(s) using candle bindings
    // Return error if initialization fails
}

// 3. Classification Function(s)
func classifyXXX(text string, config ModelConfig) (Result, error) {
    // Perform classification using initialized model
    // Return structured result
}

// 4. Main Function
func main() {
    // Parse command-line flags
    // Create config from flags
    // Initialize models
    // Run test cases
    // Print results and calculate accuracy
}

// 5. Test Cases
var testCases = []struct {
    text        string
    description string
    expected    string  // Optional: for accuracy calculation
}{
    // Test cases covering various scenarios
}
```

## Usage

Each verifier is a standalone program that can be run directly:

```bash
# PII Classifier Verifier
cd src/training/pii_model_fine_tuning
go run pii_classifier_verifier.go \
    --pii-token-model ../../../models/pii_classifier_modernbert-base_presidio_token_model \
    --cpu

# Jailbreak Classifier Verifier
cd src/training/prompt_guard_fine_tuning
go run jailbreak_classifier_verifier.go \
    --jailbreak-model ../../../models/jailbreak_classifier_modernbert-base_model \
    --similarity-model sentence-transformers/all-MiniLM-L6-v2 \
    --cpu
```

## Design Rationale

**Why not extract common code?**

1. **Small file sizes**: Each verifier is relatively small (141-283 lines)
2. **Domain-specific logic**: Significant portions are model-specific:
   - Different model initialization APIs
   - Different input/output formats
   - Different test case structures
   - Different label mappings
3. **Standalone executables**: Each is meant to be a self-contained verification tool
4. **Clear readability**: The pattern duplication actually makes each verifier easier to understand in isolation

**Benefits of the current approach:**

- Each verifier is self-documenting
- No hidden dependencies or abstractions
- Easy to modify for model-specific needs
- Clear command-line interface per verifier
- Simple to run and debug individually

## When to Consider Refactoring

Consider extracting common utilities if:

1. A new verifier type is added (4+ verifiers with same pattern)
2. Common functionality exceeds 50+ lines per verifier
3. Bugs are found that affect multiple verifiers
4. Command-line parsing becomes more complex

Potential shared utilities could include:
- Flag parsing helpers
- Test case running framework
- Result formatting utilities
- Model initialization error handling

## Related Documentation

- Model training guides: `src/training/*/README.md`
- Candle bindings: `candle-binding/README.md`
- Model evaluation: `src/training/model_eval/README.md`
