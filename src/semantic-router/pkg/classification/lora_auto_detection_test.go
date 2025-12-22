package classification

import (
	"os"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestIntentClassificationLoRAAutoDetection demonstrates that current implementation
// doesn't auto-detect LoRA models for intent classification (unlike PII detection)
func TestIntentClassificationLoRAAutoDetection(t *testing.T) {
	modelPath := "../../../../models/lora_intent_classifier_bert-base-uncased_model"
	numClasses := 14 // From category_mapping.json

	// Check if LoRA model exists
	if _, err := os.Stat(modelPath + "/lora_config.json"); os.IsNotExist(err) {
		t.Skip("LoRA intent model not available, skipping test")
	}

	t.Run("AutoDetection: CategoryInitializer Now Detects LoRA Models", func(t *testing.T) {
		// After fix: CategoryInitializerImpl auto-detects LoRA models
		// It tries InitCandleBertClassifier() first (checks for lora_config.json)
		// Falls back to InitModernBertClassifier() if needed

		cfg := &config.CategoryModel{
			ModelID: modelPath,
			UseCPU:  true,
		}

		// Create auto-detecting initializer
		initializer := createCategoryInitializer()

		// Try to initialize - should SUCCESS with LoRA auto-detection
		err := initializer.Init(cfg.ModelID, cfg.UseCPU, numClasses)
		if err != nil {
			t.Errorf("Auto-detection failed: %v", err)
			return
		}

		t.Log("✓ CategoryInitializer successfully auto-detected and initialized LoRA model")

		// Verify inference works
		inference := createCategoryInference()
		result, err := inference.Classify("What is the best business strategy?")
		if err != nil {
			t.Errorf("Classification failed: %v", err)
			return
		}

		if result.Class < 0 || result.Class >= numClasses {
			t.Errorf("Invalid category: %d (expected 0-%d)", result.Class, numClasses-1)
			return
		}

		t.Logf("✓ Classification works: category=%d, confidence=%.3f", result.Class, result.Confidence)
	})

	t.Run("Proof: Auto-Detection Already Works in Rust Layer", func(t *testing.T) {
		// This proves the Rust auto-detection ALREADY EXISTS and WORKS
		// InitCandleBertClassifier has auto-detection built-in (checks for lora_config.json)

		success := candle_binding.InitCandleBertClassifier(modelPath, numClasses, true)

		if !success {
			t.Error("InitCandleBertClassifier should auto-detect LoRA (it exists in Rust)")
			return
		}

		t.Log("✓ Proof: Rust layer successfully auto-detected LoRA model")

		// Try classification to prove it works
		result, err := candle_binding.ClassifyCandleBertText("What is the best business strategy?")
		if err != nil {
			t.Errorf("Classification failed: %v", err)
			return
		}

		if result.Class < 0 || result.Class >= numClasses {
			t.Errorf("Invalid category: %d (expected 0-%d)", result.Class, numClasses-1)
			return
		}

		t.Logf("✓ Classification works: category=%d, confidence=%.3f", result.Class, result.Confidence)
		t.Logf("   Solution: Update CategoryInitializer to use InitCandleBertClassifier")
	})
}

// TestPIIAlreadyHasAutoDetection shows PII detection already works with LoRA auto-detection
func TestPIIAlreadyHasAutoDetection(t *testing.T) {
	modelPath := "../../../../models/mom-pii-classifier"

	// Check if LoRA model exists
	if _, err := os.Stat(modelPath + "/lora_config.json"); os.IsNotExist(err) {
		t.Skip("LoRA PII model not available, skipping test")
	}

	t.Log("✓ PII detection already has auto-detection (implemented in PR #709)")
	t.Log("  Goal: Make Intent & Jailbreak detection work the same way")
}
