package services

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
)

func TestNewUnifiedClassificationService(t *testing.T) {
	// Test with nil unified classifier and nil legacy classifier (this is expected to work)
	config := &config.RouterConfig{}
	service := NewUnifiedClassificationService(nil, nil, config)

	if service == nil {
		t.Error("Expected non-nil service")
	}
	if service.classifier != nil {
		t.Error("Expected legacy classifier to be nil")
	}
	if service.unifiedClassifier != nil {
		t.Error("Expected unified classifier to be nil when passed nil")
	}
	if service.config != config {
		t.Error("Expected config to match")
	}
}

func TestNewUnifiedClassificationService_WithBothClassifiers(t *testing.T) {
	// Test with both unified and legacy classifiers
	config := &config.RouterConfig{}
	unifiedClassifier := &classification.UnifiedClassifier{}
	legacyClassifier := &classification.Classifier{}

	service := NewUnifiedClassificationService(unifiedClassifier, legacyClassifier, config)

	if service == nil {
		t.Error("Expected non-nil service")
	}
	if service.classifier != legacyClassifier {
		t.Error("Expected legacy classifier to match provided classifier")
	}
	if service.unifiedClassifier != unifiedClassifier {
		t.Error("Expected unified classifier to match provided classifier")
	}
	if service.config != config {
		t.Error("Expected config to match")
	}
}

func TestClassificationService_HasUnifiedClassifier(t *testing.T) {
	t.Run("No_classifier", func(t *testing.T) {
		service := &ClassificationService{
			unifiedClassifier: nil,
		}

		if service.HasUnifiedClassifier() {
			t.Error("Expected HasUnifiedClassifier to return false")
		}
	})

	t.Run("With_uninitialized_classifier", func(t *testing.T) {
		// Create a real UnifiedClassifier instance (uninitialized)
		classifier := &classification.UnifiedClassifier{}
		service := &ClassificationService{
			unifiedClassifier: classifier,
		}

		// Should return false because classifier is not initialized
		if service.HasUnifiedClassifier() {
			t.Error("Expected HasUnifiedClassifier to return false for uninitialized classifier")
		}
	})
}

func TestClassificationService_GetUnifiedClassifierStats(t *testing.T) {
	t.Run("Without_classifier", func(t *testing.T) {
		service := &ClassificationService{
			unifiedClassifier: nil,
		}

		stats := service.GetUnifiedClassifierStats()
		if stats["available"] != false {
			t.Errorf("Expected available=false, got %v", stats["available"])
		}
		if _, exists := stats["initialized"]; exists {
			t.Error("Expected 'initialized' key to not exist")
		}
	})

	t.Run("With_uninitialized_classifier", func(t *testing.T) {
		classifier := &classification.UnifiedClassifier{}
		service := &ClassificationService{
			unifiedClassifier: classifier,
		}

		stats := service.GetUnifiedClassifierStats()
		if stats["available"] != true {
			t.Errorf("Expected available=true, got %v", stats["available"])
		}
		if stats["initialized"] != false {
			t.Errorf("Expected initialized=false, got %v", stats["initialized"])
		}
	})
}

func TestClassificationService_ClassifyBatchUnified_ErrorCases(t *testing.T) {
	t.Run("Empty_texts", func(t *testing.T) {
		service := &ClassificationService{
			unifiedClassifier: &classification.UnifiedClassifier{},
		}

		_, err := service.ClassifyBatchUnified([]string{})
		if err == nil {
			t.Error("Expected error for empty texts")
		}
		if err.Error() != "texts cannot be empty" {
			t.Errorf("Expected 'texts cannot be empty' error, got: %v", err)
		}
	})

	t.Run("Unified_classifier_not_initialized", func(t *testing.T) {
		service := &ClassificationService{
			unifiedClassifier: nil,
		}

		texts := []string{"test"}
		_, err := service.ClassifyBatchUnified(texts)
		if err == nil {
			t.Error("Expected error for nil unified classifier")
		}
		if err.Error() != "unified classifier not initialized" {
			t.Errorf("Expected 'unified classifier not initialized' error, got: %v", err)
		}
	})

	t.Run("Classifier_not_initialized", func(t *testing.T) {
		// Use real UnifiedClassifier but not initialized
		classifier := &classification.UnifiedClassifier{}
		service := &ClassificationService{
			unifiedClassifier: classifier,
		}

		texts := []string{"test"}
		_, err := service.ClassifyBatchUnified(texts)
		if err == nil {
			t.Error("Expected error for uninitialized classifier")
		}
		// The actual error will come from the unified classifier
	})
}

func TestClassificationService_ClassifyPIIUnified_ErrorCases(t *testing.T) {
	t.Run("Unified_classifier_not_available", func(t *testing.T) {
		service := &ClassificationService{
			unifiedClassifier: nil,
		}

		_, err := service.ClassifyPIIUnified([]string{"test"})
		if err == nil {
			t.Error("Expected error for nil unified classifier")
		}
		if err.Error() != "unified classifier not initialized" {
			t.Errorf("Expected 'unified classifier not initialized' error, got: %v", err)
		}
	})
}

func TestClassificationService_ClassifySecurityUnified_ErrorCases(t *testing.T) {
	t.Run("Unified_classifier_not_available", func(t *testing.T) {
		service := &ClassificationService{
			unifiedClassifier: nil,
		}

		_, err := service.ClassifySecurityUnified([]string{"test"})
		if err == nil {
			t.Error("Expected error for nil unified classifier")
		}
		if err.Error() != "unified classifier not initialized" {
			t.Errorf("Expected 'unified classifier not initialized' error, got: %v", err)
		}
	})
}

func TestClassificationService_ClassifyIntentUnified_ErrorCases(t *testing.T) {
	t.Run("Unified_classifier_not_available_fallback", func(t *testing.T) {
		// This should fallback to the legacy ClassifyIntent method
		service := &ClassificationService{
			unifiedClassifier: nil,
			classifier:        nil, // This will return placeholder response, not error
		}

		req := IntentRequest{Text: "test"}
		result, err := service.ClassifyIntentUnified(req)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if result == nil {
			t.Error("Expected non-nil result")
		}
		// Should get placeholder response from legacy classifier
		if result.Classification.Category != "general" {
			t.Errorf("Expected placeholder category 'general', got '%s'", result.Classification.Category)
		}
		if result.RoutingDecision != "placeholder_response" {
			t.Errorf("Expected placeholder routing decision, got '%s'", result.RoutingDecision)
		}
	})

	t.Run("Classifier_not_initialized", func(t *testing.T) {
		classifier := &classification.UnifiedClassifier{}
		service := &ClassificationService{
			unifiedClassifier: classifier,
		}

		req := IntentRequest{Text: "test"}
		_, err := service.ClassifyIntentUnified(req)
		if err == nil {
			t.Error("Expected error for uninitialized classifier")
		}
		// The actual error will come from the unified classifier
	})
}

// Test data structures and basic functionality
func TestClassificationService_BasicFunctionality(t *testing.T) {
	t.Run("Service_creation", func(t *testing.T) {
		config := &config.RouterConfig{}
		service := NewClassificationService(nil, config)

		if service == nil {
			t.Error("Expected non-nil service")
		}
		if service.config != config {
			t.Error("Expected config to match")
		}
	})

	t.Run("Global_service_access", func(t *testing.T) {
		config := &config.RouterConfig{}
		service := NewClassificationService(nil, config)

		globalService := GetGlobalClassificationService()
		if globalService != service {
			t.Error("Expected global service to match created service")
		}
	})
}

// Benchmark tests for performance validation
func BenchmarkClassificationService_HasUnifiedClassifier(b *testing.B) {
	service := &ClassificationService{
		unifiedClassifier: &classification.UnifiedClassifier{},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = service.HasUnifiedClassifier()
	}
}

func BenchmarkClassificationService_GetUnifiedClassifierStats(b *testing.B) {
	service := &ClassificationService{
		unifiedClassifier: &classification.UnifiedClassifier{},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = service.GetUnifiedClassifierStats()
	}
}

// NEW TESTS
func TestClassificationService_ImageGenerationClassification(t *testing.T) {
	// Load config
	cfg, err := config.LoadConfig("../../config/config.yaml")
	if err != nil {
		t.Skipf("Skipping test - config not available: %v", err)
		return
	}

	// Update to use your new model
	cfg.Classifier.CategoryModel.ModelID = "../../models/image_class"
	cfg.Classifier.CategoryModel.CategoryMappingPath = "../../models/image_class/category_mapping.json"
	cfg.Classifier.PIIModel.PIIMappingPath = "../../" + cfg.Classifier.PIIModel.PIIMappingPath
	cfg.PromptGuard.JailbreakMappingPath = "../../" + cfg.PromptGuard.JailbreakMappingPath
	
	// Load mappings
	categoryMapping, err := classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
	if err != nil {
		t.Skipf("Skipping test - category mapping not available: %v", err)
		return
	}

	piiMapping, _ := classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
	jailbreakMapping, _ := classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)

	// Initialize classifier with all required parameters
	classifier, err := classification.NewClassifier(
		cfg,
		categoryMapping,
		piiMapping,
		jailbreakMapping,
	)
	if err != nil {
		t.Skipf("Skipping test - classifier initialization failed: %v", err)
		return
	}

	service := NewClassificationService(classifier, cfg)

	tests := []struct {
    name             string
    text             string
    expectedCategory string
    minConfidence    float64
		}{
			{
					name:             "Image_generation_dog",
					text:             "Generate a picture of a dog",
					expectedCategory: "image_generation",
					minConfidence:    0.50,
			},
			{
					name:             "Image_generation_dragon",
					text:             "generate an image of a dragon",
					expectedCategory: "image_generation",
					minConfidence:    0.50,
			},
			{
					name:             "Image_generation_sunset",
					text:             "create a sunset photo with cinematic lighting",
					expectedCategory: "image_generation",
					minConfidence:    0.50,
			},
			{
					name:             "Image_generation_paint",
					text:             "paint a portrait in oil style",
					expectedCategory: "image_generation",
					minConfidence:    0.50,
			},
			{
					name:             "Not_image_ml",
					text:             "what is machine learning",
					expectedCategory: "not_image_generation",
					minConfidence:    0.50,
			},
			{
					name:             "Not_image_python",
					text:             "how do I install python",
					expectedCategory: "not_image_generation",
					minConfidence:    0.50,
			},
			{
					name:             "Not_image_physics",
					text:             "explain quantum physics",
					expectedCategory: "not_image_generation",
					minConfidence:    0.50,
			},
	}


	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// ClassifyCategory returns 3 values based on the error
			result, confidence, err := service.classifier.ClassifyCategory(tt.text)
			if err != nil {
				t.Errorf("Classification failed: %v", err)
				return
			}

			// Log details for debugging
			t.Logf("Input: %s", tt.text)
			t.Logf("Predicted: %s (confidence: %.4f)", result, confidence)
			t.Logf("Expected: %s (min confidence: %.2f)", tt.expectedCategory, tt.minConfidence)

			// Check category
			if result != tt.expectedCategory {
				t.Errorf("FAILED: Expected category '%s', got '%s'", 
					tt.expectedCategory, result)
			}

			// Check confidence
			if confidence < tt.minConfidence {
				t.Errorf("FAILED: Confidence too low: %.4f < %.2f", 
					confidence, tt.minConfidence)
			}

			// If test passes
			if result == tt.expectedCategory && confidence >= tt.minConfidence {
				t.Logf("✓ PASSED")
			}
		})
	}
}

// Test to expose raw predictions for debugging
func TestClassificationService_DebugLogitsAndProbs(t *testing.T) {
	cfg, err := config.LoadConfig("../../config/config.yaml")
	if err != nil {
		t.Skipf("Skipping test - config not available: %v", err)
		return
	}

	cfg.Classifier.CategoryModel.ModelID = "models/image_class"
	cfg.Classifier.CategoryModel.CategoryMappingPath = "models/image_class/category_mapping.json"

	categoryMapping, err := classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
	if err != nil {
		t.Skipf("Skipping test - category mapping not available: %v", err)
		return
	}

	piiMapping, _ := classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
	jailbreakMapping, _ := classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)

	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		t.Skipf("Skipping test - classifier initialization failed: %v", err)
		return
	}

	service := NewClassificationService(classifier, cfg)

	testCases := []struct {
		text              string
		expectImageGen    bool
	}{
		{"Generate a picture of a dog", true},
		{"what is machine learning", false},
	}

	for _, tc := range testCases {
		t.Run(tc.text, func(t *testing.T) {
			result, confidence, err := service.classifier.ClassifyCategory(tc.text)
			if err != nil {
				t.Errorf("Classification failed: %v", err)
				return
			}

			t.Logf("=== DEBUG OUTPUT ===")
			t.Logf("Text: %s", tc.text)
			t.Logf("Result Category: %s", result)
			t.Logf("Result Confidence: %.6f", confidence)
			t.Logf("Expected image_generation: %v", tc.expectImageGen)

			// Check if prediction matches expectation
			isImageGen := result == "image_generation"
			if isImageGen != tc.expectImageGen {
				t.Errorf("PREDICTION MISMATCH!")
				t.Errorf("  Expected image_generation=%v", tc.expectImageGen)
				t.Errorf("  Got category=%s", result)
				t.Errorf("  Confidence=%.4f", confidence)
				t.Errorf("  This indicates a bug in the classifier!")
			}
		})
	}
}

// Test label mapping is loaded correctly
func TestClassificationService_LabelMapping(t *testing.T) {
	cfg, err := config.LoadConfig("../../config/config.yaml")
	if err != nil {
		t.Skipf("Skipping test - config not available: %v", err)
		return
	}

	cfg.Classifier.CategoryModel.ModelID = "models/image_class"
	cfg.Classifier.CategoryModel.CategoryMappingPath = "models/image_class/category_mapping.json"

	categoryMapping, err := classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
	if err != nil {
		t.Skipf("Skipping test - category mapping not available: %v", err)
		return
	}

	t.Logf("Category Mapping Loaded:")
	t.Logf("  category_to_idx: %v", categoryMapping.CategoryToIdx)
	t.Logf("  idx_to_category: %v", categoryMapping.IdxToCategory)

	// Verify the mapping is correct
	if categoryMapping.IdxToCategory["0"] != "not_image_generation" {
		t.Errorf("Expected idx 0 -> 'not_image_generation', got '%s'", categoryMapping.IdxToCategory["0"])
	}
	if categoryMapping.IdxToCategory["1"] != "image_generation" {
		t.Errorf("Expected idx 1 -> 'image_generation', got '%s'", categoryMapping.IdxToCategory["1"])
	}

	piiMapping, _ := classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
	jailbreakMapping, _ := classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)

	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		t.Skipf("Skipping test - classifier initialization failed: %v", err)
		return
	}

	// Verify basic prediction works
	result, confidence, err := classifier.ClassifyCategory("test")
	if err != nil {
		t.Errorf("Basic classification failed: %v", err)
	} else {
		t.Logf("Basic prediction works - category: %s, confidence: %.4f", 
			result, confidence)
	}
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Benchmark the classification performance
func BenchmarkClassificationService_ImageGenerationClassification(b *testing.B) {
	cfg, err := config.LoadConfig("../../config/config.yaml")
	if err != nil {
		b.Skipf("Skipping benchmark - config not available: %v", err)
		return
	}

	cfg.Classifier.CategoryModel.ModelID = "models/image_class"
	cfg.Classifier.CategoryModel.CategoryMappingPath = "models/image_class/category_mapping.json"

	categoryMapping, _ := classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
	piiMapping, _ := classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
	jailbreakMapping, _ := classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)

	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		b.Skipf("Skipping benchmark - classifier initialization failed: %v", err)
		return
	}

	service := NewClassificationService(classifier, cfg)
	text := "Generate a picture of a dog"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = service.classifier.ClassifyCategory(text)
	}
}