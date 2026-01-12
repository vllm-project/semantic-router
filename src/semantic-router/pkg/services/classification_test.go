package services

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// TestGetRecommendedModel_WithConfig tests that getRecommendedModel returns
// real model names from configuration instead of hardcoded invalid names.
func TestGetRecommendedModel_WithConfig(t *testing.T) {
	// Create a config with real decisions and model refs
	testConfig := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "default-llm-model",
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "math",
					ModelRefs: []config.ModelRef{
						{
							Model: "phi4-math-expert",
						},
					},
				},
				{
					Name: "science",
					ModelRefs: []config.ModelRef{
						{
							Model:    "mistral-science-base",
							LoRAName: "science-lora-adapter",
						},
					},
				},
				{
					Name: "code",
					ModelRefs: []config.ModelRef{
						{
							Model: "codellama-13b",
						},
					},
				},
			},
		},
	}

	service := &ClassificationService{
		classifier: nil, // No classifier - will use config fallback
		config:     testConfig,
	}

	tests := []struct {
		name             string
		category         string
		expectedModel    string
		shouldNotContain string // What should NOT be in the result
	}{
		{
			name:             "Math category should return real model",
			category:         "math",
			expectedModel:    "phi4-math-expert",
			shouldNotContain: "-specialized-model",
		},
		{
			name:             "Science category with LoRA should return LoRA name",
			category:         "science",
			expectedModel:    "science-lora-adapter",
			shouldNotContain: "-specialized-model",
		},
		{
			name:             "Code category should return real model",
			category:         "code",
			expectedModel:    "codellama-13b",
			shouldNotContain: "-specialized-model",
		},
		{
			name:             "Unknown category should return default model",
			category:         "unknown-category",
			expectedModel:    "default-llm-model",
			shouldNotContain: "-specialized-model",
		},
		{
			name:             "Case insensitive category matching",
			category:         "MATH", // Uppercase
			expectedModel:    "phi4-math-expert",
			shouldNotContain: "-specialized-model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := service.getRecommendedModel(tt.category, 0.9)

			// Verify it returns the expected model
			if result != tt.expectedModel {
				t.Errorf("getRecommendedModel(%q) = %q, want %q",
					tt.category, result, tt.expectedModel)
			}

			// Verify it does NOT contain the old buggy pattern
			if strings.Contains(result, tt.shouldNotContain) {
				t.Errorf("getRecommendedModel(%q) = %q, should NOT contain %q (old bug pattern)",
					tt.category, result, tt.shouldNotContain)
			}
		})
	}
}

// TestGetRecommendedModel_NoConfig tests fallback behavior when config is nil
func TestGetRecommendedModel_NoConfig(t *testing.T) {
	service := &ClassificationService{
		classifier: nil,
		config:     nil,
	}

	result := service.getRecommendedModel("math", 0.9)
	if result != "" {
		t.Errorf("getRecommendedModel with nil config should return empty string, got %q", result)
	}
}

// TestGetRecommendedModel_EmptyConfig tests fallback behavior with empty config
func TestGetRecommendedModel_EmptyConfig(t *testing.T) {
	service := &ClassificationService{
		classifier: nil,
		config:     &config.RouterConfig{},
	}

	result := service.getRecommendedModel("math", 0.9)
	if result != "" {
		t.Errorf("getRecommendedModel with empty config should return empty string, got %q", result)
	}
}

// TestGetRecommendedModel_NoDecisionFound tests fallback to default model
func TestGetRecommendedModel_NoDecisionFound(t *testing.T) {
	testConfig := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "default-llm-model",
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "math",
					ModelRefs: []config.ModelRef{
						{Model: "phi4-math-expert"},
					},
				},
			},
		},
	}

	service := &ClassificationService{
		classifier: nil,
		config:     testConfig,
	}

	// Test with category that doesn't exist in decisions
	result := service.getRecommendedModel("nonexistent", 0.9)
	expected := "default-llm-model"
	if result != expected {
		t.Errorf("getRecommendedModel(%q) = %q, want %q (should fallback to default)",
			"nonexistent", result, expected)
	}
}

// TestGetRecommendedModel_EmptyModelRefs tests behavior when decision exists but has no ModelRefs
func TestGetRecommendedModel_EmptyModelRefs(t *testing.T) {
	testConfig := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "default-llm-model",
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "math",
					ModelRefs: []config.ModelRef{}, // Empty ModelRefs
				},
			},
		},
	}

	service := &ClassificationService{
		classifier: nil,
		config:     testConfig,
	}

	result := service.getRecommendedModel("math", 0.9)
	expected := "default-llm-model"
	if result != expected {
		t.Errorf("getRecommendedModel(%q) with empty ModelRefs = %q, want %q (should fallback to default)",
			"math", result, expected)
	}
}
