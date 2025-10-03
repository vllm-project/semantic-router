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
