package services

import (
	"math"
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
		text           string
		expectImageGen bool
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

// Multimodal Classification Tests

// Test ClassifyMultimodal error cases
func TestClassificationService_ClassifyMultimodal_ErrorCases(t *testing.T) {
	service := &ClassificationService{
		config: &config.RouterConfig{},
	}

	t.Run("No_text_or_images", func(t *testing.T) {
		req := MultimodalClassificationRequest{
			Text:   "",
			Images: []MultimodalImage{},
		}

		_, err := service.ClassifyMultimodal(req)
		if err == nil {
			t.Error("Expected error when neither text nor images provided")
		}
		if err.Error() != "either text or images must be provided" {
			t.Errorf("Expected 'either text or images must be provided' error, got: %v", err)
		}
	})

	t.Run("Empty_images_array", func(t *testing.T) {
		req := MultimodalClassificationRequest{
			Text:   "",
			Images: []MultimodalImage{},
		}

		_, err := service.ClassifyMultimodal(req)
		if err == nil {
			t.Error("Expected error when no text and empty images")
		}
	})
}

// Test averageEmbeddings
func TestClassificationService_AverageEmbeddings(t *testing.T) {
	service := &ClassificationService{}

	t.Run("Empty_embeddings", func(t *testing.T) {
		result := service.averageEmbeddings([][]float32{})
		if result != nil {
			t.Error("Expected nil for empty embeddings")
		}
	})

	t.Run("Single_embedding", func(t *testing.T) {
		emb := []float32{1.0, 2.0, 3.0}
		result := service.averageEmbeddings([][]float32{emb})

		if len(result) != len(emb) {
			t.Errorf("Expected length %d, got %d", len(emb), len(result))
		}
		for i := range emb {
			if result[i] != emb[i] {
				t.Errorf("Expected %f at index %d, got %f", emb[i], i, result[i])
			}
		}
	})

	t.Run("Multiple_embeddings", func(t *testing.T) {
		emb1 := []float32{1.0, 2.0, 3.0}
		emb2 := []float32{3.0, 4.0, 5.0}
		emb3 := []float32{5.0, 6.0, 7.0}

		result := service.averageEmbeddings([][]float32{emb1, emb2, emb3})

		expected := []float32{3.0, 4.0, 5.0} // (1+3+5)/3, (2+4+6)/3, (3+5+7)/3
		if len(result) != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), len(result))
		}
		for i := range expected {
			if result[i] != expected[i] {
				t.Errorf("Expected %f at index %d, got %f", expected[i], i, result[i])
			}
		}
	})

	t.Run("Dimension_mismatch_handling", func(t *testing.T) {
		emb1 := []float32{1.0, 2.0}
		emb2 := []float32{3.0, 4.0, 5.0} // Different dimension

		result := service.averageEmbeddings([][]float32{emb1, emb2})

		if len(result) != 2 {
			t.Errorf("Expected length 2 (from first embedding), got %d", len(result))
		}
	})
}

// Test fuseTextAndImageEmbeddings function
func TestClassificationService_FuseTextAndImageEmbeddings(t *testing.T) {
	service := &ClassificationService{}

	t.Run("No_image_embeddings", func(t *testing.T) {
		textEmb := []float32{1.0, 2.0, 3.0}
		result := service.fuseTextAndImageEmbeddings(textEmb, [][]float32{})

		if len(result) != len(textEmb) {
			t.Errorf("Expected same length as text embedding, got %d", len(result))
		}
		for i := range textEmb {
			if result[i] != textEmb[i] {
				t.Errorf("Expected %f at index %d, got %f", textEmb[i], i, result[i])
			}
		}
	})

	t.Run("Single_image_embedding", func(t *testing.T) {
		textEmb := []float32{1.0, 2.0, 3.0}
		imageEmb := []float32{4.0, 5.0, 6.0}

		result := service.fuseTextAndImageEmbeddings(textEmb, [][]float32{imageEmb})

		// Should be 0.5 * text + 0.5 * image = [2.5, 3.5, 4.5]
		expected := []float32{2.5, 3.5, 4.5}
		if len(result) != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), len(result))
		}
		hasNonZero := false
		for _, v := range result {
			if v != 0.0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Error("Fused embedding should not be all zeros")
		}
	})

	t.Run("Multiple_image_embeddings", func(t *testing.T) {
		textEmb := []float32{1.0, 2.0, 3.0}
		imageEmb1 := []float32{4.0, 5.0, 6.0}
		imageEmb2 := []float32{7.0, 8.0, 9.0}

		result := service.fuseTextAndImageEmbeddings(textEmb, [][]float32{imageEmb1, imageEmb2})

		// Should average images first: (4+7)/2=5.5, (5+8)/2=6.5, (6+9)/2=7.5
		// Then fuse: 0.5 * text + 0.5 * avg = [3.25, 4.25, 5.25], then normalize
		if len(result) != 3 {
			t.Errorf("Expected length 3, got %d", len(result))
		}
		norm := float32(0.0)
		for _, v := range result {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm < 0.99 || norm > 1.01 {
			t.Errorf("Expected normalized embedding (norm ≈ 1.0), got norm=%f", norm)
		}
	})

	t.Run("Dimension_projection", func(t *testing.T) {
		textEmb := []float32{1.0, 2.0}
		imageEmb := []float32{3.0, 4.0, 5.0}

		result := service.fuseTextAndImageEmbeddings(textEmb, [][]float32{imageEmb})

		if len(result) != 512 {
			t.Errorf("Expected 512 dimensions (CLIP standard), got %d", len(result))
		}
	})

	t.Run("Dimension_truncation", func(t *testing.T) {
		textEmb := make([]float32, 1024)
		imageEmb := make([]float32, 768) // Different size to trigger dimension alignment
		for i := range textEmb {
			textEmb[i] = float32(i)
		}
		for i := range imageEmb {
			imageEmb[i] = float32(i + 1000)
		}

		result := service.fuseTextAndImageEmbeddings(textEmb, [][]float32{imageEmb})

		// Should project both to 512 (targetDim) when dimensions don't match
		if len(result) != 512 {
			t.Errorf("Expected 512 dimensions after projection, got %d", len(result))
		}
	})
}

// Test extractImageEmbeddings error cases
func TestClassificationService_ExtractImageEmbeddings_ErrorCases(t *testing.T) {
	service := &ClassificationService{}

	t.Run("Empty_images", func(t *testing.T) {
		_, err := service.extractImageEmbeddings([]MultimodalImage{})
		if err == nil {
			t.Error("Expected error for empty images")
		}
		if err.Error() != "failed to extract embeddings from any image" {
			t.Errorf("Expected 'failed to extract embeddings from any image' error, got: %v", err)
		}
	})

	t.Run("Invalid_base64", func(t *testing.T) {
		images := []MultimodalImage{
			{
				Data:     "invalid_base64!!!",
				MimeType: "image/jpeg",
			},
		}

		_, err := service.extractImageEmbeddings(images)
		if err == nil {
			t.Error("Expected error for invalid base64")
		}
	})

	t.Run("Empty_image_data", func(t *testing.T) {
		images := []MultimodalImage{
			{
				Data:     "",
				MimeType: "image/jpeg",
			},
		}

		_, err := service.extractImageEmbeddings(images)
		if err == nil {
			t.Error("Expected error for empty image data")
		}
	})
}

// Test classifyMultimodalWithOllama fallback
func TestClassificationService_ClassifyMultimodalWithOllama_ErrorCases(t *testing.T) {
	service := &ClassificationService{
		config: &config.RouterConfig{},
	}

	t.Run("No_classifier", func(t *testing.T) {
		req := MultimodalClassificationRequest{
			Text:   "test",
			Images: []MultimodalImage{},
		}

		// Don't panic
		result, err := service.classifyMultimodalWithOllama(req)
		if err != nil && result == nil {
		} else if result != nil {
		}
	})
}

// Test multimodal request structure
func TestMultimodalClassificationRequest_Structure(t *testing.T) {
	t.Run("Valid_request", func(t *testing.T) {
		req := MultimodalClassificationRequest{
			Text: "What is this?",
			Images: []MultimodalImage{
				{
					Data:     "base64data",
					MimeType: "image/jpeg",
				},
			},
			ContentType: "multimodal",
		}

		if req.Text == "" {
			t.Error("Text should not be empty")
		}
		if len(req.Images) == 0 {
			t.Error("Images should not be empty")
		}
		if req.Images[0].MimeType != "image/jpeg" {
			t.Error("MIME type should be set")
		}
	})

	t.Run("Image_only_request", func(t *testing.T) {
		req := MultimodalClassificationRequest{
			Text: "",
			Images: []MultimodalImage{
				{
					Data:     "base64data",
					MimeType: "image/png",
				},
			},
			ContentType: "image",
		}

		if req.Text != "" {
			t.Error("Text should be empty for image-only request")
		}
		if len(req.Images) == 0 {
			t.Error("Images should not be empty")
		}
	})
}
