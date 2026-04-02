package services

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestNewUnifiedClassificationService(t *testing.T) {
	// Test with nil unified classifier and nil legacy classifier (this is expected to work)
	config := &config.RouterConfig{}
	service := NewUnifiedClassificationService(nil, nil, config)

	if service == nil {
		t.Fatal("Expected non-nil service")
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
		t.Fatal("Expected non-nil service")
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
			t.Fatal("Expected non-nil service")
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

func TestBuildIntentResponseFromSignals_IncludesExtendedMatchedSignals(t *testing.T) {
	service := &ClassificationService{}
	req := IntentRequest{}
	signals := &classification.SignalResults{
		MatchedModalityRules:   []string{"AR"},
		MatchedAuthzRules:      []string{"premium_tier"},
		MatchedJailbreakRules:  []string{"jailbreak:block"},
		MatchedPIIRules:        []string{"pii:email"},
		MatchedReaskRules:      []string{"likely_dissatisfied"},
		MatchedProjectionRules: []string{"balance_reasoning"},
	}
	decisionResult := &decision.DecisionResult{
		Decision: &config.Decision{
			Name: "projection_route",
			ModelRefs: []config.ModelRef{{
				Model: "qwen3-8b",
			}},
		},
		Confidence: 0.91,
	}

	response := service.buildIntentResponseFromSignals(signals, decisionResult, "projection_route", 0.91, 12, req)
	require.NotNil(t, response)
	require.NotNil(t, response.MatchedSignals)

	assert.Equal(t, []string{"AR"}, response.MatchedSignals.Modality)
	assert.Equal(t, []string{"premium_tier"}, response.MatchedSignals.Authz)
	assert.Equal(t, []string{"jailbreak:block"}, response.MatchedSignals.Jailbreak)
	assert.Equal(t, []string{"pii:email"}, response.MatchedSignals.PII)
	assert.Equal(t, []string{"likely_dissatisfied"}, response.MatchedSignals.Reask)
	assert.Equal(t, []string{"balance_reasoning"}, response.MatchedSignals.Projection)
}

func TestBuildEvalResponse_ProjectionSignalsIncludedInUsedMatchedAndUnmatched(t *testing.T) {
	routerConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Projections: config.Projections{
				Mappings: []config.ProjectionMapping{
					{
						Name:   "difficulty_band",
						Source: "difficulty_score",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{
							{Name: "balance_medium"},
							{Name: "balance_reasoning"},
						},
					},
				},
			},
			Decisions: []config.Decision{
				{
					Name: "reasoning_route",
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleNode{{
							Type: "projection",
							Name: "balance_reasoning",
						}},
					},
				},
			},
		},
	}
	service := &ClassificationService{
		classifier: &classification.Classifier{Config: routerConfig},
		config:     routerConfig,
	}
	signals := &classification.SignalResults{
		MatchedProjectionRules: []string{"balance_reasoning"},
		Metrics:                &classification.SignalMetricsCollection{},
		SignalConfidences:      map[string]float64{"projection:balance_reasoning": 0.94},
	}
	decisionResult := &decision.DecisionResult{
		Decision: &routerConfig.Decisions[0],
	}

	response := service.buildEvalResponse("reason carefully", signals, decisionResult)
	require.NotNil(t, response)
	require.NotNil(t, response.DecisionResult)
	require.NotNil(t, response.DecisionResult.UsedSignals)
	require.NotNil(t, response.DecisionResult.MatchedSignals)
	require.NotNil(t, response.DecisionResult.UnmatchedSignals)

	assert.Equal(t, []string{"balance_reasoning"}, response.DecisionResult.UsedSignals.Projection)
	assert.Equal(t, []string{"balance_reasoning"}, response.DecisionResult.MatchedSignals.Projection)
	assert.Equal(t, []string{"balance_medium"}, response.DecisionResult.UnmatchedSignals.Projection)
}

func TestBuildEvalResponse_IncludesSignalValues(t *testing.T) {
	service := &ClassificationService{}
	signals := &classification.SignalResults{
		MatchedStructureRules: []string{"many_questions"},
		Metrics:               &classification.SignalMetricsCollection{},
		SignalConfidences:     map[string]float64{"structure:many_questions": 1},
		SignalValues:          map[string]float64{"structure:many_questions": 4},
	}

	response := service.buildEvalResponse("why? why? why? why?", signals, nil)
	require.NotNil(t, response)
	require.NotNil(t, response.SignalValues)
	assert.Equal(t, 4.0, response.SignalValues["structure:many_questions"])
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

func TestDetectPII_EdgeCases(t *testing.T) {
	t.Run("Empty_text_returns_error", func(t *testing.T) {
		service := &ClassificationService{classifier: nil}
		_, err := service.DetectPII(PIIRequest{Text: ""})
		require.Error(t, err)
		assert.Equal(t, "text cannot be empty", err.Error())
	})

	t.Run("Nil_classifier_returns_placeholder", func(t *testing.T) {
		service := &ClassificationService{classifier: nil}
		resp, err := service.DetectPII(PIIRequest{Text: "hello"})
		require.NoError(t, err)
		assert.False(t, resp.HasPII)
		assert.Empty(t, resp.Entities)
		assert.Equal(t, "allow", resp.SecurityRecommendation)
	})
}

func TestBuildPIIResponse_NoDetections(t *testing.T) {
	resp := newPIIResponseTestService().buildPIIResponse("hello world", []classification.PIIDetection{}, nil)

	assert.False(t, resp.HasPII)
	assert.Empty(t, resp.Entities)
	assert.Equal(t, "allow", resp.SecurityRecommendation)
	assert.Empty(t, resp.MaskedText)
}

func TestBuildPIIResponse_DefaultOptions(t *testing.T) {
	detections := samplePIIResponseDetections()
	resp := newPIIResponseTestService().buildPIIResponse(samplePIIResponseText, detections[:2], nil)

	assert.True(t, resp.HasPII)
	assert.Len(t, resp.Entities, 2)
	assert.Equal(t, "[DETECTED]", resp.Entities[0].Value)
	assert.Equal(t, "[DETECTED]", resp.Entities[1].Value)
	assert.Equal(t, 0, resp.Entities[0].StartPos)
	assert.Equal(t, 0, resp.Entities[0].EndPos)
	assert.Empty(t, resp.MaskedText)
	assert.Equal(t, "block", resp.SecurityRecommendation)
}

func TestBuildPIIResponse_RevealEntityTextOption(t *testing.T) {
	detections := samplePIIResponseDetections()

	t.Run("enabled", func(t *testing.T) {
		resp := newPIIResponseTestService().buildPIIResponse(
			samplePIIResponseText,
			detections[:1],
			&PIIOptions{RevealEntityText: true},
		)
		assert.Equal(t, "alice@test.com", resp.Entities[0].Value)
	})

	t.Run("disabled", func(t *testing.T) {
		resp := newPIIResponseTestService().buildPIIResponse(
			samplePIIResponseText,
			detections[:1],
			&PIIOptions{RevealEntityText: false},
		)
		assert.Equal(t, "[DETECTED]", resp.Entities[0].Value)
	})
}

func TestBuildPIIResponse_ReturnPositionsOption(t *testing.T) {
	detections := samplePIIResponseDetections()

	t.Run("enabled", func(t *testing.T) {
		resp := newPIIResponseTestService().buildPIIResponse(
			samplePIIResponseText,
			detections[:1],
			&PIIOptions{ReturnPositions: true},
		)
		assert.Equal(t, 13, resp.Entities[0].StartPos)
		assert.Equal(t, 29, resp.Entities[0].EndPos)
	})

	t.Run("disabled", func(t *testing.T) {
		resp := newPIIResponseTestService().buildPIIResponse(
			samplePIIResponseText,
			detections[:1],
			&PIIOptions{ReturnPositions: false},
		)
		assert.Equal(t, 0, resp.Entities[0].StartPos)
		assert.Equal(t, 0, resp.Entities[0].EndPos)
	})
}

func TestBuildPIIResponse_EntityTypeFiltering(t *testing.T) {
	detections := samplePIIResponseDetections()

	t.Run("filter_multiple_types", func(t *testing.T) {
		resp := newPIIResponseTestService().buildPIIResponse(
			samplePIIResponseText,
			detections,
			&PIIOptions{EntityTypes: []string{"EMAIL", "PHONE"}},
		)
		require.Len(t, resp.Entities, 2)
		assert.Equal(t, "EMAIL", resp.Entities[0].Type)
		assert.Equal(t, "PHONE", resp.Entities[1].Type)
	})

	t.Run("case_insensitive", func(t *testing.T) {
		resp := newPIIResponseTestService().buildPIIResponse(
			samplePIIResponseText,
			detections[:1],
			&PIIOptions{EntityTypes: []string{"email"}},
		)
		require.Len(t, resp.Entities, 1)
		assert.Equal(t, "EMAIL", resp.Entities[0].Type)
	})
}

func TestBuildPIIResponse_MaskEntities(t *testing.T) {
	service := newPIIResponseTestService()

	t.Run("basic", func(t *testing.T) {
		resp := service.buildPIIResponse(
			"Contact alice@test.com please",
			[]classification.PIIDetection{
				{EntityType: "EMAIL", Start: 8, End: 22, Text: "alice@test.com", Confidence: 0.95},
			},
			&PIIOptions{MaskEntities: true},
		)
		assert.Equal(t, "Contact [EMAIL_0] please", resp.MaskedText)
		assert.Equal(t, "[EMAIL_0]", resp.Entities[0].MaskedValue)
	})

	t.Run("same_entity_twice", func(t *testing.T) {
		resp := service.buildPIIResponse(
			"Email alice@test.com and again alice@test.com",
			[]classification.PIIDetection{
				{EntityType: "EMAIL", Start: 6, End: 20, Text: "alice@test.com", Confidence: 0.95},
				{EntityType: "EMAIL", Start: 31, End: 45, Text: "alice@test.com", Confidence: 0.93},
			},
			&PIIOptions{MaskEntities: true},
		)
		assert.Equal(t, "[EMAIL_0]", resp.Entities[0].MaskedValue)
		assert.Equal(t, "[EMAIL_0]", resp.Entities[1].MaskedValue)
		assert.Equal(t, "Email [EMAIL_0] and again [EMAIL_0]", resp.MaskedText)
	})

	t.Run("different_texts_same_type", func(t *testing.T) {
		resp := service.buildPIIResponse(
			"Email alice@test.com and bob@test.com",
			[]classification.PIIDetection{
				{EntityType: "EMAIL", Start: 6, End: 20, Text: "alice@test.com", Confidence: 0.95},
				{EntityType: "EMAIL", Start: 25, End: 37, Text: "bob@test.com", Confidence: 0.92},
			},
			&PIIOptions{MaskEntities: true},
		)
		assert.Equal(t, "[EMAIL_0]", resp.Entities[0].MaskedValue)
		assert.Equal(t, "[EMAIL_1]", resp.Entities[1].MaskedValue)
		assert.Equal(t, "Email [EMAIL_0] and [EMAIL_1]", resp.MaskedText)
	})
}

func TestBuildPIIResponse_CombinedOptions(t *testing.T) {
	resp := newPIIResponseTestService().buildPIIResponse(
		"Alice alice@test.com",
		[]classification.PIIDetection{
			{EntityType: "PERSON", Start: 0, End: 5, Text: "Alice", Confidence: 0.88},
			{EntityType: "EMAIL", Start: 6, End: 20, Text: "alice@test.com", Confidence: 0.95},
		},
		&PIIOptions{
			EntityTypes:      []string{"EMAIL"},
			ReturnPositions:  true,
			MaskEntities:     true,
			RevealEntityText: true,
		},
	)

	require.Len(t, resp.Entities, 1)
	entity := resp.Entities[0]
	assert.Equal(t, "EMAIL", entity.Type)
	assert.Equal(t, "alice@test.com", entity.Value)
	assert.Equal(t, 6, entity.StartPos)
	assert.Equal(t, 20, entity.EndPos)
	assert.Equal(t, "[EMAIL_0]", entity.MaskedValue)
	assert.Equal(t, "Alice [EMAIL_0]", resp.MaskedText)
}

func newPIIResponseTestService() *ClassificationService {
	return &ClassificationService{}
}

func samplePIIResponseDetections() []classification.PIIDetection {
	return []classification.PIIDetection{
		{EntityType: "EMAIL", Start: 13, End: 29, Text: "alice@test.com", Confidence: 0.95},
		{EntityType: "PERSON", Start: 0, End: 5, Text: "Alice", Confidence: 0.88},
		{EntityType: "PHONE", Start: 37, End: 49, Text: "555-123-4567", Confidence: 0.75},
	}
}

const samplePIIResponseText = "Alice reached alice@test.com at tel 555-123-4567"
