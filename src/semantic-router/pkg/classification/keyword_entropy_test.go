package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestKeywordClassifierWithEntropyReasoningDecision tests that keyword classifier
// returns proper reasoning decisions based on category configuration
func TestKeywordClassifierWithEntropyReasoningDecision(t *testing.T) {
	// Create a test configuration
	keywordRules := []config.KeywordRule{
		{
			Category:      "urgent_request",
			Operator:      "OR",
			Keywords:      []string{"urgent", "immediate", "asap"},
			CaseSensitive: false,
		},
		{
			Category:      "thinking",
			Operator:      "OR",
			Keywords:      []string{"think", "analyze", "reason"},
			CaseSensitive: false,
		},
	}

	categories := []config.Category{
		{
			CategoryMetadata: config.CategoryMetadata{
				Name: "urgent_request",
			},
			ModelScores: []config.ModelScore{
				{
					Model: "fast-model",
					Score: 0.9,
					ModelReasoningControl: config.ModelReasoningControl{
						UseReasoning: boolPtr(false), // No reasoning for urgent requests
					},
				},
			},
		},
		{
			CategoryMetadata: config.CategoryMetadata{
				Name: "thinking",
			},
			ModelScores: []config.ModelScore{
				{
					Model: "smart-model",
					Score: 0.9,
					ModelReasoningControl: config.ModelReasoningControl{
						UseReasoning: boolPtr(true), // Enable reasoning for thinking tasks
					},
				},
			},
		},
	}

	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			KeywordRules: keywordRules,
			Categories:   categories,
		},
	}

	// Create classifier with keyword rules
	keywordClassifier, err := NewKeywordClassifier(keywordRules)
	if err != nil {
		t.Fatalf("Failed to create keyword classifier: %v", err)
	}

	classifier := &Classifier{
		Config:            cfg,
		keywordClassifier: keywordClassifier,
	}

	// Test cases
	tests := []struct {
		name                 string
		text                 string
		expectedCategory     string
		expectedUseReasoning bool
		expectedConfidence   float64
		shouldMatch          bool
	}{
		{
			name:                 "Urgent request - no reasoning",
			text:                 "This is an urgent request",
			expectedCategory:     "urgent_request",
			expectedUseReasoning: false,
			expectedConfidence:   1.0,
			shouldMatch:          true,
		},
		{
			name:                 "Thinking task - with reasoning",
			text:                 "Please think carefully about this problem",
			expectedCategory:     "thinking",
			expectedUseReasoning: true,
			expectedConfidence:   1.0,
			shouldMatch:          true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			category, confidence, reasoningDecision, err := classifier.ClassifyCategoryWithEntropy(tt.text)
			if err != nil {
				t.Fatalf("Classification failed: %v", err)
			}

			if category != tt.expectedCategory {
				t.Errorf("Expected category %q, got %q", tt.expectedCategory, category)
			}
			if confidence != tt.expectedConfidence {
				t.Errorf("Expected confidence %f, got %f", tt.expectedConfidence, confidence)
			}
			if reasoningDecision.UseReasoning != tt.expectedUseReasoning {
				t.Errorf("Expected useReasoning %v, got %v", tt.expectedUseReasoning, reasoningDecision.UseReasoning)
			}
			if reasoningDecision.Confidence != 1.0 {
				t.Errorf("Expected reasoning decision confidence 1.0, got %f", reasoningDecision.Confidence)
			}
			if reasoningDecision.DecisionReason != "keyword_match_category_config" {
				t.Errorf("Expected decision reason 'keyword_match_category_config', got %q", reasoningDecision.DecisionReason)
			}
			if len(reasoningDecision.TopCategories) != 1 {
				t.Errorf("Expected 1 top category, got %d", len(reasoningDecision.TopCategories))
			} else if reasoningDecision.TopCategories[0].Category != tt.expectedCategory {
				t.Errorf("Expected top category %q, got %q", tt.expectedCategory, reasoningDecision.TopCategories[0].Category)
			}
		})
	}
}

// Helper function to create bool pointer
func boolPtr(b bool) *bool {
	return &b
}
