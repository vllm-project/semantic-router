package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConversationImageContentSignal(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ConversationRules: []config.ConversationRule{{
				Name: "has-images",
				Feature: config.ConversationFeature{
					Type:   "exists",
					Source: config.ConversationSource{Type: "image_content"},
				},
			}}},
		},
	}}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateConversationSignal(
		results,
		&sync.Mutex{},
		ConversationFacts{ImageContentCount: 1},
		map[string]bool{"conversation:has-images": true},
	)
	if len(results.MatchedConversationRules) != 1 ||
		results.MatchedConversationRules[0] != "has-images" {
		t.Fatalf("matched conversation rules = %v", results.MatchedConversationRules)
	}
}

func TestConversationSignalPublishesNonmatchingRawValue(t *testing.T) {
	minimum := 1.0
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ConversationRules: []config.ConversationRule{{
				Name: "has-images",
				Feature: config.ConversationFeature{
					Type:   "count",
					Source: config.ConversationSource{Type: "image_content"},
				},
				Predicate: &config.NumericPredicate{GTE: &minimum},
			}}},
		},
	}}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateConversationSignal(
		results,
		&sync.Mutex{},
		ConversationFacts{},
		map[string]bool{"conversation:has-images": true},
	)

	if results.SignalValues["conversation:has-images"] != 0 {
		t.Fatalf("signal values = %v, want published zero", results.SignalValues)
	}
	if len(results.MatchedConversationRules) != 0 {
		t.Fatalf("matched conversation rules = %v", results.MatchedConversationRules)
	}
	if results.Metrics.Conversation.Confidence != 0 {
		t.Fatalf("conversation confidence = %v, want 0", results.Metrics.Conversation.Confidence)
	}
}

func TestConversationSignalsRespectRuleScope(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ConversationRules: []config.ConversationRule{
				{
					Name: "used-images",
					Feature: config.ConversationFeature{
						Type:   "count",
						Source: config.ConversationSource{Type: "image_content"},
					},
				},
				{
					Name: "unused-images",
					Feature: config.ConversationFeature{
						Type:   "count",
						Source: config.ConversationSource{Type: "image_content"},
					},
				},
			}},
		},
	}}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateConversationSignal(
		results,
		&sync.Mutex{},
		ConversationFacts{ImageContentCount: 1},
		map[string]bool{"conversation:used-images": true},
	)

	if _, exists := results.SignalValues["conversation:unused-images"]; exists {
		t.Fatal("unused conversation rule was evaluated")
	}
}
