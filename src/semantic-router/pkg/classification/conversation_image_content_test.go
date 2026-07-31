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
	)
	if len(results.MatchedConversationRules) != 1 ||
		results.MatchedConversationRules[0] != "has-images" {
		t.Fatalf("matched conversation rules = %v", results.MatchedConversationRules)
	}
}
