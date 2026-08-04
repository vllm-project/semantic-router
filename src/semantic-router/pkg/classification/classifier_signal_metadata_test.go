package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateMetadataSignal(t *testing.T) {
	denied := "denied"
	exists := true
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{MetadataRules: []config.MetadataRule{
				{
					Name: "consent-denied",
					Key:  "consent",
					Predicate: config.MetadataPredicate{
						Equals: &denied,
					},
				},
				{
					Name: "has-cohort",
					Key:  "cohort",
					Predicate: config.MetadataPredicate{
						Exists: &exists,
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

	classifier.evaluateMetadataSignal(results, &sync.Mutex{}, RequestFacts{
		Metadata: map[string]string{"consent": "denied", "cohort": "canary"},
	}, map[string]bool{
		"metadata:consent-denied": true,
		"metadata:has-cohort":     true,
	})

	if len(results.MatchedMetadataRules) != 2 {
		t.Fatalf("matched metadata rules = %v, want two matches", results.MatchedMetadataRules)
	}
	if got := results.SignalConfidences["metadata:consent-denied"]; got != 1 {
		t.Fatalf("consent confidence = %v, want 1", got)
	}
}

func TestSignalRuleUsedNormalizesConfiguredNames(t *testing.T) {
	if !signalRuleUsed(
		map[string]bool{"metadata:consent-denied": true},
		config.SignalTypeMetadata,
		"Consent-Denied",
	) {
		t.Fatal("signalRuleUsed() should match mixed-case configured names")
	}
}
