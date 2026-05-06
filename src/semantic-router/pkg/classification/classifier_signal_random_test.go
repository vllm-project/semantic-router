package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateRandomSignalEmitsDigitValue(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					RandomRules: []config.RandomRule{{Name: "random_digit"}},
				},
			},
		},
	}
	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}
	var mu sync.Mutex

	classifier.evaluateRandomSignal(results, &mu)

	if len(results.MatchedRandomRules) != 1 || results.MatchedRandomRules[0] != "random_digit" {
		t.Fatalf("matched random rules = %v, want [random_digit]", results.MatchedRandomRules)
	}
	value, ok := results.SignalValues["random:random_digit"]
	if !ok {
		t.Fatalf("random signal value missing from %+v", results.SignalValues)
	}
	if value < 0 || value > 9 || value != float64(int(value)) {
		t.Fatalf("random value = %v, want integer in [0,9]", value)
	}
	if got := results.SignalConfidences["random:random_digit"]; got != 1.0 {
		t.Fatalf("random confidence = %v, want 1.0", got)
	}
}
