package classification

import (
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Security regression (issue #1961): include_history PII rules must detect a
// secret placed in a prior USER turn, not just system/assistant history.
func TestPIISignal_DetectsSecretInPriorUserTurn(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{
			Name:            "restricted_pii",
			Threshold:       0.7,
			IncludeHistory:  true,
			PIITypesAllowed: []string{},
		},
	}

	secretTurn := "my contact is alice@corp.example"
	currentTurn := "thanks, summarize that in one line"
	priorUserMessages := []string{secretTurn}
	nonUserMessages := []string{"Sure, here is how rotation works."}

	mockModel.setMockResponse(secretTurn, []candle_binding.TokenEntity{
		piiEntity("EMAIL", "alice@corp.example", 14, 32, 0.99),
	}, nil)

	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}
	var mu sync.Mutex

	history := historyForHistoryAwareSignals(priorUserMessages, nonUserMessages)
	classifier.evaluatePIISignal(results, &mu, currentTurn, history)

	if !results.PIIDetected {
		t.Fatalf("SECURITY: PII in a prior user turn must be detected with include_history=true (issue #1961); got PIIDetected=false")
	}
}
