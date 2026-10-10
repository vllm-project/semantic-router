package classification

import (
	"errors"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestReplayPIIEvidenceRequiresCompleteCleanClassification(t *testing.T) {
	const text = "Summarize the meeting agenda."
	for _, test := range []struct {
		name           string
		entities       []tasks.TokenEntity
		err            error
		allowed        []string
		history        []string
		includeHistory bool
		verified       bool
	}{
		{name: "clean", verified: true},
		{name: "failure allowed for routing", err: errors.New("synthetic backend unavailable")},
		{name: "incomplete scan allowed for routing", err: ErrTokenSpansTruncated},
		{name: "PII denied", entities: []tasks.TokenEntity{piiEntity("EMAIL", "alice@example.test", 0, 18, 0.99)}},
		{name: "PII allowed for routing", entities: []tasks.TokenEntity{piiEntity("EMAIL", "alice@example.test", 0, 18, 0.99)}, allowed: []string{"EMAIL"}},
		{name: "history not scanned", history: []string{"previous turn"}},
		{name: "clean history scanned", history: []string{"previous turn"}, includeHistory: true, verified: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			classifier, _, model := newTestPIIClassifier()
			classifier.Config.PIIModel.OnError = config.OnErrorAllow
			classifier.Config.PIIModel.OnUnscanned = config.OnErrorAllow
			classifier.Config.PIIRules = []config.PIIRule{{Name: "privacy", Threshold: 0.7, PIITypesAllowed: test.allowed, IncludeHistory: test.includeHistory}}
			model.setMockResponse(text, test.entities, test.err)
			for _, history := range test.history {
				model.setMockResponse(history, nil, nil)
			}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}}
			classifier.evaluatePIISignal(t.Context(), results, &sync.Mutex{}, text, test.history)
			if results.PIIContentVerified != test.verified {
				t.Fatalf("verified=%v want=%v", results.PIIContentVerified, test.verified)
			}
		})
	}
}
