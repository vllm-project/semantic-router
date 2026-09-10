package classification

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A token_spans.v1 provider that declares truncated_at returns the spans it did
// find plus ErrTokenSpansTruncated. These tests drive that result through a real
// PII rule and check what the routing decision sees under each on_error policy
// (Xunzhuo's review on #3498: the signal path used to drop every cached result
// that carried an error, so a partial scan read as a clean non-match).
func TestPIISignalTruncatedResponseRoutesThroughOnError(t *testing.T) {
	const text = "my contact is alice@corp.example and more text the provider never saw"
	email := piiEntity("EMAIL", "alice@corp.example", 14, 32, 0.99)

	cases := []struct {
		name         string
		onError      string
		entities     []candle_binding.TokenEntity
		err          error
		wantDetected bool
		wantEntities []string
	}{
		{
			name:         "allow keeps the partial spans",
			onError:      config.OnErrorAllow,
			entities:     []candle_binding.TokenEntity{email},
			err:          ErrTokenSpansTruncated,
			wantDetected: true,
			wantEntities: []string{"EMAIL"},
		},
		{
			name:         "allow with no spans before the cut reads as clean",
			onError:      config.OnErrorAllow,
			err:          ErrTokenSpansTruncated,
			wantDetected: false,
		},
		{
			name:         "block with no spans before the cut fails closed",
			onError:      config.OnErrorBlock,
			err:          ErrTokenSpansTruncated,
			wantDetected: true,
			wantEntities: []string{PIIClassificationErrorType},
		},
		{
			name:         "block keeps the partial spans and adds the error type",
			onError:      config.OnErrorBlock,
			entities:     []candle_binding.TokenEntity{email},
			err:          ErrTokenSpansTruncated,
			wantDetected: true,
			wantEntities: []string{"EMAIL", PIIClassificationErrorType},
		},
		{
			name:         "block on a backend error fails closed",
			onError:      config.OnErrorBlock,
			err:          errors.New("connection refused"),
			wantDetected: true,
			wantEntities: []string{PIIClassificationErrorType},
		},
		{
			name:         "allow on a backend error keeps the historical non-match",
			onError:      config.OnErrorAllow,
			err:          errors.New("connection refused"),
			wantDetected: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			classifier, _, mockModel := newTestPIIClassifier()
			classifier.Config.PIIModel.OnError = tc.onError
			classifier.Config.PIIRules = []config.PIIRule{{Name: "no_pii", Threshold: 0.7}}
			mockModel.setMockResponse(text, tc.entities, tc.err)

			results := &SignalResults{
				Metrics:           &SignalMetricsCollection{},
				SignalConfidences: make(map[string]float64),
				SignalValues:      make(map[string]float64),
				SignalErrors:      make(map[string]string),
			}
			var mu sync.Mutex
			classifier.evaluatePIISignal(context.Background(), results, &mu, text, nil)

			if results.PIIDetected != tc.wantDetected {
				t.Fatalf("PIIDetected = %v, want %v (entities=%v rules=%v)", results.PIIDetected, tc.wantDetected, results.PIIEntities, results.MatchedPIIRules)
			}
			for _, want := range tc.wantEntities {
				if !slices.Contains(results.PIIEntities, want) {
					t.Fatalf("PIIEntities = %v, missing %q", results.PIIEntities, want)
				}
			}
			if tc.wantDetected && !slices.Contains(results.MatchedPIIRules, "no_pii") {
				t.Fatalf("MatchedPIIRules = %v, want no_pii", results.MatchedPIIRules)
			}
		})
	}
}
