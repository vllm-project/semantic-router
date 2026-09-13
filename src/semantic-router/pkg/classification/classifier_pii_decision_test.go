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

// A valid denied entity makes a rule true even when another part of its scan
// failed. Only a match created by on_error: block is unknown to the decision
// engine and may be discarded by rules.on_unknown: no_match.
func TestPIIDetectionSurvivesIncompleteScanWithOnUnknownNoMatch(t *testing.T) {
	const text = "alice@corp.example"
	const history = "previous " + text
	email := piiEntity("EMAIL", text, 0, len(text), 0.99)
	historyEmail := piiEntity("EMAIL", text, 9, 9+len(text), 0.99)
	backendError := errors.New("connection refused")

	for _, tc := range []struct {
		name          string
		entities      []candle_binding.TokenEntity
		err           error
		historySpans  []candle_binding.TokenEntity
		historyErr    error
		allowed       []string
		threshold     float32
		wantDecision  string
		wantDeniedPII bool
	}{
		{
			name: "denied span before truncation", entities: []candle_binding.TokenEntity{email},
			err: ErrTokenSpansTruncated, wantDecision: "block_pii", wantDeniedPII: true,
		},
		{
			name: "truncation without spans", err: ErrTokenSpansTruncated,
			wantDecision: "fallback",
		},
		{
			name: "allowed span before truncation", entities: []candle_binding.TokenEntity{email},
			err: ErrTokenSpansTruncated, allowed: []string{"EMAIL"}, wantDecision: "fallback",
		},
		{
			name: "below threshold span before truncation", entities: []candle_binding.TokenEntity{email},
			err: ErrTokenSpansTruncated, threshold: 1, wantDecision: "fallback",
		},
		{
			name: "denied span with failed history", entities: []candle_binding.TokenEntity{email},
			historyErr: backendError, wantDecision: "block_pii", wantDeniedPII: true,
		},
		{
			name: "denied history with failed current message", err: backendError,
			historySpans: []candle_binding.TokenEntity{historyEmail}, wantDecision: "block_pii", wantDeniedPII: true,
		},
		{
			name: "backend error without valid evidence", err: backendError,
			wantDecision: "fallback",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			classifier, _, mockModel := newTestPIIClassifier()
			classifier.Config.PIIModel.OnError = config.OnErrorBlock
			threshold := tc.threshold
			if threshold == 0 {
				threshold = 0.7
			}
			classifier.Config.PIIRules = []config.PIIRule{{
				Name: "no_pii", Threshold: threshold, PIITypesAllowed: tc.allowed, IncludeHistory: true,
			}}
			classifier.Config.Strategy = "priority"
			classifier.Config.Decisions = []config.Decision{
				{
					Name: "block_pii", Priority: 100,
					Rules: config.RuleCombination{
						Operator: "OR", OnUnknown: config.RuleOnUnknownNoMatch,
						Conditions: []config.RuleNode{{Type: config.SignalTypePII, Name: "no_pii"}},
					},
				},
				{Name: "fallback", Priority: 1, Rules: config.RuleCombination{Operator: "AND"}},
			}
			mockModel.setMockResponse(text, tc.entities, tc.err)
			mockModel.setMockResponse(history, tc.historySpans, tc.historyErr)
			results := &SignalResults{
				Metrics: &SignalMetricsCollection{}, SignalErrors: make(map[string]string),
				SignalConfidences: make(map[string]float64), SignalValues: make(map[string]float64),
			}
			var mu sync.Mutex
			classifier.evaluatePIISignal(context.Background(), results, &mu, text, []string{history})

			result, err := classifier.EvaluateDecisionWithEngine(results)
			if err != nil {
				t.Fatalf("EvaluateDecisionWithEngine: %v", err)
			}
			if result == nil || result.Decision == nil || result.Decision.Name != tc.wantDecision {
				t.Fatalf("decision = %+v, want %s; errors=%v errorMatches=%v", result, tc.wantDecision, results.SignalErrors, results.SignalErrorMatches)
			}
			key := signalConfidenceKey(config.SignalTypePII, "no_pii")
			if results.SignalErrors[key] != piiEvaluationFailedCode {
				t.Fatalf("incomplete scan must stay visible: SignalErrors = %v", results.SignalErrors)
			}
			if !slices.Contains(results.PIIEntities, PIIClassificationErrorType) {
				t.Fatalf("missing incompleteness sentinel: PIIEntities = %v", results.PIIEntities)
			}
			if got := slices.Contains(results.PIIEntities, "EMAIL"); got != tc.wantDeniedPII {
				t.Fatalf("denied EMAIL = %v, want %v", got, tc.wantDeniedPII)
			}
			if got := results.SignalErrorMatches[key]; got != !tc.wantDeniedPII {
				t.Fatalf("error-driven match = %v, want %v", got, !tc.wantDeniedPII)
			}
		})
	}
}
