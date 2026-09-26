package classification

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type signalLimitBackend struct{ err error }

func (b signalLimitBackend) Classify(context.Context, string) (SequenceClassificationResult, error) {
	return SequenceClassificationResult{}, b.err
}

func (b signalLimitBackend) ClassifyTokens(context.Context, string) (tasks.TokenClassificationResult, error) {
	return tasks.TokenClassificationResult{}, b.err
}

// NewSignalErrorTestClassifier is available only to external tests; no native model is loaded.
func NewSignalErrorTestClassifier(err error) (*Classifier, error) {
	cfg := &config.RouterConfig{}
	cfg.PromptGuard = config.PromptGuardConfig{Enabled: true, ModelID: "test-guard", JailbreakMappingPath: "test-mapping", Threshold: .5}
	cfg.PIIModel.ModelID = "test-pii"
	cfg.PIIMappingPath = "test-pii-mapping"
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "guard", Threshold: .5}}
	cfg.PIIRules = []config.PIIRule{{Name: "private", Threshold: .5}}
	cfg.Decisions = []config.Decision{
		{Name: "guarded", Priority: 2, Rules: config.RuleNode{Type: config.SignalTypeJailbreak, Name: "guard", OnUnknown: config.RuleOnUnknownFailRequest}},
		{Name: "private", Priority: 1, Rules: config.RuleNode{Type: config.SignalTypePII, Name: "private", OnUnknown: config.RuleOnUnknownFailRequest}},
	}
	return newClassifierWithOptions(cfg,
		withJailbreak(&JailbreakMapping{LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1}, IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"}}, &MockJailbreakInitializer{}, signalLimitBackend{err}),
		withPII(&PIIMapping{LabelToIdx: map[string]int{"EMAIL_ADDRESS": 0}, IdxToLabel: map[string]string{"0": "EMAIL_ADDRESS"}}, &MockPIIInitializer{}, signalLimitBackend{err}),
	)
}

func TestSignalInputLimitAggregationPreservesPolicies(t *testing.T) {
	limit := fmt.Errorf("%w: private backend detail", binding.ErrInputLimit)
	generic := errors.New("input_limit at /private/example.bin is untyped")
	for _, policy := range []string{config.OnErrorAllow, config.OnErrorBlock} {
		for _, reverse := range []bool{false, true} {
			for _, positive := range []bool{false, true} {
				name := fmt.Sprintf("%s/reverse=%v/positive=%v", policy, reverse, positive)
				t.Run(name, func(t *testing.T) {
					c, err := NewSignalErrorTestClassifier(limit)
					require.NoError(t, err)
					c.Config.PromptGuard.OnError = policy
					c.Config.PIIModel.OnError = policy
					guard := []cachedJailbreakResult{{err: generic}, {err: limit}}
					pii := []cachedPIIResult{{err: generic}, {err: limit}}
					if reverse {
						guard[0], guard[1] = guard[1], guard[0]
						pii[0], pii[1] = pii[1], pii[0]
					}
					if positive {
						guard = append(guard, cachedJailbreakResult{result: SequenceClassificationResult{Probabilities: []float32{.9, .1}}})
						pii = append(pii, cachedPIIResult{result: tasks.TokenClassificationResult{Entities: []tasks.TokenEntity{{EntityType: "EMAIL_ADDRESS", Text: "sample", Start: 0, End: 6, Confidence: .9}}}})
					}
					result := &SignalResults{SignalConfidences: map[string]float64{}}
					c.evaluateBERTJailbreakRule(c.Config.JailbreakRules[0], []string{"sample"}, map[string][]cachedJailbreakResult{"sample": guard}, time.Now(), result, &sync.Mutex{})
					c.evaluatePIIRule(c.Config.PIIRules[0], "sample", nil, nil, false,
						map[piiCacheKey]cachedPIIContent{{source: "legacy", content: "sample"}: {results: pii}},
						time.Now(), result, &sync.Mutex{})
					require.Equal(t, "input_limit", result.SignalErrors["jailbreak:guard"])
					require.Equal(t, "input_limit", result.SignalErrors["pii:private"])
					wantMatched := positive || policy == config.OnErrorBlock
					require.Equal(t, wantMatched, len(result.MatchedJailbreakRules) > 0)
					require.Equal(t, wantMatched, len(result.MatchedPIIRules) > 0)
					require.Equal(t, !positive && policy == config.OnErrorBlock, result.SignalErrorMatches["jailbreak:guard"])
					require.Equal(t, !positive && policy == config.OnErrorBlock, result.SignalErrorMatches["pii:private"])
					if positive {
						require.InDelta(t, .9, result.SignalValues["jailbreak:guard"], .00001)
					}
				})
			}
		}
	}
}

func TestSignalInputLimitCategoricalGuardPreservesPositive(t *testing.T) {
	classifier, err := NewSignalErrorTestClassifier(binding.ErrInputLimit)
	require.NoError(t, err)
	cache := map[string][]cachedJailbreakResult{"sample": {
		{err: errors.New("untyped private failure")},
		{err: fmt.Errorf("wrapped: %w", binding.ErrInputLimit)},
		{decision: &tasks.LabelDecision{Label: "jailbreak"}},
	}}
	result := &SignalResults{SignalConfidences: map[string]float64{}}
	classifier.evaluateCategoricalJailbreakRule(classifier.Config.JailbreakRules[0], []string{"sample"}, cache, time.Now(), result, &sync.Mutex{})
	require.Equal(t, "input_limit", result.SignalErrors["jailbreak:guard"])
	require.Equal(t, []string{"guard"}, result.MatchedJailbreakRules)
	require.False(t, result.SignalErrorMatches["jailbreak:guard"])
	require.Empty(t, result.SignalValues)
}
