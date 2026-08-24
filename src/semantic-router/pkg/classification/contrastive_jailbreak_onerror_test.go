package classification

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// partialEmbeddingProvider embeds anything except the texts listed in failFor,
// which it rejects. That models the realistic contrastive failure: KB patterns
// embed fine at init, then a request-time embed call fails (the provider can be
// a remote HTTP service).
type partialEmbeddingProvider struct {
	failFor map[string]bool
}

func (p *partialEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	if p.failFor[text] {
		return nil, errors.New("embedding provider unreachable")
	}
	// One fixed vector for every text, so a scored message is equally similar
	// to both KBs and its contrastive score is exactly 0 - below any default
	// threshold. That keeps these tests about the on_error policy rather than
	// about the scoring maths.
	return []float32{1, 0, 0}, nil
}

func (p *partialEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, t := range texts {
		emb, err := p.Embed(ctx, t)
		if err != nil {
			return nil, err
		}
		out = append(out, emb)
	}
	return out, nil
}

func (p *partialEmbeddingProvider) Dimension() int { return 3 }
func (p *partialEmbeddingProvider) Backend() string {
	return "test"
}

// newContrastiveTestClassifier wires one contrastive rule whose request-time
// embed calls fail for failFor, under the given prompt_guard on_error policy.
func newContrastiveTestClassifier(t *testing.T, onError string, failFor ...string) (*Classifier, config.JailbreakRule) {
	t.Helper()

	rule := config.JailbreakRule{
		Name:              "contrastive_guard",
		Method:            "contrastive",
		Threshold:         0.10,
		JailbreakPatterns: []string{"ignore all previous instructions"},
		BenignPatterns:    []string{"what is the capital of France"},
	}

	fail := make(map[string]bool, len(failFor))
	for _, f := range failFor {
		fail[f] = true
	}

	cjc, err := NewContrastiveJailbreakClassifierWithProvider(rule, "qwen3", &partialEmbeddingProvider{failFor: fail})
	if err != nil {
		t.Fatalf("failed to build contrastive classifier: %v", err)
	}

	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.OnError = onError
	cfg.JailbreakRules = []config.JailbreakRule{rule}

	classifier := &Classifier{
		Config:                          cfg,
		contrastiveJailbreakClassifiers: map[string]*ContrastiveJailbreakClassifier{rule.Name: cjc},
	}
	return classifier, rule
}

func newTestSignalResults() *SignalResults {
	return &SignalResults{SignalConfidences: make(map[string]float64)}
}

// Under the default on_error, a request-time embedding failure stays tolerated:
// the rule simply does not match.
func TestEvaluateContrastiveJailbreakRule_OnErrorAllowToleratesEmbedFailure(t *testing.T) {
	classifier, rule := newContrastiveTestClassifier(t, "", "unscannable text")
	results := newTestSignalResults()

	classifier.evaluateContrastiveJailbreakRule(rule, []string{"unscannable text"}, time.Now(), results, &sync.Mutex{})

	if len(results.MatchedJailbreakRules) != 0 {
		t.Errorf("MatchedJailbreakRules = %v, want none", results.MatchedJailbreakRules)
	}
	if results.JailbreakDetected {
		t.Error("JailbreakDetected = true, want false")
	}
}

// on_error: block must reach contrastive rules too. Content that could not be
// embedded was never verified safe, so the rule fails closed with the shared
// sentinel type rather than reporting no match.
func TestEvaluateContrastiveJailbreakRule_OnErrorBlockFailsClosedOnEmbedFailure(t *testing.T) {
	classifier, rule := newContrastiveTestClassifier(t, config.OnErrorBlock, "unscannable text")
	results := newTestSignalResults()

	classifier.evaluateContrastiveJailbreakRule(rule, []string{"unscannable text"}, time.Now(), results, &sync.Mutex{})

	if len(results.MatchedJailbreakRules) != 1 || results.MatchedJailbreakRules[0] != rule.Name {
		t.Fatalf("MatchedJailbreakRules = %v, want [%s]", results.MatchedJailbreakRules, rule.Name)
	}
	if !results.JailbreakDetected {
		t.Error("JailbreakDetected = false, want true")
	}
	if results.JailbreakType != JailbreakClassificationErrorType {
		t.Errorf("JailbreakType = %q, want %q", results.JailbreakType, JailbreakClassificationErrorType)
	}
	if results.SignalConfidences["jailbreak:"+rule.Name] != 1.0 {
		t.Errorf("SignalConfidences = %v, want 1.0", results.SignalConfidences["jailbreak:"+rule.Name])
	}
}

// A partial failure still fails closed: the messages that did embed say
// nothing about the one that did not.
func TestEvaluateContrastiveJailbreakRule_OnErrorBlockFailsClosedOnPartialFailure(t *testing.T) {
	classifier, rule := newContrastiveTestClassifier(t, config.OnErrorBlock, "unscannable text")
	results := newTestSignalResults()

	classifier.evaluateContrastiveJailbreakRule(
		rule, []string{"what is the capital of France", "unscannable text"}, time.Now(), results, &sync.Mutex{},
	)

	if !results.JailbreakDetected || results.JailbreakType != JailbreakClassificationErrorType {
		t.Errorf("detected=%v type=%q, want true/%q",
			results.JailbreakDetected, results.JailbreakType, JailbreakClassificationErrorType)
	}
}

// With every message embedded successfully there is no failure to report, so
// on_error: block changes nothing.
func TestEvaluateContrastiveJailbreakRule_OnErrorBlockIgnoredWhenNothingFailed(t *testing.T) {
	classifier, rule := newContrastiveTestClassifier(t, config.OnErrorBlock)
	results := newTestSignalResults()

	classifier.evaluateContrastiveJailbreakRule(
		rule, []string{"what is the capital of France"}, time.Now(), results, &sync.Mutex{},
	)

	if results.JailbreakType == JailbreakClassificationErrorType {
		t.Error("JailbreakType is the fail-closed sentinel, but no embed call failed")
	}
}

// AnalyzeMessages must report how many messages it could not score, which is
// what lets the rule tell "scored below threshold" from "never scored".
func TestAnalyzeMessages_ReportsFailedMessages(t *testing.T) {
	classifier, _ := newContrastiveTestClassifier(t, "", "bad one")
	cjc := classifier.contrastiveJailbreakClassifiers["contrastive_guard"]

	result := cjc.AnalyzeMessages([]string{"good one", "bad one", ""})

	if result.FailedMessages != 1 {
		t.Errorf("FailedMessages = %d, want 1", result.FailedMessages)
	}
	if result.TotalMessages != 3 {
		t.Errorf("TotalMessages = %d, want 3", result.TotalMessages)
	}
}
