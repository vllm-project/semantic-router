package classification

import (
	"errors"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newPIISignalTestResults() *SignalResults {
	return &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}
}

func TestEvaluatePIISignalUsesToolResultSourceOnly(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{
			Name:      "tool_pii",
			Source:    config.PIISourceToolResult,
			Threshold: 0.7,
		},
	}

	toolText := "tool payload"
	userText := "user payload"
	pii := piiEntity("EMAIL", "alice@example.com", 0, 17, 0.99)
	mockModel.setMockResponse(toolText, []candle_binding.TokenEntity{pii}, nil)
	mockModel.setMockResponse(userText, []candle_binding.TokenEntity{pii}, nil)

	results := newPIISignalTestResults()
	var mu sync.Mutex
	classifier.evaluatePIISignal(results, &mu, userText, []string{"history payload"}, []string{toolText}, false)

	if !results.PIIDetected {
		t.Fatal("expected PII in the tool result to be detected")
	}
	if got := mockModel.callCount[toolText]; got != 1 {
		t.Fatalf("tool result was classified %d times, want 1", got)
	}
	if got := mockModel.callCount[userText]; got != 0 {
		t.Fatalf("user text was classified %d times by a tool-result-only rule, want 0", got)
	}
	if got := mockModel.callCount["history payload"]; got != 0 {
		t.Fatalf("history was classified %d times by a tool-result-only rule, want 0", got)
	}
}

func TestEvaluateAllSignalsWithHeadersRoutesToolResultPIIToDecisionEngine(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii", Source: config.PIISourceToolResult, Threshold: 0.7},
	}
	classifier.Config.IntelligentRouting.Decisions = []config.Decision{{
		Name:  "safe-route",
		Rules: config.RuleCombination{Type: config.SignalTypePII, Name: "tool_pii"},
	}}

	toolText := "tool payload"
	mockModel.setMockResponse(toolText, []candle_binding.TokenEntity{
		piiEntity("EMAIL", "alice@example.com", 0, 17, 0.99),
	}, nil)

	signals, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{
		Text:            "continue",
		CurrentUserText: "continue",
		ToolResultTexts: []string{toolText},
	})
	if err != nil {
		t.Fatalf("EvaluateAllSignalsWithHeaders() error = %v", err)
	}
	if signals == nil || len(signals.MatchedPIIRules) != 1 || signals.MatchedPIIRules[0] != "tool_pii" {
		t.Fatalf("matched PII rules = %#v, want [tool_pii]", signals)
	}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine() error = %v", err)
	}
	if result == nil || result.Decision.Name != "safe-route" {
		t.Fatalf("decision result = %#v, want safe-route", result)
	}
}

func TestEvaluateAllSignalsWithHeadersPreservesLegacyPIISourceScope(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "legacy_pii", Threshold: 0.7},
	}
	classifier.Config.IntelligentRouting.Decisions = []config.Decision{{
		Name:  "legacy-route",
		Rules: config.RuleCombination{Type: config.SignalTypePII, Name: "legacy_pii"},
	}}

	userText := "user payload"
	toolText := "tool payload"
	mockModel.setMockResponse(userText, []candle_binding.TokenEntity{
		piiEntity("EMAIL", "alice@example.com", 0, 17, 0.99),
	}, nil)
	mockModel.setMockResponse(toolText, []candle_binding.TokenEntity{
		piiEntity("EMAIL", "tool@example.com", 0, 16, 0.99),
	}, nil)

	signals, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{
		Text:            userText,
		CurrentUserText: userText,
		ToolResultTexts: []string{toolText},
	})
	if err != nil {
		t.Fatalf("EvaluateAllSignalsWithHeaders() error = %v", err)
	}
	if signals == nil || len(signals.MatchedPIIRules) != 1 || signals.MatchedPIIRules[0] != "legacy_pii" {
		t.Fatalf("matched PII rules = %#v, want [legacy_pii]", signals)
	}
	if got := mockModel.callCount[userText]; got != 1 {
		t.Fatalf("legacy user text was classified %d times, want 1", got)
	}
	if got := mockModel.callCount[toolText]; got != 0 {
		t.Fatalf("tool result was classified %d times by an omitted-source rule, want 0", got)
	}
}

func TestEvaluateAllSignalsWithHeadersPropagatesIncompleteToolResultScan(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii", Source: config.PIISourceToolResult, Threshold: 0.7},
	}
	classifier.Config.IntelligentRouting.Decisions = []config.Decision{{
		Name: "fail-closed",
		Rules: config.RuleCombination{
			Type:      config.SignalTypePII,
			Name:      "tool_pii",
			OnUnknown: config.RuleOnUnknownMatch,
		},
	}}

	failedToolText := "temporarily unavailable tool"
	cleanToolText := "clean tool result"
	mockModel.setMockResponse(failedToolText, nil, errors.New("backend unavailable"))
	mockModel.setMockResponse(cleanToolText, nil, nil)

	signals, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{
		Text:            "continue",
		CurrentUserText: "continue",
		ToolResultTexts: []string{failedToolText, cleanToolText},
	})
	if err != nil {
		t.Fatalf("EvaluateAllSignalsWithHeaders() error = %v", err)
	}
	if signals == nil {
		t.Fatal("EvaluateAllSignalsWithHeaders() returned nil signals")
	}
	if signals.PIIDetected {
		t.Fatal("incomplete PII scan must not report a clean scan as a PII match")
	}
	if got := signals.SignalErrors["pii:tool_pii"]; got != piiEvaluationIncompleteCode {
		t.Fatalf("PII error = %q, want %q", got, piiEvaluationIncompleteCode)
	}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine() error = %v", err)
	}
	if result == nil || result.Decision.Name != "fail-closed" {
		t.Fatalf("decision result = %#v, want fail-closed", result)
	}
}

func TestEvaluatePIISignalSharesToolResultCacheAcrossRules(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii_a", Source: config.PIISourceToolResult, Threshold: 0.7},
		{Name: "tool_pii_b", Source: config.PIISourceToolResult, Threshold: 0.9},
	}

	toolText := "repeated tool payload"
	mockModel.setMockResponse(toolText, []candle_binding.TokenEntity{
		piiEntity("EMAIL", "alice@example.com", 0, 17, 0.99),
	}, nil)

	results := newPIISignalTestResults()
	var mu sync.Mutex
	classifier.evaluatePIISignal(results, &mu, "current text", nil, []string{toolText, toolText}, false)

	if got := mockModel.callCount[toolText]; got != 1 {
		t.Fatalf("shared tool result was classified %d times, want 1", got)
	}
	if len(results.MatchedPIIRules) != 2 {
		t.Fatalf("matched %d PII rules, want 2", len(results.MatchedPIIRules))
	}
}

func TestEvaluatePIISignalReportsFailedToolResultScan(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii", Source: config.PIISourceToolResult, Threshold: 0.7},
	}
	mockModel.setMockResponse("unavailable tool", nil, errors.New("backend unavailable"))

	results := newPIISignalTestResults()
	var mu sync.Mutex
	classifier.evaluatePIISignal(results, &mu, "current text", nil, []string{"unavailable tool"}, false)

	if results.PIIDetected {
		t.Fatal("failed PII inference must not report a PII match")
	}
	if got := results.SignalErrors["pii:tool_pii"]; got != piiEvaluationFailedCode {
		t.Fatalf("PII error = %q, want %q", got, piiEvaluationFailedCode)
	}
}

func TestEvaluatePIISignalReportsIncompleteToolResultScan(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii", Source: config.PIISourceToolResult, Threshold: 0.7},
	}
	mockModel.setMockResponse("unavailable tool", nil, errors.New("backend unavailable"))
	mockModel.setMockResponse("clean tool", nil, nil)

	results := newPIISignalTestResults()
	var mu sync.Mutex
	classifier.evaluatePIISignal(results, &mu, "current text", nil, []string{"unavailable tool", "clean tool"}, false)

	if got := results.SignalErrors["pii:tool_pii"]; got != piiEvaluationIncompleteCode {
		t.Fatalf("PII error = %q, want %q", got, piiEvaluationIncompleteCode)
	}
}

func TestEvaluatePIISignalReportsIncompleteToolResultExtractionWithoutText(t *testing.T) {
	classifier, _, _ := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii", Source: config.PIISourceToolResult, Threshold: 0.7},
	}

	results := newPIISignalTestResults()
	var mu sync.Mutex
	classifier.evaluatePIISignal(results, &mu, "current text", nil, nil, true)

	if results.PIIDetected {
		t.Fatal("incomplete tool-result extraction must not report a PII match")
	}
	if got := results.SignalErrors["pii:tool_pii"]; got != piiEvaluationIncompleteCode {
		t.Fatalf("PII error = %q, want %q", got, piiEvaluationIncompleteCode)
	}
}

func TestEvaluatePIISignalPreservesPositiveMatchWhenToolResultExtractionIsIncomplete(t *testing.T) {
	classifier, _, mockModel := newTestPIIClassifier()
	classifier.Config.PIIRules = []config.PIIRule{
		{Name: "tool_pii", Source: config.PIISourceToolResult, Threshold: 0.7},
	}

	toolText := "tool payload"
	mockModel.setMockResponse(toolText, []candle_binding.TokenEntity{
		piiEntity("EMAIL", "alice@example.com", 0, 17, 0.99),
	}, nil)

	results := newPIISignalTestResults()
	var mu sync.Mutex
	classifier.evaluatePIISignal(results, &mu, "current text", nil, []string{toolText}, true)

	if !results.PIIDetected {
		t.Fatal("positive PII match must be preserved when another tool-result block is skipped")
	}
	if got := results.SignalErrors["pii:tool_pii"]; got != piiEvaluationIncompleteCode {
		t.Fatalf("PII error = %q, want %q", got, piiEvaluationIncompleteCode)
	}
}
