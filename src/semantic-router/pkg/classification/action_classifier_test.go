package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestClassifyAction(t *testing.T) {
	tests := []struct {
		name   string
		text   string
		action string
		score  float64
	}{
		{name: "generate", text: "Write a function that merges two sorted lists.", action: config.ActionGenerate, score: 1},
		{name: "explain", text: "What does this regex match?", action: config.ActionExplain, score: 1},
		{name: "fix", text: "Fix the null pointer exception in the handler.", action: config.ActionFix, score: 1},
		{name: "refactor", text: "Rename getUserData to fetchUser everywhere.", action: config.ActionRefactor, score: 1},
		{name: "test", text: "Write unit tests for the parser.", action: config.ActionTest, score: 1},
		{name: "no action phrase", text: "yes, go ahead", action: config.ActionOther, score: 1},
		{name: "earliest phrase decides", text: "Fix this bug and add a test for it.", action: config.ActionFix, score: 0.5},
		{name: "longer phrase wins at the same start", text: "Add a test for the empty input.", action: config.ActionTest, score: 1},
		{
			name:   "fenced code is not the request",
			text:   "Explain this:\n```python\ndef fix_all(items):\n    return [refactor(i) for i in items]\n```",
			action: config.ActionExplain,
			score:  1,
		},
		{name: "identifiers are not phrases", text: "What does generate_report() return?", action: config.ActionExplain, score: 1},
		{name: "case does not matter", text: "REFACTOR THE PARSER", action: config.ActionRefactor, score: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyAction(tt.text)
			if got.Action != tt.action || got.Score != tt.score {
				t.Fatalf("ClassifyAction(%q) = %+v, want action %q with score %v", tt.text, got, tt.action, tt.score)
			}
		})
	}
}

func TestActionSignalMatchesOnlyTheDeclaredAction(t *testing.T) {
	tests := []struct {
		name     string
		declared []config.ActionRule
		text     string
		matched  []string
	}{
		{
			name:     "declared action matches with its phrase share",
			declared: []config.ActionRule{{Name: config.ActionFix}, {Name: config.ActionTest}},
			text:     "Fix this bug and add a test for it.",
			matched:  []string{config.ActionFix},
		},
		{
			name:     "undeclared action matches nothing",
			declared: []config.ActionRule{{Name: config.ActionExplain}},
			text:     "Fix this bug and add a test for it.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classifier := &Classifier{Config: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{ActionRules: tt.declared}},
			}}
			results := &SignalResults{
				SignalConfidences: map[string]float64{},
				SignalValues:      map[string]float64{},
				Metrics:           &SignalMetricsCollection{},
			}

			classifier.evaluateActionSignal(results, &sync.Mutex{}, tt.text)

			if len(results.MatchedActionRules) != len(tt.matched) ||
				(len(tt.matched) == 1 && results.MatchedActionRules[0] != tt.matched[0]) {
				t.Fatalf("matched action rules = %v, want %v", results.MatchedActionRules, tt.matched)
			}
			if len(tt.matched) == 1 && results.SignalValues["action:"+tt.matched[0]] != 0.5 {
				t.Fatalf("signal values = %v, want action:%s at 0.5", results.SignalValues, tt.matched[0])
			}
			if len(results.SignalConfidences) != 0 {
				t.Fatalf("lexical action reported model confidences: %v", results.SignalConfidences)
			}
			if available := results.Metrics.Action.ConfidenceAvailable; available == nil || *available {
				t.Fatal("lexical action must report confidence as unavailable")
			}
		})
	}
}
