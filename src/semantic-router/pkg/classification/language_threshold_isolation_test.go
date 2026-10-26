package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLanguageSignal_DefaultThresholdIsolation(t *testing.T) {
	for _, tt := range []struct {
		name  string
		extra string
	}{
		{name: "single defaulted rule"},
		{name: "higher English threshold", extra: "      - name: en\n        threshold: 1\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
routing:
  signals:
    language:
      - name: fr
` + tt.extra + `  decisions:
    - name: french
      priority: 100
      rules:
        operator: OR
        conditions: [{type: language, name: fr}]
      modelRefs: [{model: local}]
    - name: fallback
      priority: 1
      rules: {operator: AND}
      modelRefs: [{model: local}]
providers:
  models:
    - name: local
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
`))
			if err != nil {
				t.Fatal(err)
			}
			language, err := NewLanguageClassifier(cfg.LanguageRules)
			if err != nil {
				t.Fatal(err)
			}
			c := &Classifier{Config: cfg, languageClassifier: language}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}}
			var mu sync.Mutex
			c.evaluateLanguageSignal(results, &mu, "Bonjour, comment allez-vous?")
			decision, err := c.EvaluateDecisionWithEngine(results)
			if err != nil || decision == nil || decision.Decision == nil {
				t.Fatalf("decision=%v error=%v", decision, err)
			}
			t.Logf("rules=%v shared_threshold=%g matched=%v confidence=%g decision=%s", cfg.LanguageRules,
				lowestLanguageThreshold(cfg.LanguageRules), results.MatchedLanguageRules,
				results.Metrics.Language.Confidence, decision.Decision.Name)
			if len(results.MatchedLanguageRules) != 1 || results.MatchedLanguageRules[0] != "fr" {
				t.Error("adding an English rule changed the French rule's default threshold behavior")
			}
			if decision.Decision.Name != "french" {
				t.Errorf("expected french decision, got %q", decision.Decision.Name)
			}
		})
	}
}

func TestLanguageSignal_PerRuleThresholds(t *testing.T) {
	const lowConfidenceFrench = "comment"
	language, err := NewLanguageClassifier(nil)
	if err != nil {
		t.Fatal(err)
	}
	detected, err := language.ClassifyWithThreshold(lowConfidenceFrench, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if detected.LanguageCode != "fr" || detected.Confidence < 0.1 || detected.Confidence >= defaultLanguageThreshold {
		t.Fatalf("expected French fixture confidence in [0.1, %g), got %+v", defaultLanguageThreshold, detected)
	}

	for _, tt := range []struct {
		name      string
		text      string
		rules     []config.LanguageRule
		wantMatch string
	}{
		{
			name:  "lower English threshold does not loosen defaulted French rule",
			text:  lowConfidenceFrench,
			rules: []config.LanguageRule{{Name: "fr"}, {Name: "en", Threshold: 0.1}},
		},
		{
			name:      "explicit lower French threshold matches",
			text:      lowConfidenceFrench,
			rules:     []config.LanguageRule{{Name: "fr", Threshold: 0.1}, {Name: "en"}},
			wantMatch: "fr",
		},
		{
			name:  "single defaulted French rule rejects low confidence",
			text:  lowConfidenceFrench,
			rules: []config.LanguageRule{{Name: "fr"}},
		},
		{
			name:      "single explicit lower threshold matches",
			text:      lowConfidenceFrench,
			rules:     []config.LanguageRule{{Name: "fr", Threshold: 0.1}},
			wantMatch: "fr",
		},
		{
			name:  "single explicit higher threshold rejects",
			text:  "Bonjour, comment allez-vous?",
			rules: []config.LanguageRule{{Name: "fr", Threshold: 1}},
		},
		{
			name:      "defaulted English rule matches detector fallback",
			text:      lowConfidenceFrench,
			rules:     []config.LanguageRule{{Name: "en"}},
			wantMatch: "en",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.LanguageRules = tt.rules
			c := &Classifier{Config: cfg, languageClassifier: language}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}}
			var mu sync.Mutex
			c.evaluateLanguageSignal(results, &mu, tt.text)
			if tt.wantMatch == "" {
				if len(results.MatchedLanguageRules) != 0 {
					t.Fatalf("expected no language match, got %v (confidence=%g)", results.MatchedLanguageRules, results.Metrics.Language.Confidence)
				}
			} else if len(results.MatchedLanguageRules) != 1 || results.MatchedLanguageRules[0] != tt.wantMatch {
				t.Fatalf("expected language match %q, got %v", tt.wantMatch, results.MatchedLanguageRules)
			}
		})
	}
}

func TestLowestLanguageThreshold(t *testing.T) {
	for _, tt := range []struct {
		name  string
		rules []config.LanguageRule
		want  float32
	}{
		{name: "no rules", want: 0},
		{name: "defaulted rule", rules: []config.LanguageRule{{Name: "fr"}}, want: defaultLanguageThreshold},
		{name: "single explicit higher", rules: []config.LanguageRule{{Name: "en", Threshold: 1}}, want: 1},
		{name: "single explicit lower", rules: []config.LanguageRule{{Name: "en", Threshold: 0.1}}, want: 0.1},
		{name: "default before higher", rules: []config.LanguageRule{{Name: "fr"}, {Name: "en", Threshold: 1}}, want: defaultLanguageThreshold},
		{name: "default after higher", rules: []config.LanguageRule{{Name: "en", Threshold: 1}, {Name: "fr"}}, want: defaultLanguageThreshold},
		{name: "default with lower", rules: []config.LanguageRule{{Name: "fr"}, {Name: "en", Threshold: 0.1}}, want: 0.1},
		{name: "explicit minimum", rules: []config.LanguageRule{{Name: "fr", Threshold: 0.4}, {Name: "en", Threshold: 0.8}}, want: 0.4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := lowestLanguageThreshold(tt.rules); got != tt.want {
				t.Fatalf("expected threshold %g, got %g", tt.want, got)
			}
		})
	}
}
