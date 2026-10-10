package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

func topicContinuityTestConfig(rules ...TopicContinuityRule) *RouterConfig {
	return &RouterConfig{IntelligentRouting: IntelligentRouting{Signals: Signals{TopicContinuityRules: rules}}}
}

func decodeTopicContinuityRule(t *testing.T, document string) TopicContinuityRule {
	t.Helper()
	var rule TopicContinuityRule
	if err := yaml.Unmarshal([]byte(document), &rule); err != nil {
		t.Fatal(err)
	}
	return rule
}

func TestTopicContinuityDefaults(t *testing.T) {
	rule := decodeTopicContinuityRule(t, "name: topic_boundary\n")
	if err := ValidateTopicContinuityRuleContract(rule); err != nil {
		t.Fatalf("defaults should validate: %v", err)
	}
	cfg := rule.EvalConfig()
	want := topiccontinuity.EvalConfig{
		Name: "topic_boundary",
		Policy: topiccontinuity.HistoryPolicy{
			Limits:           topiccontinuity.Limits{MaxPriorTurns: 8, MaxTurnBytes: 16384, MaxInputBytes: 147456},
			IncludeAssistant: true,
		},
		Continuation: 0.35,
		Change:       0.08,
	}
	if cfg != want {
		t.Fatalf("EvalConfig = %+v, want %+v", cfg, want)
	}
}

func TestTopicContinuityPresenceAwareFields(t *testing.T) {
	rule := decodeTopicContinuityRule(t, `
name: strict
include_assistant: false
thresholds:
  change: 0
`)
	if err := ValidateTopicContinuityRuleContract(rule); err != nil {
		t.Fatalf("explicit false and zero should validate: %v", err)
	}
	cfg := rule.EvalConfig()
	if cfg.Policy.IncludeAssistant || cfg.Change != 0 || cfg.Continuation != 0.35 {
		t.Fatalf("explicit values not preserved: %+v", cfg)
	}
}

func TestTopicContinuityDerivedInputBytes(t *testing.T) {
	rule := decodeTopicContinuityRule(t, "name: r\nlimits:\n  max_prior_turns: 3\n  max_turn_bytes: 1024\n")
	if got := rule.EffectiveLimits().MaxInputBytes; got != 4*1024 {
		t.Fatalf("derived max_input_bytes = %d, want %d", got, 4*1024)
	}
	over := decodeTopicContinuityRule(t, "name: r\nlimits:\n  max_prior_turns: 32\n  max_turn_bytes: 65536\n")
	err := ValidateTopicContinuityRuleContract(over)
	if err == nil || !strings.Contains(err.Error(), "set an explicit limits.max_input_bytes") {
		t.Fatalf("expected explicit-total error, got %v", err)
	}
	explicit := decodeTopicContinuityRule(t,
		"name: r\nlimits:\n  max_prior_turns: 32\n  max_turn_bytes: 65536\n  max_input_bytes: 1048576\n")
	if err := ValidateTopicContinuityRuleContract(explicit); err != nil {
		t.Fatalf("explicit bounded total should validate: %v", err)
	}
}

func TestTopicContinuityRejectsInvalidRules(t *testing.T) {
	cases := map[string]string{
		"name: \"\"\n":                                               "name is required",
		"name: \" padded \"\n":                                       "surrounding whitespace",
		"name: r\nthresholds:\n  continuation: 1\n":                  "0 <= change < continuation < 1",
		"name: r\nthresholds:\n  change: 0.5\n":                      "0 <= change < continuation < 1",
		"name: r\nthresholds:\n  change: -0.1\n":                     "0 <= change < continuation < 1",
		"name: r\nlimits:\n  max_prior_turns: 33\n":                  "max_prior_turns must be in",
		"name: r\nlimits:\n  max_turn_bytes: 100\n":                  "max_turn_bytes must be in",
		"name: r\nlimits:\n  max_input_bytes: 2048\n":                "must be at least limits.max_turn_bytes",
		"name: r\nlimits:\n  max_prior_turns: -1\n":                  "must not be negative",
		"name: r\nlimits:\n  max_input_bytes: 2000000\n":             "max_input_bytes must be in",
		"name: r\nlimits:\n  max_input_bytes: 512\n":                 "max_input_bytes must be in",
		"name: r\nthresholds:\n  continuation: .nan\n":               "0 <= change < continuation < 1",
		"name: r\nthresholds:\n  continuation: 0.2\n  change: 0.2\n": "0 <= change < continuation < 1",
	}
	for document, want := range cases {
		err := ValidateTopicContinuityRuleContract(decodeTopicContinuityRule(t, document))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: expected %q, got %v", document, want, err)
		}
	}
}

func TestValidateTopicContinuityContractsRuleCountAndDuplicates(t *testing.T) {
	rules := make([]TopicContinuityRule, TopicContinuityMaxRules+1)
	for i := range rules {
		rules[i] = TopicContinuityRule{Name: string(rune('a' + i))}
	}
	if err := validateTopicContinuityContracts(topicContinuityTestConfig(rules...)); err == nil ||
		!strings.Contains(err.Error(), "at most 8 rules") {
		t.Fatalf("expected rule-count error, got %v", err)
	}
	if err := validateTopicContinuityContracts(topicContinuityTestConfig(rules[:TopicContinuityMaxRules]...)); err != nil {
		t.Fatalf("8 rules should validate: %v", err)
	}
	dup := topicContinuityTestConfig(TopicContinuityRule{Name: "x"}, TopicContinuityRule{Name: "x"})
	if err := validateTopicContinuityContracts(dup); err == nil || !strings.Contains(err.Error(), "duplicate name") {
		t.Fatalf("expected duplicate error, got %v", err)
	}
}

func TestTopicContinuityIsNotDecisionReferenceable(t *testing.T) {
	cfg := topicContinuityTestConfig(TopicContinuityRule{Name: "topic_boundary"})
	cfg.Decisions = []Decision{{
		Name:  "fresh",
		Rules: RuleNode{Type: SignalTypeTopicContinuity, Name: "topic_boundary"},
	}}
	err := validateDecisionSignalReferences(cfg)
	if err == nil || !strings.Contains(err.Error(), "not decision-referenceable") {
		t.Fatalf("expected decision rejection, got %v", err)
	}
	if entry, ok := LookupSignalCatalog(SignalTypeTopicContinuity); !ok || entry.DecisionReferenceable {
		t.Fatalf("catalog entry = %+v, %v", entry, ok)
	}
}

func TestTopicContinuityIsNotAProjectionInput(t *testing.T) {
	err := validateProjectionScoreInput("s", ProjectionScoreInput{Type: SignalTypeTopicContinuity, Name: "topic_boundary"},
		map[string]map[string]struct{}{}, topicContinuityTestConfig(), map[string]string{})
	if err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected projection rejection, got %v", err)
	}
}
