package decision

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Exercise the maintained policy itself rather than a duplicate test-only tree.
func maintainedRecipeEngine(t *testing.T, name string) *DecisionEngine {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", name, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return NewDecisionEngine(cfg.KeywordRules, cfg.EmbeddingRules, cfg.Categories, cfg.Decisions, config.RoutingStrategyPriority)
}

func TestBalanceQuestionPolicy(t *testing.T) {
	engine := maintainedRecipeEngine(t, "balance")
	lanes := []struct{ label, lane string }{
		{"legal", "premium_legal"},
		{"proof", "formal_math_proof"},
		{"argument", "reasoning_deep"},
		{"systems", "complex_specialist"},
		{"science", "complex_specialist"},
		{"code", "medium_code_general"},
		{"health", "verified_health"},
		{"explain", "medium_explainer"},
		{"creative", "medium_creative"},
		{"facts", "fast_qa"},
		{"chat", "simple_general"},
	}
	var tests []evaluateSignalsCase
	for _, l := range lanes {
		tests = append(tests, evaluateSignalsCase{
			name:             "task " + l.label + " picks " + l.lane,
			signals:          &SignalMatches{DecisionRules: []string{"task:" + l.label}},
			expectedDecision: l.lane,
		})
	}
	unknown := map[string]string{}
	for _, name := range []string{"task", "verification", "correction", "clarify"} {
		unknown["decision:"+name] = "decision_rejected"
	}
	tests = append(tests, []evaluateSignalsCase{
		{
			name:             "an explanation with a source request is verified",
			signals:          &SignalMatches{DecisionRules: []string{"task:explain", "verification"}},
			expectedDecision: "verified_explainer",
		},
		{
			name:             "a source request alone picks no lane",
			signals:          &SignalMatches{DecisionRules: []string{"verification"}},
			expectedDecision: "casual_chat",
		},
		{
			name:             "a correction needs an earlier answer",
			signals:          &SignalMatches{DecisionRules: []string{"task:code", "correction"}},
			expectedDecision: "medium_code_general",
		},
		{
			name: "a correction after an answer outranks the code lane",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:code", "correction"},
			},
			expectedDecision: "feedback_wrong_answer_verified",
		},
		{
			name: "the legal lane outranks a correction",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:legal", "correction"},
			},
			expectedDecision: "premium_legal",
		},
		{
			name: "a clarification after an answer outranks the explainer lane",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:explain", "clarify"},
			},
			expectedDecision: "feedback_need_clarification",
		},
		{
			name: "a clarification that is also a correction recovers",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:chat", "clarify", "correction"},
			},
			expectedDecision: "feedback_wrong_answer_verified",
		},
		{
			name: "an image goes to the visual lane whatever the task",
			signals: &SignalMatches{
				ConversationRules: []string{"has_image"},
				DecisionRules:     []string{"task:legal"},
			},
			expectedDecision: "omni",
		},
		{
			name: "unknown answers fall back",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				SignalErrors:      unknown,
			},
			expectedDecision: "casual_chat",
		},
	}...)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestPrivacyQuestionsKeepWorkLocal(t *testing.T) {
	engine := maintainedRecipeEngine(t, "privacy")
	unknown := map[string]string{
		"decision:attack":        "decision_rejected",
		"decision:confidential":  "decision_rejected",
		"decision:personal_data": "decision_rejected",
		"decision:depth":         "decision_rejected",
	}
	tests := []evaluateSignalsCase{
		{name: "an attack outranks private material", signals: &SignalMatches{DecisionRules: []string{"attack", "confidential"}}, expectedDecision: "local_security_containment"},
		{name: "private material stays local", signals: &SignalMatches{DecisionRules: []string{"confidential"}}, expectedDecision: "local_privacy_policy"},
		{name: "personal data stays local", signals: &SignalMatches{DecisionRules: []string{"personal_data"}}, expectedDecision: "local_privacy_policy"},
		{name: "deep reasoning on private material stays local", signals: &SignalMatches{DecisionRules: []string{"depth", "confidential"}}, expectedDecision: "local_privacy_policy"},
		{name: "deep reasoning with personal data stays local", signals: &SignalMatches{DecisionRules: []string{"depth", "personal_data"}}, expectedDecision: "local_privacy_policy"},
		{name: "only non-sensitive deep reasoning leaves", signals: &SignalMatches{DecisionRules: []string{"depth"}}, expectedDecision: "cloud_frontier_reasoning"},
		{name: "ordinary work stays local", signals: &SignalMatches{}, expectedDecision: "local_standard"},
		{name: "an unanswered request stays on the privacy lane", signals: &SignalMatches{SignalErrors: unknown}, expectedDecision: "local_privacy_policy"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestAgentQuestionsKeepPrivacyPriority(t *testing.T) {
	engine := maintainedRecipeEngine(t, "agent")
	answers := func(difficulty float64, rules ...string) *SignalMatches {
		return &SignalMatches{
			DecisionRules: rules,
			SignalValues:  map[string]float64{"decision:difficulty": difficulty},
		}
	}
	tests := []evaluateSignalsCase{
		{name: "an attack outranks privacy", signals: answers(0.5, "attack", "confidential"), expectedDecision: "local_security_containment"},
		{name: "private material outranks code work", signals: answers(1.7, "confidential", "domain:code", "difficulty"), expectedDecision: "local_privacy_policy"},
		{name: "personal data outranks a specialist field", signals: answers(1.9, "personal_data", "domain:legal_health", "difficulty"), expectedDecision: "local_privacy_policy"},
		{name: "simple math takes the fast path", signals: answers(0.3, "domain:math_science"), expectedDecision: "simple_math_fast_path"},
		{name: "math that needs reasoning takes the STEM lane", signals: answers(3.4, "domain:math_science", "difficulty"), expectedDecision: "domain_stem_research"},
		{name: "law and health keep their lane", signals: answers(1.9, "domain:legal_health", "difficulty"), expectedDecision: "domain_legal_health"},
		{name: "hard software work takes the complex code lane", signals: answers(2.9, "domain:code", "difficulty"), expectedDecision: "domain_code_complex"},
		{name: "medium software work takes the code lane", signals: answers(1.7, "domain:code", "difficulty"), expectedDecision: "domain_code"},
		{name: "a simple software question stays local", signals: answers(0.6, "domain:code"), expectedDecision: "simple_general"},
		{name: "business keeps its lane", signals: answers(1.9, "domain:business", "difficulty"), expectedDecision: "domain_business"},
		{name: "deep general reasoning escalates", signals: answers(3.0, "domain:general", "difficulty"), expectedDecision: "complex_general"},
		{name: "a general comparison takes the medium lane", signals: answers(1.8, "domain:general", "difficulty"), expectedDecision: "medium_general"},
		{name: "no answer falls to the local lane", signals: &SignalMatches{}, expectedDecision: "simple_general"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}
