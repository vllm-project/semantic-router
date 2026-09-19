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

func TestBalanceTaskIntentPolicy(t *testing.T) {
	engine := maintainedRecipeEngine(t, "balance")
	tests := []evaluateSignalsCase{
		{
			name: "short health source request keeps verification",
			signals: &SignalMatches{
				DomainRules:     []string{"health"},
				KeywordRules:    []string{"verification_markers"},
				ProjectionRules: []string{"balance_simple"},
			},
			expectedDecision: "verified_health",
		},
		{
			name: "architecture domain and semantic intent",
			signals: &SignalMatches{
				DomainRules:    []string{"computer science"},
				EmbeddingRules: []string{"architecture_design"},
			},
			expectedDecision: "complex_specialist",
		},
		{
			name: "architecture language and semantic intent survive topic uncertainty",
			signals: &SignalMatches{
				DomainRules:    []string{"business"},
				KeywordRules:   []string{"architecture_markers"},
				EmbeddingRules: []string{"architecture_design"},
			},
			expectedDecision: "complex_specialist",
		},
		{
			name: "tradeoffs alone do not establish systems design",
			signals: &SignalMatches{
				DomainRules:  []string{"business"},
				KeywordRules: []string{"architecture_markers"},
			},
			expectedDecision: "casual_chat",
		},
		{
			name: "normative argument does not require medium difficulty",
			signals: &SignalMatches{
				DomainRules:     []string{"law"},
				KeywordRules:    []string{"normative_topic_markers", "argument_request_markers"},
				ProjectionRules: []string{"balance_simple"},
			},
			expectedDecision: "reasoning_deep",
		},
		{
			name: "actual legal risk retains priority",
			signals: &SignalMatches{
				DomainRules:     []string{"law"},
				KeywordRules:    []string{"normative_topic_markers", "argument_request_markers", "legal_risk_markers"},
				ComplexityRules: []string{"legal_risk:hard"},
			},
			expectedDecision: "premium_legal",
		},
		{
			name: "explicit explanation and sources survive unnamed topic",
			signals: &SignalMatches{
				DomainRules:     []string{"other"},
				KeywordRules:    []string{"explanation_request_markers", "verification_markers"},
				ProjectionRules: []string{"balance_simple"},
			},
			expectedDecision: "verified_explainer",
		},
		{
			name: "ordinary explanation needs no redundant effort gate",
			signals: &SignalMatches{
				DomainRules:     []string{"psychology"},
				KeywordRules:    []string{"explanation_request_markers"},
				ProjectionRules: []string{"balance_simple"},
			},
			expectedDecision: "medium_explainer",
		},
		{
			name: "fictional drafting is a task",
			signals: &SignalMatches{
				KeywordRules:    []string{"drafting_action_markers", "creative_form_markers"},
				ProjectionRules: []string{"balance_simple"},
			},
			expectedDecision: "medium_creative",
		},
		{
			name: "tone without drafting does not establish creative work",
			signals: &SignalMatches{
				KeywordRules:    []string{"personal_tone_markers"},
				ContextRules:    []string{"short_context"},
				StructureRules:  []string{"low_question_density"},
				ProjectionRules: []string{"balance_simple"},
			},
			expectedDecision: "simple_general",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestAgentTaskIntentKeepsPrivacyPriority(t *testing.T) {
	engine := maintainedRecipeEngine(t, "agent")
	tests := []evaluateSignalsCase{
		{
			name: "editing code with uncertain embedding",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				KeywordRules:    []string{"code_edit_action", "code_artifact"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "domain_code",
		},
		{
			name: "software topic alone is not coding",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "simple_general",
		},
		{
			name: "business semantics retain commercial lane",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				EmbeddingRules:  []string{"business_analysis"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "domain_business",
		},
		{
			name: "privacy outranks editing",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				KeywordRules:    []string{"code_edit_action", "code_artifact"},
				ProjectionRules: []string{"policy_privacy_local_only", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "local_privacy_policy",
		},
		{
			name: "daily comparison is not a business request",
			signals: &SignalMatches{
				DomainRules:     []string{"business"},
				KeywordRules:    []string{"comparison_request"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "medium_general",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}
