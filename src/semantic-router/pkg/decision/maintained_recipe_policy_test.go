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
			name: "health semantics retain verification when topic is uncertain",
			signals: &SignalMatches{
				DomainRules:     []string{"other"},
				EmbeddingRules:  []string{"health_guidance"},
				KeywordRules:    []string{"verification_markers", "reference_heavy_markers"},
				ContextRules:    []string{"short_context"},
				ProjectionRules: []string{"balance_simple", "verification_required"},
			},
			expectedDecision: "verified_health",
		},
		{
			name: "health semantics alone do not request verification",
			signals: &SignalMatches{
				EmbeddingRules: []string{"health_guidance"},
			},
			expectedDecision: "casual_chat",
		},
		{
			name: "health correction keeps recovery priority",
			signals: &SignalMatches{
				EmbeddingRules:    []string{"health_guidance"},
				KeywordRules:      []string{"verification_markers"},
				UserFeedbackRules: []string{"wrong_answer"},
				ProjectionRules:   []string{"verification_required", "feedback_correction_verified"},
			},
			expectedDecision: "feedback_wrong_answer_verified",
		},
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
