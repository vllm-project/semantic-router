package controllers

import (
	"context"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildCanonicalConfigAppliesOperatorSpecFamilies(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Config: vllmv1alpha1.ConfigSpec{
				Strategy:               "priority",
				DefaultReasoningEffort: "high",
				ReasoningFamilies: map[string]vllmv1alpha1.ReasoningFamily{
					"qwen3": {Type: "reasoning_effort", Parameter: "think"},
				},
				Tools: &vllmv1alpha1.ToolsConfig{
					Enabled:             true,
					TopK:                7,
					SimilarityThreshold: "0.4",
					ToolsDBPath:         "/config/tools.json",
					FallbackToEmpty:     true,
				},
				Classifier: &vllmv1alpha1.ClassifierConfig{
					CategoryModel: &vllmv1alpha1.CategoryModelConfig{
						ModelID:             "domain-classifier",
						Threshold:           "0.6",
						UseCPU:              true,
						CategoryMappingPath: "/config/domain.yaml",
					},
					PIIModel: &vllmv1alpha1.PIIModelConfig{
						ModelID:        "pii-classifier",
						Threshold:      "0.7",
						PIIMappingPath: "/config/pii.yaml",
					},
				},
				ComplexityRules: []vllmv1alpha1.ComplexityRulesConfig{
					{
						Name:      "code",
						Threshold: "0.55",
						Hard:      vllmv1alpha1.ComplexityCandidates{Candidates: []string{"debug a race"}},
						Easy:      vllmv1alpha1.ComplexityCandidates{Candidates: []string{"say hello"}},
						Composer: &vllmv1alpha1.RuleComposition{
							Operator: "AND",
							Conditions: []vllmv1alpha1.CompositionCondition{
								{Type: "domain", Name: "engineering"},
							},
						},
					},
				},
			},
		},
	}

	canonical, err := r.buildCanonicalConfig(context.Background(), sr)
	if err != nil {
		t.Fatalf("buildCanonicalConfig failed: %v", err)
	}

	assertOperatorRouterProviderConfig(t, canonical)
	assertOperatorToolsConfig(t, canonical.Global.Integrations.Tools)
	assertOperatorClassifierConfig(t, canonical.Global.ModelCatalog.Modules.Classifier)
	assertOperatorComplexityConfig(t, canonical.Routing.Signals.Complexity)
}

func TestBuildCanonicalConfigAppliesCanonicalRoutingOverride(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Config: vllmv1alpha1.ConfigSpec{
				Routing: rawCanonicalRoutingJSON(t, `{
					"signals": {
						"conversation": [
							{
								"name": "active_tool_use",
								"feature": {
									"type": "count",
									"source": {"type": "assistant_tool_cycle"}
								},
								"predicate": {"gte": 1}
							}
						],
						"events": [
							{
								"name": "critical_payment_event",
								"event_types": ["payment_failed"],
								"severities": ["critical"],
								"temporal": true
							}
						]
					},
					"projections": {
						"scores": [
							{
								"name": "session_risk_score",
								"method": "weighted_sum",
								"inputs": [
									{"type": "event", "name": "critical_payment_event", "weight": 1.0, "value_source": "binary"}
								]
							}
						],
						"mappings": [
							{
								"name": "session_risk_band",
								"source": "session_risk_score",
								"method": "threshold_bands",
								"outputs": [
									{"name": "risk_high", "gte": 0.8}
								]
							}
						]
					},
					"decisions": [
						{
							"name": "agentic_workflow_route",
							"rules": {
								"operator": "AND",
								"conditions": [
									{"type": "conversation", "name": "active_tool_use"},
									{"type": "event", "name": "critical_payment_event"},
									{"type": "projection", "name": "risk_high"}
								]
							},
							"modelRefs": [
								{"model": "fast-model"},
								{"model": "deep-model"}
							],
							"algorithm": {
								"type": "hybrid",
								"hybrid": {
									"elo_weight": 0.4,
									"router_dc_weight": 0.4,
									"automix_weight": 0.2,
									"normalize_scores": true
								}
							}
						}
					]
				}`),
			},
		},
	}

	canonical, err := r.buildCanonicalConfig(context.Background(), sr)
	if err != nil {
		t.Fatalf("buildCanonicalConfig failed: %v", err)
	}

	assertCanonicalV03RoutingConfig(t, canonical)
}

func TestBuildCanonicalConfigPreservesDecisionAlgorithm(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Config: vllmv1alpha1.ConfigSpec{
				Decisions: []vllmv1alpha1.DecisionConfig{
					{
						Name: "hybrid-route",
						Rules: vllmv1alpha1.RuleCombinationConfig{
							Operator: "AND",
							Conditions: []vllmv1alpha1.RuleConditionConfig{
								{Type: "event", Name: "critical_payment_event"},
							},
						},
						ModelRefs: []vllmv1alpha1.ModelRefConfig{
							{Model: "fast-model"},
							{Model: "deep-model"},
						},
						Algorithm: rawCanonicalRoutingJSON(t, `{
							"type": "hybrid",
							"hybrid": {
								"elo_weight": 0.4,
								"router_dc_weight": 0.4,
								"automix_weight": 0.2,
								"normalize_scores": true
							}
						}`),
					},
				},
			},
		},
	}

	canonical, err := r.buildCanonicalConfig(context.Background(), sr)
	if err != nil {
		t.Fatalf("buildCanonicalConfig failed: %v", err)
	}

	if len(canonical.Routing.Decisions) != 1 {
		t.Fatalf("expected one decision, got %#v", canonical.Routing.Decisions)
	}
	algorithm := canonical.Routing.Decisions[0].Algorithm
	if algorithm == nil || algorithm.Type != "hybrid" || algorithm.Hybrid == nil {
		t.Fatalf("expected hybrid algorithm to survive typed decision conversion, got %#v", algorithm)
	}
	if algorithm.Hybrid.EloWeight != 0.4 || !algorithm.Hybrid.NormalizeScores {
		t.Fatalf("expected hybrid fields to survive typed decision conversion, got %#v", algorithm.Hybrid)
	}
	if canonical.Routing.Decisions[0].Rules.Conditions[0].Type != "event" {
		t.Fatalf("expected event condition to survive typed decision conversion, got %#v", canonical.Routing.Decisions[0].Rules)
	}
}

func rawCanonicalRoutingJSON(t *testing.T, raw string) *apiextensionsv1.JSON {
	t.Helper()

	return &apiextensionsv1.JSON{Raw: []byte(raw)}
}

func assertOperatorRouterProviderConfig(t *testing.T, canonical *routerconfig.CanonicalConfig) {
	t.Helper()

	if canonical.Global.Router.Strategy != "priority" {
		t.Fatalf("expected priority strategy, got %q", canonical.Global.Router.Strategy)
	}
	if canonical.Providers.Defaults.DefaultReasoningEffort != "high" {
		t.Fatalf("unexpected default reasoning effort: %q", canonical.Providers.Defaults.DefaultReasoningEffort)
	}
	family := canonical.Providers.Defaults.ReasoningFamilies["qwen3"]
	if family.Type != "reasoning_effort" || family.Parameter != "think" {
		t.Fatalf("unexpected reasoning family: %#v", family)
	}
}

func assertCanonicalV03RoutingConfig(t *testing.T, canonical *routerconfig.CanonicalConfig) {
	t.Helper()

	assertCanonicalConversationSignal(t, canonical.Routing.Signals.Conversation)
	assertCanonicalEventSignal(t, canonical.Routing.Signals.EventRules)
	assertCanonicalProjectionConfig(t, canonical.Routing.Projections)
	assertCanonicalAgenticWorkflowDecision(t, canonical.Routing.Decisions)
}

func assertCanonicalConversationSignal(t *testing.T, signals []routerconfig.ConversationRule) {
	t.Helper()

	if len(signals) != 1 {
		t.Fatalf("expected one conversation signal, got %#v", signals)
	}
	conversation := signals[0]
	if conversation.Name != "active_tool_use" || conversation.Feature.Type != "count" {
		t.Fatalf("unexpected conversation signal: %#v", conversation)
	}
	if conversation.Predicate == nil || conversation.Predicate.GTE == nil || *conversation.Predicate.GTE != 1 {
		t.Fatalf("unexpected conversation predicate: %#v", conversation.Predicate)
	}
}

func assertCanonicalEventSignal(t *testing.T, signals []routerconfig.EventRule) {
	t.Helper()

	if len(signals) != 1 {
		t.Fatalf("expected one event signal, got %#v", signals)
	}
	event := signals[0]
	if event.Name != "critical_payment_event" || len(event.EventTypes) != 1 || event.EventTypes[0] != "payment_failed" {
		t.Fatalf("unexpected event signal: %#v", event)
	}
}

func assertCanonicalProjectionConfig(t *testing.T, projections routerconfig.CanonicalProjections) {
	t.Helper()

	if len(projections.Scores) != 1 || len(projections.Mappings) != 1 {
		t.Fatalf("expected projection score and mapping, got %#v", projections)
	}
	if projections.Mappings[0].Outputs[0].Name != "risk_high" {
		t.Fatalf("unexpected projection mapping: %#v", projections.Mappings[0])
	}
}

func assertCanonicalAgenticWorkflowDecision(t *testing.T, decisions []routerconfig.Decision) {
	t.Helper()

	if len(decisions) != 1 {
		t.Fatalf("expected one decision, got %#v", decisions)
	}
	decision := decisions[0]
	if decision.Rules.Operator != "AND" || len(decision.Rules.Conditions) != 3 {
		t.Fatalf("unexpected decision rules: %#v", decision.Rules)
	}
	if decision.Rules.Conditions[0].Type != "conversation" ||
		decision.Rules.Conditions[1].Type != "event" ||
		decision.Rules.Conditions[2].Type != "projection" {
		t.Fatalf("unexpected decision condition types: %#v", decision.Rules.Conditions)
	}
	if decision.Algorithm == nil || decision.Algorithm.Type != "hybrid" || decision.Algorithm.Hybrid == nil {
		t.Fatalf("expected hybrid algorithm, got %#v", decision.Algorithm)
	}
	if decision.Algorithm.Hybrid.EloWeight != 0.4 || !decision.Algorithm.Hybrid.NormalizeScores {
		t.Fatalf("expected hybrid fields, got %#v", decision.Algorithm.Hybrid)
	}
}

func assertOperatorToolsConfig(t *testing.T, tools routerconfig.ToolsConfig) {
	t.Helper()

	if tools.TopK != 7 || tools.ToolsDBPath != "/config/tools.json" || !tools.FallbackToEmpty {
		t.Fatalf("unexpected tools config: %#v", tools)
	}
	if tools.SimilarityThreshold == nil || *tools.SimilarityThreshold < 0.399 || *tools.SimilarityThreshold > 0.401 {
		t.Fatalf("unexpected tools threshold: %#v", tools.SimilarityThreshold)
	}
}

func assertOperatorClassifierConfig(t *testing.T, classifier routerconfig.CanonicalClassifierModule) {
	t.Helper()

	if classifier.Domain.ModelID != "domain-classifier" || classifier.Domain.CategoryMappingPath != "/config/domain.yaml" {
		t.Fatalf("unexpected domain classifier: %#v", classifier.Domain)
	}
	if classifier.PII.ModelID != "pii-classifier" || classifier.PII.PIIMappingPath != "/config/pii.yaml" {
		t.Fatalf("unexpected PII classifier: %#v", classifier.PII)
	}
}

func assertOperatorComplexityConfig(t *testing.T, rules []routerconfig.ComplexityRule) {
	t.Helper()

	if len(rules) != 1 {
		t.Fatalf("expected one complexity rule, got %#v", rules)
	}
	rule := rules[0]
	if rule.Name != "code" || rule.Threshold < 0.549 || rule.Threshold > 0.551 {
		t.Fatalf("unexpected complexity rule: %#v", rule)
	}
	if rule.Composer == nil || rule.Composer.Operator != "AND" || len(rule.Composer.Conditions) != 1 {
		t.Fatalf("unexpected complexity composer: %#v", rule.Composer)
	}
	condition := rule.Composer.Conditions[0]
	if condition.Type != "domain" || condition.Name != "engineering" {
		t.Fatalf("unexpected composer condition: %#v", condition)
	}
}
