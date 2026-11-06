package controllers

import (
	"context"
	"strings"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/utils/ptr"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildCanonicalConfigAppliesOperatorDefaults(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Config: vllmv1alpha1.ConfigSpec{
				Strategy:        "priority",
				ReasoningEffort: "high",
				Tools: &vllmv1alpha1.ToolsConfig{
					Enabled:             true,
					TopK:                7,
					SimilarityThreshold: "0.4",
					ToolsDBPath:         "/config/tools.json",
					FallbackToEmpty:     true,
				},
				ComplexityRules: []vllmv1alpha1.ComplexityRulesConfig{
					{Name: "code", Description: "Code difficulty"},
				},
				ExternalModels: []vllmv1alpha1.ExternalModelConfig{
					{
						Name:      "rewriter",
						ModelRole: routerconfig.ModelRoleMemoryRewrite,
						ModelName: "rewriter-v1",
						Endpoint:  vllmv1alpha1.ExternalModelEndpoint{Address: "rewriter.default.svc", Port: 8080, Protocol: "http"},
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
	assertOperatorComplexityConfig(t, canonical.Routing.Signals.Complexity)
	assertOperatorExternalModels(t, canonical.Global.ModelCatalog.External)
}

// The CRD's external_models block must land on global.model_catalog.external
// with the router's own field names.
func assertOperatorExternalModels(t *testing.T, external []routerconfig.ExternalModelConfig) {
	t.Helper()
	if len(external) != 1 {
		t.Fatalf("external models were not carried onto the canonical config: %#v", external)
	}
	model := external[0]
	if model.Name != "rewriter" || model.ModelRole != routerconfig.ModelRoleMemoryRewrite || model.ModelName != "rewriter-v1" {
		t.Fatalf("unexpected external model: %#v", model)
	}
	if model.ModelEndpoint.Address != "rewriter.default.svc" || model.ModelEndpoint.Port != 8080 || model.ModelEndpoint.Protocol != "http" {
		t.Fatalf("external model endpoint did not survive conversion: %#v", model.ModelEndpoint)
	}
}

func TestOperatorResponseCacheConfigNormalizesLegacyAndRejectsConflict(t *testing.T) {
	legacy := &vllmv1alpha1.SemanticCacheConfig{Enabled: true}
	got, err := operatorResponseCacheConfig(vllmv1alpha1.ConfigSpec{SemanticCache: legacy})
	if err != nil || got != legacy {
		t.Fatalf("legacy response cache = %v, %v", got, err)
	}
	_, err = operatorResponseCacheConfig(vllmv1alpha1.ConfigSpec{
		ResponseCache: legacy,
		SemanticCache: legacy.DeepCopy(),
	})
	if err == nil {
		t.Fatal("operatorResponseCacheConfig() accepted canonical and legacy fields")
	}
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
						"keywords": [
							{
								"name": "critical_payment_event",
								"operator": "AND",
								"keywords": ["payment_failed", "critical"]
							}
						]
					},
					"projections": {
						"scores": [
							{
								"name": "session_risk_score",
								"method": "weighted_sum",
								"inputs": [
									{"type": "keyword", "name": "critical_payment_event", "weight": 1.0, "value_source": "binary"}
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
									{"type": "keyword", "name": "critical_payment_event"},
									{"type": "projection", "name": "risk_high"}
								]
							},
							"modelRefs": [
								{"model": "fast-model"},
								{"model": "deep-model"}
							],
							"algorithm": {
								"type": "multi_factor",
								"multi_factor": {
									"weights": {"quality": 0.6, "cost": 0.4}
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
						Name: "weighted-route",
						Rules: vllmv1alpha1.RuleCombinationConfig{
							Operator:  "AND",
							OnUnknown: "fail_request",
							Conditions: []vllmv1alpha1.RuleConditionConfig{
								{Type: "keyword", Name: "critical_payment_event"},
							},
						},
						ModelRefs: []vllmv1alpha1.ModelRefConfig{
							{Model: "fast-model"},
							{Model: "deep-model"},
						},
						Algorithm: rawCanonicalRoutingJSON(t, `{
							"type": "multi_factor",
							"multi_factor": {
								"weights": {"quality": 0.6, "cost": 0.4}
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
	if algorithm == nil || algorithm.Type != "multi_factor" || algorithm.MultiFactor == nil || algorithm.MultiFactor.Weights == nil {
		t.Fatalf("expected multi_factor algorithm to survive typed decision conversion, got %#v", algorithm)
	}
	if algorithm.MultiFactor.Weights.Quality != 0.6 || algorithm.MultiFactor.Weights.Cost != 0.4 {
		t.Fatalf("expected multi_factor weights to survive typed decision conversion, got %#v", algorithm.MultiFactor)
	}
	if canonical.Routing.Decisions[0].Rules.Conditions[0].Type != "keyword" {
		t.Fatalf("expected keyword condition to survive typed decision conversion, got %#v", canonical.Routing.Decisions[0].Rules)
	}
	if canonical.Routing.Decisions[0].Rules.OnUnknown != "fail_request" {
		t.Fatalf("expected on_unknown to survive typed decision conversion, got %#v", canonical.Routing.Decisions[0].Rules)
	}
}

func TestBuildCanonicalConfigCarriesDecisionReliabilityAndFallback(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Config: vllmv1alpha1.ConfigSpec{
				Decisions: []vllmv1alpha1.DecisionConfig{{
					Name: "long_report",
					Rules: vllmv1alpha1.RuleCombinationConfig{
						Operator:   "AND",
						Conditions: []vllmv1alpha1.RuleConditionConfig{{Type: "keyword", Name: "long_report"}},
					},
					ModelRefs: []vllmv1alpha1.ModelRefConfig{{Model: "small"}, {Model: "large"}},
					Reliability: &vllmv1alpha1.DecisionReliabilityConfig{
						TotalTimeout: "600s", PerTryTimeout: "300s", RetryCount: ptr.To[int32](1),
						RetryOn: "reset", RetriableStatusCodes: []int32{429},
					},
					Fallback: &vllmv1alpha1.DecisionFallbackConfig{
						Enabled: ptr.To(true), MaxAttempts: 2, TotalTimeout: "900s", PerAttemptTimeout: "450s",
						RetryableStatusCodes: []int32{502, 503},
					},
				}},
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
	decision := canonical.Routing.Decisions[0]
	reliability := decision.Reliability
	if reliability == nil || reliability.TotalTimeout != "600s" || reliability.PerTryTimeout != "300s" ||
		reliability.RetryCount == nil || *reliability.RetryCount != 1 || reliability.RetryOn != "reset" ||
		len(reliability.RetriableStatusCodes) != 1 || reliability.RetriableStatusCodes[0] != 429 {
		t.Fatalf("decision reliability = %#v, want the CR's block", reliability)
	}
	fallback := decision.Fallback
	if fallback == nil || fallback.Enabled == nil || !*fallback.Enabled || fallback.MaxAttempts != 2 ||
		fallback.TotalTimeout != 900*time.Second || fallback.PerAttemptTimeout != 450*time.Second ||
		len(fallback.RetryableStatusCodes) != 2 {
		t.Fatalf("decision fallback = %#v, want the CR's block", fallback)
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
}

func assertCanonicalV03RoutingConfig(t *testing.T, canonical *routerconfig.CanonicalConfig) {
	t.Helper()

	assertCanonicalConversationSignal(t, canonical.Routing.Signals.Conversation)
	assertCanonicalKeywordSignal(t, canonical.Routing.Signals.Keywords)
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

func assertCanonicalKeywordSignal(t *testing.T, signals []routerconfig.KeywordRule) {
	t.Helper()

	if len(signals) != 1 {
		t.Fatalf("expected one keyword signal, got %#v", signals)
	}
	keyword := signals[0]
	if keyword.Name != "critical_payment_event" || keyword.Operator != "AND" ||
		len(keyword.Keywords) != 2 || keyword.Keywords[0] != "payment_failed" {
		t.Fatalf("unexpected keyword signal: %#v", keyword)
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
		decision.Rules.Conditions[1].Type != "keyword" ||
		decision.Rules.Conditions[2].Type != "projection" {
		t.Fatalf("unexpected decision condition types: %#v", decision.Rules.Conditions)
	}
	if decision.Algorithm == nil || decision.Algorithm.Type != "multi_factor" || decision.Algorithm.MultiFactor == nil || decision.Algorithm.MultiFactor.Weights == nil {
		t.Fatalf("expected multi_factor algorithm, got %#v", decision.Algorithm)
	}
	if decision.Algorithm.MultiFactor.Weights.Quality != 0.6 || decision.Algorithm.MultiFactor.Weights.Cost != 0.4 {
		t.Fatalf("expected multi_factor weights, got %#v", decision.Algorithm.MultiFactor)
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

func assertOperatorComplexityConfig(t *testing.T, rules []routerconfig.ComplexityRule) {
	t.Helper()

	if len(rules) != 1 || rules[0].Name != "code" || rules[0].Description != "Code difficulty" {
		t.Fatalf("unexpected complexity rules: %#v", rules)
	}
}

// The modules and fields only the removed classifiers read are refused, not
// dropped, so a resource that writes them learns why.
func TestOperatorRefusesTheRemovedClassifierFields(t *testing.T) {
	r := &SemanticRouterReconciler{}
	block := func(raw string) *apiextensionsv1.JSON { return &apiextensionsv1.JSON{Raw: []byte(raw)} }
	for field, spec := range map[string]vllmv1alpha1.ConfigSpec{
		"config.prompt_guard":     {PromptGuard: block(`{"threshold":"0.6"}`)},
		"config.classifier":       {Classifier: block(`{"pii_model":{"threshold":"0.3"}}`)},
		"config.complexity_model": {ComplexityModel: block(`{}`)},
	} {
		canonical := &routerconfig.CanonicalConfig{Global: &routerconfig.CanonicalGlobal{}}
		if err := r.applyOperatorModelCatalog(canonical, spec); err == nil || !strings.Contains(err.Error(), field+" is removed") {
			t.Fatalf("%s: %v", field, err)
		}
	}
	if _, err := r.convertComplexityRules([]vllmv1alpha1.ComplexityRulesConfig{{Name: "code", Threshold: "0.5"}}); err == nil || !strings.Contains(err.Error(), "no longer scores example banks") {
		t.Fatalf("a complexity threshold: %v", err)
	}
}
