package controllers

import (
	"context"
	"testing"

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
