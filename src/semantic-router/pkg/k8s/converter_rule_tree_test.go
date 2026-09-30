package k8s

import (
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestConvertedRouteRejectsMultiChildNot pins the Kubernetes load path to the
// same rule-tree contract as file config. The IntelligentRoute CRD pins the
// operator enum but allows up to 50 conditions under NOT, so a multi-child NOT
// used to reconcile into a decision that could never match; it must now fail
// the same normalization the reconciler runs through ParseYAMLBytes and
// ValidateKubernetesConfigContracts, naming the decision and node path.
func TestConvertedRouteRejectsMultiChildNot(t *testing.T) {
	pool := testPoolWithModels(v1alpha1.ModelConfig{Name: "test-model"})
	route := testRouteWithKeywords(
		[]v1alpha1.KeywordSignal{
			{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}},
			{Name: "billing", Operator: "OR", Keywords: []string{"invoice"}},
		},
		v1alpha1.Decision{
			Name:     "not-urgent-billing",
			Priority: 100,
			Signals: v1alpha1.SignalCombination{
				Operator: "NOT",
				Conditions: []v1alpha1.SignalCondition{
					{Type: "keyword", Name: "urgent"},
					{Type: "keyword", Name: "billing"},
				},
			},
			ModelRefs: []v1alpha1.ModelRef{{Model: "test-model"}},
		},
	)

	canonical, err := NewCRDConverter().Convert(pool, route, &config.CanonicalConfig{})
	if err != nil {
		t.Fatalf("Convert() error = %v", err)
	}
	canonicalBytes, err := yamlv3.Marshal(canonical)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}

	_, err = config.ParseYAMLBytes(canonicalBytes)
	if err == nil {
		t.Fatal("expected the converted route to fail rule-tree validation")
	}
	want := `decision "not-urgent-billing": rules: NOT requires exactly one child condition, got 2`
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("expected error containing %q, got: %v", want, err)
	}
}

// TestConvertedRouteAcceptsUnaryNot is the positive half: a CRD-authored unary
// NOT reconciles into a normalized tree the engine evaluates as NOT.
func TestConvertedRouteAcceptsUnaryNot(t *testing.T) {
	pool := testPoolWithModels(v1alpha1.ModelConfig{Name: "test-model"})
	route := testRouteWithKeywords(
		[]v1alpha1.KeywordSignal{{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}}},
		v1alpha1.Decision{
			Name:     "not-urgent",
			Priority: 100,
			Signals: v1alpha1.SignalCombination{
				Operator:   "NOT",
				Conditions: []v1alpha1.SignalCondition{{Type: "keyword", Name: "urgent"}},
			},
			ModelRefs: []v1alpha1.ModelRef{{Model: "test-model"}},
		},
	)

	canonical, err := NewCRDConverter().Convert(pool, route, &config.CanonicalConfig{})
	if err != nil {
		t.Fatalf("Convert() error = %v", err)
	}
	canonicalBytes, err := yamlv3.Marshal(canonical)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}

	cfg, err := config.ParseYAMLBytes(canonicalBytes)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if err := config.ValidateKubernetesConfigContracts(cfg); err != nil {
		t.Fatalf("ValidateKubernetesConfigContracts() error = %v", err)
	}
	if len(cfg.Decisions) != 1 || cfg.Decisions[0].Rules.Operator != config.RuleOperatorNot {
		t.Fatalf("decisions = %+v, want one decision with a normalized NOT root", cfg.Decisions)
	}
}
