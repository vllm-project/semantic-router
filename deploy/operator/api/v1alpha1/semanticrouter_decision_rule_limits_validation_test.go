package v1alpha1

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

func TestDecisionRuleLimitWebhook(t *testing.T) {
	for _, canonical := range []bool{false, true} {
		resource := &SemanticRouter{Spec: SemanticRouterSpec{Config: ConfigSpec{
			DecisionRuleLimits: &DecisionRuleLimitsConfig{MaxNodes: ptr.To(1)},
		}}}
		if canonical {
			resource.Spec.Config.Routing = &apiextensionsv1.JSON{Raw: []byte(`{"decisions":[{"name":"bounded","rules":{"operator":"AND","conditions":[{"type":"keyword","name":"urgent"}]}}]}`)}
		} else {
			resource.Spec.Config.Decisions = []DecisionConfig{{Name: "bounded", Rules: RuleCombinationConfig{
				Operator: "AND", Conditions: []RuleConditionConfig{{Type: "keyword", Name: "urgent"}},
			}}}
		}
		for _, validate := range []func() error{
			func() error { _, err := resource.ValidateCreate(context.Background(), resource); return err },
			func() error { _, err := resource.ValidateUpdate(context.Background(), resource, resource); return err },
		} {
			err := validate()
			if err == nil || !strings.Contains(err.Error(), "node count 2 exceeds max_nodes=1") {
				t.Fatalf("expected admission budget error, got %v", err)
			}
		}
		resource.Spec.Config.DecisionRuleLimits.MaxNodes = ptr.To(2)
		if err := resource.ValidateDecisionRuleLimits(); err != nil {
			t.Fatal(err)
		}
		resource.Spec.Config.DecisionRuleLimits.MaxDepth = ptr.To(0)
		if err := resource.ValidateDecisionRuleLimits(); err == nil || !strings.Contains(err.Error(), "positive integer") {
			t.Fatalf("expected nonpositive setting rejection, got %v", err)
		}
	}
}

func TestDecisionRuleLimitGeneratedSchemas(t *testing.T) {
	for _, relative := range []string{"config/crd/bases", "bundle/manifests"} {
		t.Run(relative, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("../..", relative, "vllm.ai_semanticrouters.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			var crd apiextensionsv1.CustomResourceDefinition
			if err = yaml.Unmarshal(data, &crd); err != nil {
				t.Fatal(err)
			}
			limits := crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["spec"].Properties["config"].Properties["decision_rule_limits"]
			for name, defaultValue := range map[string]string{"max_depth": "16", "max_nodes": "256"} {
				property := limits.Properties[name]
				if property.Type != "integer" || property.Minimum == nil || *property.Minimum != 1 || property.Default == nil || string(property.Default.Raw) != defaultValue {
					t.Fatalf("%s lost positive integer/default contract: %+v", name, property)
				}
			}
			var internal apiextensions.JSONSchemaProps
			if err = apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(&limits, &internal, nil); err != nil {
				t.Fatal(err)
			}
			validator, _, err := validation.NewSchemaValidator(&internal)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"max_depth", "max_nodes"} {
				for _, value := range []interface{}{0, -1, 1.5, true, "32"} {
					if errs := validation.ValidateCustomResource(field.NewPath("decision_rule_limits"), map[string]interface{}{name: value}, validator); len(errs) == 0 {
						t.Errorf("schema admitted invalid %s=%v", name, value)
					}
				}
				if errs := validation.ValidateCustomResource(field.NewPath("decision_rule_limits"), map[string]interface{}{name: 32}, validator); len(errs) > 0 {
					t.Errorf("schema rejected positive override: %v", errs)
				}
			}
		})
	}
}
