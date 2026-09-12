package v1alpha1

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	schemacel "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/yaml"
)

// The API server's own CEL evaluator, run against the generated CRD schema,
// so a rule that is present but wrong fails here instead of at kubectl apply.
// Limits mirror the API server defaults (k8s.io/apiserver/pkg/apis/cel:
// PerCallLimit and RuntimeCELCostBudget) without importing that package.
func loadCRDValidator(t *testing.T) (*schema.Structural, *schemacel.Validator) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"))
	if err != nil {
		t.Fatalf("read CRD: %v", err)
	}
	var crd apiextensionsv1.CustomResourceDefinition
	if err = yaml.Unmarshal(data, &crd); err != nil {
		t.Fatalf("parse CRD: %v", err)
	}
	if len(crd.Spec.Versions) == 0 || crd.Spec.Versions[0].Schema == nil {
		t.Fatal("CRD carries no schema")
	}
	var internal apiextensions.JSONSchemaProps
	if err = apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(crd.Spec.Versions[0].Schema.OpenAPIV3Schema, &internal, nil); err != nil {
		t.Fatalf("convert schema: %v", err)
	}
	structural, err := schema.NewStructural(&internal)
	if err != nil {
		t.Fatalf("structural schema: %v", err)
	}
	validator := schemacel.NewValidator(structural, true, 1_000_000)
	if validator == nil {
		t.Fatal("CRD schema has no x-kubernetes-validations; the CEL rules are gone")
	}
	return structural, validator
}

func celErrors(t *testing.T, structural *schema.Structural, validator *schemacel.Validator, cr string) []string {
	t.Helper()
	var obj map[string]interface{}
	if err := yaml.Unmarshal([]byte(cr), &obj); err != nil {
		t.Fatalf("parse CR: %v", err)
	}
	errs, _ := validator.Validate(context.Background(), field.NewPath(""), structural, obj, nil, 10_000_000)
	out := make([]string, 0, len(errs))
	for _, e := range errs {
		out = append(out, e.Error())
	}
	return out
}

// A backend contract must match the signal that reads it. The shared backend
// block lists every contract any consumer accepts, so each consumer narrows
// it; without that, a CR the API server admits produces a router config the
// router rejects at load.
func TestCRDRefusesContractsTheConsumerCannotRead(t *testing.T) {
	structural, validator := loadCRDValidator(t)
	const complexity = `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  config:
    complexity_model:
      backend: {protocol: http_classify, model: scorer, contract: %s}
`
	const pii = `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  config:
    classifier:
      pii_model:
        backend: {protocol: http_classify, model: pii-spans%s}
`
	cases := []struct {
		name    string
		cr      string
		wantErr string // empty means admitted
	}{
		{"complexity reads score.v1", strings.Replace(complexity, "%s", "score.v1", 1), ""},
		{"complexity reads label_distribution.v1", strings.Replace(complexity, "%s", "label_distribution.v1", 1), ""},
		{"complexity refuses token_spans.v1", strings.Replace(complexity, "%s", "token_spans.v1", 1), "token_spans.v1 is the PII contract"},
		{"pii reads token_spans.v1", strings.Replace(pii, "%s", ", contract: token_spans.v1", 1), ""},
		{"pii may omit the contract", strings.Replace(pii, "%s", "", 1), ""},
		{"pii refuses score.v1", strings.Replace(pii, "%s", ", contract: score.v1", 1), "PII reads token_spans.v1 only"},
		{"pii refuses label_distribution.v1", strings.Replace(pii, "%s", ", contract: label_distribution.v1", 1), "PII reads token_spans.v1 only"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := celErrors(t, structural, validator, tc.cr)
			if tc.wantErr == "" {
				if len(errs) != 0 {
					t.Fatalf("admitted CR was refused: %v", errs)
				}
				return
			}
			if len(errs) == 0 {
				t.Fatalf("CR was admitted; the router would reject the generated config")
			}
			if !strings.Contains(strings.Join(errs, "\n"), tc.wantErr) {
				t.Fatalf("refused for another reason: %v", errs)
			}
		})
	}
}

// The rule Theo added on #3542 is exercised the same way: complexity must state
// a contract because it reads two shapes.
func TestCRDRefusesComplexityBackendWithoutContract(t *testing.T) {
	structural, validator := loadCRDValidator(t)
	errs := celErrors(t, structural, validator, `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  config:
    complexity_model:
      backend: {protocol: http_classify, model: scorer}
`)
	if !strings.Contains(strings.Join(errs, "\n"), "backend.contract must be stated") {
		t.Fatalf("complexity backend without contract was admitted: %v", errs)
	}
}
