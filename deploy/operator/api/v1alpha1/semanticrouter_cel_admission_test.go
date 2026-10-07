package v1alpha1

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	schemacel "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	kubejson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/yaml"
)

// The API server's own CEL evaluator, run against the generated CRD schema,
// so a rule that is present but wrong fails here instead of at kubectl apply.
// Limits mirror the API server defaults (k8s.io/apiserver/pkg/apis/cel:
// PerCallLimit and RuntimeCELCostBudget) without importing that package.
func loadCRDValidator(t *testing.T) (*schema.Structural, *schemacel.Validator) {
	t.Helper()
	structural := loadCRDStructural(t, filepath.Join("..", "..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"))
	validator := schemacel.NewValidator(structural, true, 1_000_000)
	if validator == nil {
		t.Fatal("CRD schema has no x-kubernetes-validations; the CEL rules are gone")
	}
	return structural, validator
}

// loadCRDStructural reads a generated CRD copy into the structural schema the
// API server validates and prunes custom resources with.
func loadCRDStructural(t *testing.T, path string) *schema.Structural {
	t.Helper()
	data, err := os.ReadFile(path)
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
	return structural
}

func celErrors(t *testing.T, structural *schema.Structural, validator *schemacel.Validator, cr string) []string {
	t.Helper()
	var obj map[string]interface{}
	data, err := yaml.YAMLToJSON([]byte(cr))
	if err != nil {
		t.Fatalf("parse CR: %v", err)
	}
	// API-server unstructured decoding preserves integer values. Standard
	// encoding/json would make them float64, which CEL correctly rejects.
	if err := kubejson.Unmarshal(data, &obj); err != nil {
		t.Fatalf("decode CR: %v", err)
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

// The Operator derives the gateway mode from spec.gateway, so args that set it
// would contradict the ports and probes it renders.
func TestCRDRefusesGatewayModeFlagsInArgs(t *testing.T) {
	structural, validator := loadCRDValidator(t)
	const cr = `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  args: [%s]
`
	for _, test := range []struct {
		args    string
		refused bool
	}{
		{`"--secure=false"`, false},
		{`"--secure=false", "-gateway=extproc"`, true},
		{`"--gateway", "standalone"`, true},
		{`"-listener-address=127.0.0.1"`, true},
		{`"-gateway-mode-is-not-a-flag"`, false},
	} {
		errs := celErrors(t, structural, validator, strings.Replace(cr, "%s", test.args, 1))
		refused := len(errs) > 0 && strings.Contains(strings.Join(errs, "; "), "spec.args must not set -gateway")
		if refused != test.refused {
			t.Errorf("args [%s]: refused = %v, errors = %v", test.args, refused, errs)
		}
	}
}

// The bounds on spec.args are what let the API server estimate the rule's
// cost; the largest list they admit must also fit the per-call CEL budget, or
// a valid CR would be refused when it is written.
func TestCRDAdmitsTheLargestArgsItsBoundsAllow(t *testing.T) {
	structural, validator := loadCRDValidator(t)
	args := structural.Properties["spec"].Properties["args"]
	if args.ValueValidation == nil || args.ValueValidation.MaxItems == nil ||
		args.Items == nil || args.Items.ValueValidation == nil || args.Items.ValueValidation.MaxLength == nil {
		t.Fatal("spec.args must bound its length and each item's, or the API server refuses the CRD")
	}
	item := strconv.Quote("--secure=" + strings.Repeat("x", int(*args.Items.ValueValidation.MaxLength)-len("--secure=")))
	items := make([]string, *args.ValueValidation.MaxItems)
	for i := range items {
		items[i] = item
	}
	errs := celErrors(t, structural, validator, `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  args: [`+strings.Join(items, ", ")+`]
`)
	if len(errs) != 0 {
		t.Fatalf("%d args of %d characters were refused: %v", len(items), *args.Items.ValueValidation.MaxLength, errs)
	}
}
