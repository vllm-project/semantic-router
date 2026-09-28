package v1alpha1

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	kubejson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/yaml"
)

// The API server's OpenAPI validator, run against the generated CRD schema,
// so a negative streamed body limit is refused at admission.
func TestCRDValidatesStreamedBodyLimits(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"))
	if err != nil {
		t.Fatalf("read CRD: %v", err)
	}
	var crd apiextensionsv1.CustomResourceDefinition
	if err = yaml.Unmarshal(data, &crd); err != nil {
		t.Fatalf("parse CRD: %v", err)
	}
	var internal apiextensions.JSONSchemaProps
	if err = apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(crd.Spec.Versions[0].Schema.OpenAPIV3Schema, &internal, nil); err != nil {
		t.Fatalf("convert schema: %v", err)
	}
	validator, _, err := validation.NewSchemaValidator(&internal)
	if err != nil {
		t.Fatalf("schema validator: %v", err)
	}

	cases := []struct {
		name    string
		body    string
		wantErr string // empty means admitted
	}{
		{"enabled with limits", "{enabled: true, max_bytes: 10485760, timeout_sec: 30}", ""},
		{"zero disables the limits", "{enabled: true, max_bytes: 0, timeout_sec: 0}", ""},
		{"negative max_bytes", "{enabled: true, max_bytes: -1}", "spec.config.streamed_body.max_bytes"},
		{"negative timeout_sec", "{enabled: true, timeout_sec: -1}", "spec.config.streamed_body.timeout_sec"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cr := "apiVersion: vllm.ai/v1alpha1\nkind: SemanticRouter\nmetadata: {name: r}\n" +
				"spec:\n  config:\n    streamed_body: " + tc.body + "\n"
			raw, err := yaml.YAMLToJSON([]byte(cr))
			if err != nil {
				t.Fatalf("parse CR: %v", err)
			}
			var obj map[string]interface{}
			if err := kubejson.Unmarshal(raw, &obj); err != nil {
				t.Fatalf("decode CR: %v", err)
			}
			errs := validation.ValidateCustomResource(field.NewPath(""), obj, validator)
			if tc.wantErr == "" {
				if len(errs) != 0 {
					t.Fatalf("admitted CR was refused: %v", errs)
				}
				return
			}
			if !strings.Contains(errs.ToAggregate().Error(), tc.wantErr) {
				t.Fatalf("want refusal naming %q, got %v", tc.wantErr, errs)
			}
		})
	}
}
