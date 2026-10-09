package v1alpha1

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	"sigs.k8s.io/yaml"
)

func TestPIITokenWindowAdmission(t *testing.T) {
	cases := []struct {
		name, config string
		refuse       bool
	}{
		{"omitted", `{}`, false},
		{"nullable", `{"window":null}`, false},
		{"whole input", `{"max_sequence_length":32768}`, false},
		{"explicit window", `{"max_sequence_length":32768,"window":{"size":512,"overlap":64}}`, false},
		{"zero limit", `{"max_sequence_length":0,"window":{"size":512}}`, false},
		{"local window", `{"window":{"size":128}}`, false},
		{"remote", `{"backend":{}}`, false},
		{"remote window", `{"backend":{},"window":{"size":128}}`, true},
		{"remote budget", `{"backend":{},"max_sequence_length":32768}`, true},
		{"negative limit", `{"max_sequence_length":-1}`, true},
		{"empty", `{"window":{}}`, true},
		{"large implicit", `{"window":{"size":513}}`, true},
		{"large explicit", `{"max_sequence_length":64,"window":{"size":128}}`, true},
		{"negative overlap", `{"window":{"size":128,"overlap":-1}}`, true},
		{"overlap equals size", `{"window":{"size":128,"overlap":128}}`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg PIIModelConfig
			if err := json.Unmarshal([]byte(tc.config), &cfg); err != nil {
				t.Fatal(err)
			}
			sr := &SemanticRouter{Spec: SemanticRouterSpec{Config: ConfigSpec{
				Classifier: &ClassifierConfig{PIIModel: &cfg},
			}}}
			_, createErr := sr.ValidateCreate(context.Background(), sr)
			_, updateErr := sr.ValidateUpdate(context.Background(), &SemanticRouter{}, sr)
			if (createErr != nil) != tc.refuse || (updateErr != nil) != tc.refuse {
				t.Fatalf("create=%v update=%v, refuse=%v", createErr, updateErr, tc.refuse)
			}
		})
	}
}

func TestPIITokenWindowCELAndPruning(t *testing.T) {
	structural, validator := loadCRDValidator(t)
	cases := []struct {
		config string
		refuse bool
	}{
		{`{window: null}`, false},
		{`{max_sequence_length: 32768, window: {size: 512, overlap: 64}}`, false},
		{`{window: {size: 512}}`, false},
		{`{backend: {name: remote, protocol: http_classify}}`, false},
		{`{backend: {name: remote, protocol: http_classify}, window: {size: 128}}`, true},
		{`{backend: {name: remote, protocol: http_classify}, max_sequence_length: 1024}`, true},
		{`{max_sequence_length: 64, window: {size: 128}}`, true},
		{`{window: {size: 128, overlap: 128}}`, true},
	}
	for _, tc := range cases {
		cr := "apiVersion: vllm.ai/v1alpha1\nkind: SemanticRouter\nmetadata: {name: synthetic}\nspec:\n  config:\n    classifier:\n      pii_model: " + tc.config
		if errs := celErrors(t, structural, validator, cr); (len(errs) > 0) != tc.refuse {
			t.Errorf("%s errors=%v, refuse=%v", tc.config, errs, tc.refuse)
		}
	}
	var obj map[string]interface{}
	if err := yaml.Unmarshal([]byte(`spec: {config: {classifier: {pii_model: {use_mmbert_32k: true, max_sequence_length: 32768, window: {size: 512, overlap: 64}}}}}`), &obj); err != nil {
		t.Fatal(err)
	}
	pruning.Prune(obj, structural, true)
	data, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range []string{`"max_sequence_length":32768`, `"size":512`, `"overlap":64`} {
		if !strings.Contains(string(data), token) {
			t.Fatalf("CRD pruned %s: %s", token, data)
		}
	}
	if strings.Contains(string(data), "use_mmbert_32k") {
		t.Fatalf("CRD kept the retired use_mmbert_32k selector: %s", data)
	}
}

func TestPIITokenWindowDeepCopy(t *testing.T) {
	original := &PIIModelConfig{Window: &PromptGuardWindowConfig{Size: 512, Overlap: 64}}
	copy := original.DeepCopy()
	copy.Window.Overlap = 32
	if original.Window.Overlap != 64 || copy.Window == original.Window {
		t.Fatal("PII window geometry aliases its DeepCopy")
	}
	if (&PIIModelConfig{}).DeepCopy().Window != nil {
		t.Fatal("DeepCopy injected an omitted PII window")
	}
}

func TestPIITokenWindowCRDMirrors(t *testing.T) {
	var schemas []apiextensionsv1.JSONSchemaProps
	for _, directory := range []string{"config/crd/bases", "bundle/manifests"} {
		data, err := os.ReadFile(filepath.Join("..", "..", directory, "vllm.ai_semanticrouters.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		var crd apiextensionsv1.CustomResourceDefinition
		if err := yaml.Unmarshal(data, &crd); err != nil {
			t.Fatal(err)
		}
		pii := crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["spec"].Properties["config"].Properties["classifier"].Properties["pii_model"]
		schemas = append(schemas, pii)
		for _, field := range []string{"window", "max_sequence_length"} {
			property, exists := pii.Properties[field]
			if !exists || property.Default != nil {
				t.Fatalf("%s lost %s or injected its default", directory, field)
			}
		}
		for _, field := range []string{"use_mmbert_32k", "use_modernbert"} {
			if _, exists := pii.Properties[field]; exists {
				t.Fatalf("%s still declares the retired %s selector", directory, field)
			}
		}
		window := pii.Properties["window"]
		if !window.Nullable || window.Properties["size"].Default != nil || window.Properties["overlap"].Default != nil {
			t.Fatalf("%s changed omission/null window semantics", directory)
		}
	}
	if !reflect.DeepEqual(schemas[0], schemas[1]) {
		t.Fatal("installed CRD and OLM bundle disagree on PII")
	}
}
