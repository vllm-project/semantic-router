package v1alpha1

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/yaml"
)

func TestModelRuntimeCRDAndSample(t *testing.T) {
	for _, path := range []string{"config/crd/bases/vllm.ai_semanticrouters.yaml", "bundle/manifests/vllm.ai_semanticrouters.yaml"} {
		data, err := os.ReadFile(filepath.Join("../..", path))
		if err != nil {
			t.Fatal(err)
		}
		var crd apiextensionsv1.CustomResourceDefinition
		if err := yaml.Unmarshal(data, &crd); err != nil {
			t.Fatal(err)
		}
		config := crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["spec"].Properties["config"]
		for _, name := range []string{"model_deployments", "model_bindings", "model_admission", "routing"} {
			field := config.Properties[name]
			if field.Type != "object" || field.XPreserveUnknownFields == nil || !*field.XPreserveUnknownFields {
				t.Fatalf("%s prunes router-owned %s", path, name)
			}
		}
		// A removed block is kept whole, so reconciliation refuses it by name
		// instead of the API server pruning it.
		for _, name := range []string{"prompt_guard", "classifier", "complexity_model"} {
			field := config.Properties[name]
			if field.Type != "object" || field.XPreserveUnknownFields == nil || !*field.XPreserveUnknownFields {
				t.Fatalf("%s prunes the removed %s", path, name)
			}
		}
	}
	data, err := os.ReadFile("../../config/samples/vllm.ai_v1alpha1_semanticrouter_model_runtime.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var sample SemanticRouter
	if err := yaml.Unmarshal(data, &sample); err != nil {
		t.Fatal(err)
	}
	if sample.Spec.Config.ModelDeployments == nil || sample.Spec.Config.ModelAdmission == nil || sample.Spec.Config.Routing == nil {
		t.Fatal("sample lost model runtime declarations")
	}
	if _, err := sample.ValidateCreate(context.Background(), &sample); err != nil {
		t.Fatal(err)
	}
}
