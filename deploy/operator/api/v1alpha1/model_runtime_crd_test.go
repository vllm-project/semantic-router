package v1alpha1

import (
	"context"
	"os"
	"path/filepath"
	"slices"
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
		for _, name := range []string{"model_deployments", "model_admission", "routing"} {
			field := config.Properties[name]
			if field.Type != "object" || field.XPreserveUnknownFields == nil || !*field.XPreserveUnknownFields {
				t.Fatalf("%s prunes router-owned %s", path, name)
			}
		}
		backend := config.Properties["prompt_guard"].Properties["backend"]
		if !slices.ContainsFunc(backend.Properties["protocol"].Enum, func(value apiextensionsv1.JSON) bool { return string(value.Raw) == `"http_chat"` }) {
			t.Fatalf("%s rejects chat guard protocol", path)
		}
		if !slices.ContainsFunc(backend.Properties["contract"].Enum, func(value apiextensionsv1.JSON) bool { return string(value.Raw) == `"label_decision.v1"` }) {
			t.Fatalf("%s rejects unscored guard verdicts", path)
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
