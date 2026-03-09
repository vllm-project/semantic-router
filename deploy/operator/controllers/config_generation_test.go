package controllers

import (
	"context"
	"testing"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGenerateConfigYAML_DoesNotSynthesizeProvidersFromVLLMEndpoints(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			VLLMEndpoints: []vllmv1alpha1.VLLMEndpointSpec{
				{
					Name:            "qwen-primary",
					Model:           "qwen3-4b",
					ReasoningFamily: "qwen3",
					Backend: vllmv1alpha1.VLLMBackend{
						Type: "service",
						Service: &vllmv1alpha1.ServiceBackend{
							Name: "qwen-svc",
							Port: 8000,
						},
					},
				},
			},
		},
	}

	r := &SemanticRouterReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).Build(),
		Scheme: s,
	}

	configYAML, err := r.generateConfigYAML(context.Background(), sr)
	if err != nil {
		t.Fatalf("generateConfigYAML() failed: %v", err)
	}

	var config map[string]interface{}
	if err := yaml.Unmarshal([]byte(configYAML), &config); err != nil {
		t.Fatalf("yaml.Unmarshal() failed: %v", err)
	}

	if _, ok := config["providers"]; ok {
		t.Fatalf("did not expect synthesized providers block in operator-generated runtime config: %#v", config["providers"])
	}

	if _, ok := config["vllm_endpoints"]; !ok {
		t.Fatalf("expected vllm_endpoints in operator-generated runtime config: %#v", config)
	}

	modelConfig, ok := config["model_config"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected model_config in operator-generated runtime config: %#v", config)
	}

	entry, ok := modelConfig["qwen3-4b"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected qwen3-4b model entry in model_config: %#v", modelConfig)
	}

	preferredEndpoints, ok := entry["preferred_endpoints"].([]interface{})
	if !ok || len(preferredEndpoints) != 1 || preferredEndpoints[0] != "qwen-primary" {
		t.Fatalf("expected preferred_endpoints to stay on runtime model_config entry, got %#v", entry["preferred_endpoints"])
	}
}

func TestGenerateConfigYAML_CompilesAuthoringConfig(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			AuthoringConfig: &apiextensionsv1.JSON{
				Raw: []byte(`{
					"version":"v0.1",
					"listeners":[{"name":"grpc-50051","address":"0.0.0.0","port":50051,"timeout":"300s"}],
					"signals":{"keywords":[{"name":"billing_keywords","operator":"contains","keywords":["invoice","refund"],"case_sensitive":false}]},
					"providers":{
						"models":[{"name":"qwen3-4b","reasoning_family":"qwen3","endpoints":[{"name":"primary","weight":100,"endpoint":"vllm-qwen.default.svc.cluster.local:8000"}]}],
						"default_model":"qwen3-4b",
						"reasoning_families":{"qwen3":{"type":"effort","parameter":"reasoning_effort"}},
						"default_reasoning_effort":"medium"
					},
					"decisions":[{"name":"billing-route","priority":90,"rules":{"operator":"AND","conditions":[{"type":"keyword","name":"billing_keywords"}]},"modelRefs":[{"model":"qwen3-4b","use_reasoning":false}]}]
				}`),
			},
			Config: vllmv1alpha1.ConfigSpec{
				API: &vllmv1alpha1.APIConfig{
					BatchClassification: &vllmv1alpha1.BatchClassificationConfig{
						MaxBatchSize: 32,
					},
				},
			},
		},
	}

	r := &SemanticRouterReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).Build(),
		Scheme: s,
	}

	configYAML, err := r.generateConfigYAML(context.Background(), sr)
	if err != nil {
		t.Fatalf("generateConfigYAML() failed: %v", err)
	}

	var config map[string]interface{}
	if err := yaml.Unmarshal([]byte(configYAML), &config); err != nil {
		t.Fatalf("yaml.Unmarshal() failed: %v", err)
	}

	if _, ok := config["providers"]; ok {
		t.Fatalf("did not expect providers block in compiled operator runtime config: %#v", config["providers"])
	}
	if got, ok := config["default_model"].(string); !ok || got != "qwen3-4b" {
		t.Fatalf("expected default_model from authoringConfig, got %#v", config["default_model"])
	}
	if _, ok := config["keyword_rules"]; !ok {
		t.Fatalf("expected keyword_rules from authoringConfig, got %#v", config)
	}
	if _, ok := config["vllm_endpoints"]; !ok {
		t.Fatalf("expected vllm_endpoints from authoringConfig, got %#v", config)
	}
	if _, ok := config["api"]; !ok {
		t.Fatalf("expected non-overlapping config.api to merge on top of authoringConfig, got %#v", config)
	}
}
