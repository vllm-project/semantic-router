package config

import (
	"strings"
	"testing"
)

func TestGlobalConfigContractsValidateHallucinationBackend(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel.Backend = "grpc"

	err := runConfigContractValidators(cfg, globalConfigContractValidators)
	if err == nil {
		t.Fatal("expected global config validation to reject an unknown hallucination backend")
	}
	if !strings.Contains(err.Error(), "hallucination detector backend") {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestValidateHallucinationBackend_DefaultsToTheLocalDetector(t *testing.T) {
	cfg := &HallucinationModelConfig{ModelID: "models/mom-halugate-detector"}
	if err := ValidateHallucinationBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NormalizedBackend() != HallucinationBackendLocal {
		t.Errorf("NormalizedBackend() = %q, want model_runtime", cfg.NormalizedBackend())
	}
}

func TestValidateHallucinationBackend_NormalizesCaseAndWhitespace(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "  Model_Runtime "}
	if err := ValidateHallucinationBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NormalizedBackend() != HallucinationBackendLocal {
		t.Errorf("NormalizedBackend() = %q, want model_runtime", cfg.NormalizedBackend())
	}
}

func TestValidateHallucinationBackend_RejectsUnknownBackend(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "grpc"}
	if err := ValidateHallucinationBackend(cfg); err == nil {
		t.Fatalf("expected error for unknown backend, got nil")
	}
}

func TestValidateHallucinationBackend_AcceptsTheProjectionsEndpointMarker(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "Endpoint"}
	if err := ValidateHallucinationBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NormalizedBackend() != HallucinationBackendEndpoint {
		t.Errorf("NormalizedBackend() = %q, want endpoint", cfg.NormalizedBackend())
	}
}

func TestNormalizedBackend_DefaultsWhenEmpty(t *testing.T) {
	cfg := &HallucinationModelConfig{}
	if got := cfg.NormalizedBackend(); got != HallucinationBackendLocal {
		t.Errorf("NormalizedBackend() = %q, want model_runtime", got)
	}
}

func TestCompileModelBindingsLocalDefaultLeavesHallucinationUnbound(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel = HallucinationModelConfig{ModelID: "models/mom-halugate-detector"}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := plan.Lookup(DefaultRecipeName, "hallucination_detector"); ok {
		t.Fatal("the local default is the module's canonical local binding, not a plan entry")
	}
}

func TestCompileModelBindingsResolvesAnExplicitHallucinationBinding(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ExternalModels = []ExternalModelConfig{{Name: "grounding", ModelName: "grounding-spans", ModelRole: ModelRoleClassification, ModelEndpoint: ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 9000}}}
	cfg.ModelDeployments = map[string]ModelDeployment{"grounding": {Provider: "http", ExternalModel: "grounding"}}
	cfg.ModelBindings = map[string]ModelBinding{"hallucination_detector": {Deployment: "grounding", Adapter: RemoteClassifierProtocolHTTPClassify, Contract: RemoteClassifierContractTokenSpans}}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, _ := plan.Lookup(DefaultRecipeName, "hallucination_detector")
	if spec.Deployment.ExternalModel != "grounding" || spec.Binding.Adapter != RemoteClassifierProtocolHTTPClassify {
		t.Fatalf("explicit binding = %+v", spec)
	}
}
