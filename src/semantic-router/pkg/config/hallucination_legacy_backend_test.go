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

func TestValidateHallucinationBackend_DefaultsToCandle(t *testing.T) {
	cfg := &HallucinationModelConfig{ModelID: "models/mom-halugate-detector"}
	if err := ValidateHallucinationBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NormalizedBackend() != HallucinationBackendCandle {
		t.Errorf("NormalizedBackend() = %q, want candle", cfg.NormalizedBackend())
	}
}

func TestValidateHallucinationBackend_NormalizesCaseAndWhitespace(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "  Candle "}
	if err := ValidateHallucinationBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NormalizedBackend() != HallucinationBackendCandle {
		t.Errorf("NormalizedBackend() = %q, want candle", cfg.NormalizedBackend())
	}
}

func TestValidateHallucinationBackend_RejectsUnknownBackend(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "grpc"}
	if err := ValidateHallucinationBackend(cfg); err == nil {
		t.Fatalf("expected error for unknown backend, got nil")
	}
}

func TestValidateHallucinationBackend_EndpointRequiresEndpointURL(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "endpoint", ModelID: "m"}
	if err := ValidateHallucinationBackend(cfg); err == nil {
		t.Fatalf("expected error when endpoint is missing")
	}
}

func TestValidateHallucinationBackend_EndpointRejectsRelativeURL(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "endpoint", Endpoint: "127.0.0.1:8077/v1", ModelID: "m"}
	if err := ValidateHallucinationBackend(cfg); err == nil {
		t.Fatalf("expected error for non-absolute endpoint URL")
	}
}

func TestValidateHallucinationBackend_EndpointRejectsSurroundingWhitespace(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "endpoint", Endpoint: " http://127.0.0.1:8077/v1 ", ModelID: "m"}
	if err := ValidateHallucinationBackend(cfg); err == nil {
		t.Fatalf("expected error for endpoint with surrounding whitespace")
	}
}

func TestValidateHallucinationBackend_EndpointRequiresModelID(t *testing.T) {
	cfg := &HallucinationModelConfig{Backend: "endpoint", Endpoint: "http://127.0.0.1:8077/v1"}
	if err := ValidateHallucinationBackend(cfg); err == nil {
		t.Fatalf("expected error when model_id is missing")
	}
}

func TestValidateHallucinationBackend_EndpointValid(t *testing.T) {
	cfg := &HallucinationModelConfig{
		Backend:  "Endpoint",
		Endpoint: "http://127.0.0.1:8077/v1",
		ModelID:  "KRLabsOrg/lettucedect-v2-qwen-2b",
	}
	if err := ValidateHallucinationBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NormalizedBackend() != HallucinationBackendEndpoint {
		t.Errorf("NormalizedBackend() = %q, want endpoint", cfg.NormalizedBackend())
	}
}

func TestNormalizedBackend_DefaultsWhenEmpty(t *testing.T) {
	cfg := &HallucinationModelConfig{}
	if got := cfg.NormalizedBackend(); got != HallucinationBackendCandle {
		t.Errorf("NormalizedBackend() = %q, want candle", got)
	}
}

func TestCompileModelBindingsDesugarsLegacyHallucinationEndpoint(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel = HallucinationModelConfig{Backend: "endpoint", Endpoint: "http://127.0.0.1:8077/v1", ModelID: "detector"}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := plan.Lookup(DefaultRecipeName, "hallucination_detector")
	if !ok {
		t.Fatal("legacy backend: endpoint must compile into a hallucination_detector binding")
	}
	if spec.Deployment.Provider != "http" || spec.Deployment.ExternalModel != LegacyHallucinationEndpointModel ||
		spec.Binding.Adapter != RemoteClassifierProtocolHTTPChat || spec.Binding.Contract != RemoteClassifierContractTokenSpans {
		t.Fatalf("desugared binding = %+v", spec)
	}
	if got := LegacyHallucinationExternalModel(&cfg.HallucinationMitigation.HallucinationModel); got.ModelName != "detector" || got.ModelEndpoint.Address != "http://127.0.0.1:8077/v1" {
		t.Fatalf("external = %+v", got)
	}
}

func TestCompileModelBindingsCandleLeavesHallucinationUnbound(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel = HallucinationModelConfig{ModelID: "models/mom-halugate-detector"}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := plan.Lookup(DefaultRecipeName, "hallucination_detector"); ok {
		t.Fatal("the candle default is the module's canonical local binding, not a plan entry")
	}
}

func TestCompileModelBindingsExplicitHallucinationBindingWins(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel = HallucinationModelConfig{Backend: "endpoint", Endpoint: "http://127.0.0.1:8077/v1", ModelID: "detector"}
	cfg.ExternalModels = []ExternalModelConfig{{Name: "grounding", ModelName: "grounding-spans", ModelRole: ModelRoleClassification, ModelEndpoint: ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 9000}}}
	cfg.ModelDeployments = map[string]ModelDeployment{"grounding": {Provider: "http", ExternalModel: "grounding"}}
	cfg.ModelBindings = map[string]ModelBinding{"hallucination_detector": {Deployment: "grounding", Adapter: RemoteClassifierProtocolHTTPClassify, Contract: RemoteClassifierContractTokenSpans}}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, _ := plan.Lookup(DefaultRecipeName, "hallucination_detector")
	if spec.Deployment.ExternalModel != "grounding" || spec.Binding.Adapter != RemoteClassifierProtocolHTTPClassify {
		t.Fatalf("explicit binding must win over the legacy scalar, got %+v", spec)
	}
}

func TestCompileModelBindingsRejectsUnknownLegacyHallucinationBackend(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel = HallucinationModelConfig{Backend: "grpc", ModelID: "detector"}
	if _, err := CompileModelBindings(cfg); err == nil || !strings.Contains(err.Error(), "hallucination detector backend") {
		t.Fatalf("err = %v", err)
	}
}
