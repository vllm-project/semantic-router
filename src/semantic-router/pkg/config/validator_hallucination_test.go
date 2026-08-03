package config

import "testing"

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
