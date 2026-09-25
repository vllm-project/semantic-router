package config

import (
	"strings"
	"testing"
)

// remotePromptGuardConfig builds an enabled prompt_guard using a remote
// backend, plus the guardrail external model that backend needs.
func remotePromptGuardConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.Backend = &RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Contract: RemoteClassifierContractLabelDistribution,
		Model:    "guard",
	}
	cfg.PromptGuard.JailbreakMappingPath = "models/x/jailbreak_type_mapping.json"
	cfg.ExternalModels = []ExternalModelConfig{{
		Name:          "guard",
		Provider:      "openai",
		ModelRole:     ModelRoleGuardrail,
		ModelName:     "guard-model",
		ModelEndpoint: ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
	}}
	return cfg
}

func TestValidatePromptGuardBackend_RemoteBackendAcceptsAWiredGuardrail(t *testing.T) {
	if err := validatePromptGuardBackend(remotePromptGuardConfig()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// A remote backend with no matching external model makes
// IsPromptGuardEnabled() return false, which silently skips the whole
// jailbreak signal - so on_error: block becomes a no-op instead of failing
// closed. That must be a config error, not a silent downgrade.
func TestValidatePromptGuardBackend_RemoteBackendRequiresGuardrailModel(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.ExternalModels = nil

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when no external model matches backend.model")
	}
	if !strings.Contains(err.Error(), "guard") {
		t.Errorf("error %q should name backend.model guard", err)
	}
}

// A guardrail model whose role matches but whose llm_model_name is empty
// fails the same way: IsPromptGuardEnabled() returns false and the guardrail
// silently never runs.
func TestValidatePromptGuardBackend_RemoteBackendRequiresGuardrailModelName(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Backend.Protocol = RemoteClassifierProtocolHTTPChat
	cfg.PromptGuard.Backend.Contract = RemoteClassifierContractLabelDecision
	cfg.ExternalModels[0].ModelName = ""

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when the guardrail external model has no llm_model_name")
	}
	if !strings.Contains(err.Error(), "llm_model_name") {
		t.Errorf("error %q should name the missing field llm_model_name", err)
	}
}

func TestValidatePromptGuardBackend_RemoteBackendRequiresGuardrailAddress(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.ExternalModels[0].ModelEndpoint.Address = ""

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when the guardrail external model has no llm_endpoint.address")
	}
	if !strings.Contains(err.Error(), "llm_endpoint") {
		t.Errorf("error %q should name the missing field llm_endpoint.address", err)
	}
}

func TestValidatePromptGuardBackend_RequiresJailbreakMappingPath(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.JailbreakMappingPath = ""

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when enabled backend has no jailbreak_mapping_path")
	}
	if !strings.Contains(err.Error(), "jailbreak_mapping_path") {
		t.Errorf("error %q should name the missing field jailbreak_mapping_path", err)
	}
}

// The local (variant) backend needs no external model at all.
func TestValidatePromptGuardBackend_LocalVariantNeedsNoGuardrail(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Backend = nil
	cfg.PromptGuard.Variant = PromptGuardVariantMmBERT32K
	cfg.PromptGuard.ModelID = "models/jailbreak"
	cfg.ExternalModels = nil

	if err := validatePromptGuardBackend(cfg); err != nil {
		t.Fatalf("unexpected error for the local variant backend: %v", err)
	}
}

// The wiring checks are additive: the pre-existing backend-selection checks
// must still fire through the same entry point.
func TestValidatePromptGuardBackend_StillRejectsVariantAndBackendTogether(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Variant = PromptGuardVariantMmBERT32K

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when variant and backend are both set")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error %q should report the mutual exclusion", err)
	}
}

func TestValidatePromptGuardBackend_StillRejectsUnknownOnError(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.OnError = "fail"

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error for an unrecognized on_error value")
	}
	if !strings.Contains(err.Error(), "on_error") {
		t.Errorf("error %q should name on_error", err)
	}
}

func TestValidatePromptGuardNamedBackend(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Backend = &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPChat, Contract: RemoteClassifierContractLabelDecision, Model: "guard"}
	if err := validatePromptGuardBackend(cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.IsPromptGuardEnabled() {
		t.Fatal("named backend was silently disabled")
	}
	cfg.PromptGuard.Backend.Contract = RemoteClassifierContractLabelDistribution
	if err := validatePromptGuardBackend(cfg); err == nil {
		t.Fatal("chat cannot declare probabilities")
	}
	cfg.PromptGuard.Backend.Contract = RemoteClassifierContractLabelDecision
	cfg.PromptGuard.Backend.Model = "unknown"
	if err := validatePromptGuardBackend(cfg); err == nil {
		t.Fatal("guard fell back to first matching role")
	}
}

func TestCanonicalPromptGuardNamedBackendReplacesInheritedVariant(t *testing.T) {
	raw := MustStructuredPayload(map[string]interface{}{"model_catalog": map[string]interface{}{"modules": map[string]interface{}{"prompt_guard": map[string]interface{}{"backend": map[string]interface{}{"protocol": "http_chat", "model": "guard", "contract": "label_decision.v1"}}}}})
	model := PromptGuardConfig{Variant: PromptGuardVariantMmBERT32K}
	if err := normalizeCanonicalPromptGuardBackend(&model, raw); err != nil || model.Variant != "" {
		t.Fatalf("model=%+v err=%v", model, err)
	}
	legacy := MustStructuredPayload(map[string]interface{}{"model_catalog": map[string]interface{}{"modules": map[string]interface{}{"prompt_guard": map[string]interface{}{"protocol": "http_chat"}}}})
	if err := normalizeCanonicalPromptGuardBackend(&model, legacy); err == nil || !strings.Contains(err.Error(), "migrate") {
		t.Fatalf("legacy protocol should require migration: %v", err)
	}
}
