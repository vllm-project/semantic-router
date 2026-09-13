package config

import (
	"strings"
	"testing"
)

// remotePromptGuardConfig builds an enabled prompt_guard using a remote
// protocol, plus the guardrail external model that protocol needs.
func remotePromptGuardConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.Protocol = PromptGuardProtocolHTTPClassify
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

func TestValidatePromptGuardBackend_RemoteProtocolAcceptsAWiredGuardrail(t *testing.T) {
	if err := validatePromptGuardBackend(remotePromptGuardConfig()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// A remote protocol with no guardrail external model makes
// IsPromptGuardEnabled() return false, which silently skips the whole
// jailbreak signal - so on_error: block becomes a no-op instead of failing
// closed. That must be a config error, not a silent downgrade.
func TestValidatePromptGuardBackend_RemoteProtocolRequiresGuardrailModel(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.ExternalModels = nil

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when no external model has model_role: guardrail")
	}
	if !strings.Contains(err.Error(), ModelRoleGuardrail) {
		t.Errorf("error %q should name the missing model_role: %s", err, ModelRoleGuardrail)
	}
}

// A guardrail model whose role matches but whose llm_model_name is empty
// fails the same way: IsPromptGuardEnabled() returns false and the guardrail
// silently never runs.
func TestValidatePromptGuardBackend_RemoteProtocolRequiresGuardrailModelName(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.ExternalModels[0].ModelName = ""

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when the guardrail external model has no llm_model_name")
	}
	if !strings.Contains(err.Error(), "llm_model_name") {
		t.Errorf("error %q should name the missing field llm_model_name", err)
	}
}

func TestValidatePromptGuardBackend_RemoteProtocolRequiresGuardrailAddress(t *testing.T) {
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

// An empty jailbreak_mapping_path is the same class of fail-open, but the
// operator path reaches it through a serialization bug rather than an author's
// mistake (see validatePromptGuardWiring's comment), so requiring it here would
// break every operator deployment at startup. Guard the deliberate omission so
// nobody re-adds the check without fixing that first.
func TestValidatePromptGuardBackend_DoesNotRequireJailbreakMappingPath(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.JailbreakMappingPath = ""

	if err := validatePromptGuardBackend(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// A disabled prompt_guard carries no wiring requirement.
func TestValidatePromptGuardBackend_DisabledNeedsNoGuardrail(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Enabled = false
	cfg.ExternalModels = nil
	cfg.PromptGuard.JailbreakMappingPath = ""

	if err := validatePromptGuardBackend(cfg); err != nil {
		t.Fatalf("unexpected error for a disabled prompt_guard: %v", err)
	}
}

// The local (variant) backend needs no external model at all.
func TestValidatePromptGuardBackend_LocalVariantNeedsNoGuardrail(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Protocol = ""
	cfg.PromptGuard.Variant = PromptGuardVariantMmBERT32K
	cfg.PromptGuard.ModelID = "models/jailbreak"
	cfg.ExternalModels = nil

	if err := validatePromptGuardBackend(cfg); err != nil {
		t.Fatalf("unexpected error for the local variant backend: %v", err)
	}
}

// The wiring checks are additive: the pre-existing backend-selection checks
// must still fire through the same entry point.
func TestValidatePromptGuardBackend_StillRejectsVariantAndProtocolTogether(t *testing.T) {
	cfg := remotePromptGuardConfig()
	cfg.PromptGuard.Variant = PromptGuardVariantMmBERT32K

	err := validatePromptGuardBackend(cfg)
	if err == nil {
		t.Fatal("expected an error when variant and protocol are both set")
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
	cfg.PromptGuard.Protocol = ""
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
