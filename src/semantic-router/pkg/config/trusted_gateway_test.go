package config

import (
	"strings"
	"testing"
)

func trustedGatewayConfig(backendRefs string, enabled bool) string {
	gatewayCredential := ""
	if enabled {
		gatewayCredential = `
global:
  services:
    authz:
      gateway_credential:
        enabled: true
`
	}
	return `
version: v0.3
providers:
  models:
    - name: gateway-model
      provider_model_id: gateway-model
      api_format: openai
      backend_refs:
` + backendRefs + `
routing: {}
` + gatewayCredential
}

func TestTrustedGatewayBackendMaterializesWithoutAStaticKey(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(trustedGatewayConfig(`
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          trusted_gateway: true
`, true)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	profile, err := cfg.GetProviderProfileForEndpoint("gateway-model_gateway")
	if err != nil {
		t.Fatal(err)
	}
	if !profile.TrustedGateway {
		t.Fatalf("profile = %+v, want TrustedGateway", profile)
	}
	if key := cfg.GetModelAccessKey("gateway-model"); key != "" {
		t.Fatalf("a trusted gateway has no static key, got %q", key)
	}
	if !cfg.Authz.GatewayCredential.Enabled || cfg.Authz.GatewayCredential.GetHeader() != DefaultGatewayCredentialHeader {
		t.Fatalf("gateway credential contract = %+v", cfg.Authz.GatewayCredential)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	var ref *CanonicalBackendRef
	for _, model := range exported.Providers.Models {
		if model.Name == "gateway-model" && len(model.BackendRefs) == 1 {
			ref = &model.BackendRefs[0]
		}
	}
	if ref == nil {
		t.Fatalf("exported providers = %+v", exported.Providers.Models)
	}
	if !ref.TrustedGateway || ref.APIKey != "" || ref.APIKeyEnv != "" {
		t.Fatalf("exported backend ref = %+v, want trusted_gateway without a key", *ref)
	}
}

func TestTrustedGatewayBackendIsRejectedWhenTheContractIsOff(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(trustedGatewayConfig(`
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          trusted_gateway: true
`, false)))
	if err == nil || !strings.Contains(err.Error(), "gateway_credential.enabled") {
		t.Fatalf("error = %v, want the authz contract requirement", err)
	}
}

func TestTrustedGatewayBackendRejectsAStaticKey(t *testing.T) {
	for _, key := range []string{"api_key: shared-secret", "api_key_env: GATEWAY_KEY"} {
		_, err := ParseYAMLBytes([]byte(trustedGatewayConfig(`
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          trusted_gateway: true
          `+key+`
`, true)))
		if err == nil || !strings.Contains(err.Error(), "cannot combine with api_key") {
			t.Fatalf("%s: error = %v, want the static key rejection", key, err)
		}
	}
}

func TestTrustedGatewayBackendNeedsAnAuthHeader(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(trustedGatewayConfig(`
        - name: gateway
          provider: ollama
          endpoint: http://gateway.example:11434/v1
          trusted_gateway: true
`, true)))
	if err == nil || !strings.Contains(err.Error(), "needs an auth header") {
		t.Fatalf("error = %v, want the auth header requirement", err)
	}
}

func TestTrustedGatewayAndStaticBackendsDoNotShareACluster(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(trustedGatewayConfig(`
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          trusted_gateway: true
        - name: direct
          provider: openai-compatible
          endpoint: https://direct.example/v1
          api_key: static-secret
`, true)))
	if err == nil || !strings.Contains(err.Error(), "credential source differ") {
		t.Fatalf("error = %v, want the credential source difference", err)
	}
}

func TestGatewayCredentialHeaderIsLowercasedWithADefault(t *testing.T) {
	if got := (GatewayCredentialConfig{Enabled: true}).GetHeader(); got != "x-vsr-gateway-credential" {
		t.Fatalf("default header = %q", got)
	}
	if got := (GatewayCredentialConfig{Enabled: true, Header: " X-Gateway-Token "}).GetHeader(); got != "x-gateway-token" {
		t.Fatalf("configured header = %q", got)
	}
}
