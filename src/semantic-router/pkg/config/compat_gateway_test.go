package config

import (
	"strings"
	"testing"
)

func compatGatewayConfig(apiFormat, backendRefs string) []byte {
	return []byte(`
version: v0.3
providers:
  models:
    - name: relay
      provider_model_id: relay-model
      api_format: ` + apiFormat + `
      backend_refs:
` + backendRefs + `
routing: {}
`)
}

func TestCompatGatewayReachesTheProviderProfile(t *testing.T) {
	tests := []struct {
		name        string
		apiFormat   string
		backendRefs string
	}{
		{
			name: "anthropic backend pointed at a Messages gateway", apiFormat: "anthropic",
			backendRefs: `
        - provider: anthropic
          base_url: https://gateway.example.com/anthropic
          compat_gateway: true
`,
		},
		{
			name: "anthropic-compatible gateway", apiFormat: "anthropic",
			backendRefs: `
        - provider: anthropic-compatible
          base_url: https://gateway.example.com/anthropic
          compat_gateway: true
`,
		},
		{
			name: "openai-compatible relay", apiFormat: "openai",
			backendRefs: `
        - provider: openai-compatible
          base_url: https://relay.example.com/v1
          compat_gateway: true
`,
		},
		{
			name: "deprecated spelling on a vllm relay", apiFormat: "openai",
			backendRefs: `
        - provider: vllm
          base_url: https://relay.example.com/v1
          openai_compat_gateway: true
`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes(compatGatewayConfig(test.apiFormat, test.backendRefs))
			if err != nil {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}
			if len(cfg.ProviderProfiles) != 1 {
				t.Fatalf("provider profiles = %+v, want one", cfg.ProviderProfiles)
			}
			for name, profile := range cfg.ProviderProfiles {
				if !profile.CompatGateway {
					t.Fatalf("profile %s lost compat_gateway: %+v", name, profile)
				}
			}
		})
	}
}

func TestCompatGatewayIsRejectedWhereItHasNoEffect(t *testing.T) {
	tests := []struct {
		name        string
		apiFormat   string
		backendRefs string
		want        string
	}{
		{
			name: "provider with its own response contract", apiFormat: "openai",
			backendRefs: `
        - provider: azure-openai
          base_url: https://my-resource.openai.azure.com/openai/v1
          compat_gateway: true
`,
			want: "providers.models[relay].backend_refs[0]: compat_gateway has no effect on azure-openai backends",
		},
		{
			name: "deprecated spelling on a provider with its own contract", apiFormat: "anthropic",
			backendRefs: `
        - provider: snowflake-cortex
          base_url: https://my-account.snowflakecomputing.com/api/v2/cortex/v1
          openai_compat_gateway: true
`,
			want: "compat_gateway has no effect on snowflake-cortex backends",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(compatGatewayConfig(test.apiFormat, test.backendRefs))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestValidateCompatGatewayRejectsWireFormatsWithoutARelayContract(t *testing.T) {
	err := ValidateCompatGateway("systemone-compatible", "vllm-sr/systemone@1")
	if err == nil || !strings.Contains(err.Error(), "only to OpenAI and Anthropic wire formats") {
		t.Fatalf("ValidateCompatGateway() error = %v", err)
	}
	for _, protocol := range []string{"openai/chat-completions@1", "openai/responses@1", "anthropic/messages@1"} {
		if err := ValidateCompatGateway("vllm", protocol); err != nil {
			t.Fatalf("ValidateCompatGateway(vllm, %s) error = %v", protocol, err)
		}
	}
}

// Export writes the canonical spelling, whichever one the source used.
func TestCompatGatewayExportsTheCanonicalSpelling(t *testing.T) {
	ref := canonicalBackendRefFromRuntime("relay", VLLMEndpoint{Name: "relay_primary"}, "", ProviderProfile{
		Type: "anthropic", BaseURL: "https://gateway.example.com/anthropic", CompatGateway: true,
	})
	if !ref.CompatGateway || ref.OpenAICompatGateway {
		t.Fatalf("exported backend ref = %+v, want only compat_gateway set", ref)
	}
}
