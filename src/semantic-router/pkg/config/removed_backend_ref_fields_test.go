package config

import (
	"strings"
	"testing"
)

func TestRemovedBackendRefFieldsFailWithAHint(t *testing.T) {
	for _, field := range []string{"compat_gateway", "openai_compat_gateway"} {
		t.Run(field, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: claude-via-gateway
      provider_model_id: example-claude
      api_format: anthropic
      backend_refs:
        - name: gateway
          provider: anthropic-compatible
          endpoint: https://gateway.example.com/anthropic
          ` + field + `: true
recipes:
  - name: default
`))
			want := "providers.models[claude-via-gateway].backend_refs[0]: " + field +
				" was removed; gateway response fields are accepted by default, so delete it"
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, want)
			}
		})
	}
}

// Without the removed field the same backend loads: gateway fields need no
// setting.
func TestGatewayBackendNeedsNoSetting(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: claude-via-gateway
      provider_model_id: example-claude
      api_format: anthropic
      backend_refs:
        - name: gateway
          provider: anthropic-compatible
          endpoint: https://gateway.example.com/anthropic
recipes:
  - name: default
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if len(cfg.ProviderProfiles) != 1 {
		t.Fatalf("provider profiles = %+v, want one", cfg.ProviderProfiles)
	}
	for name, profile := range cfg.ProviderProfiles {
		if !strings.EqualFold(profile.Type, "anthropic-compatible") {
			t.Fatalf("profile %s type = %q", name, profile.Type)
		}
	}
}
