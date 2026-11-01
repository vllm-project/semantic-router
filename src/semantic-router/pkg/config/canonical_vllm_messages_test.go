package config

import "testing"

// vllm serve answers /v1/messages behind the same bearer token as its OpenAI
// routes, so a Messages model on a vLLM backend keeps the vllm provider, which
// is also the provider a backend ref gets when it names none.
func TestCanonicalVLLMBackendServesAnthropicMessages(t *testing.T) {
	for _, provider := range []string{"", "vllm"} {
		providerLine := ""
		if provider != "" {
			providerLine = "\n          provider: " + provider
		}
		cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: agent
      provider_model_id: qwen35-35b-a3b
      api_format: anthropic
      backend_refs:
        - endpoint: 127.0.0.1:18102
          protocol: http` + providerLine + `
          api_key: static-secret
routing: {}
`))
		if err != nil {
			t.Fatalf("provider %q: %v", provider, err)
		}
		profile, ok := cfg.ProviderProfiles["agent_primary"]
		if !ok {
			t.Fatalf("provider %q: materialized provider profile is missing", provider)
		}
		if profile.Type != "vllm" || profile.Protocol != "anthropic/messages@1" {
			t.Fatalf("provider %q: profile = (%q, %q), want vllm over Messages", provider, profile.Type, profile.Protocol)
		}
		header, prefix, err := profile.ResolveAuthHeader()
		if err != nil || header != "Authorization" || prefix != "Bearer" {
			t.Fatalf("provider %q: auth = (%q, %q, %v), want the bearer token vLLM checks", provider, header, prefix, err)
		}
		path, err := profile.ResolveCreatePath("")
		if err != nil || path != "/v1/messages" {
			t.Fatalf("provider %q: create path = %q, %v", provider, path, err)
		}
	}
}
