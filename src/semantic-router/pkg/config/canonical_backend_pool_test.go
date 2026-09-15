package config

import (
	"fmt"
	"strings"
	"testing"
)

func TestCanonicalBackendPoolAcceptsHomogeneousReplicas(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(canonicalBackendPoolConfig(`
        - name: primary
          provider: vllm
          endpoint: 10.0.0.1:8000/v1
          protocol: http
          api_key_env: POOL_API_KEY
          weight: 75
        - name: secondary
          provider: vllm
          endpoint: 10.0.0.2:8001/v1
          protocol: http
          api_key_env: POOL_API_KEY
          weight: 25
`)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if got := len(cfg.GetEndpointsForModel("replica-pool")); got != 2 {
		t.Fatalf("replica endpoint count = %d, want 2", got)
	}
}

func TestCanonicalBackendPoolRejectsHeterogeneousRequestSemantics(t *testing.T) {
	tests := []struct {
		name        string
		backendRefs string
		want        string
	}{
		{
			name: "provider",
			backendRefs: `
        - name: primary
          provider: vllm
          endpoint: 10.0.0.1:8000
        - name: secondary
          provider: sglang
          endpoint: 10.0.0.2:8000
`,
			want: "provider differ",
		},
		{
			name: "base path",
			backendRefs: `
        - name: primary
          provider: vllm
          endpoint: 10.0.0.1:8000/v1
        - name: secondary
          provider: vllm
          endpoint: 10.0.0.2:8000/compatible/v1
`,
			want: "base path",
		},
		{
			name: "credential source",
			backendRefs: `
        - name: primary
          provider: vllm
          endpoint: 10.0.0.1:8000
          api_key_env: PRIMARY_API_KEY
        - name: secondary
          provider: vllm
          endpoint: 10.0.0.2:8000
          api_key_env: SECONDARY_API_KEY
`,
			want: "credential source",
		},
		{
			name: "extra headers",
			backendRefs: `
        - name: primary
          provider: vllm
          endpoint: 10.0.0.1:8000
          extra_headers:
            X-Tenant: one
        - name: secondary
          provider: vllm
          endpoint: 10.0.0.2:8000
          extra_headers:
            X-Tenant: two
`,
			want: "extra headers",
		},
		{
			name: "TLS server name",
			backendRefs: `
        - name: primary
          provider: vllm
          base_url: https://api-1.example.test/v1
        - name: secondary
          provider: vllm
          base_url: https://api-2.example.test/v1
`,
			want: "TLS server name",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(canonicalBackendPoolConfig(test.backendRefs)))
			if err == nil {
				t.Fatal("ParseYAMLBytes() unexpectedly accepted heterogeneous pool")
			}
			if !strings.Contains(err.Error(), "cannot share one Envoy cluster") ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %q, want heterogeneous %s error", err, test.want)
			}
		})
	}
}

func TestCanonicalBackendPoolErrorDoesNotExposeCredential(t *testing.T) {
	const firstSecret = "first-secret-must-not-leak"
	const secondSecret = "second-secret-must-not-leak"
	_, err := ParseYAMLBytes([]byte(canonicalBackendPoolConfig(fmt.Sprintf(`
        - name: primary
          provider: vllm
          endpoint: 10.0.0.1:8000
          api_key: %s
        - name: secondary
          provider: vllm
          endpoint: 10.0.0.2:8000
          api_key: %s
`, firstSecret, secondSecret))))
	if err == nil {
		t.Fatal("ParseYAMLBytes() unexpectedly accepted distinct credentials")
	}
	if strings.Contains(err.Error(), firstSecret) || strings.Contains(err.Error(), secondSecret) {
		t.Fatalf("ParseYAMLBytes() leaked a credential in error: %q", err)
	}
	if !strings.Contains(err.Error(), "credential source") {
		t.Fatalf("ParseYAMLBytes() error = %q, want credential source", err)
	}
}

func canonicalBackendPoolConfig(backendRefs string) string {
	return `
version: v0.3
providers:
  defaults:
    model: replica-pool
  models:
    - name: replica-pool
      provider_model_id: private-model
      api_format: openai
      backend_refs:
` + backendRefs + `
routing: {}
`
}
