package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNormalizeEmbeddingProviderURLCanonicalizesWithoutSecrets(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "case and credentials",
			raw:  " HTTPS://user:password@EXAMPLE.COM/ModelPath/?api_key=query-secret#fragment ",
			want: "https://example.com/ModelPath",
		},
		{
			name: "root path",
			raw:  "http://EXAMPLE.COM///",
			want: "http://example.com",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := normalizeEmbeddingProviderURL(test.raw)
			if got != test.want {
				t.Fatalf("normalizeEmbeddingProviderURL(%q) = %q, want %q", test.raw, got, test.want)
			}
			for _, secret := range []string{"password", "query-secret", "fragment"} {
				if strings.Contains(got, secret) {
					t.Fatalf("normalized URL contains secret %q: %q", secret, got)
				}
			}
		})
	}
}

func TestNormalizeEmbeddingProviderURLMalformedFallbackRedactsSecrets(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "invalid escape",
			raw:  " https://user:password@example.com/%zz?api_key=query-secret#fragment ",
			want: "https://example.com/%zz",
		},
		{
			name: "schemeless endpoint",
			raw:  "user:password@example.com/%zz?api_key=query-secret",
			want: "example.com/%zz",
		},
		{
			name: "malformed scheme",
			raw:  "https:/user:password@example.com/%zz#fragment",
			want: "https:/example.com/%zz",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := normalizeEmbeddingProviderURL(test.raw)
			if got != test.want {
				t.Fatalf("malformed URL fallback = %q, want %q", got, test.want)
			}
			for _, secret := range []string{"password", "query-secret", "fragment"} {
				if strings.Contains(got, secret) {
					t.Fatalf("malformed URL fallback contains secret %q: %q", secret, got)
				}
			}
		})
	}
}

func TestToolsEmbeddingProviderIdentityDoesNotIncludeEndpointSecrets(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					Backend:         config.EmbeddingBackendOpenAICompatible,
					ModelType:       config.EmbeddingModelTypeRemote,
					TargetDimension: 3,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL:    "https://user:password@example.com/ModelPath/?api_key=query-secret",
					Model:      "embedding-model",
					APIKeyEnv:  "EMBEDDING_API_KEY",
					Dimensions: 3,
				},
			},
		},
	}

	identity := toolsEmbeddingProviderIdentity(cfg)
	if strings.Contains(identity, "password") || strings.Contains(identity, "query-secret") {
		t.Fatalf("provider identity contains endpoint secret: %q", identity)
	}
	if !strings.Contains(identity, "https://example.com/ModelPath") {
		t.Fatalf("provider identity lost the canonical endpoint: %q", identity)
	}
}
