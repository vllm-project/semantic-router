package config

import (
	"strings"
	"testing"
)

func TestValidateEmbeddingContracts_AcceptsRemoteEmbeddingProviderLimits(t *testing.T) {
	cfg := remoteEmbeddingRouterConfig(EmbeddingEndpointConfig{
		BaseURL:        "https://example.test/v1",
		Model:          "text-embedding-3-small",
		APIKeyEnv:      EmbeddingAPIKeyEnvName,
		TimeoutSeconds: EmbeddingEndpointMaxTimeoutSeconds,
		MaxRetries:     EmbeddingEndpointMaxRetries,
	})
	if err := validateEmbeddingContracts(cfg); err != nil {
		t.Fatalf("validateEmbeddingContracts() error = %v", err)
	}
}

func TestValidateEmbeddingContracts_RejectsRemoteEmbeddingProviderLimits(t *testing.T) {
	tests := []struct {
		name     string
		endpoint EmbeddingEndpointConfig
		want     string
	}{
		{
			name: "unsupported URL scheme",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "ftp://example.test/v1", Model: "embedding-model",
			},
			want: "endpoint.base_url",
		},
		{
			name: "URL userinfo",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "https://user:password@example.test/v1", Model: "embedding-model",
			},
			want: "endpoint.base_url must not include userinfo",
		},
		{
			name: "URL query",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "https://example.test/v1?api_key=secret", Model: "embedding-model",
			},
			want: "endpoint.base_url must not include query or fragment",
		},
		{
			name: "URL fragment",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "https://example.test/v1#private", Model: "embedding-model",
			},
			want: "endpoint.base_url must not include query or fragment",
		},
		{
			name: "unrelated secret environment",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "https://example.test/v1", Model: "embedding-model",
				APIKeyEnv: "VLLM_SR_LOOPER_SHARED_SECRET",
			},
			want: "endpoint.api_key_env must be VLLM_SR_EMBEDDING_API_KEY",
		},
		{
			name: "credential over HTTP",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "http://example.test/v1", Model: "embedding-model",
				APIKeyEnv: EmbeddingAPIKeyEnvName,
			},
			want: "endpoint.base_url must use https",
		},
		{
			name: "timeout above limit",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "https://example.test/v1", Model: "embedding-model",
				TimeoutSeconds: EmbeddingEndpointMaxTimeoutSeconds + 1,
			},
			want: "endpoint.timeout_seconds must not exceed",
		},
		{
			name: "retries above limit",
			endpoint: EmbeddingEndpointConfig{
				BaseURL: "https://example.test/v1", Model: "embedding-model",
				MaxRetries: EmbeddingEndpointMaxRetries + 1,
			},
			want: "endpoint.max_retries must not exceed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateEmbeddingContracts(remoteEmbeddingRouterConfig(tt.endpoint))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateEmbeddingContracts() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestValidateEmbeddingContracts_DoesNotExposeRemoteEndpointURL(t *testing.T) {
	const sensitiveMarker = "DO-NOT-EXPOSE-URL-VALUE"
	for _, baseURL := range []string{
		"https://user:" + sensitiveMarker + "@example.test/v1",
		"https://example.test/v1?token=" + sensitiveMarker,
		"https://example.test/v1#" + sensitiveMarker,
	} {
		cfg := remoteEmbeddingRouterConfig(EmbeddingEndpointConfig{
			BaseURL: baseURL, Model: "embedding-model",
		})
		err := validateEmbeddingContracts(cfg)
		if err == nil {
			t.Fatal("validateEmbeddingContracts() returned nil error")
		}
		if strings.Contains(err.Error(), sensitiveMarker) || strings.Contains(err.Error(), baseURL) {
			t.Fatalf("validateEmbeddingContracts() leaked base_url content: %v", err)
		}
	}
}

func remoteEmbeddingRouterConfig(endpoint EmbeddingEndpointConfig) *RouterConfig {
	return &RouterConfig{
		InlineModels: InlineModels{
			EmbeddingModels: EmbeddingModels{
				EmbeddingConfig: HNSWConfig{
					Backend: EmbeddingBackendOpenAICompatible, ModelType: EmbeddingModelTypeRemote,
				},
				Endpoint: endpoint,
			},
		},
	}
}
