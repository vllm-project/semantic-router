package config

import "strings"

const (
	EmbeddingBackendCandle           = "candle"
	EmbeddingBackendOpenVINO         = "openvino"
	EmbeddingBackendOpenAICompatible = "openai_compatible"

	EmbeddingModelTypeQwen3  = "qwen3"
	EmbeddingModelTypeRemote = "remote"
)

// EmbeddingEndpointConfig defines an external embedding provider endpoint.
type EmbeddingEndpointConfig struct {
	BaseURL        string `yaml:"base_url,omitempty"`
	Model          string `yaml:"model,omitempty"`
	APIKeyEnv      string `yaml:"api_key_env,omitempty"`
	TimeoutSeconds int    `yaml:"timeout_seconds,omitempty"`
	MaxRetries     int    `yaml:"max_retries,omitempty"`
	Dimensions     int    `yaml:"dimensions,omitempty"`
}

func (e EmbeddingModels) EmbeddingBackend() string {
	backend := normalizeEmbeddingBackend(e.EmbeddingConfig.Backend)
	if backend != "" {
		return backend
	}
	if strings.EqualFold(strings.TrimSpace(e.EmbeddingConfig.ModelType), EmbeddingModelTypeRemote) {
		return EmbeddingBackendOpenAICompatible
	}
	return EmbeddingBackendCandle
}

func (e EmbeddingModels) UsesRemoteEmbeddingBackend() bool {
	return e.EmbeddingBackend() == EmbeddingBackendOpenAICompatible
}

func (e EmbeddingEndpointConfig) IsConfigured() bool {
	return strings.TrimSpace(e.BaseURL) != "" || strings.TrimSpace(e.Model) != ""
}

func normalizeEmbeddingBackend(backend string) string {
	return strings.ToLower(strings.TrimSpace(backend))
}
