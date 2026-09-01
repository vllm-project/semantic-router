package config

import "strings"

const (
	EmbeddingBackendCandle           = "candle"
	EmbeddingBackendOpenVINO         = "openvino"
	EmbeddingBackendOpenAICompatible = "openai_compatible"

	EmbeddingModelTypeQwen3      = "qwen3"
	EmbeddingModelTypeRemote     = "remote"
	EmbeddingModelTypeMultiModal = "multimodal"
)

// IsMultiModalEmbeddingModelType reports whether embedding_config.model_type names
// the multimodal checkpoint. Only then does embedding_config.target_dimension
// describe the multimodal model; for every other model type it describes the text
// encoder and must not be checked against the multimodal ladder.
func IsMultiModalEmbeddingModelType(modelType string) bool {
	return strings.EqualFold(strings.TrimSpace(modelType), EmbeddingModelTypeMultiModal)
}

// EmbeddingEndpointConfig defines an external embedding provider endpoint.
type EmbeddingEndpointConfig struct {
	BaseURL          string `yaml:"base_url,omitempty"`
	Model            string `yaml:"model,omitempty"`
	APIKeyEnv        string `yaml:"api_key_env,omitempty"`
	TimeoutSeconds   int    `yaml:"timeout_seconds,omitempty"`
	MaxRetries       int    `yaml:"max_retries,omitempty"`
	MaxResponseBytes int64  `yaml:"max_response_bytes,omitempty"`
	Dimensions       int    `yaml:"dimensions,omitempty"`
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
