package config

import "strings"

const (
	EmbeddingBackendCandle           = "candle"
	EmbeddingBackendOpenVINO         = "openvino"
	EmbeddingBackendOpenAICompatible = "openai_compatible"

	EmbeddingModelTypeQwen3  = "qwen3"
	EmbeddingModelTypeRemote = "remote"

	// EmbeddingEndpointMaxTimeoutSeconds bounds conversion to time.Duration and
	// prevents a stalled provider from holding router work indefinitely.
	EmbeddingEndpointMaxTimeoutSeconds = 3600
	// EmbeddingEndpointMaxRetries bounds total attempts and retry backoff work.
	EmbeddingEndpointMaxRetries = 10
	// EmbeddingAPIKeyEnvName is the only credential source accepted by the
	// remote embedding client. A dedicated name prevents the endpoint config
	// from repurposing unrelated process secrets as outbound credentials.
	// #nosec G101 -- This constant is an environment variable name, not a credential.
	EmbeddingAPIKeyEnvName = "VLLM_SR_EMBEDDING_API_KEY"
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

// IsValidEmbeddingAPIKeyEnv reports whether name is the dedicated remote
// embedding credential source.
func IsValidEmbeddingAPIKeyEnv(name string) bool {
	return name == EmbeddingAPIKeyEnvName
}

func normalizeEmbeddingBackend(backend string) string {
	return strings.ToLower(strings.TrimSpace(backend))
}
