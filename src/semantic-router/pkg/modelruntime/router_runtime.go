package modelruntime

import (
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type EmbeddingRuntimeState struct {
	Embeddings        *embedding.Set
	AnyReady          bool
	EmbeddingProvider *EmbeddingProviderRuntimeState
}

type EmbeddingProviderRuntimeState struct {
	Mode           string
	Backend        string
	Model          string
	Dimension      int
	APIKeyEnv      string
	APIKeyEnvSet   *bool
	Healthy        *bool
	LastProbeError string
	LastCheckedAt  string
}

func remoteEmbeddingProviderRuntimeStateFromConfig(cfg *config.RouterConfig) *EmbeddingProviderRuntimeState {
	if cfg == nil {
		return nil
	}
	// embeddings.default names the openai_embeddings deployment; a deprecated
	// embeddings.semantic.endpoint is read as one.
	cfg = config.WithModelResources(cfg)
	deployment, ok := cfg.ModelDeployments[cfg.EmbeddingDeployment]
	if !ok || deployment.Provider != config.OpenAIEmbeddingsProvider {
		return nil
	}
	apiKeyEnv := ""
	if deployment.Auth != nil {
		apiKeyEnv = strings.TrimSpace(deployment.Auth.APIKeyEnv)
	}
	var apiKeyEnvSet *bool
	if apiKeyEnv != "" {
		value := os.Getenv(apiKeyEnv) != ""
		apiKeyEnvSet = &value
	}
	dimension := deployment.DeclaredDimensions()
	if dimension == 0 {
		dimension = cfg.EmbeddingModels.EmbeddingConfig.TargetDimension
	}
	return &EmbeddingProviderRuntimeState{
		Mode:         "remote",
		Backend:      config.EmbeddingBackendOpenAICompatible,
		Model:        strings.TrimSpace(deployment.Model),
		Dimension:    dimension,
		APIKeyEnv:    apiKeyEnv,
		APIKeyEnvSet: apiKeyEnvSet,
	}
}

func remoteEmbeddingProviderProbeStatus(
	cfg *config.RouterConfig,
	provider embedding.Provider,
	dimension int,
	probeErr error,
) *EmbeddingProviderRuntimeState {
	status := remoteEmbeddingProviderRuntimeStateFromConfig(cfg)
	if status == nil {
		status = &EmbeddingProviderRuntimeState{Mode: "remote"}
	}
	if provider != nil {
		status.Backend = provider.Backend()
	}
	if dimension > 0 {
		status.Dimension = dimension
	}
	healthy := probeErr == nil
	status.Healthy = &healthy
	status.LastCheckedAt = time.Now().UTC().Format(time.RFC3339)
	if probeErr != nil {
		status.LastProbeError = probeErr.Error()
	}
	return status
}
