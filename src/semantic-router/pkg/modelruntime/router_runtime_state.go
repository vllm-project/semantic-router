package modelruntime

import (
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// EmbeddingRuntimeState records which embedding capabilities are ready after
// runtime preparation and the health of any remote provider.
type EmbeddingRuntimeState struct {
	AnyReady          bool
	ToolsReady        bool
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

type embeddingStateTracker struct {
	mu    sync.Mutex
	state EmbeddingRuntimeState
}

func newEmbeddingStateTracker(state EmbeddingRuntimeState) *embeddingStateTracker {
	return &embeddingStateTracker{state: cloneEmbeddingRuntimeState(state)}
}

func (t *embeddingStateTracker) markAnyReady() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.AnyReady = true
}

func (t *embeddingStateTracker) markToolsReady() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.AnyReady = true
	t.state.ToolsReady = true
}

func (t *embeddingStateTracker) markEmbeddingProvider(status *EmbeddingProviderRuntimeState) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.EmbeddingProvider = cloneEmbeddingProviderRuntimeState(status)
}

func (t *embeddingStateTracker) snapshot() EmbeddingRuntimeState {
	t.mu.Lock()
	defer t.mu.Unlock()
	return cloneEmbeddingRuntimeState(t.state)
}

func cloneEmbeddingRuntimeState(state EmbeddingRuntimeState) EmbeddingRuntimeState {
	state.EmbeddingProvider = cloneEmbeddingProviderRuntimeState(state.EmbeddingProvider)
	return state
}

func cloneEmbeddingProviderRuntimeState(status *EmbeddingProviderRuntimeState) *EmbeddingProviderRuntimeState {
	if status == nil {
		return nil
	}
	clone := *status
	if status.APIKeyEnvSet != nil {
		value := *status.APIKeyEnvSet
		clone.APIKeyEnvSet = &value
	}
	if status.Healthy != nil {
		value := *status.Healthy
		clone.Healthy = &value
	}
	return &clone
}

func remoteEmbeddingProviderRuntimeStateFromConfig(cfg *config.RouterConfig) *EmbeddingProviderRuntimeState {
	if cfg == nil || !cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return nil
	}

	apiKeyEnv := strings.TrimSpace(cfg.EmbeddingModels.Endpoint.APIKeyEnv)
	var apiKeyEnvSet *bool
	if apiKeyEnv != "" {
		value := os.Getenv(apiKeyEnv) != ""
		apiKeyEnvSet = &value
	}

	dimension := cfg.EmbeddingModels.Endpoint.Dimensions
	if dimension == 0 {
		dimension = cfg.EmbeddingModels.EmbeddingConfig.TargetDimension
	}

	return &EmbeddingProviderRuntimeState{
		Mode:         "remote",
		Backend:      cfg.EmbeddingModels.EmbeddingBackend(),
		Model:        strings.TrimSpace(cfg.EmbeddingModels.Endpoint.Model),
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
