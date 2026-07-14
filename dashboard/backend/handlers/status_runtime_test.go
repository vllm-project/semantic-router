package handlers

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestRuntimeStatusFromStateIncludesEmbeddingProviderStatus(t *testing.T) {
	apiKeyEnvSet := false
	healthy := false
	state := &startupstatus.State{
		Phase:   "ready",
		Ready:   true,
		Message: "Router ready",
		EmbeddingProvider: &startupstatus.EmbeddingProviderStatus{
			Mode:           "remote",
			Backend:        config.EmbeddingBackendOpenAICompatible,
			Model:          "text-embedding-3-small",
			Dimension:      1536,
			APIKeyEnv:      "VLLM_SR_EMBEDDING_API_KEY",
			APIKeyEnvSet:   &apiKeyEnvSet,
			Healthy:        &healthy,
			LastProbeError: "embedding API key env is not set",
			LastCheckedAt:  "2026-07-08T00:00:00Z",
		},
	}

	runtime := runtimeStatusFromState(state)
	if runtime.EmbeddingProvider == nil {
		t.Fatal("expected embedding provider status")
	}
	if runtime.EmbeddingProvider == state.EmbeddingProvider {
		t.Fatal("expected embedding provider status to be cloned")
	}
	if runtime.EmbeddingProvider.APIKeyEnv != "VLLM_SR_EMBEDDING_API_KEY" {
		t.Fatalf("api key env = %q", runtime.EmbeddingProvider.APIKeyEnv)
	}
	if runtime.EmbeddingProvider.APIKeyEnvSet == nil || *runtime.EmbeddingProvider.APIKeyEnvSet {
		t.Fatalf("api key env set = %v, want false", runtime.EmbeddingProvider.APIKeyEnvSet)
	}
	if runtime.EmbeddingProvider.Healthy == nil || *runtime.EmbeddingProvider.Healthy {
		t.Fatalf("healthy = %v, want false", runtime.EmbeddingProvider.Healthy)
	}
	if runtime.EmbeddingProvider.Dimension != 1536 {
		t.Fatalf("dimension = %d, want 1536", runtime.EmbeddingProvider.Dimension)
	}
}
