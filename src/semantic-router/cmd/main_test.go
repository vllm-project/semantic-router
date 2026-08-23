package main

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestStartupEmbeddingProviderStatusMapsRedactedRuntimeState(t *testing.T) {
	apiKeyEnvSet := true
	healthy := true
	status := startupEmbeddingProviderStatus(modelruntime.EmbeddingRuntimeState{
		EmbeddingProvider: &modelruntime.EmbeddingProviderRuntimeState{
			Mode:          "remote",
			Backend:       config.EmbeddingBackendOpenAICompatible,
			Model:         "text-embedding-3-small",
			Dimension:     1536,
			APIKeyEnv:     "OPENAI_API_KEY",
			APIKeyEnvSet:  &apiKeyEnvSet,
			Healthy:       &healthy,
			LastCheckedAt: "2026-07-08T00:00:00Z",
		},
	})

	if status == nil {
		t.Fatal("expected startup embedding provider status")
	}
	if status.APIKeyEnv != "OPENAI_API_KEY" {
		t.Fatalf("api key env = %q", status.APIKeyEnv)
	}
	if status.APIKeyEnvSet == nil || !*status.APIKeyEnvSet {
		t.Fatalf("api key env set = %v, want true", status.APIKeyEnvSet)
	}
	if status.Healthy == nil || !*status.Healthy {
		t.Fatalf("healthy = %v, want true", status.Healthy)
	}
}

func TestMarkRouterReadyIncludesEmbeddingProviderStatus(t *testing.T) {
	healthy := true
	writer := &recordingStartupWriter{}
	markRouterReady(writer, &startupstatus.EmbeddingProviderStatus{
		Mode:      "remote",
		Backend:   config.EmbeddingBackendOpenAICompatible,
		Model:     "text-embedding-3-small",
		Dimension: 1536,
		Healthy:   &healthy,
	})

	if writer.state.Phase != "ready" || !writer.state.Ready {
		t.Fatalf("ready state = %+v", writer.state)
	}
	if writer.state.EmbeddingProvider == nil {
		t.Fatal("expected embedding provider in ready state")
	}
	if writer.state.EmbeddingProvider.Model != "text-embedding-3-small" {
		t.Fatalf("embedding provider model = %q", writer.state.EmbeddingProvider.Model)
	}
}

type recordingStartupWriter struct {
	state startupstatus.State
}

func (w *recordingStartupWriter) Write(state startupstatus.State) error {
	w.state = state
	return nil
}
