package main

import (
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestApplyKubernetesConfigUpdateEnsuresModelsBeforeReplace(t *testing.T) {
	restoreKubernetesUpdateSeams := stubKubernetesUpdateSeams(t)
	defer restoreKubernetesUpdateSeams()

	cfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	order := make([]string, 0, 2)

	ensureKubernetesConfigModels = func(got *config.RouterConfig) error {
		order = append(order, "ensure")
		if got != cfg {
			t.Fatalf("ensureKubernetesConfigModels() cfg = %p, want %p", got, cfg)
		}
		return nil
	}
	replaceKubernetesRuntimeConfig = func(got *config.RouterConfig) {
		order = append(order, "replace")
		if got != cfg {
			t.Fatalf("replaceKubernetesRuntimeConfig() cfg = %p, want %p", got, cfg)
		}
	}

	if err := applyKubernetesConfigUpdate(cfg); err != nil {
		t.Fatalf("applyKubernetesConfigUpdate() error = %v", err)
	}

	wantOrder := []string{"ensure", "replace"}
	if !reflect.DeepEqual(order, wantOrder) {
		t.Fatalf("applyKubernetesConfigUpdate() order = %v, want %v", order, wantOrder)
	}
}

func TestApplyKubernetesConfigUpdateSkipsReplaceOnEnsureFailure(t *testing.T) {
	restoreKubernetesUpdateSeams := stubKubernetesUpdateSeams(t)
	defer restoreKubernetesUpdateSeams()

	cfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	ensureKubernetesConfigModels = func(got *config.RouterConfig) error {
		if got != cfg {
			t.Fatalf("ensureKubernetesConfigModels() cfg = %p, want %p", got, cfg)
		}
		return errors.New("download failed")
	}
	replaceKubernetesRuntimeConfig = func(got *config.RouterConfig) {
		t.Fatalf("replaceKubernetesRuntimeConfig() should not be called on ensure failure")
	}

	err := applyKubernetesConfigUpdate(cfg)
	if err == nil {
		t.Fatal("applyKubernetesConfigUpdate() error = nil, want failure")
	}
	if got := err.Error(); got != "failed to ensure models for kubernetes config update: download failed" {
		t.Fatalf("applyKubernetesConfigUpdate() error = %q", got)
	}
}

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

func stubKubernetesUpdateSeams(t *testing.T) func() {
	t.Helper()

	originalEnsure := ensureKubernetesConfigModels
	originalReplace := replaceKubernetesRuntimeConfig

	return func() {
		ensureKubernetesConfigModels = originalEnsure
		replaceKubernetesRuntimeConfig = originalReplace
	}
}
