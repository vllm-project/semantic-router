package main

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestKubernetesUpdateProtectsPublishedArtifactBeforeDownload(t *testing.T) {
	restore := stubKubernetesUpdateSeams(t)
	defer restore()
	artifact := t.TempDir()
	makeConfig := func(revision string) *config.RouterConfig {
		cfg := &config.RouterConfig{MoMRegistry: map[string]string{artifact: "test/model"}}
		cfg.CategoryModel.ModelID = artifact
		cfg.CategoryMappingPath = filepath.Join(artifact, "labels.json")
		cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}}}
		cfg.ModelDeployments = map[string]config.ModelDeployment{"intent": {Provider: "candle", Artifact: artifact, Revision: revision}}
		cfg.ModelBindings = map[string]config.ModelBinding{"domain_classifier": {Deployment: "intent", Contract: "label_distribution.v1", Adapter: "mmbert32k"}}
		return cfg
	}
	current := makeConfig(strings.Repeat("a", 40))
	candidate := makeConfig(strings.Repeat("b", 40))
	ensureKubernetesConfigModels = func(context.Context, *config.RouterConfig, startupstatus.StatusWriter) error {
		t.Fatal("candidate download must not modify published artifacts")
		return nil
	}
	activate := func(context.Context, *config.RouterConfig) error {
		t.Fatal("unsafe candidate was published")
		return nil
	}
	err := applyKubernetesConfigUpdate(context.Background(), candidate, activate, nil, func() *config.RouterConfig { return current })
	if err == nil || !strings.Contains(err.Error(), "in use") {
		t.Fatalf("update error = %v, want live artifact rejection", err)
	}
}

func TestApplyKubernetesConfigUpdateEnsuresModelsBeforeReplace(t *testing.T) {
	restoreKubernetesUpdateSeams := stubKubernetesUpdateSeams(t)
	defer restoreKubernetesUpdateSeams()

	cfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	order := make([]string, 0, 2)

	ensureKubernetesConfigModels = func(_ context.Context, got *config.RouterConfig, _ startupstatus.StatusWriter) error {
		order = append(order, "ensure")
		if got != cfg {
			t.Fatalf("ensureKubernetesConfigModels() cfg = %p, want %p", got, cfg)
		}
		return nil
	}
	activate := func(_ context.Context, got *config.RouterConfig) error {
		order = append(order, "replace")
		if got != cfg {
			t.Fatalf("activation cfg = %p, want %p", got, cfg)
		}
		return nil
	}

	if err := applyKubernetesConfigUpdate(context.Background(), cfg, activate, nil); err != nil {
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
	ensureKubernetesConfigModels = func(_ context.Context, got *config.RouterConfig, _ startupstatus.StatusWriter) error {
		if got != cfg {
			t.Fatalf("ensureKubernetesConfigModels() cfg = %p, want %p", got, cfg)
		}
		return errors.New("download failed")
	}
	activate := func(_ context.Context, got *config.RouterConfig) error {
		t.Fatalf("activation should not be called on ensure failure")
		return nil
	}

	err := applyKubernetesConfigUpdate(context.Background(), cfg, activate, nil)
	if err == nil {
		t.Fatal("applyKubernetesConfigUpdate() error = nil, want failure")
	}
	if got := err.Error(); got != "failed to ensure models for kubernetes config update: download failed" {
		t.Fatalf("applyKubernetesConfigUpdate() error = %q", got)
	}
}

func TestApplyKubernetesConfigUpdateDoesNotPublishAfterCancellation(t *testing.T) {
	restoreKubernetesUpdateSeams := stubKubernetesUpdateSeams(t)
	defer restoreKubernetesUpdateSeams()

	ctx, cancel := context.WithCancel(context.Background())
	ensureKubernetesConfigModels = func(context.Context, *config.RouterConfig, startupstatus.StatusWriter) error {
		cancel()
		return nil
	}
	activate := func(context.Context, *config.RouterConfig) error {
		t.Fatal("activation called after cancellation")
		return nil
	}

	err := applyKubernetesConfigUpdate(ctx, &config.RouterConfig{}, activate, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("applyKubernetesConfigUpdate() error = %v, want context canceled", err)
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

	return func() {
		ensureKubernetesConfigModels = originalEnsure
	}
}

func TestApplyKubernetesConfigUpdateReturnsActivationFailure(t *testing.T) {
	restore := stubKubernetesUpdateSeams(t)
	defer restore()
	ensureKubernetesConfigModels = func(context.Context, *config.RouterConfig, startupstatus.StatusWriter) error { return nil }
	failure := errors.New("candidate warmup failed")
	err := applyKubernetesConfigUpdate(context.Background(), &config.RouterConfig{}, func(context.Context, *config.RouterConfig) error { return failure }, nil)
	if !errors.Is(err, failure) {
		t.Fatalf("activation failure = %v", err)
	}
}

func TestKubernetesNamespaceUsesPodNamespace(t *testing.T) {
	t.Setenv("POD_NAMESPACE", "router-test")
	if got := kubernetesNamespaceDefault(); got != "router-test" {
		t.Fatal(got)
	}
	t.Setenv("POD_NAMESPACE", "")
	if got := kubernetesNamespaceDefault(); got != "default" {
		t.Fatal(got)
	}
}
