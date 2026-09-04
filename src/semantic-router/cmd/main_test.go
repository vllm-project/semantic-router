package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestRunRouterProcessShutsDownManagementServerWhenStartupIsCancelled(t *testing.T) {
	apiPort := freePort(t)
	routerPort := freePort(t)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configYAML := fmt.Sprintf(`version: v0.3
global:
  services:
    management_api:
      bind_address: 127.0.0.1
      port: %d
      remote_exposure: false
      auth:
        mode: disabled
`, apiPort)
	if err := os.WriteFile(configPath, []byte(configYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan error, 1)
	go func() {
		done <- runRouterProcess(ctx, runtimeOptions{
			configPath:  configPath,
			port:        routerPort,
			apiPort:     apiPort,
			apiBind:     "127.0.0.1",
			metricsPort: 0,
			enableAPI:   true,
		})
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("runRouterProcess() error = %v, want context canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("runRouterProcess() did not finish after startup cancellation")
	}

	requireListenerReleased(t, "management", fmt.Sprintf("127.0.0.1:%d", apiPort))
}

func TestRunRouterProcessLoadedShutdownIsBoundedAndReleasesListener(t *testing.T) {
	routerPort := freePort(t)
	metricsPort := freePort(t)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configYAML, err := os.ReadFile(filepath.Join("..", "..", "..", "e2e", "config", "config.agent-smoke.cpu.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, configYAML, 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- runRouterProcess(ctx, runtimeOptions{
			configPath:  configPath,
			port:        routerPort,
			metricsPort: metricsPort,
			enableAPI:   false,
		})
	}()

	address := fmt.Sprintf("127.0.0.1:%d", routerPort)
	waitForProcessListener(t, done, address)

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("runRouterProcess() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("loaded Router shutdown exceeded its bound")
	}

	requireListenerReleased(t, "ExtProc", address)
	requireListenerReleased(t, "metrics", fmt.Sprintf("127.0.0.1:%d", metricsPort))
}

func TestShutdownRouterProcessBoundsSlowHookAndPreservesErrors(t *testing.T) {
	hookErr := errors.New("hook failed")
	tracingErr := errors.New("tracing failed")
	hooks := []func(context.Context) error{
		func(ctx context.Context) error {
			<-ctx.Done()
			return errors.Join(hookErr, ctx.Err())
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	err := shutdownRouterProcess(ctx, nil, nil, nil, &hooks, func(context.Context) error {
		return tracingErr
	})
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("shutdownRouterProcess() took %s, want at most 1s", elapsed)
	}
	for _, want := range []error{hookErr, context.DeadlineExceeded, tracingErr} {
		if !errors.Is(err, want) {
			t.Errorf("shutdownRouterProcess() error = %v, want errors.Is(_, %v)", err, want)
		}
	}
}

func waitForProcessListener(t *testing.T, done <-chan error, address string) {
	t.Helper()
	startupDeadline := time.Now().Add(5 * time.Second)
	for {
		select {
		case err := <-done:
			t.Fatalf("Router exited before ExtProc became ready: %v", err)
		default:
		}
		connection, err := net.DialTimeout("tcp", address, 25*time.Millisecond)
		if err == nil {
			_ = connection.Close()
			return
		}
		if time.Now().After(startupDeadline) {
			t.Fatal("ExtProc listener did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func requireListenerReleased(t *testing.T, name, address string) {
	t.Helper()
	listener, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("%s listener remained open after shutdown: %v", name, err)
	}
	_ = listener.Close()
}

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
