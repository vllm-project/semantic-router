package modelruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingRuntimeTasksUseOnlyRemoteProviderWhenConfigured(t *testing.T) {
	cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
	paths := resolveEmbeddingPaths(cfg)

	_, tasks, _ := embeddingRuntimeTasks(cfg, "test", paths)
	if len(tasks) != 1 {
		t.Fatalf("embeddingRuntimeTasks() returned %d task(s), want 1", len(tasks))
	}
	if tasks[0].Name != "router.embedding.remote_provider" {
		t.Fatalf("task name = %q, want router.embedding.remote_provider", tasks[0].Name)
	}
}

func TestEmbeddingRuntimeTasksRemoteConfigWithCandleOverrideUsesLocalTask(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendCandle)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "mmbert")
	cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
	paths := resolveEmbeddingPaths(cfg)

	_, tasks, _ := embeddingRuntimeTasks(cfg, "test", paths)
	if len(tasks) != 1 || tasks[0].Name != "router.embedding.unified_factory" {
		t.Fatalf("local override tasks = %+v, want only router.embedding.unified_factory", tasks)
	}
}

func TestPrepareRouterRuntimeOpenVINOOverrideDoesNotClaimReadyOrProbeRemote(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenVINO)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "mmbert")
	cfg := remoteEmbeddingRuntimeConfig("http://must-not-be-contacted.invalid/v1")

	state, err := PrepareRouterRuntime(context.Background(), cfg, PrepareRouterRuntimeOptions{Component: "test"})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime: %v", err)
	}
	if state.AnyReady || state.ToolsReady || state.EmbeddingProvider != nil {
		t.Fatalf("OpenVINO handoff state = %+v, want explicitly not-ready without remote provider", state)
	}
}

func TestPrepareRouterRuntimeProbesRemoteEmbeddingProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeRuntimeEmbeddingResponse(t, w, []float64{0.1, 0.2})
	}))
	defer server.Close()
	t.Setenv("REMOTE_EMBEDDING_API_KEY", "test-secret")
	cfg := remoteEmbeddingRuntimeConfig(server.URL + "/v1")
	cfg.EmbeddingModels.Endpoint.APIKeyEnv = "REMOTE_EMBEDDING_API_KEY"

	state, err := PrepareRouterRuntime(context.Background(), cfg, PrepareRouterRuntimeOptions{
		Component:      "test-router",
		MaxParallelism: 1,
	})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() error = %v", err)
	}
	if !state.AnyReady || !state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %+v, want AnyReady and ToolsReady", state)
	}
	assertReadyRemoteEmbeddingProviderStatus(t, state.EmbeddingProvider)
}

func TestPrepareRouterRuntimeReportsRemoteEmbeddingProbeFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad key", http.StatusUnauthorized)
	}))
	defer server.Close()

	var failedEvent *Event
	state, err := PrepareRouterRuntime(context.Background(), remoteEmbeddingRuntimeConfig(server.URL+"/v1"), PrepareRouterRuntimeOptions{
		Component:      "test-router",
		MaxParallelism: 1,
		OnEvent: func(event Event) {
			if event.Task == "router.embedding.remote_provider" && event.Status == TaskFailed {
				copyEvent := event
				failedEvent = &copyEvent
			}
		},
	})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() returned error for best-effort remote failure: %v", err)
	}
	if state.AnyReady || state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %+v, want not ready", state)
	}
	assertFailedRemoteEmbeddingProviderStatus(t, state.EmbeddingProvider)
	assertRemoteEmbeddingFailureEvent(t, failedEvent)
}

func assertReadyRemoteEmbeddingProviderStatus(t *testing.T, provider *EmbeddingProviderRuntimeState) {
	t.Helper()
	if provider == nil {
		t.Fatal("expected embedding provider status")
	}
	if provider.Mode != "remote" || provider.Backend != config.EmbeddingBackendOpenAICompatible {
		t.Fatalf("embedding provider status = %+v, want remote openai-compatible", provider)
	}
	if provider.Model != "BAAI/bge-m3" {
		t.Fatalf("embedding provider model = %q", provider.Model)
	}
	if provider.APIKeyEnv != "REMOTE_EMBEDDING_API_KEY" {
		t.Fatalf("embedding provider api key env = %q", provider.APIKeyEnv)
	}
	assertRemoteEmbeddingProviderDimensionAndTimestamp(t, provider)
	assertBoolPtr(t, provider.APIKeyEnvSet, true, "embedding provider api key env set")
	assertBoolPtr(t, provider.Healthy, true, "embedding provider healthy")
	if provider.LastProbeError != "" {
		t.Fatalf("embedding provider last probe error = %q, want empty", provider.LastProbeError)
	}
}

func assertFailedRemoteEmbeddingProviderStatus(t *testing.T, provider *EmbeddingProviderRuntimeState) {
	t.Helper()
	if provider == nil {
		t.Fatal("expected embedding provider status")
	}
	assertRemoteEmbeddingProviderDimensionAndTimestamp(t, provider)
	assertBoolPtr(t, provider.Healthy, false, "embedding provider healthy")
	if provider.LastProbeError == "" {
		t.Fatal("expected embedding provider last probe error")
	}
}

func assertRemoteEmbeddingProviderDimensionAndTimestamp(t *testing.T, provider *EmbeddingProviderRuntimeState) {
	t.Helper()
	if provider.Dimension != 2 {
		t.Fatalf("embedding provider dimension = %d, want 2", provider.Dimension)
	}
	if provider.LastCheckedAt == "" {
		t.Fatal("expected embedding provider last checked timestamp")
	}
}

func assertBoolPtr(t *testing.T, got *bool, want bool, label string) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s = %v, want %v", label, got, want)
	}
}

func assertRemoteEmbeddingFailureEvent(t *testing.T, failedEvent *Event) {
	t.Helper()
	if failedEvent == nil || failedEvent.Error == nil {
		t.Fatal("expected failed remote provider event with error")
	}
	if !strings.Contains(failedEvent.Error.Error(), "authentication failed") {
		t.Fatalf("remote provider event error = %v, want authentication failure", failedEvent.Error)
	}
}

func remoteEmbeddingRuntimeConfig(baseURL string) *config.RouterConfig {
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
				EmbeddingConfig: config.HNSWConfig{
					Backend:         config.EmbeddingBackendOpenAICompatible,
					ModelType:       config.EmbeddingModelTypeRemote,
					TargetDimension: 2,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL: baseURL,
					Model:   "BAAI/bge-m3",
				},
			},
		},
	}
}

func writeRuntimeEmbeddingResponse(t *testing.T, w http.ResponseWriter, embedding []float64) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"data": []map[string]interface{}{
			{"index": 0, "embedding": embedding},
		},
	}); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}
